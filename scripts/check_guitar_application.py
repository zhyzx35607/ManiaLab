"""Run complete real guitar imports/inference/exports through local source HTTP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.server import create_app
from scripts.verify_portable_guitar import check_guitar, sha
from werkzeug.serving import make_server


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--probe', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    work = args.output.resolve()
    if work.exists():
        raise ValueError('Preserve prior evidence; use a new output directory')
    work.mkdir(parents=True)
    app = http = thread = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    checks, passed = [], False

    def start():
        nonlocal app, http, thread
        app = create_app(ROOT, work/'app')
        http = make_server('127.0.0.1', 0, app, threaded=True)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()

    def stop():
        nonlocal app, http, thread
        if http is not None:
            http.shutdown(); thread.join(); http.server_close()
            app.extensions['executor'].shutdown()
            app = http = thread = None

    def request(path, payload=None, upload=None):
        body, headers = None, {}
        if upload:
            field, name, content = upload
            boundary = 'ManiaLabGuitarVerification'
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
                'Content-Type: application/octet-stream\r\n\r\n').encode()+content+f'\r\n--{boundary}--\r\n'.encode()
            headers['Content-Type'] = 'multipart/form-data; boundary='+boundary
        elif payload is not None:
            body = json.dumps(payload).encode()
            headers['Content-Type'] = 'application/json'
        if body is not None:
            headers['X-ManiaLab-Token'] = app.config['TOKEN']
        with opener.open(urllib.request.Request(f'http://127.0.0.1:{http.server_port}'+path,
                data=body, headers=headers), timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if 'application/json' in response.headers.get('Content-Type', '') else raw

    def wait(job, timeout=240):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            current = request('/api/jobs/'+job['id'])
            if current['status'] in ('done', 'error', 'cancelled'):
                return current
            time.sleep(.25)
        raise TimeoutError('Guitar application check exceeded job timeout')

    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT/'manialab').glob('*.py'))}
    try:
        start()
        check_guitar(request, wait, start, stop, work, args.probe, checks)
        assert all(sha(ROOT/p) == digest for p, digest in sources.items()), 'Sources changed during verification'
        passed = True
    finally:
        stop()
        (work/'report.json').write_text(json.dumps(dict(passed=passed, sources=sources,
            probe_report_sha256=sha(args.probe/'report.json'), checks=checks,
            packaged=False, musical_acceptance=False), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
