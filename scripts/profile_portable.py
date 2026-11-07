"""Measure the released EXE on a caller-supplied song without uploading audio.

Requires psutil in the evaluation environment, not in the application.
The work folder includes the private song/project and must not be published.
Process-tree RSS sums may count shared pages more than once; private bytes
are recorded separately. Polling produces observed peaks, not hard bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request


def run(executable, audio, work, generate_bpm=None, generate_offset=0):
    import psutil

    executable, audio, work = executable.resolve(), audio.resolve(), work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    data = work / 'isolated-data'
    data.mkdir(exist_ok=True)
    marker = data / 'current-url.txt'
    marker.unlink(missing_ok=True)
    environment = {k: v for k, v in os.environ.items() if k not in (
        'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'MANIALAB_DEEP_PYTHON', 'MANIALAB_DEEP_MODEL')}
    environment['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32')
    environment.update(HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9',
                       NO_PROXY='127.0.0.1,localhost', PYTHONUTF8='1')
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    stop = threading.Event()
    samples = []
    stage = 'startup'
    started = time.monotonic()
    process = None
    monitor = None
    with (work / 'launch.log').open('w', encoding='utf-8') as log:
        try:
            process = subprocess.Popen([str(executable), '--no-browser', '--data-dir', str(data)],
                cwd=work, env=environment, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
            parent = psutil.Process(process.pid)

            def observe():
                while not stop.is_set():
                    rss = private = 0
                    try:
                        children = parent.children(recursive=True)
                    except psutil.Error:
                        break
                    for p in [parent, *children]:
                        try:
                            memory = p.memory_info()
                            rss += memory.rss
                            private += getattr(memory, 'private', 0)
                        except psutil.Error:
                            pass
                    samples.append(dict(seconds=round(time.monotonic() - started, 2), stage=stage,
                                        rss_bytes=rss, private_bytes=private, processes=1 + len(children)))
                    stop.wait(.25)

            monitor = threading.Thread(target=observe, daemon=True)
            monitor.start()
            deadline = time.monotonic() + 60
            while not marker.exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Portable startup failed; inspect launch.log')
                time.sleep(.1)
            url = marker.read_text(encoding='utf-8').strip()
            while True:
                try:
                    html = http.open(url, timeout=2).read().decode()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.1)
            token = re.search(r'name="manialab-token" content="([^"]+)"', html).group(1)

            def request(path, payload=None, upload=False):
                headers = {}
                body = None
                if upload:
                    boundary = 'ManiaLabResourceProfileBoundary'
                    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="profile{audio.suffix}"\r\n'
                            'Content-Type: application/octet-stream\r\n\r\n').encode() + audio.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
                    headers['Content-Type'] = 'multipart/form-data; boundary=' + boundary
                elif payload is not None:
                    body = json.dumps(payload).encode()
                    headers['Content-Type'] = 'application/json'
                if body is not None:
                    headers['X-ManiaLab-Token'] = token
                with http.open(urllib.request.Request(url + path, data=body, headers=headers), timeout=30) as response:
                    value = response.read()
                    return json.loads(value) if 'application/json' in response.headers.get('Content-Type', '') else value

            def wait(job):
                deadline = time.monotonic() + 3600
                last_notice = time.monotonic()
                while time.monotonic() < deadline:
                    row = request('/api/jobs/' + job['id'])
                    if row['status'] in ('done', 'error', 'cancelled'):
                        if row['status'] != 'done':
                            raise RuntimeError(str(row))
                        return
                    if time.monotonic() - last_notice > 25:
                        print(json.dumps(dict(stage=stage, elapsed=round(time.monotonic()-started),
                                              message=row.get('message'))), flush=True)
                        last_notice = time.monotonic()
                    time.sleep(.25)
                raise TimeoutError('Profile job exceeded one hour')

            status = request('/api/status')
            project = request('/api/import', upload=True)
            base = '/api/projects/' + project['id']
            runs = []
            for label in ('cold_analysis', 'cached_analysis'):
                stage = label
                begin = time.monotonic()
                wait(request(base + '/analyze', {'engine': 'deep', 'settings': {}, 'audio_shift_ms': 22.5}))
                state = request(base)
                analysis = state['analysis']
                elapsed = time.monotonic() - begin
                if bool(analysis['cached']) != (label == 'cached_analysis'):
                    raise AssertionError('Unexpected analysis cache state')
                measured = [s for s in samples if s['stage'] == label]
                runs.append(dict(stage=label, seconds=round(elapsed, 2), candidates=len(analysis['candidates']),
                    peak_tree_rss_bytes=max(s['rss_bytes'] for s in measured),
                    peak_tree_private_bytes=max(s['private_bytes'] for s in measured),
                    maximum_processes=max(s['processes'] for s in measured), cached=analysis['cached']))
                print(json.dumps(runs[-1]), flush=True)
            project_dir = data / 'projects' / project['id']
            generation = None
            if generate_bpm is not None:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from manialab.domain import parse_osu, validate
                from manialab.timing import grid_report
                stage = 'generate_export'
                begin = time.monotonic()
                tempo = dict(bpm=generate_bpm, offset=generate_offset, division=4)
                wait(request(base + '/generate', dict(mode='audio', settings={'keys': 10},
                    audio_shift_ms=22.5, timing_confirmed=True, **tempo)))
                exported = request(base + '/export/osu').decode('utf-8')
                notes = parse_osu(exported)['notes']
                structure = validate(notes, 10)
                grid = grid_report(notes, tempo)
                assert structure['valid'] and grid['off_grid'] == 0
                (work / 'generated.osu').write_text(exported, encoding='utf-8')
                measured = [s for s in samples if s['stage'] == stage]
                generation = dict(seconds=round(time.monotonic() - begin, 2), keys=10,
                    supplied_timing=tempo, structure=structure, grid=grid,
                    peak_tree_rss_bytes=max(s['rss_bytes'] for s in measured),
                    peak_tree_private_bytes=max(s['private_bytes'] for s in measured))
                print(json.dumps(generation), flush=True)
            report = dict(version=status['version'], duration_seconds=analysis['duration'],
                executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                logical_cpu_count=psutil.cpu_count(), installed_memory_bytes=psutil.virtual_memory().total,
                sample_interval_seconds=.25, runs=runs,
                project_disk_bytes=sum(p.stat().st_size for p in project_dir.rglob('*') if p.is_file()),
                measurement_note='Single-host observed process-tree peaks; shared RSS pages may be counted more than once. Browser is not started. No guaranteed minimum RAM or universal runtime claim.',
                generation=generation, samples=samples)
            (work / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in report.items() if k != 'samples'}), flush=True)
        finally:
            stop.set()
            if monitor:
                monitor.join(timeout=2)
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--generate-bpm', type=float)
    parser.add_argument('--generate-offset', type=int, default=0)
    args = parser.parse_args()
    if os.name != 'nt':
        raise SystemExit('This profiler is for the Windows distribution.')
    run(args.executable, args.audio, args.work, args.generate_bpm, args.generate_offset)
