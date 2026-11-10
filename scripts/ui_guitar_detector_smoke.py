"""Exercise optional guitar controls in a real browser, without neural inference."""
from contextlib import ExitStack
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flask import jsonify, request
from manialab import server as application
from werkzeug.serving import make_server


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    folder = args.output.resolve()
    if folder.exists() or not folder.is_relative_to((ROOT / 'test-data').resolve()):
        raise ValueError('Use a new local test directory')
    folder.mkdir(parents=True)
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(16000 * 2))
    fixture = folder / 'fixture.wav'
    fixture.write_bytes(stream.getvalue())
    files = ['manialab/server.py', 'static/app.js', 'static/index.html',
             'static/guitar-detector.js', 'static/provided-stems.js',
             'scripts/ui_guitar_detector_smoke.py', 'scripts/ui_guitar_detector_smoke.cjs']
    identities = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    calls, controls = [], {'mode': 'success'}

    def analyze(path, work, model, settings, *args, **kwargs):
        deep = len(args) == 3
        separator, progress, cancelled = args if deep else (None, *args)
        if isinstance(separator, application.ProvidedStemSeparator):
            separation = separator.report
        elif separator is not None:
            separation = dict(source='model', profile=separator.profile,
                              voices=['drums', 'bass', 'vocals', 'other', 'guitar', 'piano']
                              if separator.profile == 'six' else ['drums', 'bass', 'vocals', 'other'])
        else:
            separation = None
        selected = kwargs.get('guitar_detector') is not None
        releases = kwargs.get('allow_guitar_predicted_releases', False)
        engine = ('provided' if separation and separation.get('source') == 'provided' else
                  'deep6' if separation and separation.get('profile') == 'six' else
                  'deep' if deep else 'basic')
        calls.append(dict(engine=engine, guitar_engine='fl' if selected else 'general',
                          allow_guitar_predicted_releases=releases, mode=controls['mode']))
        mode = controls['mode']
        progress(20, '界面测试：等待模拟吉他采音')
        deadline = time.monotonic() + (20 if mode == 'cancel' else .5)
        while time.monotonic() < deadline:
            if cancelled():
                raise InterruptedError('界面测试已取消')
            time.sleep(.02)
        if mode == 'cancel':
            raise ValueError('The UI did not cancel the simulated job')
        if mode == 'error':
            raise ValueError('界面测试模拟采音失败')
        result = dict(revision=8, duration=1, candidates=[], phrase_candidates=[], waveform=[],
                      tempo=dict(bpm=120, offset=0, stable=True, division=4), cached=False,
                      elapsed_seconds=.5, guitar_engine='fl' if selected else 'general',
                      allow_guitar_predicted_releases=releases,
                      piano_engine='specialist' if kwargs.get('piano_detector') else 'general',
                      instrument_engine='tsumugi' if kwargs.get('instrument_detector') else 'general',
                      allow_predicted_releases=kwargs.get('allow_predicted_releases', False),
                      voices=[dict(voice=v, candidates=0) for v in separation['voices']] if separation else [])
        if separation:
            result['separation'] = separation
        return result

    def provided(mix, sources, work, progress, cancelled):
        # Upload and persistence are real; this smoke does not decode or transcribe.
        return dict(source='provided', voices=list(sources), duration_seconds=1,
                    sample_rate=16000, set_sha256='synthetic-guitar-ui-fixture')

    with ExitStack() as stack:
        stack.enter_context(patch.object(application, 'analyze', analyze))
        stack.enter_context(patch.object(application, 'analyze_deep', analyze))
        stack.enter_context(patch.object(application, 'prepare_provided_stems', provided))
        stack.enter_context(patch.object(application.GuitarDetector, 'status', lambda self:
                                        dict(available=True, default=False, predicted_releases_default=False)))
        stack.enter_context(patch.object(application.InstrumentDetector, 'status',
                                        lambda self: dict(available=True, default=False)))
        stack.enter_context(patch.object(application.PianoDetector, 'status', lambda self: dict(available=True)))
        stack.enter_context(patch.object(application.Separator, 'status',
                                        lambda self: dict(available=True, profiles={'six': {'available': True}})))
        app = application.create_app(ROOT, folder)

        @app.post('/__smoke/mode')
        def mode():
            value = request.json['mode']
            if value not in ('success', 'cancel', 'error'):
                raise ValueError('Unknown smoke mode')
            controls['mode'] = value
            return jsonify(ok=True)

        host = make_server('127.0.0.1', 0, app, threaded=True)
        thread = threading.Thread(target=host.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(['node', str(ROOT / 'scripts/ui_guitar_detector_smoke.cjs'),
                                     f'http://127.0.0.1:{host.server_port}', str(folder), str(fixture)],
                                    capture_output=True, text=True, encoding='utf-8', timeout=240,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            (folder / 'browser.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
            (folder / 'backend-calls.json').write_text(json.dumps(calls, indent=2), encoding='utf-8')
            if result.returncode:
                raise ValueError(result.stdout + '\n' + result.stderr)
            report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
            actual = [{k: row[k] for k in ('engine', 'guitar_engine', 'allow_guitar_predicted_releases')}
                      for row in calls]
            if actual != report['submitted_options']:
                raise ValueError('Browser request and backend routing differed')
            if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                   for name, digest in identities.items()):
                raise ValueError('The UI or harness changed during verification')
            report.update(backend_routing_equal=True, backend_calls=calls, code_sha256=identities)
            (folder / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            host.shutdown()
            thread.join()
            host.server_close()
            app.extensions['executor'].shutdown()


if __name__ == '__main__':
    main()
