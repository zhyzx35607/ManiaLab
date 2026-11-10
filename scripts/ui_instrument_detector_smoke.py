"""Exercise specialist controls in a real browser; model computation is mocked."""
from contextlib import ExitStack
import argparse
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
    calls, controls = [], {'mode': 'success'}

    def analyze(path, work, model, settings, *args, **kwargs):
        deep = len(args) == 3
        separator, progress, cancelled = args if deep else (None, *args)
        if isinstance(separator, application.ProvidedStemSeparator):
            separation = separator.report
        elif separator is not None:
            separation = dict(source='model', profile=separator.profile,
                              voices=['bass', 'other', 'piano'] if separator.profile == 'six' else ['bass', 'other'])
        else:
            separation = None
        selected = kwargs.get('instrument_detector') is not None
        releases = kwargs.get('allow_predicted_releases', False)
        calls.append(dict(engine='provided' if separation and separation.get('source') == 'provided' else
                          'deep6' if separation and separation.get('profile') == 'six' else 'deep' if deep else 'basic',
                          instrument_engine='tsumugi' if selected else 'general',
                          allow_predicted_releases=releases, mode=controls['mode']))
        mode = controls['mode']
        progress(20, '界面测试：等待模拟采音')
        deadline = time.monotonic() + (20 if mode == 'cancel' else .4)
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
                      elapsed_seconds=.4, instrument_engine='tsumugi' if selected else 'general',
                      allow_predicted_releases=releases,
                      piano_engine='specialist' if kwargs.get('piano_detector') else 'general',
                      voices=[dict(voice=v, candidates=0) for v in separation['voices']] if separation else [])
        if separation:
            result['separation'] = separation
        return result

    def provided(mix, sources, work, progress, cancelled):
        # The UI/import API is real; decoding and model execution are outside this smoke.
        return dict(source='provided', voices=list(sources), duration_seconds=1,
                    sample_rate=16000, set_sha256='synthetic-ui-fixture')

    with ExitStack() as stack:
        stack.enter_context(patch.object(application, 'analyze', analyze))
        stack.enter_context(patch.object(application, 'analyze_deep', analyze))
        stack.enter_context(patch.object(application, 'prepare_provided_stems', provided))
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
            result = subprocess.run(['node', str(ROOT / 'scripts/ui_instrument_detector_smoke.cjs'),
                                     f'http://127.0.0.1:{host.server_port}', str(folder), str(fixture)],
                                    capture_output=True, text=True, encoding='utf-8', timeout=180,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            (folder / 'browser.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
            (folder / 'backend-calls.json').write_text(json.dumps(calls, indent=2), encoding='utf-8')
            if result.returncode:
                raise ValueError(result.stdout + '\n' + result.stderr)
            report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
            actual = [{k: row[k] for k in ('engine', 'instrument_engine', 'allow_predicted_releases')}
                      for row in calls]
            if actual != report['submitted_options']:
                raise ValueError('Browser request and backend routing differed')
            report['backend_routing_equal'] = True
            report['backend_calls'] = calls
            (folder / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            host.shutdown()
            thread.join()
            host.server_close()
            app.extensions['executor'].shutdown()


if __name__ == '__main__':
    main()
