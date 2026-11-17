"""Disposable synthetic project for UI checks; never runs a neural model."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.server import create_app
from manialab.timing import grid_diagnostics
from waitress import serve


def timing_fixture(projects, title):
    pid, folder = projects.create()
    with wave.open(str(folder / 'files/synthetic.wav'), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(22050)
        output.writeframes(b'\0\0' * 22050 * 12)
    shutil.copyfile(folder / 'files/synthetic.wav', folder / 'preview.wav')
    notes = [Note(1115+i*500, 1865+i*500, pitch=60+i%8, source='pitch', id=f'timing{i}') for i in range(20)]
    projects.save(dict(id=pid, title=title, artist='Fixture', mode='audio', settings=parameters({'ln_ratio':1}),
        variants=[], active=-1, charts=[], audio='synthetic.wav', original=None, input_notes=[], audio_shift_ms=22.5,
        analysis=dict(revision=3, duration=12, candidates=[n.data() for n in notes],
            timing_attacks=[n.data() for n in notes], waveform=[0]*100,
            tempo=dict(bpm=120, offset=115, division=4, stable=True,
                reliability='起音与网格整体一致；仍需试听确认',
                sections=grid_diagnostics(notes,120,115,4)))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='manialab-ui-') as work:
        app = create_app(ROOT, work)
        projects = app.extensions['projects']
        pid, folder = projects.create()
        with wave.open(str(folder / 'files/synthetic.wav'), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(22050)
            output.writeframes(b'\0\0' * 22050 * 12)
        notes = [Note(1000 + i * 500, 1250 + i * 500, pitch=60+i%8, source='pitch', id=str(i)).data() for i in range(20)]
        projects.save(dict(id=pid, title='UI synthetic', artist='Fixture', mode='audio', settings=parameters(),
            variants=[], active=-1, charts=[], audio='synthetic.wav', original=None, input_notes=[], audio_shift_ms=0,
            analysis=dict(revision=3, duration=12, candidates=notes, timing_attacks=notes,
                tempo=dict(bpm=120, offset=0, division=4, stable=True), waveform=[0]*100)))
        timing_fixture(projects, 'UI timing clock')
        timing_fixture(projects, 'UI timing wave')
        timing_fixture(projects, 'UI timing saved')
        timing_fixture(projects, 'UI timing style')
        try:
            serve(app, host='127.0.0.1', port=args.port, threads=4)
        finally:
            app.extensions['executor'].shutdown()


if __name__ == '__main__':
    main()
