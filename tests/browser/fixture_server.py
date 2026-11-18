"""Disposable synthetic project for UI checks; never runs a neural model."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.generation import generate_variant
from manialab import server
from manialab.planner import validate_plan
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
    return pid


def precise_timing_fixture(projects):
    pid = timing_fixture(projects, 'UI timing precision')
    state = projects.load(pid)
    state['analysis']['tempo'].update(
        bpm=120,
        offset=115.25,
        stable=False,
        candidates=[dict(bpm=120, offset=115.25, division=4, coverage=.95)],
        precise_candidates=[dict(
            bpm=120.37, offset=115.25, division=4, coverage=.95,
            precision_suggestion=True, rounding_drift_ms=37,
        )],
    )
    projects.save(state)
    return pid


def audio_meter_fixture(projects):
    pid = timing_fixture(projects, 'UI audio meter')
    state = projects.load(pid)
    beats = [115 + i * 500 for i in range(24)]
    state['analysis']['separation'] = dict(profile='four', beat_evidence=dict(beats_ms=beats, downbeats_ms=beats[::3]))
    projects.save(state)


def ln_remap_fixture(projects):
    pid = timing_fixture(projects, 'UI LN navigation remap')
    state = projects.load(pid)
    state.update(mode='remap', input_keys=10, input_timing=[
        '115,500,3', '1615,-50,4,2,0,70,0,0', '3115,400,2,2,0,70,1,0', '3115,250,5,2,0,70,1,0'],
        input_notes=[dict(n, source='import', lane=i%10)
                     for i,n in enumerate(state['analysis']['candidates'])])
    projects.save(state)


def evidence_fixture(projects, title):
    pid, folder = projects.create()
    with wave.open(str(folder / 'files/synthetic.wav'), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(22050)
        output.writeframes(b'\0\0' * 22050 * 12)
    shutil.copyfile(folder / 'files/synthetic.wav', folder / 'preview.wav')
    stem_folder = folder / 'stems-provided' / ('a' * 64)
    stem_folder.mkdir(parents=True)
    for voice in ('other', 'piano'):
        shutil.copyfile(folder / 'preview.wav', stem_folder / f'{voice}.wav')
    notes = [Note(1040+i*500, 1710+i*500, pitch=60+i%8, source='pitch', voice='other',
        id=f'evidence{i}', release_reason='model_predicted_offset') for i in range(14)]
    notes += [Note(9000, 9600, source='pitch', voice='other', id='unobserved', release_reason='input_boundary'),
              Note(10000, 10250, source='pitch', voice='piano', id='unknown')]
    projects.save(dict(id=pid, title=title, artist='Fixture', mode='audio', settings=parameters({'ln_ratio':1}),
        variants=[], active=-1, charts=[], audio='synthetic.wav', original=None, input_notes=[], audio_shift_ms=22.5,
        analysis=dict(revision=13, duration=12, candidates=[n.data() for n in notes], phrase_candidates=[],
            timing_attacks=[n.data() for n in notes], waveform=[0]*100,
            voices=[dict(voice=v) for v in ('other', 'piano')],
            separation=dict(source='provided', voices=['other', 'piano'], set_sha256='a'*64),
            structure=dict(boundaries=[0,12000], segments=[dict(start_ms=0,end_ms=12000)]),
            tempo=dict(bpm=120, offset=0, division=4, stable=True))))
    return pid


def legacy_region_fixture(projects):
    pid = evidence_fixture(projects, 'UI evidence legacy region')
    state = projects.load(pid)
    generate_variant(state, dict(mode='audio', timing_confirmed=True), state['settings'],
        dict(bpm=120, offset=-22, division=4, audio_shift_ms=22.5), lambda *_:None, lambda:False)
    # Model a saved legacy chart followed by reanalysis. Only this disposable
    # fixture changes; the browser then exercises the real regional API.
    del state['variants'][state['active']]['source_evidence']
    next(n for n in state['analysis']['candidates'] if n['id']=='evidence0')['end'] = 2310
    projects.save(state)


def plan_fixture(projects, title):
    pid = evidence_fixture(projects, title)
    state = projects.load(pid)
    state['analysis']['segments'] = state['analysis']['structure']['segments']
    state['plan'] = dict(preferences={'density':4}, segments=[], explanation='Existing fixture plan')
    projects.save(state)


def regional_audit_fixture(projects, title):
    pid = timing_fixture(projects, title)
    state = projects.load(pid)
    notes = [Note(1000, 1800, id='locked', source='pitch', voice='piano'),
             Note(3500, 4200, id='region', source='pitch', voice='bass')]
    state['audio_shift_ms'] = 0
    state['analysis'].update(duration=5, candidates=[n.data() for n in notes],
        timing_attacks=[n.data() for n in notes], phrase_candidates=[],
        voices=[dict(voice=v) for v in ('piano', 'bass')],
        structure=dict(boundaries=[0,3000,5000],
            segments=[dict(start_ms=0,end_ms=3000),dict(start_ms=3000,end_ms=5000)]),
        tempo=dict(bpm=120, offset=0, division=4, stable=True))
    projects.save(state)
    return pid


def reanalyzed_regional_fixture(projects):
    pid = regional_audit_fixture(projects, 'UI regional audit reanalyzed')
    state = projects.load(pid)
    generate_variant(state, dict(mode='audio', timing_confirmed=True), state['settings'],
        dict(bpm=120, offset=0, division=4, audio_shift_ms=0), lambda *_:None, lambda:False)
    # Simulate a later analysis reusing the old ordinal ID for another onset.
    # The UI then performs the real regional generate/save operation.
    current = next(n for n in state['analysis']['candidates'] if n['id']=='locked')
    current.update(start=3000, end=3300, pitch=72)
    state['analysis']['timing_attacks'] = state['analysis']['candidates']
    projects.save(state)


def install_fixture_planner(app):
    # Replace only this disposable server's external call. Real plan HTTP,
    # queued jobs, cancellation, validation and project commits still execute.
    waiting, release = threading.Event(), threading.Event()
    def fixture_plan(endpoint, model, key, instruction, segments, current, review=None):
        if endpoint != 'fixture://planner' or model not in ('success', 'error', 'cancel'):
            raise ValueError('Fixture planner rejects external requests')
        if model == 'error':
            raise ValueError('Deterministic fixture planner error')
        if model == 'cancel':
            release.clear()
            waiting.set()
            if not release.wait(15):
                raise ValueError('Fixture planner release timed out')
            waiting.clear()
        return validate_plan(dict(preferences={'density':11}, segments=[],
            explanation='Deterministic fixture plan'), segments)
    app.extensions['planner'] = fixture_plan
    @app.get('/__fixture/planner-state')
    def planner_state():
        return dict(waiting=waiting.is_set())
    @app.post('/__fixture/planner-release')
    def planner_release():
        release.set()
        return dict(ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='manialab-ui-') as work:
        app = create_app(ROOT, work)
        install_fixture_planner(app)
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
        precise_timing_fixture(projects)
        audio_meter_fixture(projects)
        timing_fixture(projects, 'UI LN navigation audio')
        ln_remap_fixture(projects)
        for title in ('UI evidence review', 'UI evidence edits', 'UI evidence snapshots',
                      'UI draft expression', 'UI draft structure', 'UI draft recommendation'):
            evidence_fixture(projects, title)
        legacy_region_fixture(projects)
        for action in ('clear', 'success', 'error', 'cancel', 'scope'):
            plan_fixture(projects, f'UI plan {action}')
        for action in ('current', 'legacy', 'full'):
            regional_audit_fixture(projects, f'UI regional audit {action}')
        reanalyzed_regional_fixture(projects)
        try:
            serve(app, host='127.0.0.1', port=args.port, threads=4)
        finally:
            app.extensions['executor'].shutdown()


if __name__ == '__main__':
    main()
