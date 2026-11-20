"""Disposable synthetic project for UI checks; never runs a neural model."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from manialab.arrange import parameters
from manialab.chart_workspace import audio_work
from manialab.domain import Note
from manialab.generation import generate_variant
from manialab import server
from manialab.planner import validate_plan
from manialab.server import create_app
from manialab.timing import grid_diagnostics
from waitress import serve


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def offset_rounding_fixture(projects):
    pid = timing_fixture(projects, 'UI timing offset rounding')
    state = projects.load(pid)
    state['audio_shift_ms'] = 0
    state['analysis']['tempo'].update(
        offset=21,
        candidates=[dict(bpm=120, offset=21, division=4, coverage=.95)],
        precise_candidates=[],
    )
    projects.save(state)
    return pid


def audio_meter_fixture(projects):
    pid = timing_fixture(projects, 'UI audio meter')
    state = projects.load(pid)
    beats = [115 + i * 500 for i in range(24)]
    state['analysis']['separation'] = dict(profile='four', audio_sha256='a'*64,
        model_sha256='b'*64, input_processing=dict(revision=2), algorithm_revision=1,
        beat_evidence=dict(beats_ms=beats, downbeats_ms=beats[::3]))
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
    decoded_files = {voice:dict(wav_sha256=sha256(stem_folder/f'{voice}.wav'))
                     for voice in ('other', 'piano')}
    notes = [Note(1040+i*500, 1710+i*500, pitch=60+i%8, source='pitch', voice='other',
        id=f'evidence{i}', release_reason='model_predicted_offset') for i in range(14)]
    notes += [Note(9000, 9600, source='pitch', voice='other', id='unobserved', release_reason='input_boundary'),
              Note(10000, 10250, source='pitch', voice='piano', id='unknown')]
    projects.save(dict(id=pid, title=title, artist='Fixture', mode='audio', settings=parameters({'ln_ratio':1}),
        variants=[], active=-1, charts=[], audio='synthetic.wav', original=None, input_notes=[], audio_shift_ms=22.5,
        analysis=dict(revision=13, duration=12, candidates=[n.data() for n in notes], phrase_candidates=[],
            timing_attacks=[n.data() for n in notes], waveform=[0]*100,
            voices=[dict(voice=v) for v in ('other', 'piano')],
            separation=dict(source='provided', audio_sha256='c'*64, voices=['other', 'piano'],
                set_sha256='a'*64, decoded_files=decoded_files),
            structure=dict(boundaries=[0,12000], segments=[dict(start_ms=0,end_ms=12000)]),
            tempo=dict(bpm=120, offset=0, division=4, stable=True))))
    return pid


def evidence_source_identity_fixture(projects):
    pid = evidence_fixture(projects, 'UI evidence source identity')
    state = projects.load(pid)
    generate_variant(state, dict(mode='audio', timing_confirmed=True), state['settings'],
        dict(bpm=120, offset=-22, division=4, audio_shift_ms=22.5), lambda *_:None, lambda:False)
    state['analysis']['separation'] = dict(
        source='provided', audio_sha256='c'*64, voices=['other', 'piano'], set_sha256='b'*64)
    state['analysis']['candidates'] = [
        note | {'start': note['start'] + 400, 'end': note['end'] + 400}
        for note in state['analysis']['candidates']
    ]
    work = audio_work(projects.path(pid), state)
    for identity, sample in (('a'*64, 2205), ('b'*64, 13230)):
        samples = bytearray(22050 * 12 * 2)
        start = sample * 2
        samples[start:start+2] = b'\x00\x7f'
        path = work / 'stems-provided' / identity / 'other.wav'
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(22050)
            output.writeframes(samples)
        shutil.copyfile(projects.path(pid)/'preview.wav', path.parent/'piano.wav')
    references = (
        (state['variants'][state['active']]['source_separation']['separation'],
            work/'stems-provided'/('a'*64)),
        (state['analysis']['separation'], work/'stems-provided'/('b'*64)),
    )
    for separation, folder in references:
        separation['decoded_files'] = {
            voice:dict(wav_sha256=sha256(folder/f'{voice}.wav'))
            for voice in ('other', 'piano')
        }
    projects.save(state)
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
    waiting = threading.Event()
    def fixture_plan(endpoint, model, key, instruction, segments, current, review=None, cancelled=None):
        if endpoint != 'fixture://planner' or model not in ('success', 'error', 'cancel'):
            raise ValueError('Fixture planner rejects external requests')
        if model == 'error':
            raise ValueError('Deterministic fixture planner error')
        if model == 'cancel':
            waiting.set()
            try:
                while True:
                    if cancelled is not None and cancelled():
                        raise InterruptedError('已取消，原有谱面已保留')
                    time.sleep(0.01)
            finally:
                waiting.clear()
        return validate_plan(dict(preferences={'density':11}, segments=[],
            explanation='Deterministic fixture plan'), segments)
    app.extensions['planner'] = fixture_plan
    @app.get('/__fixture/planner-state')
    def planner_state():
        return dict(waiting=waiting.is_set())


def include_fixture_projects_in_recent(projects):
    fixture_ids = tuple(sorted(path.parent.name for path in projects.root.glob('*/state.json')))
    recent = projects.recent

    def recent_with_fixtures():
        def project_entry(project_id):
            try:
                project_path = projects.path(project_id) / 'state.json'
                state = projects.load(project_id)
                modified = project_path.stat().st_mtime
                if not isinstance(state, dict) or state.get('id') != project_id:
                    return None
                return modified, {
                    'id': project_id, 'title': state['title'], 'mode': state['mode'],
                }
            except (OSError, TypeError, ValueError, KeyError):
                return None

        try:
            recent_projects = recent()
        except (OSError, TypeError, ValueError, KeyError):
            paths = []
            for path in projects.root.glob('*/state.json'):
                try:
                    paths.append((path.stat().st_mtime, path.parent.name))
                except OSError:
                    continue
            recent_projects = []
            for _, project_id in sorted(paths, reverse=True):
                entry = project_entry(project_id)
                if entry is not None:
                    recent_projects.append(entry[1])
                    if len(recent_projects) == 30:
                        break

        entries = {}
        for project in recent_projects:
            if isinstance(project, dict) and 'id' in project:
                entry = project_entry(project['id'])
                if entry is not None:
                    entries[project['id']] = entry

        for project_id in fixture_ids:
            entry = project_entry(project_id)
            if entry is not None:
                entries[project_id] = entry

        return [project for _, project in sorted(entries.values(), key=lambda item: item[0], reverse=True)]

    projects.recent = recent_with_fixtures


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
        offset_rounding_fixture(projects)
        audio_meter_fixture(projects)
        timing_fixture(projects, 'UI LN navigation audio')
        ln_remap_fixture(projects)
        for title in ('UI evidence review', 'UI evidence edits', 'UI evidence snapshots',
                      'UI draft expression', 'UI draft structure', 'UI draft recommendation'):
            evidence_fixture(projects, title)
        evidence_source_identity_fixture(projects)
        legacy_region_fixture(projects)
        for action in ('clear', 'success', 'error', 'cancel', 'scope'):
            plan_fixture(projects, f'UI plan {action}')
        for action in ('current', 'legacy', 'full', 'edit'):
            regional_audit_fixture(projects, f'UI regional audit {action}')
        reanalyzed_regional_fixture(projects)
        include_fixture_projects_in_recent(projects)
        try:
            serve(app, host='127.0.0.1', port=args.port, threads=4)
        finally:
            app.extensions['executor'].shutdown()


if __name__ == '__main__':
    main()
