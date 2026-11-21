"""Reference-derived structure changes commit only usable audio boundaries."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import zipfile

import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import propose_expression
from manialab.generation import generate_variant
from manialab.server import create_app
from manialab.structure import edit_boundaries
from manialab.timing import chart_clock_notes
from tests.test_server import wait_job


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def structured_project(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    try:
        state = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(b'synthetic metadata fixture'), 'song.wav')}).json
        notes = [Note(1000, 1750, source='pitch', voice='other', id='early'),
                 Note(6000, 6750, source='pitch', voice='other', id='late')]
        analysis = dict(revision=14, duration=10, candidates=[n.data() for n in notes],
            phrase_candidates=[], timing_attacks=[],
            tempo=dict(bpm=120, offset=0, division=4, stable=True))
        analysis['structure'] = edit_boundaries({}, [4000], analysis)
        analysis['segments'] = analysis['structure']['segments']
        state['analysis'] = analysis
        prefs = parameters({'ln_ratio': 1})
        generate_variant(state, dict(mode='audio', timing_confirmed=True), prefs,
            dict(bpm=120, offset=0, division=4, audio_shift_ms=0),
            lambda *_: None, lambda: False)
        state['expression'] = propose_expression(analysis, prefs)
        state['expression']['source'] = 'manual'
        state['expression']['sections'][0]['voices']['other']['mode'] = 'onsets'
        state['plan'] = dict(preferences={}, explanation='Keep the saved section preference',
            segments=[dict(start_ms=0, end_ms=4000, preferences=dict(density=3))])
        projects = app.extensions['projects']
        projects.save(state)
        yield client, headers, projects, state['id']
    finally:
        app.extensions['executor'].shutdown()


@pytest.mark.parametrize('action,mark_field', [
    ('bookmarks', 'bookmarks'), ('kiai', 'kiai_transitions'),
])
@pytest.mark.parametrize('marks,shift', [
    ([30000], 50), ([0], -50), ([0], .1), ([10000], -.1),
])
def test_unusable_reference_marks_preserve_existing_structure_and_intent(
        structured_project, action, mark_field, marks, shift):
    client, headers, projects, pid = structured_project
    state = projects.load(pid)
    state['reference_marks'] = {mark_field: marks}
    projects.save(state)
    before = deepcopy(projects.load(pid))

    response = client.post(f'/api/projects/{pid}/structure', headers=headers,
        json=dict(action=action, audio_shift_ms=shift))
    assert response.status_code == 200
    result = wait_job(client, response.json)

    assert result['status'] == 'error'
    assert '不在当前音频范围内' in result['message']
    assert projects.load(pid) == before


@pytest.mark.parametrize('action,mark_field', [
    ('bookmarks', 'bookmarks'), ('kiai', 'kiai_transitions'),
])
@pytest.mark.parametrize('shift', [-50, 50])
def test_reference_structure_uses_audio_clock_and_invalidates_only_current_intent(
        structured_project, action, mark_field, shift):
    client, headers, projects, pid = structured_project
    state = projects.load(pid)
    state['reference_marks'] = {mark_field: [3000, 7000]}
    projects.save(state)
    before = deepcopy(projects.load(pid))

    response = client.post(f'/api/projects/{pid}/structure', headers=headers,
        json=dict(action=action, audio_shift_ms=shift))
    assert response.status_code == 200
    assert wait_job(client, response.json)['status'] == 'done'
    after = projects.load(pid)

    assert after['analysis']['structure']['boundaries'] == [0, 3000 + shift, 7000 + shift, 10000]
    assert after['analysis']['segments'] == after['analysis']['structure']['segments']
    assert after['analysis']['candidates'] == before['analysis']['candidates']
    assert after['variants'] == before['variants']
    assert after['active'] == before['active']
    assert 'plan' not in after
    assert after['expression']['source'] == 'automatic'
    assert [(row['start_ms'], row['end_ms']) for row in after['expression']['sections']] == [
        (0, 3000 + shift), (3000 + shift, 7000 + shift), (7000 + shift, 10000)]


@pytest.mark.parametrize('action,mark_field', [
    ('bookmarks', 'bookmarks'), ('kiai', 'kiai_transitions'),
])
@pytest.mark.parametrize('mark,shift,expected', [
    (3200, 22.5, 3223),
    (3201, -22.5, 3179),
])
def test_reference_clock_half_millisecond_uses_chart_rounding(
        structured_project, action, mark_field, mark, shift, expected):
    client, headers, projects, pid = structured_project
    state = projects.load(pid)
    state['reference_marks'] = {mark_field: [mark]}
    projects.save(state)

    response = client.post(f'/api/projects/{pid}/structure', headers=headers,
        json=dict(action=action, audio_shift_ms=shift))
    assert response.status_code == 200
    assert wait_job(client, response.json)['status'] == 'done'
    after = client.get(f'/api/projects/{pid}').json

    chart_expected = chart_clock_notes(
        [Note(mark, mark, source='pitch', voice='mix', id='mark')], -shift)[0].start
    assert chart_expected == expected
    assert after['analysis']['structure']['boundaries'] == [0, expected, 10000]


@pytest.mark.parametrize('shift,boundary,expected', [
    (22.5, 3201, 3179),
    (-22.5, 3200, 3223),
])
def test_osu_and_osz_bookmark_export_use_chart_rounding(
        structured_project, shift, boundary, expected):
    client, headers, projects, pid = structured_project
    state = projects.load(pid)
    variant = state['variants'][state['active']]
    variant['audio_shift_ms'] = shift
    variant['selection']['structure']['boundaries'] = [0, boundary, 10000]
    audio_path = projects.path(pid) / 'files' / state['audio']
    variant['source_audio_sha256'] = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    projects.save(state)

    chart_expected = chart_clock_notes(
        [Note(boundary, boundary, source='pitch', voice='mix', id='mark')], shift)[0].start
    assert chart_expected == expected
    osu = client.get(f'/api/projects/{pid}/export/osu', headers=headers)
    assert osu.status_code == 200

    osz = client.get(f'/api/projects/{pid}/export/osz', headers=headers)
    assert osz.status_code == 200
    with zipfile.ZipFile(io.BytesIO(osz.data)) as archive:
        chart, = (name for name in archive.namelist() if name.lower().endswith('.osu'))
        assert archive.read(chart) == osu.data
    assert f'Bookmarks: {expected}' in osu.data.decode().splitlines()


@pytest.mark.parametrize('boundaries', [[4000, 4100], [-1]])
def test_rejected_manual_boundaries_preserve_saved_structure_and_intent(
        structured_project, boundaries):
    client, headers, projects, pid = structured_project
    before = deepcopy(projects.load(pid))
    response = client.post(f'/api/projects/{pid}/structure', headers=headers,
        json=dict(action='edit', boundaries=boundaries))
    assert response.status_code == 200
    assert wait_job(client, response.json)['status'] == 'error'
    assert projects.load(pid) == before
