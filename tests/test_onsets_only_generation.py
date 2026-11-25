import hashlib
import io
from copy import deepcopy
from pathlib import Path
import zipfile

import pytest

from manialab.arrange import parameters
from manialab.domain import Note, export_osu, parse_osu
from manialab.expression import propose_expression
from manialab.server import create_app
from manialab.storage import Projects
from tests.test_server import wait_job
from tests.test_structure_expression import analysis


@pytest.mark.parametrize('structured', [False, True])
def test_zero_global_ln_ratio_overrides_saved_phrase_and_segment_preferences(tmp_path, structured):
    root = Path(__file__).parents[1]
    app = create_app(root, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    audio = b'placeholder'
    state = client.post('/api/import', data={
        'files': (io.BytesIO(audio), 'song.wav'),
    }, headers=headers).json
    project_url = f"/api/projects/{state['id']}"

    notes = [
        Note(1000, 1400, pitch=60, source='pitch', voice='vocals', id='v1'),
        Note(1450, 1850, pitch=62, source='pitch', voice='vocals', id='v2'),
        Note(3000, 3700, pitch=48, source='pitch', voice='piano', id='p1'),
    ]
    source = analysis(notes)
    source['revision'] = 3
    source['audio_sha256'] = hashlib.sha256(audio).hexdigest()
    source['phrase_candidates'] = [dict(
        notes[0].data(), end=2300, phrase=True, support=['v1', 'v2'],
        support_coverage=1., release_reason='vocal_activity_decay',
    )]
    source['tempo'].update(stable=True, division=4)
    state['analysis'] = source

    expression = propose_expression(source, parameters({'ln_ratio': .8, 'phrase_ln': 1}))
    expression['source'] = 'manual'
    expression['sections'][0]['voices']['vocals']['mode'] = 'phrase'
    expression['sections'][0]['voices']['piano']['mode'] = 'sustain'
    if structured:
        state['expression'] = expression
    else:
        source.pop('structure')
    state['plan'] = {'segments': [dict(
        start_ms=0, end_ms=12000,
        preferences={'ln_ratio': 1., 'phrase_ln': 1.},
    )]}
    Projects(tmp_path / 'projects').save(state)
    saved_analysis_before = deepcopy(client.get(project_url).json['analysis'])

    try:
        # Regenerate an existing LN candidate, rather than only testing a fresh
        # project whose UI has never restored saved settings or expression.
        payload = {
            'mode': 'audio', 'settings': {'ln_ratio': 1, 'phrase_ln': 1},
            'bpm': 120, 'offset': 0, 'division': 4,
            'audio_shift_ms': 0, 'timing_confirmed': True,
        }
        response = client.post(project_url + '/generate', json=payload, headers=headers)
        assert response.status_code == 200, response.json
        assert wait_job(client, response.json)['status'] == 'done'
        previous_state = client.get(project_url).json
        previous = deepcopy(previous_state['variants'][previous_state['active']])
        assert previous['report']['ln'] > 0

        response = client.post(project_url + '/generate', json={
            'mode': 'audio',
            'settings': {'ln_ratio': 0, 'phrase_ln': 0},
            'bpm': 120, 'offset': 0, 'division': 4,
            'audio_shift_ms': 0, 'timing_confirmed': True,
        }, headers=headers)
        assert response.status_code == 200, response.json
        job = wait_job(client, response.json)
        assert job['status'] == 'done', job

        saved = client.get(project_url).json
        candidate = saved['variants'][saved['active']]
        assert candidate['report']['ln'] == 0
        assert candidate['candidate_id'] != previous['candidate_id']
        assert {note['id'] for note in candidate['notes']} == {'v1', 'v2', 'p1'}
        assert all(note['start'] == note['end'] for note in candidate['notes'])
        assert saved['variants'][0] == previous
        assert saved['analysis'] == saved_analysis_before
        if structured:
            assert saved['expression']['source'] == 'manual'
            assert saved['expression']['sections'][0]['voices']['vocals']['mode'] == 'phrase'
        assert saved['plan']['segments'][0]['preferences']['ln_ratio'] == 1.

        query = '?candidate_id=' + candidate['candidate_id']
        chart = client.get(project_url + '/export/osu' + query)
        package = client.get(project_url + '/export/osz' + query)
        assert chart.status_code == package.status_code == 200
        parsed = parse_osu(chart.get_data(as_text=True))
        assert parsed['notes'] and not any(note.hold for note in parsed['notes'])
        with zipfile.ZipFile(io.BytesIO(package.data)) as archive:
            assert archive.read('ManiaLab.osu') == chart.data
            assert archive.read('song.wav') == audio
        assert client.get(project_url + '/export/osz?candidate_id=' + previous['candidate_id']).status_code == 400
    finally:
        app.extensions['executor'].shutdown()


def test_zero_ln_settings_do_not_remove_imported_holds_in_strict_remap(tmp_path):
    root = Path(__file__).parents[1]
    app = create_app(root, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    notes = [Note(1000, 3000, 0, id='hold'), Note(4000, 4000, 1, id='tap')]
    original = export_osu(notes, 10, root / 'vendor/mapperatorinator/template.osu')
    state = client.post('/api/import', data={
        'files': (io.BytesIO(original.encode()), 'chart.osu'),
    }, headers=headers).json
    project_url = f"/api/projects/{state['id']}"
    try:
        response = client.post(project_url + '/generate', json={
            'mode': 'remap', 'settings': {'keys': 7, 'ln_ratio': 0, 'phrase_ln': 0},
        }, headers=headers)
        assert response.status_code == 200, response.json
        assert wait_job(client, response.json)['status'] == 'done'
        saved = client.get(project_url).json
        candidate = saved['variants'][saved['active']]
        assert candidate['report']['ln'] == 1
        exported = parse_osu(client.get(project_url + '/export/osu').get_data(as_text=True))
        assert [(note.start, note.end) for note in exported['notes']] == [(1000, 3000), (4000, 4000)]
    finally:
        app.extensions['executor'].shutdown()
