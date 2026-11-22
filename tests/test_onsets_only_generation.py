import io
from pathlib import Path

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import propose_expression
from manialab.server import create_app
from manialab.storage import Projects
from tests.test_server import wait_job
from tests.test_structure_expression import analysis


def test_zero_global_ln_ratio_overrides_saved_phrase_and_segment_preferences(tmp_path):
    root = Path(__file__).parents[1]
    app = create_app(root, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    state = client.post('/api/import', data={
        'files': (io.BytesIO(b'placeholder'), 'song.wav'),
    }, headers=headers).json
    project_url = f"/api/projects/{state['id']}"

    notes = [
        Note(1000, 1400, pitch=60, source='pitch', voice='vocals', id='v1'),
        Note(1450, 1850, pitch=62, source='pitch', voice='vocals', id='v2'),
        Note(3000, 3700, pitch=48, source='pitch', voice='piano', id='p1'),
    ]
    source = analysis(notes)
    source['revision'] = 3
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
    state['expression'] = expression
    state['plan'] = {'segments': [dict(
        start_ms=0, end_ms=12000,
        preferences={'ln_ratio': 1., 'phrase_ln': 1.},
    )]}
    Projects(tmp_path / 'projects').save(state)

    try:
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
        assert {note['id'] for note in candidate['notes']} == {'v1', 'v2', 'p1'}
        assert all(note['start'] == note['end'] for note in candidate['notes'])
        assert saved['expression']['source'] == 'manual'
        assert saved['expression']['sections'][0]['voices']['vocals']['mode'] == 'phrase'
        assert saved['plan']['segments'][0]['preferences']['ln_ratio'] == 1.
    finally:
        app.extensions['executor'].shutdown()
