import copy
import io
import time
from pathlib import Path

import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.planner import validate_plan
from manialab.server import create_app
from manialab.structure import edit_boundaries


def _analysis(notes, boundary):
    result = dict(
        revision=3,
        duration=12,
        tempo=dict(bpm=120, offset=0, division=4, stable=True),
        candidates=[note.data() for note in notes],
        phrase_candidates=[],
    )
    result['structure'] = edit_boundaries({}, [boundary], result)
    result['segments'] = result['structure']['segments']
    return result


def _wait(client, response):
    for _ in range(500):
        job = client.get('/api/jobs/' + response['id']).json
        if job['status'] in ('done', 'error', 'cancelled'):
            return job
        time.sleep(.02)
    raise AssertionError('task timed out')


@pytest.mark.parametrize('analysis_succeeds', [True, False], ids=['replaced', 'failed'])
def test_reanalysis_only_drops_external_plan_on_success(tmp_path, monkeypatch, analysis_succeeds):
    root = Path(__file__).resolve().parents[1]
    app = create_app(root, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    try:
        imported = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(b'synthetic audio'), 'song.wav')})
        assert imported.status_code == 200, imported.json
        pid = imported.json['id']
        base = f'/api/projects/{pid}'
        projects = app.extensions['projects']

        old_analysis = _analysis([
            Note(2000, 2600, source='pitch', voice='piano', id='old-early'),
            Note(9000, 9600, source='pitch', voice='piano', id='old-late'),
        ], boundary=4000)
        state = projects.load(pid)
        state['analysis'] = old_analysis
        state['settings'] = parameters()
        projects.save(state)

        def old_section_plan(_endpoint, _model, _key, _instruction, segments,
                             _current, *, review=None, cancelled=None):
            return validate_plan({
                'preferences': {},
                'segments': [
                    {'index': 0, 'preferences': {'piano_weight': 0}},
                    {'index': 1, 'preferences': {'piano_weight': 1}},
                ],
                'explanation': 'Old section-bound plan',
            }, segments)

        app.extensions['planner'] = old_section_plan
        planned = client.post(base + '/plan', headers=headers, json={
            'endpoint': 'fixture://planner', 'model': 'fixture', 'settings': {},
        })
        assert planned.status_code == 200, planned.json
        assert _wait(client, planned.json)['status'] == 'done'
        old_plan = projects.load(pid)['plan']
        assert [row['start_ms'] for row in old_plan['segments']] == [0, 4000]

        new_notes = [
            Note(6000, 6600, source='pitch', voice='piano', id='new-middle'),
            Note(9000, 9600, source='pitch', voice='piano', id='new-late'),
        ]
        new_analysis = _analysis(new_notes, boundary=8000)

        def fake_analyze(_path, _work, _model, _settings, _progress, _cancelled):
            if not analysis_succeeds:
                raise ValueError('Synthetic analysis failure')
            return copy.deepcopy(new_analysis)

        monkeypatch.setattr(app.extensions['analysis_services'], 'analyze', fake_analyze)
        analyzed = client.post(base + '/analyze', headers=headers, json={'settings': {}})
        assert analyzed.status_code == 200, analyzed.json
        job = _wait(client, analyzed.json)
        current = client.get(base).json
        if not analysis_succeeds:
            assert job['status'] == 'error', job
            assert current['plan'] == old_plan
            assert current['analysis']['structure']['boundaries'] == [0, 4000, 12000]
            return
        assert job['status'] == 'done', job
        assert current['analysis']['structure']['boundaries'] == [0, 8000, 12000]

        generated = client.post(base + '/generate', headers=headers, json={
            'mode': 'audio', 'settings': {'keys': 6}, 'bpm': 120, 'offset': 0,
            'division': 4, 'audio_shift_ms': 0, 'timing_confirmed': True,
        })
        assert generated.status_code == 200, generated.json
        assert _wait(client, generated.json)['status'] == 'done'
        current = client.get(base).json
        assert {note['id'] for note in current['variants'][-1]['notes']} == {
            'new-middle', 'new-late',
        }
        assert 'plan' not in current
    finally:
        app.extensions['executor'].shutdown()
