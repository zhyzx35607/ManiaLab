"""A fixed-BPM refit keeps the requested clock precision across its API."""
from copy import deepcopy
import io
from pathlib import Path

import pytest

from manialab.domain import Note
from manialab.server import create_app


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('bpm,offset,shift', [
    (123.456789, -37.5, 22.5),
    (123.456789, -1.5, -50),
    (120, -37.5, 0),
    (10, -600000, 100),
    (1000, 600000, -100),
])
def test_fixed_bpm_refit_returns_the_same_precision_in_response_and_diagnostics(
        tmp_path, bpm, offset, shift):
    app = create_app(ROOT, tmp_path)
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        state = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(b'synthetic metadata fixture'), 'song.wav')}).json
        times = [round(115 + index * 60000 / bpm / 4) for index in range(28)]
        state['analysis'] = dict(revision=14, duration=(times[-1] + 1000) / 1000,
            candidates=[], timing_attacks=[Note(time, time, id=str(index)).data()
                for index, time in enumerate(times)],
            tempo=dict(bpm=120, offset=115, division=4, stable=True))
        state['plan'] = dict(explanation='Timing inspection must not commit a plan change')
        projects = app.extensions['projects']
        projects.save(state)
        before = deepcopy(projects.load(state['id']))

        response = client.post(f"/api/projects/{state['id']}/timing-refit", headers=headers,
            json=dict(bpm=bpm, offset=offset, division=4, audio_shift_ms=shift))

        assert response.status_code == 200, response.json
        report = response.json
        assert report['bpm'] == bpm
        assert report['clock'] == dict(bpm=bpm, offset=report['offset'],
            division=4, audio_shift_ms=shift)
        assert type(report['offset']) is int
        assert report['window_clock'] == 'audio'
        assert 0 <= report['coverage'] <= 1
        assert projects.load(state['id']) == before
    finally:
        app.extensions['executor'].shutdown()
