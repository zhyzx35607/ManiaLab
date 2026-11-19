"""Cross-route task failures use synthetic WAVs and deterministic barriers."""
import io
from pathlib import Path
import threading

import numpy as np
import pytest
from scipy.io import wavfile

from manialab.chart_workspace import audio_work
from manialab.server import create_app
from tests.test_server import wait_job


ROOT = Path(__file__).resolve().parents[1]


def _wave_bytes(pulse):
    samples = np.zeros(22050 * 2, dtype=np.int16)
    samples[pulse:pulse + 100] = 12000
    output = io.BytesIO()
    wavfile.write(output, 22050, samples)
    return output.getvalue()


@pytest.fixture
def stem_project(tmp_path):
    app = create_app(ROOT, tmp_path / 'data')
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    service = app.extensions['task_service']
    projects = app.extensions['projects']
    try:
        imported = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(_wave_bytes(2205)), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        state = projects.load(pid)
        state['analysis'] = {'sentinel': 'previous analysis remains selected'}
        projects.save(state)
        base = f'/api/projects/{pid}'

        def upload(pulse):
            response = client.post(base + '/provided-stems', headers=headers,
                data={'piano': (io.BytesIO(_wave_bytes(pulse)), 'piano.wav')})
            assert response.status_code == 200, response.json
            return response.json

        assert wait_job(client, upload(2205))['status'] == 'done'
        saved = projects.load(pid)
        work = audio_work(projects.path(pid), saved)
        roots = [work / 'provided-uploads', work / 'stems-provided']
        previous = [set(root.iterdir()) for root in roots]
        old_wav = next(iter(previous[1])) / 'piano.wav'
        old_bytes = old_wav.read_bytes()
        yield app, client, headers, pid, upload, saved, roots, previous, old_wav, old_bytes
    finally:
        service.shutdown()


@pytest.mark.parametrize('failure', ['cancel-after-decode', 'save-after-decode'])
def test_provided_stem_late_failure_preserves_old_set_and_releases_worker(
        stem_project, monkeypatch, failure):
    app, client, headers, pid, upload, saved, roots, previous, old_wav, old_bytes = stem_project
    projects = app.extensions['projects']
    services = app.extensions['analysis_services']
    decoded, release = threading.Event(), threading.Event()
    prepare = services.prepare_provided_stems
    save = projects.save
    new_keys = []

    def finish_decode(*args, **kwargs):
        report = prepare(*args, **kwargs)
        new_keys.append(report['set_sha256'])
        decoded.set()
        assert release.wait(5)
        return report

    monkeypatch.setattr(services, 'prepare_provided_stems', finish_decode)
    if failure == 'save-after-decode':
        def fail_new_set(state):
            if state.get('provided_stems', {}).get('set_sha256') in new_keys:
                raise OSError('synthetic final state commit failure')
            return save(state)
        monkeypatch.setattr(projects, 'save', fail_new_set)
    try:
        job = upload(4410)
        assert decoded.wait(5)
        assert (roots[1] / new_keys[0] / 'separation.json').is_file()
        if failure == 'cancel-after-decode':
            response = client.post(f"/api/jobs/{job['id']}/cancel", headers=headers)
            assert response.status_code == 200
        release.set()
        result = wait_job(client, job)
    finally:
        release.set()

    assert result['status'] == ('cancelled' if failure == 'cancel-after-decode' else 'error')
    assert projects.load(pid) == saved
    assert [set(root.iterdir()) for root in roots] == previous
    assert old_wav.read_bytes() == old_bytes
    assert app.extensions['task_service']._idle.is_set()
    monkeypatch.setattr(services, 'prepare_provided_stems', prepare)
    monkeypatch.setattr(projects, 'save', save)
    recovered = wait_job(client, upload(4410))
    assert recovered['status'] == 'done'
    assert projects.load(pid)['provided_stems']['set_sha256'] != saved['provided_stems']['set_sha256']


def test_queued_stem_import_cancel_does_not_decode_and_keeps_old_set(stem_project, monkeypatch):
    app, client, headers, pid, upload, saved, roots, previous, old_wav, old_bytes = stem_project
    service = app.extensions['task_service']
    services = app.extensions['analysis_services']
    submit = service.executor.submit
    prepare = services.prepare_provided_stems
    queued, decode_calls = [], []
    monkeypatch.setattr(service.executor, 'submit', lambda operation: queued.append(operation))
    monkeypatch.setattr(services, 'prepare_provided_stems', lambda *args, **kwargs: decode_calls.append(True))
    job = upload(4410)
    assert job['status'] == 'queued' and len(queued) == 1
    response = client.post(f"/api/jobs/{job['id']}/cancel", headers=headers)
    assert response.status_code == 200
    queued[0]()

    assert service.get(job['id'])['status'] == 'cancelled'
    assert not decode_calls
    assert app.extensions['projects'].load(pid) == saved
    assert [set(root.iterdir()) for root in roots] == previous
    assert old_wav.read_bytes() == old_bytes
    assert service._idle.is_set()
    monkeypatch.setattr(service.executor, 'submit', submit)
    monkeypatch.setattr(services, 'prepare_provided_stems', prepare)
    assert wait_job(client, upload(4410))['status'] == 'done'
