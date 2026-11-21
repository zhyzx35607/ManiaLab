"""Cross-route task failures use synthetic WAVs and deterministic barriers."""
from copy import deepcopy
import io
from pathlib import Path
import threading

import numpy as np
import pytest
from scipy.io import wavfile

from manialab.arrange import parameters
from manialab.chart_workspace import audio_work
from manialab.domain import Note
from manialab.server import create_app
from manialab.stems import separation_identity
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


def test_regional_generation_save_failure_preserves_parent_and_releases_worker(
        stem_project, monkeypatch):
    app, client, headers, pid, *_ = stem_project
    projects = app.extensions['projects']
    service = app.extensions['task_service']
    state = projects.load(pid)
    separation = state['provided_stems']
    reference = dict(identity=separation_identity(separation), separation=separation)
    old_evidence = {
        'locked': {
            'detected': dict(start=500, end=750, voice='piano', pitch=60,
                source='pitch', strength=1.0, release_reason='frame_decay'),
            'expressed': dict(start=500, end=750, phrase=False,
                release_reason='frame_decay', support=[], support_coverage=0.0),
        },
    }
    settings = parameters(dict(ln_ratio=0, phrase_ln=0))
    parent = dict(mode='audio', timing_revision=3, created=1.0, settings=settings,
        bpm=120, offset=0, division=4, audio_shift_ms=0, meter=4,
        notes=[Note(500, 750, 0, source='pitch', id='locked', voice='piano',
            release_reason='frame_decay').data()],
        report={}, selection={}, region=None, source_separation=reference,
        source_evidence=old_evidence)
    region = [1000, 2000]
    state['analysis'] = dict(revision=3, duration=2,
        candidates=[
            Note(500, 875, id='locked', source='pitch', voice='piano',
                release_reason='frame_decay').data(),
            Note(1250, 1500, id='regional', source='pitch', voice='piano',
                release_reason='frame_decay').data(),
        ], phrase_candidates=[], separation=separation)
    state['variants'] = [parent]
    state['active'] = 0
    projects.save(state)
    before = projects.load(pid)
    base = f'/api/projects/{pid}'
    payload = dict(mode='audio', settings=settings, timing_confirmed=True,
        bpm=120, offset=0, division=4, audio_shift_ms=0, region=region)
    save = projects.save
    failed_children = []

    def fail_regional_commit(candidate_state):
        variants = candidate_state.get('variants', [])
        if (len(variants) == 2 and candidate_state.get('active') == 1
                and variants[1].get('region') == region):
            failed_children.append(deepcopy(variants[1]))
            raise OSError('synthetic regional candidate commit failure')
        return save(candidate_state)

    monkeypatch.setattr(projects, 'save', fail_regional_commit)
    response = client.post(base + '/generate', json=payload, headers=headers)
    assert response.status_code == 200, response.json
    failed = wait_job(client, response.json)

    assert failed['status'] == 'error'
    assert 'result' not in failed
    assert len(failed_children) == 1
    attempted = failed_children[0]
    assert attempted['region'] == region
    assert attempted['source_separation'] == reference
    assert attempted['source_evidence']['locked'] == old_evidence['locked']
    assert attempted['source_evidence']['regional']['detected']['start'] == 1250
    after_failure = projects.load(pid)
    assert after_failure == before
    assert after_failure['variants'] == before['variants']
    assert after_failure['active'] == before['active'] == 0
    assert after_failure['variants'][0]['source_evidence'] == old_evidence
    assert after_failure['variants'][0]['source_separation'] == reference
    assert service._idle.is_set()

    monkeypatch.setattr(projects, 'save', save)
    recovered = wait_job(client, client.post(base + '/generate', json=payload,
        headers=headers).json)
    assert recovered['status'] == 'done'
    committed = projects.load(pid)
    assert committed['variants'][0] == before['variants'][0]
    child = committed['variants'][committed['active']]
    assert committed['active'] == 1 and len(committed['variants']) == 2
    assert child['region'] == region
    assert child['source_separation'] == reference
    assert child['source_evidence']['locked'] == old_evidence['locked']
    assert child['source_evidence']['regional']['detected']['start'] == 1250
    assert service._idle.is_set()


def test_http_cancel_during_post_commit_cleanup_keeps_successful_result(stem_project):
    app, client, headers, pid, *_ = stem_project
    projects = app.extensions['projects']
    service = app.extensions['task_service']
    before = projects.load(pid)
    cleanup_started, release_cleanup = threading.Event(), threading.Event()
    cleanup_arguments = []

    def operation(state, progress, cancelled):
        state['lifecycle_commit_marker'] = 'saved before cleanup'
        return {'committed': True}

    def cleanup(committed):
        cleanup_arguments.append(committed)
        cleanup_started.set()
        assert release_cleanup.wait(15)

    job = service.submit(pid, operation, cleanup=cleanup)
    try:
        assert cleanup_started.wait(15)
        saved = projects.load(pid)
        assert saved['lifecycle_commit_marker'] == 'saved before cleanup'
        assert service.get(job['id'])['status'] == 'running'
        assert 'result' not in service.get(job['id'])

        response = client.post(f"/api/jobs/{job['id']}/cancel", headers=headers)
        assert response.status_code == 200
        assert response.json == {'ok': True}
        pending = service.get(job['id'])
        assert pending['status'] == 'running' and pending['cancel'] is True
        assert projects.load(pid) == saved
    finally:
        release_cleanup.set()

    result = wait_job(client, job)
    expected = deepcopy(before)
    expected['lifecycle_commit_marker'] = 'saved before cleanup'
    assert result['status'] == 'done'
    assert result['result'] == {'committed': True}
    assert result['cancel'] is False
    assert projects.load(pid) == expected
    assert cleanup_arguments == [True]
    assert service._idle.is_set()

    recovered = service.submit(pid, lambda *_: {'worker_reused': True})
    recovery = wait_job(client, recovered)
    assert recovery['status'] == 'done'
    assert recovery['result'] == {'worker_reused': True}
    assert service._idle.is_set()


def test_analysis_cancel_reports_snapshot_cleanup_failure_and_reuses_worker(
        tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path / 'data')
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    projects = app.extensions['projects']
    service = app.extensions['task_service']
    source = tmp_path / 'source.wav'
    source.write_bytes(_wave_bytes(2205))
    imported = client.post('/api/import', headers=headers,
        data={'files': (io.BytesIO(source.read_bytes()), 'mix.wav')})
    assert imported.status_code == 200, imported.json
    pid = imported.json['id']
    state = projects.load(pid)
    state['analysis'] = {'sentinel': 'previous analysis stays committed'}
    projects.save(state)

    snapshot_ready, release_analysis = threading.Event(), threading.Event()
    snapshot_paths = []
    real_unlink = Path.unlink

    def fail_snapshot_unlink(path, *args, **kwargs):
        if path.name.startswith('.analysis-input-'):
            snapshot_paths.append(path)
            raise OSError('synthetic snapshot deletion denied')
        return real_unlink(path, *args, **kwargs)

    def cancelled_analysis(path, work, model, settings, progress, cancelled):
        from manialab.audio import _audio_snapshot

        with _audio_snapshot(path, work, cancelled) as (snapshot, audio_hash):
            snapshot_paths.append(snapshot)
            assert len(audio_hash) == 64
            snapshot_ready.set()
            assert release_analysis.wait(5)
            raise InterruptedError('已取消分析')

    monkeypatch.setattr(app.extensions['analysis_services'], 'analyze', cancelled_analysis)
    monkeypatch.setattr(Path, 'unlink', fail_snapshot_unlink)
    try:
        response = client.post(f'/api/projects/{pid}/analyze', json={}, headers=headers)
        assert response.status_code == 200, response.json
        job = response.json
        assert snapshot_ready.wait(5)

        cancel = client.post(f"/api/jobs/{job['id']}/cancel", headers=headers)
        assert cancel.status_code == 200
    finally:
        release_analysis.set()

    result = wait_job(client, job)
    try:
        assert result['status'] == 'error'
        assert result['cleanup_failed'] is True
        assert '资源清理失败' in result['message']
        assert str(snapshot_paths[0]) in result['message']
        assert '请手动删除' in result['message']
        assert projects.load(pid) == state
        assert len(snapshot_paths) >= 2
        assert snapshot_paths[0].exists()
        assert service._idle.is_set()

        recovery = service.submit(pid, lambda *_: {'worker_reused': True})
        assert wait_job(client, recovery)['status'] == 'done'
    finally:
        Path.unlink = real_unlink
        for path in set(snapshot_paths):
            real_unlink(path, missing_ok=True)
        service.shutdown()


def test_post_replace_cleanup_error_does_not_roll_back_committed_resource(
        stem_project, monkeypatch):
    app, client, headers, pid, upload, saved, roots, previous, old_wav, old_bytes = stem_project
    projects = app.extensions['projects']
    service = app.extensions['task_service']
    state_path = projects.path(pid) / 'state.json'
    temporary = state_path.with_suffix('.tmp')
    unlink = Path.unlink

    def fail_if_unlinked_after_replace(path, *args, **kwargs):
        if path == temporary:
            raise OSError('synthetic post-replace temporary cleanup failure')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fail_if_unlinked_after_replace)
    job = upload(4410)
    result = wait_job(client, job)

    assert result['status'] == 'done'
    committed = projects.load(pid)
    new_key = committed['provided_stems']['set_sha256']
    assert new_key != saved['provided_stems']['set_sha256']
    assert (roots[1] / new_key / 'separation.json').is_file()
    assert (roots[1] / new_key / 'piano.wav').is_file()
    assert previous[0] <= set(roots[0].iterdir())
    assert previous[1] <= set(roots[1].iterdir())
    assert old_wav.read_bytes() == old_bytes
    assert service._idle.is_set()

    monkeypatch.setattr(Path, 'unlink', unlink)
    recovered = service.submit(pid, lambda *_: {'worker_reused': True})
    recovery = wait_job(client, recovered)
    assert recovery['status'] == 'done'
    assert recovery['result'] == {'worker_reused': True}
    assert service._idle.is_set()
