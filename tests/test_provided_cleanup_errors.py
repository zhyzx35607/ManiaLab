"""Locked new stem files must not turn failed rollback into silent success."""
import io
import os
from pathlib import Path
import threading
import time
import wave

import numpy as np
import pytest

from manialab.chart_workspace import audio_work
from manialab.provided_stems import provided_stem_identity
from manialab.server import create_app


ROOT = Path(__file__).resolve().parents[1]


def _wave(pulse):
    samples = bytearray(8000 * 2 * 2)
    samples[pulse * 2:pulse * 2 + 2] = b'\x00\x60'
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        audio.writeframes(samples)
    output.seek(0)
    return output


def _finished(client, job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = client.get('/api/jobs/' + job['id']).json
        if result['status'] not in ('queued', 'running'):
            return result
        time.sleep(.01)
    pytest.fail('Synthetic stem import did not finish')


@pytest.mark.parametrize('failure', ['cancelled', 'decode-failed', 'prepared'])
def test_provided_input_snapshot_cleanup_reports_path_without_committing(
        tmp_path, monkeypatch, failure):
    from manialab import provided_stems

    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    service = app.extensions['task_service']
    projects = app.extensions['projects']
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    snapshots = set()
    entered, proceed = threading.Event(), threading.Event()
    remove = provided_stems.shutil.rmtree

    def upload(pid):
        response = client.post(f'/api/projects/{pid}/provided-stems', headers=headers,
            data={'piano': (io.BytesIO(b'synthetic piano'), 'piano.wav')})
        assert response.status_code == 200, response.json
        return response.json

    def fake_decode(path, work, *, cancelled):
        snapshots.add(Path(path).parent)
        if failure == 'cancelled':
            entered.set()
            assert proceed.wait(5)
            assert cancelled()
            raise InterruptedError('已取消声轨导入')
        if failure == 'decode-failed':
            raise ValueError('synthetic invalid stem')
        return np.zeros(800, dtype=np.float32)

    def fail_snapshot_cleanup(path, *args, **kwargs):
        if Path(path) in snapshots:
            raise PermissionError('synthetic input snapshot is locked')
        return remove(path, *args, **kwargs)

    try:
        imported = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(b'synthetic mix'), 'mix.wav')})
        assert imported.status_code == 200, imported.json
        pid = imported.json['id']
        saved = projects.load(pid)
        saved['analysis'] = {'sentinel': 'keep the previous analysis'}
        projects.save(saved)
        work = audio_work(projects.path(pid), saved)
        with monkeypatch.context() as patched:
            patched.setattr(provided_stems, 'decode', fake_decode)
            patched.setattr(provided_stems.shutil, 'rmtree', fail_snapshot_cleanup)
            job = upload(pid)
            if failure == 'cancelled':
                try:
                    assert entered.wait(5)
                    response = client.post(f"/api/jobs/{job['id']}/cancel", headers=headers)
                    assert response.status_code == 200, response.json
                finally:
                    proceed.set()
            result = _finished(client, job)

        snapshot, = snapshots
        assert result['status'] == 'error'
        assert result['cancel'] is (failure == 'cancelled')
        assert result['cleanup_failed'] is True
        assert str(snapshot) in result['message']
        assert '请手动删除' in result['message']
        assert 'synthetic input snapshot is locked' in result['message']
        if failure == 'cancelled':
            assert '已取消声轨导入' in result['message']
            assert '取消后资源清理失败' in result['message']
        if failure == 'decode-failed':
            assert 'synthetic invalid stem' in result['message']
        assert (snapshot / 'mix.wav').read_bytes() == b'synthetic mix'
        assert (snapshot / 'piano.wav').read_bytes() == b'synthetic piano'
        assert projects.load(pid) == saved
        assert not list((work / 'provided-uploads').iterdir())
        assert not list((work / 'stems-provided').iterdir())

        monkeypatch.setattr(provided_stems, 'decode', lambda *args, **kwargs: np.zeros(800, dtype=np.float32))
        recovered = _finished(client, upload(pid))
        assert recovered['status'] == 'done', recovered
        assert not recovered.get('cleanup_failed', False)
        assert projects.load(pid)['provided_stems']['voices'] == ['piano']
        assert service.shutdown() is False
    finally:
        proceed.set()
        service.shutdown()


@pytest.mark.parametrize('failure', ['cancelled', 'save-failed'])
@pytest.mark.parametrize('locked_part', ['upload', 'decoded', 'both'])
def test_stem_rollback_reports_locked_files_and_attempts_both_new_directories(
        tmp_path, monkeypatch, failure, locked_part):
    from manialab import analysis_routes

    app = create_app(ROOT, tmp_path)
    service = app.extensions['task_service']
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (_wave(0), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        base = '/api/projects/' + pid
        projects = app.extensions['projects']
        services = app.extensions['analysis_services']

        def upload(pulse):
            response = client.post(base + '/provided-stems', headers=headers,
                data={'piano': (_wave(pulse), 'uploaded.wav')})
            assert response.status_code == 200, response.json
            return response.json

        assert _finished(client, upload(800))['status'] == 'done'
        saved = projects.load(pid)
        saved['analysis'] = {'sentinel': 'keep the previous analysis'}
        projects.save(saved)
        work = audio_work(projects.path(pid), saved)
        upload_root, decoded_root = work / 'provided-uploads', work / 'stems-provided'
        old_uploads, old_sets = set(upload_root.iterdir()), set(decoded_root.iterdir())
        old_wav = next(iter(old_sets)) / 'piano.wav'
        old_bytes = old_wav.read_bytes()
        prepare, save = services.prepare_provided_stems, projects.save
        unlink = analysis_routes.shutil.os.unlink
        new_keys, attempted = [], []

        def decode_then_fail(mix, sources, work, progress, cancelled, *, on_new_folder=None):
            report = prepare(mix, sources, work, progress, cancelled,
                             on_new_folder=on_new_folder)
            new_keys.append(report['set_sha256'])
            if failure == 'cancelled':
                cancelled.cancel()
            return report

        def fail_new_save(state):
            if state.get('provided_stems', {}).get('set_sha256') in new_keys:
                raise OSError('synthetic final save failure')
            return save(state)

        def locked_unlink(path, *args, **kwargs):
            path = Path(path)
            name = path.name
            attempted.append(name)
            lock_upload = (locked_part in ('upload', 'both') and name == 'uploaded.wav'
                           and path.parent.parent.parent == upload_root)
            lock_decoded = (locked_part in ('decoded', 'both') and name == 'piano.wav'
                            and path.parent.parent == decoded_root)
            if lock_upload or lock_decoded:
                raise PermissionError('synthetic file is locked', str(path))
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(services, 'prepare_provided_stems', decode_then_fail)
        if failure == 'save-failed':
            monkeypatch.setattr(projects, 'save', fail_new_save)
        monkeypatch.setattr(analysis_routes.shutil.os, 'unlink', locked_unlink)
        try:
            result = _finished(client, upload(1600))
        finally:
            monkeypatch.setattr(analysis_routes.shutil.os, 'unlink', unlink)
            monkeypatch.setattr(services, 'prepare_provided_stems', prepare)
            monkeypatch.setattr(projects, 'save', save)

        assert {'uploaded.wav', 'piano.wav'} <= set(attempted)
        assert projects.load(pid) == saved
        assert old_wav.read_bytes() == old_bytes
        assert all(path.is_dir() for path in old_uploads | old_sets)
        new_uploads = set(upload_root.iterdir()) - old_uploads
        new_decoded = decoded_root / new_keys[0]
        if locked_part in ('upload', 'both'):
            assert len(new_uploads) == 1
            assert (next(iter(new_uploads)) / 'piano' / 'uploaded.wav').is_file()
        else:
            assert not new_uploads
        if locked_part in ('decoded', 'both'):
            assert (new_decoded / 'piano.wav').is_file()
        else:
            assert not new_decoded.exists()
        assert result['status'] == 'error'
        assert result.get('cleanup_failed') is True
        assert '资源清理失败' in result['message']
        assert service._idle.is_set()
        assert service._shutdown_cancellation.cleanup_failures
        recovered = _finished(client, upload(2400))
        assert recovered['status'] == 'done'
        assert not recovered.get('cleanup_failed', False)
        assert old_wav.read_bytes() == old_bytes
        assert service.shutdown() is False
    finally:
        service.shutdown()


def test_failed_rebuild_preserves_preexisting_invalid_cache_directory(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    service = app.extensions['task_service']
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (_wave(0), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        base = '/api/projects/' + pid
        projects = app.extensions['projects']

        def upload(pulse):
            response = client.post(base + '/provided-stems', headers=headers,
                data={'piano': (_wave(pulse), 'uploaded.wav')})
            assert response.status_code == 200, response.json
            return response.json

        assert _finished(client, upload(800))['status'] == 'done'
        saved = projects.load(pid)
        work = audio_work(projects.path(pid), saved)
        source = tmp_path / 'uploaded.wav'
        source.write_bytes(_wave(1600).getvalue())
        mix = projects.path(pid) / 'files' / saved['audio']
        key = provided_stem_identity(mix, {'piano': source})[1]
        invalid_folder = work / 'stems-provided' / key
        invalid_folder.mkdir(parents=True)
        (invalid_folder / 'separation.json').write_text('[]', encoding='utf-8')
        assert key != saved['provided_stems']['set_sha256']

        def fail_decode(*args, **kwargs):
            raise OSError('synthetic decoder failure')

        monkeypatch.setattr('manialab.provided_stems.decode', fail_decode)
        result = _finished(client, upload(1600))

        assert result['status'] == 'error'
        assert invalid_folder.is_dir()
        assert (invalid_folder / 'separation.json').read_text(encoding='utf-8') == '[]'
        assert not list((work / 'stems-provided').glob('.provided-build-*'))
        assert projects.load(pid) == saved
        assert service._idle.is_set()
    finally:
        service.shutdown()


def test_failed_same_key_rebuild_preserves_the_saved_decoded_folder(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    service = app.extensions['task_service']
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (_wave(0), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        base = '/api/projects/' + pid
        projects = app.extensions['projects']

        def upload():
            response = client.post(base + '/provided-stems', headers=headers,
                data={'piano': (_wave(800), 'uploaded.wav')})
            assert response.status_code == 200, response.json
            return response.json

        assert _finished(client, upload())['status'] == 'done'
        saved = projects.load(pid)
        work = audio_work(projects.path(pid), saved)
        decoded_folder = work / 'stems-provided' / saved['provided_stems']['set_sha256']
        original_files = {
            name: (decoded_folder / name).read_bytes()
            for name in ('piano.npy', 'piano.wav')
        }
        (decoded_folder / 'separation.json').write_text('[]', encoding='utf-8')

        def fail_decode(*args, **kwargs):
            raise OSError('synthetic decoder failure')

        monkeypatch.setattr('manialab.provided_stems.decode', fail_decode)
        result = _finished(client, upload())

        assert result['status'] == 'error'
        assert projects.load(pid) == saved
        assert decoded_folder.is_dir()
        assert {name: (decoded_folder / name).read_bytes() for name in original_files} == original_files
        assert (decoded_folder / 'separation.json').read_text(encoding='utf-8') == '[]'
        assert not list((work / 'stems-provided').glob('.provided-build-*'))
    finally:
        service.shutdown()


def test_save_failure_restores_same_key_cache_and_manifest(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    service = app.extensions['task_service']
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (_wave(0), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        base = '/api/projects/' + pid
        projects = app.extensions['projects']

        def upload():
            response = client.post(base + '/provided-stems', headers=headers,
                data={'piano': (_wave(800), 'uploaded.wav')})
            assert response.status_code == 200, response.json
            return response.json

        assert _finished(client, upload())['status'] == 'done'
        saved = projects.load(pid)
        work = audio_work(projects.path(pid), saved)
        decoded_folder = work / 'stems-provided' / saved['provided_stems']['set_sha256']
        (decoded_folder / 'separation.json').write_text('[]', encoding='utf-8')
        original_files = {
            path.relative_to(decoded_folder): path.read_bytes()
            for path in decoded_folder.rglob('*') if path.is_file()
        }
        save = projects.save

        def fail_save(state):
            if state.get('provided_stems', {}).get('set_sha256') == saved['provided_stems']['set_sha256']:
                raise OSError('synthetic project save failure')
            return save(state)

        monkeypatch.setattr(projects, 'save', fail_save)
        result = _finished(client, upload())

        assert result['status'] == 'error'
        assert projects.load(pid) == saved
        assert decoded_folder.is_dir()
        assert {
            path.relative_to(decoded_folder): path.read_bytes()
            for path in decoded_folder.rglob('*') if path.is_file()
        } == original_files
        assert not list((work / 'stems-provided').glob('.provided-backup-*'))
        assert service._idle.is_set()
    finally:
        service.shutdown()


def test_save_failure_reports_backup_when_new_cache_cannot_be_removed(tmp_path, monkeypatch):
    from manialab import analysis_routes

    app = create_app(ROOT, tmp_path)
    service = app.extensions['task_service']
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (_wave(0), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        base = '/api/projects/' + pid
        projects = app.extensions['projects']

        def upload():
            response = client.post(base + '/provided-stems', headers=headers,
                data={'piano': (_wave(800), 'uploaded.wav')})
            assert response.status_code == 200, response.json
            return response.json

        assert _finished(client, upload())['status'] == 'done'
        saved = projects.load(pid)
        work = audio_work(projects.path(pid), saved)
        canonical = work / 'stems-provided' / saved['provided_stems']['set_sha256']
        (canonical / 'separation.json').write_text('[]', encoding='utf-8')
        original_files = {
            path.relative_to(canonical): path.read_bytes()
            for path in canonical.rglob('*') if path.is_file()
        }
        save = projects.save
        rmtree = analysis_routes.shutil.rmtree
        backups = []

        def fail_save(state):
            if state.get('provided_stems', {}).get('set_sha256') == saved['provided_stems']['set_sha256']:
                raise OSError('synthetic project save failure')
            return save(state)

        def fail_remove_new_canonical(path, *args, **kwargs):
            target = Path(path)
            if target == canonical:
                backups.extend(canonical.parent.glob('.provided-backup-*'))
                raise PermissionError('synthetic new cache is locked', str(target))
            return rmtree(path, *args, **kwargs)

        monkeypatch.setattr(projects, 'save', fail_save)
        monkeypatch.setattr(analysis_routes.shutil, 'rmtree', fail_remove_new_canonical)
        result = _finished(client, upload())

        assert result['status'] == 'error'
        assert result.get('cleanup_failed') is True
        assert len(backups) == 1
        assert backups[0].is_dir()
        assert {
            path.relative_to(backups[0]): path.read_bytes()
            for path in backups[0].rglob('*') if path.is_file()
        } == original_files
        assert str(backups[0]) in result['message']
        assert str(canonical) in result['message']
        assert canonical.is_dir()
        assert projects.load(pid) == saved
        assert service._idle.is_set()
        assert service.shutdown() is False
    finally:
        service.shutdown()


def test_committed_rebuild_reports_backup_when_old_cache_cannot_be_removed(tmp_path, monkeypatch):
    from manialab import analysis_routes

    app = create_app(ROOT, tmp_path)
    service = app.extensions['task_service']
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (_wave(0), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        base = '/api/projects/' + pid
        projects = app.extensions['projects']

        def upload():
            response = client.post(base + '/provided-stems', headers=headers,
                data={'piano': (_wave(800), 'uploaded.wav')})
            assert response.status_code == 200, response.json
            return response.json

        assert _finished(client, upload())['status'] == 'done'
        saved = projects.load(pid)
        work = audio_work(projects.path(pid), saved)
        canonical = work / 'stems-provided' / saved['provided_stems']['set_sha256']
        (canonical / 'separation.json').write_text('[]', encoding='utf-8')
        original_files = {
            path.relative_to(canonical): path.read_bytes()
            for path in canonical.rglob('*') if path.is_file()
        }
        rmtree = analysis_routes.shutil.rmtree
        backups = []

        def fail_remove_old_backup(path, *args, **kwargs):
            target = Path(path)
            if target.parent == canonical.parent and target.name.startswith('.provided-backup-'):
                backups.append(target)
                raise PermissionError('synthetic old cache is locked', str(target))
            return rmtree(path, *args, **kwargs)

        monkeypatch.setattr(analysis_routes.shutil, 'rmtree', fail_remove_old_backup)
        result = _finished(client, upload())

        assert result['status'] == 'done'
        assert result.get('cleanup_failed') is True
        assert result['result']['voices'] == ['piano']
        assert len(backups) == 1
        assert backups[0].is_dir()
        assert {
            path.relative_to(backups[0]): path.read_bytes()
            for path in backups[0].rglob('*') if path.is_file()
        } == original_files
        assert '结果已保存' in result['message']
        assert str(backups[0]) in result['message']
        assert projects.load(pid)['provided_stems']['set_sha256'] == saved['provided_stems']['set_sha256']
        assert service._idle.is_set()
        assert service.shutdown() is False
    finally:
        service.shutdown()


def test_failed_cache_restore_exposes_backup_path_in_task_message(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    service = app.extensions['task_service']
    try:
        client = app.test_client()
        headers = {'X-ManiaLab-Token': app.config['TOKEN']}
        imported = client.post('/api/import', headers=headers,
            data={'files': (_wave(0), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        base = '/api/projects/' + pid
        projects = app.extensions['projects']

        def upload():
            response = client.post(base + '/provided-stems', headers=headers,
                data={'piano': (_wave(800), 'uploaded.wav')})
            assert response.status_code == 200, response.json
            return response.json

        assert _finished(client, upload())['status'] == 'done'
        saved = projects.load(pid)
        work = audio_work(projects.path(pid), saved)
        canonical = work / 'stems-provided' / saved['provided_stems']['set_sha256']
        (canonical / 'separation.json').write_text('[]', encoding='utf-8')
        original_files = {
            path.relative_to(canonical): path.read_bytes()
            for path in canonical.rglob('*') if path.is_file()
        }
        replace = os.replace
        restore_attempts = []

        def fail_publish_and_restore(source, target):
            source, target = Path(source), Path(target)
            if target == canonical and source.name.startswith('.provided-build-'):
                raise OSError('synthetic cache publish failure')
            if target == canonical and source.name.startswith('.provided-backup-'):
                restore_attempts.append((source, target))
                raise OSError('synthetic cache restore failure')
            return replace(source, target)

        monkeypatch.setattr('manialab.provided_stems.os.replace', fail_publish_and_restore)
        result = _finished(client, upload())

        assert result['status'] == 'error'
        assert result.get('cleanup_failed') is True
        assert len(restore_attempts) == 2
        backups = list((work / 'stems-provided').glob('.provided-backup-*'))
        assert len(backups) == 1
        assert backups[0].is_dir()
        assert {
            path.relative_to(backups[0]): path.read_bytes()
            for path in backups[0].rglob('*') if path.is_file()
        } == original_files
        assert str(backups[0]) in result['message']
        assert str(canonical) in result['message']
        assert projects.load(pid) == saved
        assert service._idle.is_set()
        assert service.shutdown() is False
    finally:
        service.shutdown()
