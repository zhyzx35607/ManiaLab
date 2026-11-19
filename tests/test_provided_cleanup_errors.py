"""Locked new stem files must not turn failed rollback into silent success."""
import io
from pathlib import Path
import time
import wave

import pytest

from manialab.chart_workspace import audio_work
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

        def decode_then_fail(mix, sources, work, progress, cancelled):
            report = prepare(mix, sources, work, progress, cancelled)
            new_keys.append(report['set_sha256'])
            if failure == 'cancelled':
                cancelled.cancel()
            return report

        def fail_new_save(state):
            if state.get('provided_stems', {}).get('set_sha256') in new_keys:
                raise OSError('synthetic final save failure')
            return save(state)

        def locked_unlink(path, *args, **kwargs):
            name = Path(path).name
            attempted.append(name)
            locked_names = ({'uploaded.wav', 'piano.wav'} if locked_part == 'both' else
                            {'uploaded.wav' if locked_part == 'upload' else 'piano.wav'})
            if name in locked_names:
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
