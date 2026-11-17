"""Uploads must preserve source identity and leave export references usable."""
import io
import logging
from pathlib import Path
import threading
import time
import zipfile

import numpy as np
import pytest
from scipy.io import wavfile

from manialab import server
from manialab.domain import Note, export_osu, parse_osu
from manialab.storage import extract_osz


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'vendor/mapperatorinator/template.osu'


def package(entries):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        for name, content in entries:
            archive.writestr(name, content)
    data.seek(0)
    return data


def wave(seconds=1):
    data = io.BytesIO()
    wavfile.write(data, 22050, np.zeros(round(22050 * seconds), dtype=np.int16))
    data.seek(0)
    return data


def wait_job(client, job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = client.get('/api/jobs/' + job['id']).json
        if result['status'] not in ('queued', 'running'):
            return result
        time.sleep(.01)
    raise AssertionError('task did not finish')


@pytest.fixture
def api(tmp_path):
    app = server.create_app(ROOT, tmp_path)
    try:
        yield app.test_client(), {'X-ManiaLab-Token': app.config['TOKEN']}, app
    finally:
        app.extensions['executor'].shutdown()


@pytest.mark.parametrize('entries', [
    [('new.wav', b'new'), ('SONG.WAV', b'changed')],
    [('new.wav', b'new'), ('song.wav/child.wav', b'child')],
    [('new.wav', b'new'), ('song.wav/', b'')],
])
def test_archive_conflict_does_not_touch_existing_or_write_earlier_members(tmp_path, entries):
    original = tmp_path / 'song.wav'
    original.write_bytes(b'original')
    with pytest.raises(ValueError, match='同名|冲突'):
        extract_osz(package(entries), tmp_path)
    assert original.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [original]


@pytest.mark.parametrize('entries', [
    [('nested/child.wav', b'child'), ('NESTED', b'file')],
    [('nested', b'file'), ('NESTED/child.wav', b'child')],
    [('nested/', b''), ('NESTED', b'file')],
    [('song.wav', b'first'), ('SONG.WAV', b'second')],
])
def test_archive_rejects_internal_file_directory_and_case_collisions(tmp_path, entries):
    with pytest.raises(ValueError, match='同名|冲突'):
        extract_osz(package(entries), tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('inputs', [
    'audio_then_archive', 'archive_then_audio', 'two_archives',
    'archive_directory_then_audio', 'audio_then_archive_directory',
])
def test_import_rejects_cross_input_collisions_and_keeps_other_projects(api, tmp_path, inputs):
    client, headers, app = api
    previous = client.post('/api/import', data={'files': (io.BytesIO(b'previous'), 'old.wav')}, headers=headers).json
    first = (io.BytesIO(b'first'), 'song.wav')
    second = (package([('SONG.WAV', b'second')]), 'second.osz')
    if inputs == 'archive_then_audio':
        first, second = second, first
    elif inputs == 'two_archives':
        first = (package([('song.wav', b'first')]), 'first.osz')
    elif inputs == 'archive_directory_then_audio':
        first, second = (package([('SONG.WAV/', b'')]), 'first.osz'), first
    elif inputs == 'audio_then_archive_directory':
        second = (package([('SONG.WAV/', b'')]), 'second.osz')
    response = client.post('/api/import', data={'files': [first, second]}, headers=headers)
    assert response.status_code == 400
    assert [p.name for p in (tmp_path / 'projects').iterdir()] == [previous['id']]
    assert (tmp_path / 'projects' / previous['id'] / 'files/old.wav').read_bytes() == b'previous'


def test_import_does_not_parse_osu_named_directories_as_charts(api):
    client, headers, _ = api
    response = client.post('/api/import', data={'files': (package([
        ('unused.osu/', b''), ('song.wav', b'audio'),
    ]), 'song.osz')}, headers=headers)
    assert response.status_code == 200
    assert response.json['audio'] == 'song.wav'
    assert response.json['charts'] == []


def test_nested_chart_supplemental_audio_export_closes_reference_and_preserves_objects(api):
    client, headers, _ = api
    original = export_osu([Note(1000, 2000, 0), Note(3000, 3000, 1)], 6, TEMPLATE, audio='song.wav')
    original = original.replace('0.000,500.00000000,4,2,0,70,1,0',
                                '0.000,500.00000000,4,2,0,70,1,0\n2500,-100,4,2,1,60,0,1')
    original = original.replace('2000:0:0:0:0:', '2000:2:3:4:50:hit.wav')
    uploaded = client.post('/api/import', data={'files': [
        (package([('nested/chart.osu', original), ('nested/hit.wav', b'custom-hit')]), 'charts.osz'),
        (io.BytesIO(b'supplemental-audio'), 'song.wav'),
    ]}, headers=headers)
    assert uploaded.status_code == 200
    pid = uploaded.json['id']
    result = wait_job(client, client.post(f'/api/projects/{pid}/generate',
        json={'mode': 'remap', 'settings': {'keys': 6}}, headers=headers).json)
    assert result['status'] == 'done', result
    response = client.get(f'/api/projects/{pid}/export/osz')
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        name = 'nested/ManiaLab.osu'
        parsed = parse_osu(archive.read(name).decode())
        reference = (Path(name).parent / parsed['general']['AudioFilename']).as_posix()
        assert archive.read(reference) == b'supplemental-audio'
        assert archive.read('nested/hit.wav') == b'custom-hit'
        assert len(archive.namelist()) == len(set(archive.namelist()))
    source = parse_osu(original)
    assert parsed['timing'] == source['timing']
    assert sorted(n.raw.split(',', 1)[1] for n in parsed['notes']) == sorted(n.raw.split(',', 1)[1] for n in source['notes'])


def import_mix(client, headers):
    response = client.post('/api/import', data={'files': (wave(), 'mix.wav')}, headers=headers)
    assert response.status_code == 200
    return response.json['id']


def test_provided_stems_validate_all_uploads_before_saving(api, tmp_path):
    client, headers, _ = api
    pid = import_mix(client, headers)
    response = client.post(f'/api/projects/{pid}/provided-stems', data={
        'piano': (wave(), 'piano.wav'), 'guitar': (io.BytesIO(b'invalid'), 'guitar.txt'),
    }, headers=headers)
    assert response.status_code == 400
    assert not (tmp_path / 'projects' / pid / 'provided-uploads').exists()


def test_failed_provided_stems_clean_new_files_and_preserve_previous_set(api, tmp_path):
    client, headers, _ = api
    pid = import_mix(client, headers)
    endpoint = f'/api/projects/{pid}/provided-stems'
    success = wait_job(client, client.post(endpoint, data={'piano': (wave(), 'piano.wav')}, headers=headers).json)
    assert success['status'] == 'done'
    saved = client.get(f'/api/projects/{pid}').json
    work = tmp_path / 'projects' / pid
    old_set = work / 'stems-provided' / saved['provided_stems']['set_sha256']
    old_files = {p.name: p.read_bytes() for p in old_set.iterdir()}
    old_uploads = list((work / 'provided-uploads').iterdir())
    failed = wait_job(client, client.post(endpoint, data={'piano': (wave(.5), 'short.wav')}, headers=headers).json)
    assert failed['status'] == 'error'
    assert client.get(f'/api/projects/{pid}').json == saved
    assert list((work / 'provided-uploads').iterdir()) == old_uploads
    assert list((work / 'stems-provided').iterdir()) == [old_set]
    assert {p.name: p.read_bytes() for p in old_set.iterdir()} == old_files


def test_cancelled_provided_stems_remove_new_decoded_folder_before_job_finishes(api, tmp_path, monkeypatch):
    client, headers, _ = api
    pid = import_mix(client, headers)
    ready, release = threading.Event(), threading.Event()
    original = server.prepare_provided_stems
    def prepare(*args):
        report = original(*args)
        ready.set()
        assert release.wait(5)
        return report
    monkeypatch.setattr(server, 'prepare_provided_stems', prepare)
    job = client.post(f'/api/projects/{pid}/provided-stems', data={'piano': (wave(), 'piano.wav')}, headers=headers).json
    try:
        assert ready.wait(5)
        assert client.post('/api/jobs/' + job['id'] + '/cancel', headers=headers).status_code == 200
    finally:
        release.set()
    assert wait_job(client, job)['status'] == 'cancelled'
    state = client.get(f'/api/projects/{pid}').json
    assert 'provided_stems' not in state
    work = tmp_path / 'projects' / pid
    assert not list((work / 'provided-uploads').iterdir())
    assert not list((work / 'stems-provided').iterdir())


def test_request_and_background_failures_use_persistent_logger(api, tmp_path, monkeypatch):
    client, headers, app = api
    path = tmp_path / 'audit-errors.log'
    handler = logging.FileHandler(path, encoding='utf-8')
    server.logger.addHandler(handler)
    try:
        pid = import_mix(client, headers)
        def fail_load(*args):
            raise RuntimeError('request failure sentinel')
        with monkeypatch.context() as request_patch:
            request_patch.setattr(app.extensions['projects'], 'load', fail_load)
            assert client.get(f'/api/projects/{pid}').status_code == 500
        def fail(*args):
            raise RuntimeError('background failure sentinel')
        monkeypatch.setattr(server, 'prepare_provided_stems', fail)
        result = wait_job(client, client.post(f'/api/projects/{pid}/provided-stems',
            data={'piano': (wave(), 'piano.wav')}, headers=headers).json)
        assert result['status'] == 'error'
        handler.flush()
        content = path.read_text(encoding='utf-8')
        assert 'Request failed' in content and 'request failure sentinel' in content
        assert 'Project task failed' in content and 'background failure sentinel' in content
    finally:
        server.logger.removeHandler(handler)
        handler.close()
