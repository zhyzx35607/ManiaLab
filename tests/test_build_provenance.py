import json

import pytest

from scripts.build_provenance import sha256, verify_worker, worker_sources


def test_packaging_rejects_stale_worker_sources_and_executable(tmp_path):
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    for name in ('separate_audio.py', 'worker_lifetime.py'):
        (scripts/name).write_text('# original', encoding='utf-8')
    directory = tmp_path/'assets/deep'
    worker = directory/'ManiaLabAudio/ManiaLabAudio.exe'
    worker.parent.mkdir(parents=True)
    worker.write_bytes(b'original executable')
    manifest = directory/'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='unrecorded'):
        verify_worker(tmp_path, directory)
    identity = dict(worker_sources=worker_sources(tmp_path), worker_sha256=sha256(worker))
    manifest.write_text(json.dumps(identity), encoding='utf-8')
    assert verify_worker(tmp_path, directory) == identity
    (scripts/'separate_audio.py').write_text('# float input fix', encoding='utf-8')
    with pytest.raises(ValueError, match='sources changed'):
        verify_worker(tmp_path, directory)
    (scripts/'separate_audio.py').write_text('# original', encoding='utf-8')
    worker.write_bytes(b'old executable')
    with pytest.raises(ValueError, match='executable'):
        verify_worker(tmp_path, directory)
