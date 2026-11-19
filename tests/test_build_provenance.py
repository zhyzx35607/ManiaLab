import json

import pytest

from scripts.build_provenance import sha256, verify_worker, worker_sources
from scripts.worker_inventory import worker_inventory


def test_packaging_rejects_stale_worker_sources_and_executable(tmp_path):
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    for name in ('separate_audio.py', 'separate_six.py', 'worker_lifetime.py'):
        (scripts/name).write_text('# original', encoding='utf-8')
    directory = tmp_path/'assets/deep'
    worker = directory/'ManiaLabAudio/ManiaLabAudio.exe'
    worker.parent.mkdir(parents=True)
    worker.write_bytes(b'original executable')
    manifest = directory/'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='unrecorded'):
        verify_worker(tmp_path, directory)
    identity = dict(worker_sources=worker_sources(tmp_path), worker_sha256=sha256(worker),
        worker_files=worker_inventory(worker.parent))
    manifest.write_text(json.dumps(identity), encoding='utf-8')
    assert verify_worker(tmp_path, directory) == identity
    (scripts/'separate_audio.py').write_text('# float input fix', encoding='utf-8')
    with pytest.raises(ValueError, match='sources changed'):
        verify_worker(tmp_path, directory)
    (scripts/'separate_audio.py').write_text('# original', encoding='utf-8')
    (scripts/'separate_six.py').write_text('# six-source change', encoding='utf-8')
    with pytest.raises(ValueError, match='sources changed'):
        verify_worker(tmp_path, directory)
    (scripts/'separate_six.py').write_text('# original', encoding='utf-8')
    worker.write_bytes(b'old executable')
    with pytest.raises(ValueError, match='executable'):
        verify_worker(tmp_path, directory)


def test_instrument_package_needs_matching_real_audio_checks(tmp_path, monkeypatch):
    from manialab import instrument_bundle as bundle
    from scripts.build_provenance import verify_instrument_worker
    directory = tmp_path/'instruments'
    executable = directory/'ManiaLabInstrument/ManiaLabInstrument.exe'
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b'packaged worker')
    sources = {'scripts/transcribe_instruments.py': 'fixed source hash'}
    monkeypatch.setattr(bundle, 'runtime_identity', lambda root: sources)
    monkeypatch.setattr(bundle, 'verify_source', lambda path: {'upstream.py': 'fixed upstream hash'})
    monkeypatch.setattr(bundle, 'verify_model', lambda path, model: None)
    verification = dict(worker_sha256=sha256(executable), worker_sources=sources,
        worker_files=worker_inventory(executable.parent),
        models=bundle.MODELS, portable_runtime_verified=True,
        cases=[dict(model=model, allow_predicted_releases=release,
                    raw_evidence_equal=True, cache_reused=True)
               for model in bundle.MODELS for release in (False, True)])
    report_path = directory/'verification.json'
    manifest = dict(worker_sha256=sha256(executable), worker_sources=sources,
        worker_files=worker_inventory(executable.parent),
        portable_runtime_verified=True, verified_models=list(bundle.MODELS),
        source_files={'upstream.py': 'fixed upstream hash'})

    def save():
        report_path.write_text(json.dumps(verification), encoding='utf-8')
        manifest['verification_sha256'] = sha256(report_path)
        (directory/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    save()
    assert verify_instrument_worker(tmp_path, directory) == manifest
    report_path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='evidence'):
        verify_instrument_worker(tmp_path, directory)
    save()
    verification['worker_sha256'] = 'stale worker'
    save()
    with pytest.raises(ValueError, match='another build'):
        verify_instrument_worker(tmp_path, directory)
    verification['worker_sha256'] = sha256(executable)
    (executable.parent/'runtime.dll').write_bytes(b'new runtime with unchanged exe')
    manifest['worker_files'] = worker_inventory(executable.parent)
    save()
    with pytest.raises(ValueError, match='another build'):
        verify_instrument_worker(tmp_path, directory)
    verification['worker_files'] = manifest['worker_files']
    verification['cases'].pop()
    save()
    with pytest.raises(ValueError, match='both models'):
        verify_instrument_worker(tmp_path, directory)
