import ast
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from scripts import build_provenance
from scripts.build_provenance import sha256, verify_worker, worker_sources
from scripts.package_assets import MAPPER_INSTALLER_FILES, capabilities
from scripts.worker_inventory import worker_inventory


def test_source_snapshot_paths_deduplicates_installer_python_files(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    installer_scripts = [name for name in MAPPER_INSTALLER_FILES if name.endswith('.py')]
    for name in installer_scripts:
        (assets / name).write_text('# synthetic installer source', encoding='utf-8')

    for profile in ('standard', 'extended'):
        paths = build_provenance.source_snapshot_paths(tmp_path, capabilities(profile))
        keys = [path.relative_to(tmp_path).as_posix() for path in paths]
        assert len(keys) == len(set(keys))
        for name in installer_scripts:
            assert keys.count(f'assets/{name}') == 1


def test_source_snapshot_paths_includes_docs_and_rejects_links(tmp_path, monkeypatch):
    documentation = tmp_path / 'docs' / 'guide'
    documentation.mkdir(parents=True)
    source = documentation / 'quick-start.md'
    source.write_text('portable guide', encoding='utf-8')

    paths = build_provenance.source_snapshot_paths(tmp_path, capabilities('core'))
    assert source in paths

    original_is_symlink = Path.is_symlink

    def mark_documentation_link(path):
        return path == source or original_is_symlink(path)

    monkeypatch.setattr(Path, 'is_symlink', mark_documentation_link)
    with pytest.raises(ValueError, match='Source tree must not contain links'):
        build_provenance.source_snapshot_paths(tmp_path, capabilities('core'))


def test_archive_payload_inventory_covers_final_tree_except_root_manifest(tmp_path):
    package = tmp_path / 'ManiaLab'
    runtime = package / '_internal' / 'runtime.dll'
    guide = package / 'docs' / 'release-manifest.json'
    runtime.parent.mkdir(parents=True)
    guide.parent.mkdir(parents=True)
    runtime.write_bytes(b'runtime')
    guide.write_bytes(b'nested manifest is ordinary payload')
    (package / 'release-manifest.json').write_bytes(b'self reference is excluded')

    inventory = build_provenance.archive_payload_inventory(package)

    assert inventory == {
        'files': {
            'ManiaLab/_internal/runtime.dll': hashlib.sha256(b'runtime').hexdigest(),
            'ManiaLab/docs/release-manifest.json': hashlib.sha256(
                b'nested manifest is ordinary payload'
            ).hexdigest(),
        },
        'directories': ['ManiaLab/', 'ManiaLab/_internal/', 'ManiaLab/docs/'],
    }


@pytest.mark.parametrize('escape', ['link', 'resolved_path'])
def test_archive_payload_inventory_rejects_links_and_resolved_escapes(
    tmp_path, monkeypatch, escape
):
    package = tmp_path / 'ManiaLab'
    package.mkdir()
    payload = package / 'runtime.dll'
    payload.write_bytes(b'runtime')

    if escape == 'link':
        original_is_symlink = Path.is_symlink

        def mark_payload_link(path):
            return path == payload or original_is_symlink(path)

        monkeypatch.setattr(Path, 'is_symlink', mark_payload_link)
        message = 'must not contain links'
    else:
        original_resolve = Path.resolve
        outside = tmp_path / 'outside' / 'runtime.dll'

        def resolve_outside_payload(path, *args, **kwargs):
            if path == payload:
                return outside
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'resolve', resolve_outside_payload)
        message = 'escapes its root'

    with pytest.raises(ValueError, match=message):
        build_provenance.archive_payload_inventory(package)


def test_build_source_snapshot_rejects_resolved_paths_outside_root_before_hashing(
    tmp_path, monkeypatch
):
    source_root = tmp_path / 'source'
    source_root.mkdir()
    source_path = source_root / 'assets' / 'installer.py'
    outside_path = tmp_path / 'outside.py'
    build_script = Path(build_provenance.__file__).resolve().parents[1] / 'scripts' / 'build_windows.py'
    source = ast.parse(build_script.read_text(encoding='utf-8'))
    function = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == 'source_snapshot'
    )
    hash_calls = []

    def synthetic_sha256(path):
        hash_calls.append(path)
        return 'a' * 64

    namespace = {
        'ROOT': source_root,
        'enabled': capabilities('standard'),
        'source_snapshot_paths': lambda root, enabled: (source_path,),
        'sha256': synthetic_sha256,
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'source-snapshot-only', 'exec'), namespace)

    original_resolve = Path.resolve

    def resolve_external_source(path, *args, **kwargs):
        if path == source_path:
            return outside_path
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', resolve_external_source)
    with pytest.raises(ValueError, match='escapes build root'):
        namespace['source_snapshot']()

    assert hash_calls == []


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


def test_uv_download_checks_pinned_archive_binary_and_reported_target(tmp_path, monkeypatch):
    executable = b'synthetic uv executable'
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, 'w') as archive:
        archive.writestr('uv.exe', executable)
    archive_bytes = archive_bytes.getvalue()
    release = {
        'version': '1.2.3',
        'target': 'x86_64-pc-windows-msvc',
        'archive_name': 'uv-test.zip',
        'archive_bytes': len(archive_bytes),
        'archive_sha256': hashlib.sha256(archive_bytes).hexdigest(),
        'executable_name': 'uv.exe',
        'executable_bytes': len(executable),
        'executable_sha256': hashlib.sha256(executable).hexdigest(),
    }
    monkeypatch.setattr(
        build_provenance.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(stdout='uv 1.2.3 (commit x86_64-pc-windows-msvc)'),
    )
    destination = tmp_path / 'inputs' / 'uv.exe'
    identity = build_provenance.download_uv_binary(
        destination,
        release=release,
        opener=lambda url, timeout: io.BytesIO(archive_bytes),
    )

    assert destination.read_bytes() == executable
    assert identity['version'] == '1.2.3'
    assert identity['archive_sha256'] == release['archive_sha256']
    assert identity['sha256'] == release['executable_sha256']


def test_uv_download_rejects_wrong_archive_or_executable(tmp_path):
    executable = b'synthetic uv executable'
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, 'w') as archive:
        archive.writestr('uv.exe', executable)
    archive_bytes = archive_bytes.getvalue()
    release = {
        'version': '1.2.3',
        'target': 'x86_64-pc-windows-msvc',
        'archive_name': 'uv-test.zip',
        'archive_bytes': len(archive_bytes),
        'archive_sha256': '0' * 64,
        'executable_name': 'uv.exe',
        'executable_bytes': len(executable),
        'executable_sha256': hashlib.sha256(executable).hexdigest(),
    }
    with pytest.raises(ValueError, match='archive checksum'):
        build_provenance.download_uv_binary(
            tmp_path / 'uv.exe', release=release,
            opener=lambda url, timeout: io.BytesIO(archive_bytes),
        )

    release['archive_sha256'] = hashlib.sha256(archive_bytes).hexdigest()
    release['executable_sha256'] = '1' * 64
    with pytest.raises(ValueError, match='executable checksum'):
        build_provenance.download_uv_binary(
            tmp_path / 'uv.exe', release=release,
            opener=lambda url, timeout: io.BytesIO(archive_bytes),
        )
