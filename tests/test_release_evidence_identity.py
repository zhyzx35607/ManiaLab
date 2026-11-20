"""Bind synthetic release reports to actual fixture bytes, without running an EXE."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from manialab import __version__
from scripts import collect_release_validation as collector
from scripts import package_assets, verify_shutdown


def _sha(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture(params=['core', 'standard'])
def release_fixture(tmp_path, monkeypatch, request):
    profile = request.param
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'launcher.py').write_bytes(b'synthetic source fixture')
    build = tmp_path / 'build'
    executable = build / 'dist/ManiaLab/ManiaLab.exe'
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b'synthetic executable fixture; never executed')
    model_bytes = b'synthetic model fixture; never loaded'
    model = dict(filename='basic-pitch.onnx', version='synthetic', url='synthetic-fixture',
        sha256=_sha(model_bytes))
    # Redirect only source/git identity and pinned model metadata to controlled fixtures.
    # Production ZIP, checksum, report, and synthetic model-byte validation still run.
    monkeypatch.setattr(collector, 'ROOT', source)
    monkeypatch.setattr(collector.subprocess, 'check_output', lambda *args, **kwargs: 'synthetic-commit')
    monkeypatch.setattr(collector, 'validate_evidence', lambda *args, **kwargs: {
        'schema': 'manialab-pytest-evidence-v1',
        'source_commit': 'synthetic-commit',
        'source_digest_before': 'a' * 64,
    })
    monkeypatch.setattr(package_assets, 'CORE_MODEL', model)
    declared = package_assets.capabilities(profile)
    identity = dict(build_profile=profile, capabilities=declared,
        executable_sha256=_sha(executable.read_bytes()))
    manifest = dict(identity, version=__version__,
        sources={'launcher.py': _sha((source / 'launcher.py').read_bytes())}, core_models=model,
        worker_sha256='synthetic-worker' if profile != 'core' else None,
        worker_sources={} if profile != 'core' else None, worker_files=[] if profile != 'core' else None,
        piano_model={'sha256': 'synthetic-piano', 'bytes': 1} if profile != 'core' else None,
        guitar_model=None, instrument_worker=None)
    (executable.parent / 'release-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    identity = dict(identity, release_manifest_sha256=_sha(
        (executable.parent / 'release-manifest.json').read_bytes()))
    package = build / 'release'
    package.mkdir()
    archive = package / (package_assets.archive_stem(__version__, profile) + '.zip')
    with zipfile.ZipFile(archive, 'w') as output:
        output.write(executable, 'ManiaLab/ManiaLab.exe')
        output.writestr('ManiaLab/release-manifest.json', json.dumps(manifest))
        output.writestr('ManiaLab/_internal/assets/basic-pitch.onnx', model_bytes)
        output.writestr('ManiaLab/_internal/assets/model-manifest.json', json.dumps({
            'basic_pitch': {key: value for key, value in model.items() if key != 'filename'}}))
        output.writestr('ManiaLab/_internal/assets/arrangement-profiles.json', '{}')
    junit = tmp_path / 'junit.xml'
    junit.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>', encoding='utf-8')
    pytest_evidence = tmp_path / 'pytest-evidence.json'
    pytest_evidence.write_text('{}', encoding='utf-8')
    portable = tmp_path / 'portable.json'
    checks = [dict(identity, check='release_manifest'), {'check': 'startup'}, {'check': 'restart'}]
    if profile == 'core':
        checks += [{'check': name} for name in (
            'basic_analysis', 'provided_stem_analysis_export', 'bad_audio_recovery')]
    else:
        checks += [dict(check='piano_onnx_runtime', sha256='synthetic-piano', bytes=1,
            providers=['CPUExecutionProvider'])]
        checks += [{'check': name} for name in ('deep_analysis', 'six_stem_analysis_export', 'cancel_and_failure')]
    portable.write_text(json.dumps(checks), encoding='utf-8')
    shutdown = tmp_path / 'shutdown.json'
    shutdown_data = dict(identity, version=__version__, passed=True, worker_expected=profile != 'core',
        application_exited=True, worker_stopped_on_parent_exit=profile != 'core')
    if profile == 'core':
        shutdown_data.update(worker_processes_before_exit=[], worker_processes_after_exit=[])
    shutdown.write_text(json.dumps(shutdown_data), encoding='utf-8')
    return SimpleNamespace(build=build, executable=executable, manifest=manifest, identity=identity,
        junit=junit, pytest_evidence=pytest_evidence, portable=portable, shutdown=shutdown,
        checks=checks, shutdown_data=shutdown_data)


def _collect(fixture):
    return collector.collect(fixture.build, fixture.junit, fixture.portable, fixture.shutdown,
        pytest_evidence=fixture.pytest_evidence)


def test_matching_release_reports_are_accepted_for_the_same_fixture_bytes(release_fixture):
    report, output, _ = _collect(release_fixture)
    assert report['executable_sha256'] == _sha(release_fixture.executable.read_bytes())
    assert report['verification']['portable']['passed'] is True
    assert report['verification']['shutdown']['passed'] is True
    assert output.is_file()


@pytest.mark.parametrize('report_kind', ['portable', 'shutdown'])
@pytest.mark.parametrize('field', ['executable_sha256', 'build_profile', 'capabilities',
    'release_manifest_sha256'])
def test_reports_for_other_bytes_or_capabilities_are_rejected(release_fixture, report_kind, field):
    fixture = release_fixture
    document = json.loads(getattr(fixture, report_kind).read_text(encoding='utf-8'))
    identity = document[0] if report_kind == 'portable' else document
    identity[field] = {'executable_sha256': _sha(b'another executable'),
        'build_profile': 'extended', 'capabilities': {'audio': False},
        'release_manifest_sha256': _sha(b'another release manifest')}[field]
    getattr(fixture, report_kind).write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(SystemExit, match='identity'):
        _collect(fixture)


@pytest.mark.parametrize('report_kind', ['portable', 'shutdown'])
@pytest.mark.parametrize('field', ['executable_sha256', 'release_manifest_sha256'])
def test_legacy_reports_without_build_identity_are_rejected(release_fixture, report_kind, field):
    fixture = release_fixture
    document = json.loads(getattr(fixture, report_kind).read_text(encoding='utf-8'))
    identity = document[0] if report_kind == 'portable' else document
    del identity[field]
    getattr(fixture, report_kind).write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(SystemExit, match='identity'):
        _collect(fixture)


def test_duplicate_portable_identity_checks_are_not_ambiguously_accepted(release_fixture):
    fixture = release_fixture
    fixture.portable.write_text(json.dumps(fixture.checks + [fixture.checks[0]]), encoding='utf-8')
    with pytest.raises(SystemExit, match='exactly one'):
        _collect(fixture)


def test_shutdown_report_version_must_match_the_verified_manifest(release_fixture):
    fixture = release_fixture
    fixture.shutdown_data['version'] = 'a different release'
    fixture.shutdown.write_text(json.dumps(fixture.shutdown_data), encoding='utf-8')
    with pytest.raises(SystemExit, match='identity'):
        _collect(fixture)


def test_shutdown_report_identity_uses_file_bytes_and_rejects_changed_files(release_fixture):
    fixture = release_fixture
    identity = verify_shutdown.shutdown_report_identity(fixture.executable, fixture.manifest)
    assert identity == dict(fixture.identity, version=__version__)
    fixture.executable.write_bytes(b'changed executable fixture')
    with pytest.raises(ValueError, match='identity'):
        verify_shutdown.shutdown_report_identity(fixture.executable, fixture.manifest)


def test_same_executable_with_changed_source_manifest_rejects_old_reports(release_fixture):
    fixture = release_fixture
    source = collector.ROOT / 'static/app.js'
    source.parent.mkdir()
    source.write_bytes(b'updated synthetic static behavior')
    fixture.manifest['sources']['static/app.js'] = _sha(source.read_bytes())
    manifest_path = fixture.executable.parent / 'release-manifest.json'
    manifest_path.write_text(json.dumps(fixture.manifest, indent=2), encoding='utf-8')
    archive = next((fixture.build / 'release').glob('*.zip'))
    with zipfile.ZipFile(archive) as package:
        files = {name: package.read(name) for name in package.namelist()}
    files['ManiaLab/release-manifest.json'] = manifest_path.read_bytes()
    files['ManiaLab/_internal/static/app.js'] = source.read_bytes()
    with zipfile.ZipFile(archive, 'w') as package:
        for name, content in files.items():
            package.writestr(name, content)
    assert _sha(fixture.executable.read_bytes()) == fixture.identity['executable_sha256']
    with pytest.raises(SystemExit, match='Portable verification identity'):
        _collect(fixture)

    # Updating only the portable report must still reject the old shutdown evidence.
    fixture.checks[0]['release_manifest_sha256'] = _sha(manifest_path.read_bytes())
    fixture.portable.write_text(json.dumps(fixture.checks), encoding='utf-8')
    with pytest.raises(SystemExit, match='Shutdown verification identity'):
        _collect(fixture)


def test_zip_manifest_bytes_must_match_the_verified_manifest(release_fixture):
    fixture = release_fixture
    archive = next((fixture.build / 'release').glob('*.zip'))
    with zipfile.ZipFile(archive) as package:
        files = {name: package.read(name) for name in package.namelist()}
    manifest_bytes = files['ManiaLab/release-manifest.json']
    files['ManiaLab/release-manifest.json'] = b'\n' + manifest_bytes
    with zipfile.ZipFile(archive, 'w') as package:
        for name, content in files.items():
            package.writestr(name, content)
    with pytest.raises(SystemExit, match='manifest bytes'):
        _collect(fixture)
