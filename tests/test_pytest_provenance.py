import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

from scripts import pytest_provenance


def test_source_snapshot_includes_release_metadata_without_hashing_model_weights():
    names = {relative for relative, _ in pytest_provenance._source_files(pytest_provenance.ROOT)}
    assert 'assets/piano-model.json' in names
    assert 'assets/guitar-model.json' in names
    assert 'assets/model-manifest.json' in names
    assert 'assets/basic-pitch.onnx' in names
    assert 'assets/run_mapper.py' in names
    assert 'assets/arrangement-profiles.json' in names
    assert 'assets/deep-models.json' in names
    assert 'licenses/HFGuitar-Model-Card.md' in names
    assert 'licenses/HFGuitar-Source-Setup.py.txt' in names
    assert 'licenses/FFmpeg-build-info.txt' not in names
    assert 'assets/deep/hdemucs_high_trained.pt' not in names
    assert 'docs/tsumugi-attack-recovery-protocol.md' in names
    assert 'assets/mapperatorinator-source.zip' not in names
    assert {
        'tests/fixtures/https_proxy_tls/ca.pem',
        'tests/fixtures/https_proxy_tls/origin-cert.pem',
        'tests/fixtures/https_proxy_tls/origin-key.pem',
        'tests/fixtures/https_proxy_tls/proxy-cert.pem',
        'tests/fixtures/https_proxy_tls/proxy-key.pem',
    } <= names


def test_source_snapshot_hashes_nonignored_pem_and_omits_gitignored_pem(tmp_path):
    root = tmp_path
    subprocess.run(['git', 'init', '--quiet', str(root)], check=True)
    fixture_dir = root / 'tests' / 'fixtures' / 'https_proxy_tls'
    fixture_dir.mkdir(parents=True)
    tracked_candidate = fixture_dir / 'origin-cert.pem'
    ignored = fixture_dir / 'ignored.pem'
    (root / '.gitignore').write_text(
        'tests/fixtures/https_proxy_tls/ignored.pem\n', encoding='utf-8'
    )
    tracked_candidate.write_bytes(b'test certificate one')
    ignored.write_bytes(b'ignored certificate')

    names = {relative for relative, _ in pytest_provenance._source_files(root)}
    assert 'tests/fixtures/https_proxy_tls/origin-cert.pem' in names
    assert 'tests/fixtures/https_proxy_tls/ignored.pem' not in names
    before = pytest_provenance.source_snapshot(root)

    tracked_candidate.write_bytes(b'test certificate two')
    after = pytest_provenance.source_snapshot(root)

    assert before['digest'] != after['digest']
    assert before['files'] == after['files'] == 2


def test_source_snapshot_excludes_synthetic_user_data_under_repository_ignores(tmp_path):
    root = tmp_path
    subprocess.run(['git', 'init', '--quiet', str(root)], check=True)
    (root / '.gitignore').write_bytes((pytest_provenance.ROOT / '.gitignore').read_bytes())

    ignored_paths = (
        'test-data/user-audio.wav',
        'test-data/reference.osu',
        'test-data/reference.osz',
        'test-data/model.pt',
        'tests/artifacts/audio.wav',
        'tests/artifacts/map.osu',
        'tests/artifacts/package.osz',
        'tests/artifacts/model.pth',
        'tests/artifacts/model.th',
        'tests/artifacts/model.ckpt',
        'assets/deep/model.pt',
        'assets/deep/model.pth',
        'assets/deep/model.th',
        'assets/deep/model.ckpt',
        'assets/runtime/uv.exe',
        'assets/instruments/checkpoints/model.pth',
    )
    for relative in ignored_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic ignored placeholder')

    basic_pitch = root / 'assets' / 'basic-pitch.onnx'
    basic_pitch.parent.mkdir(parents=True, exist_ok=True)
    basic_pitch.write_bytes(b'synthetic pinned Basic Pitch placeholder')

    names = {relative for relative, _ in pytest_provenance._source_files(root)}
    assert names == {'.gitignore', 'assets/basic-pitch.onnx'}
    assert names.isdisjoint(ignored_paths)


def _pytest_subprocess(case, *extra):
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-c', str(pytest_provenance.ROOT / 'pytest.ini'), str(case), *extra],
        cwd=pytest_provenance.ROOT, capture_output=True, text=True, check=False,
    )


def test_autoloaded_hook_writes_valid_sidecar_after_real_small_pytest(tmp_path):
    case = pytest_provenance.ROOT / 'tests' / ('_pytest_provenance_smoke_' + uuid.uuid4().hex + '.py')
    junit = tmp_path / 'smoke.xml'
    try:
        case.write_text('def test_tiny():\n    assert 2 + 2 == 4\n', encoding='utf-8')
        result = _pytest_subprocess(case, f'--junitxml={junit}')
        sidecar = junit.with_name(junit.name + '.provenance.json')
        assert result.returncode == 0, result.stdout + result.stderr
        assert junit.is_file() and sidecar.is_file(), result.stdout + result.stderr
        evidence = pytest_provenance.validate_evidence(sidecar, junit)
        assert evidence['exit_code'] == 0
        assert evidence['argv'][-1] == f'--junitxml={junit}'
    finally:
        case.unlink(missing_ok=True)


def test_autoloaded_hook_fails_when_sources_change_during_pytest(tmp_path):
    case = pytest_provenance.ROOT / 'tests' / ('_pytest_provenance_mutation_' + uuid.uuid4().hex + '.py')
    junit = tmp_path / 'mutation.xml'
    try:
        case.write_text(
            'from pathlib import Path' + chr(10)
            + 'def test_mutates_its_own_source():' + chr(10)
            + '    Path(__file__).write_text("# changed", encoding="utf-8")' + chr(10),
            encoding='utf-8',
        )
        result = _pytest_subprocess(case, f'--junitxml={junit}')
        sidecar = junit.with_name(junit.name + '.provenance.json')
        assert result.returncode != 0, result.stdout + result.stderr
        assert sidecar.is_file()
        evidence = json.loads(sidecar.read_text(encoding='utf-8'))
        assert evidence['source_digest_before'] != evidence['source_digest_after']
        assert evidence['exit_code'] == 0
        with pytest.raises(ValueError, match='source (digest|file count)'):
            pytest_provenance.validate_evidence(sidecar, junit)
    finally:
        case.unlink(missing_ok=True)


def test_autoloaded_hook_does_not_create_sidecar_without_junit_option(tmp_path):
    case = tmp_path / 'no_junit_case.py'
    case.write_text('def test_tiny():\n    assert True\n', encoding='utf-8')
    result = _pytest_subprocess(case)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.glob('*.provenance.json'))


def test_sessionfinish_writes_failure_sidecar_when_junit_is_missing(tmp_path):
    from _pytest.stash import Stash

    junit = tmp_path / 'missing.xml'
    evidence = junit.with_name(junit.name + '.provenance.json')
    stash = Stash()
    snapshot = pytest_provenance.source_snapshot()
    stash[pytest_provenance._SESSION_STATE] = {
        'junit_path': junit,
        'evidence_path': evidence,
        'source_commit': pytest_provenance.source_commit(),
        'snapshot_before': snapshot,
        'started_at': pytest_provenance.utc_now(),
        'argv': [sys.executable, '-m', 'pytest'],
    }
    session = SimpleNamespace(config=SimpleNamespace(stash=stash), exitstatus=pytest.ExitCode.OK)
    finish = pytest_provenance.pytest_sessionfinish(session, pytest.ExitCode.OK)
    next(finish)
    with pytest.raises(StopIteration):
        next(finish)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert evidence.is_file()
    sidecar = json.loads(evidence.read_text(encoding='utf-8'))
    assert sidecar['junit_sha256'] == ''
    assert sidecar['source_digest_before'] == sidecar['source_digest_after']
    with pytest.raises(ValueError, match='junit_sha256|JUnit'):
        pytest_provenance.validate_evidence(evidence, junit)
