import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_verified_pytest as runner
from scripts import pytest_provenance


def _snapshot(digest='a' * 64, files=12):
    return {'digest': digest, 'files': files}


def test_source_snapshot_includes_release_metadata_without_hashing_model_weights():
    names = {relative for relative, _ in pytest_provenance._source_files(pytest_provenance.ROOT)}
    assert 'assets/piano-model.json' in names
    assert 'assets/guitar-model.json' in names
    assert 'assets/model-manifest.json' in names
    assert 'assets/basic-pitch.onnx' in names
    assert 'docs/tsumugi-attack-recovery-protocol.md' in names
    assert 'assets/mapperatorinator-source.zip' not in names


def test_verified_runner_writes_source_and_junit_identity_without_running_pytest(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    junit = root / 'reports' / 'junit.xml'
    sidecar = root / 'reports' / 'evidence.json'
    junit_bytes = b'<testsuite tests="1" failures="0"/>'
    calls = []
    snapshots = iter([_snapshot(), _snapshot()])

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        junit.parent.mkdir(parents=True, exist_ok=True)
        junit.write_bytes(junit_bytes)
        return SimpleNamespace(returncode=0)

    times = iter(['2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z'])
    monkeypatch.setattr(runner, 'source_snapshot', lambda path: next(snapshots))
    monkeypatch.setattr(runner, 'source_commit', lambda path: 'b' * 40)
    monkeypatch.setattr(runner, 'utc_now', lambda: next(times))
    monkeypatch.setattr(runner.subprocess, 'run', fake_run)

    result = runner.run(Path('reports/junit.xml'), Path('reports/evidence.json'),
        ['tests/synthetic_test.py', '-k', 'small'], root=root)

    assert result == 0
    assert len(calls) == 1
    command, cwd, check = calls[0]
    assert command[:4] == [runner.sys.executable, '-m', 'pytest', '-q']
    assert command[4:6] == ['tests/synthetic_test.py', '-k']
    assert command[-1] == f'--junitxml={junit}'
    assert cwd == root and check is False
    evidence = json.loads(sidecar.read_text(encoding='utf-8'))
    assert evidence == {
        'schema': pytest_provenance.SOURCE_SCHEMA,
        'source_commit': 'b' * 40,
        'argv': command,
        'junit_sha256': hashlib.sha256(junit_bytes).hexdigest(),
        'source_digest_before': 'a' * 64,
        'source_digest_after': 'a' * 64,
        'exit_code': 0,
        'started_at': '2026-01-01T00:00:00Z',
        'completed_at': '2026-01-01T00:00:01Z',
        'file_count': 12,
    }


def test_runner_fails_when_sources_change_during_pytest_but_keeps_evidence(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    junit = root / 'junit.xml'
    sidecar = root / 'pytest-evidence.json'
    snapshots = iter([_snapshot('a' * 64), _snapshot('c' * 64)])
    monkeypatch.setattr(runner, 'source_snapshot', lambda path: next(snapshots))
    monkeypatch.setattr(runner, 'source_commit', lambda path: 'd' * 40)
    monkeypatch.setattr(runner, 'utc_now', lambda: '2026-01-01T00:00:00Z')

    def fake_run(command, *, cwd, check):
        junit.write_text('<testsuite/>', encoding='utf-8')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    assert runner.run(junit, sidecar, [], root=root) == 1
    evidence = json.loads(sidecar.read_text(encoding='utf-8'))
    assert evidence['source_digest_before'] == 'a' * 64
    assert evidence['source_digest_after'] == 'c' * 64
    assert evidence['exit_code'] == 0
    assert evidence['source_digest_before'] != evidence['source_digest_after']


def test_runner_fails_when_junit_is_missing_and_persists_failure_sidecar(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    junit = root / 'missing' / 'junit.xml'
    sidecar = root / 'missing' / 'evidence.json'
    snapshots = iter([_snapshot(), _snapshot()])
    monkeypatch.setattr(runner, 'source_snapshot', lambda path: next(snapshots))
    monkeypatch.setattr(runner, 'source_commit', lambda path: 'e' * 40)
    monkeypatch.setattr(runner, 'utc_now', lambda: '2026-01-01T00:00:00Z')
    monkeypatch.setattr(runner.subprocess, 'run',
        lambda *args, **kwargs: SimpleNamespace(returncode=0))

    assert runner.run(junit, sidecar, [], root=root) == 1
    evidence = json.loads(sidecar.read_text(encoding='utf-8'))
    assert evidence['junit_sha256'] == ''
    assert evidence['source_digest_before'] == evidence['source_digest_after']
    with pytest.raises(ValueError, match='junit_sha256|JUnit'):
        pytest_provenance.validate_evidence(sidecar, junit, root=root)


def test_runner_passes_through_pytest_failure_and_records_report_hash(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    junit = root / 'junit.xml'
    sidecar = root / 'evidence.json'
    content = b'<testsuite tests="1" failures="1"/>'
    monkeypatch.setattr(runner, 'source_snapshot', lambda path: _snapshot())
    monkeypatch.setattr(runner, 'source_commit', lambda path: 'f' * 40)
    monkeypatch.setattr(runner, 'utc_now', lambda: '2026-01-01T00:00:00Z')

    def fake_run(command, *, cwd, check):
        junit.write_bytes(content)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    assert runner.run(junit, sidecar, [], root=root) == 1
    evidence = json.loads(sidecar.read_text(encoding='utf-8'))
    assert evidence['exit_code'] == 1
    assert evidence['junit_sha256'] == hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match='failed pytest|nonzero'):
        pytest_provenance.validate_evidence(sidecar, junit, root=root)
