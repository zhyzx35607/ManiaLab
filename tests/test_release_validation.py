import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from manialab import __version__
from scripts import pytest_provenance
from scripts.collect_release_validation import collect
from scripts.package_assets import CORE_FILES, capabilities, verify_core_model_files

ROOT = Path(__file__).resolve().parents[1]


def write_pytest_evidence(path, junit_report):
    snapshot = pytest_provenance.source_snapshot(ROOT)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    evidence = {
        "schema": pytest_provenance.SOURCE_SCHEMA,
        "source_commit": commit,
        "argv": ["-m", "pytest", "-q", "--junitxml", str(junit_report)],
        "junit_sha256": hashlib.sha256(junit_report.read_bytes()).hexdigest(),
        "source_digest_before": snapshot["digest"],
        "source_digest_after": snapshot["digest"],
        "exit_code": 0,
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "file_count": snapshot["files"],
    }
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path, evidence

def make_release_fixture(root, profile='standard'):
    build_root = root / "build"
    executable = build_root / "dist" / "ManiaLab" / "ManiaLab.exe"
    package = build_root / "release"
    executable.parent.mkdir(parents=True)
    package.mkdir(parents=True)
    executable.write_bytes(b"portable-executable")
    manifest = {
        "version": __version__,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "sources": {},
        "build_profile": profile,
        "capabilities": capabilities(profile),
        "core_models": verify_core_model_files(ROOT / 'assets'),
        "piano_model": {"sha256": "expected-model-sha", "bytes": 123} if profile != 'core' else None,
        "worker_sha256": "expected-worker-sha" if profile != 'core' else None,
        "worker_sources": {} if profile != 'core' else None,
        "worker_files": [] if profile != 'core' else None,
        "guitar_model": None,
        "instrument_worker": None,
    }
    (executable.parent / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    from scripts.package_assets import archive_stem
    archive_path = package / f"{archive_stem(__version__, profile)}.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(executable, "ManiaLab/ManiaLab.exe")
        archive.writestr(
            "ManiaLab/release-manifest.json",
            json.dumps(manifest),
        )
        for name in CORE_FILES:
            archive.write(ROOT / 'assets' / name, 'ManiaLab/_internal/assets/' + name)
    reports = root / "reports"
    reports.mkdir()
    pytest_report = reports / "junit.xml"
    pytest_report.write_text(
        '<testsuites tests="3" failures="0" errors="0" skipped="1">'
        '<testsuite tests="3" failures="0" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    pytest_evidence, _ = write_pytest_evidence(reports / "pytest-evidence.json", pytest_report)
    portable_report = reports / "portable.json"
    portable_checks = [
        {"check": "release_manifest", "build_profile": profile,
         "capabilities": manifest["capabilities"],
         "executable_sha256": manifest["executable_sha256"],
         "release_manifest_sha256": hashlib.sha256(
             (executable.parent / "release-manifest.json").read_bytes()).hexdigest()},
        {"check": "startup"},
        {"check": "restart"},
    ]
    if profile == 'core':
        portable_checks += [
            {"check": "basic_analysis"},
            {"check": "provided_stem_analysis_export"},
            {"check": "bad_audio_recovery"},
        ]
    else:
        portable_checks += [
            {
                "check": "piano_onnx_runtime",
                "sha256": "expected-model-sha",
                "bytes": 123,
                "providers": ["CPUExecutionProvider"],
            },
            {"check": "deep_analysis"},
            {"check": "six_stem_analysis_export"},
            {"check": "cancel_and_failure"},
        ]
    portable_report.write_text(json.dumps(portable_checks), encoding="utf-8")
    shutdown_report = reports / "shutdown.json"
    shutdown = {
        "passed": True,
        "version": manifest["version"],
        "build_profile": profile,
        "capabilities": manifest["capabilities"],
        "executable_sha256": manifest["executable_sha256"],
        "release_manifest_sha256": hashlib.sha256(
            (executable.parent / "release-manifest.json").read_bytes()).hexdigest(),
        "worker_expected": profile != 'core',
        "application_exited": True,
        "worker_stopped_on_parent_exit": profile != 'core',
    }
    if profile == 'core':
        shutdown.update(worker_processes_before_exit=[], worker_processes_after_exit=[])
    shutdown_report.write_text(
        json.dumps(shutdown), encoding="utf-8"
    )
    return build_root, pytest_report, portable_report, shutdown_report, archive_path, pytest_evidence


def test_collect_release_validation_writes_hashes_and_sanitized_summary(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = make_release_fixture(tmp_path)
    report, output, checksums = collect(
        build_root, pytest_report, portable_report, shutdown_report,
        pytest_evidence=pytest_evidence,
    )

    assert report["version"] == __version__
    assert report["archive"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert report["verification"]["pytest"]["tests"] == 3
    assert report["verification"]["pytest"]["skipped"] == 1
    assert report["verification"]["shutdown"]["passed"] is True
    assert report["verification"]["piano_model"]["cpu_smoke_passed"] is True
    assert report["limitations"]["musical_acceptance"] is False
    assert "song.wav" not in output.read_text(encoding="utf-8")
    assert archive.name in checksums.read_text(encoding="utf-8")


def test_collect_core_validation_requires_core_inventory_and_runtime_checks(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = make_release_fixture(
        tmp_path, profile='core'
    )
    report, output, checksums = collect(build_root, pytest_report, portable_report, shutdown_report,
        pytest_evidence=pytest_evidence)
    assert archive.name.endswith('-core.zip')
    assert report['build_profile'] == 'core'
    assert report['verification']['piano_model'] is None
    assert report['limitations']['separation_weights_bundled'] is False
    assert report['limitations']['musical_acceptance'] is False
    assert report['verification']['shutdown']['worker_expected'] is False
    assert report['verification']['shutdown']['application_exited'] is True
    assert output.name.endswith('-core-validation.json')
    assert archive.name in checksums.read_text(encoding='utf-8')


def test_collect_core_validation_rejects_declared_deep_components(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = make_release_fixture(
        tmp_path, profile='core'
    )
    manifest_path = build_root / 'dist' / 'ManiaLab' / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['worker_sha256'] = 'unexpected-worker'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    for report_path in (portable_report, shutdown_report):
        document = json.loads(report_path.read_text(encoding='utf-8'))
        identity = document[0] if isinstance(document, list) else document
        identity['release_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(document), encoding='utf-8')
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('ManiaLab/ManiaLab.exe', (build_root / 'dist/ManiaLab/ManiaLab.exe').read_bytes())
        bundle.writestr('ManiaLab/release-manifest.json', json.dumps(manifest))
        for name in CORE_FILES:
            bundle.write(ROOT / 'assets' / name, 'ManiaLab/_internal/assets/' + name)

    with pytest.raises(SystemExit, match='Core manifest declares'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_core_validation_rejects_changed_basic_pitch_bytes(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive, pytest_evidence = make_release_fixture(
        tmp_path, profile='core'
    )
    manifest = json.loads(
        (build_root / 'dist' / 'ManiaLab' / 'release-manifest.json').read_text(encoding='utf-8')
    )
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.write(build_root / 'dist/ManiaLab/ManiaLab.exe', 'ManiaLab/ManiaLab.exe')
        bundle.writestr('ManiaLab/release-manifest.json', json.dumps(manifest))
        for name in CORE_FILES:
            asset = ROOT / 'assets' / name
            if name == 'basic-pitch.onnx':
                bundle.writestr('ManiaLab/_internal/assets/' + name, b'changed model')
            else:
                bundle.write(asset, 'ManiaLab/_internal/assets/' + name)

    with pytest.raises(SystemExit, match='Basic Pitch model checksum'):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_missing_portable_checks(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    portable_report.write_text(json.dumps([{"check": "release_manifest"}]), encoding="utf-8")

    with pytest.raises(SystemExit, match="missing required checks"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_piano_smoke_for_other_model(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    checks = json.loads(portable_report.read_text(encoding="utf-8"))
    piano_check = next(item for item in checks if item["check"] == "piano_onnx_runtime")
    piano_check["sha256"] = "different-model-sha"
    portable_report.write_text(json.dumps(checks), encoding="utf-8")

    with pytest.raises(SystemExit, match="Piano runtime smoke does not match"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_missing_pytest_evidence(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, _ = make_release_fixture(tmp_path)

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=tmp_path / "missing-pytest-evidence.json")


def test_collect_release_validation_rejects_legacy_pytest_evidence_schema(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    sidecar = json.loads(pytest_evidence.read_text(encoding="utf-8"))
    sidecar["schema"] = "legacy-junit-without-source-identity"
    pytest_evidence.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_replaced_junit_bytes(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    pytest_report.write_text(
        '<testsuites tests="4" failures="0" errors="0" skipped="0">'
        '<testsuite tests="4" failures="0" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_junit_from_different_source_digest(tmp_path, monkeypatch):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    snapshot = pytest_provenance.source_snapshot(ROOT)
    monkeypatch.setattr(pytest_provenance, "source_snapshot", lambda root: {
        "digest": "f" * 64 if snapshot["digest"] != "f" * 64 else "e" * 64,
        "files": snapshot["files"],
    })

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_source_changes_during_pytest(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    sidecar = json.loads(pytest_evidence.read_text(encoding="utf-8"))
    sidecar["source_digest_after"] = "f" * 64 if sidecar["source_digest_before"] != "f" * 64 else "e" * 64
    pytest_evidence.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)


def test_collect_release_validation_rejects_failed_pytest_sidecar(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _, pytest_evidence = make_release_fixture(tmp_path)
    sidecar = json.loads(pytest_evidence.read_text(encoding="utf-8"))
    sidecar["exit_code"] = 1
    pytest_evidence.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(SystemExit, match="pytest evidence"):
        collect(build_root, pytest_report, portable_report, shutdown_report,
            pytest_evidence=pytest_evidence)
