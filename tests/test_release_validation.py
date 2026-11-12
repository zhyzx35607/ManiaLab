import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from manialab import __version__
from scripts.collect_release_validation import collect


def make_release_fixture(root):
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
    }
    (executable.parent / "release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    archive_path = package / f"ManiaLab-{__version__}-windows-x64.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(executable, "ManiaLab/ManiaLab.exe")
        archive.writestr(
            "ManiaLab/release-manifest.json",
            json.dumps(manifest),
        )
    reports = root / "reports"
    reports.mkdir()
    pytest_report = reports / "junit.xml"
    pytest_report.write_text(
        '<testsuites tests="3" failures="0" errors="0" skipped="1">'
        '<testsuite tests="3" failures="0" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    portable_report = reports / "portable.json"
    portable_report.write_text(json.dumps([
        {"check": name} for name in (
            "release_manifest",
            "deep_analysis",
            "six_stem_analysis_export",
            "cancel_and_failure",
        )
    ]), encoding="utf-8")
    shutdown_report = reports / "shutdown.json"
    shutdown_report.write_text(
        json.dumps({"worker_stopped_on_parent_exit": True}), encoding="utf-8"
    )
    return build_root, pytest_report, portable_report, shutdown_report, archive_path


def test_collect_release_validation_writes_hashes_and_sanitized_summary(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, archive = make_release_fixture(tmp_path)
    report, output, checksums = collect(
        build_root, pytest_report, portable_report, shutdown_report
    )

    assert report["version"] == __version__
    assert report["archive"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert report["verification"]["pytest"]["tests"] == 3
    assert report["verification"]["pytest"]["skipped"] == 1
    assert report["verification"]["shutdown"]["passed"] is True
    assert report["limitations"]["musical_acceptance"] is False
    assert "song.wav" not in output.read_text(encoding="utf-8")
    assert archive.name in checksums.read_text(encoding="utf-8")


def test_collect_release_validation_rejects_missing_portable_checks(tmp_path):
    build_root, pytest_report, portable_report, shutdown_report, _ = make_release_fixture(tmp_path)
    portable_report.write_text(json.dumps([{"check": "release_manifest"}]), encoding="utf-8")

    with pytest.raises(SystemExit, match="missing required checks"):
        collect(build_root, pytest_report, portable_report, shutdown_report)
