"""Validate a portable build and write sanitized, version-specific release evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab import __version__
from scripts.package_assets import (
    archive_stem, capabilities, verify_core_model_payload, verify_core_payload,
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(build_root, pytest_report, portable_report, shutdown_report):
    build_root = Path(build_root).resolve()
    package_dir = build_root / "release"
    executable = build_root / "dist" / "ManiaLab" / "ManiaLab.exe"
    manifest_path = executable.parent / "release-manifest.json"
    if not all(path.is_file() for path in (executable, manifest_path)):
        raise SystemExit("Portable executable or manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest.get("build_profile")
    try:
        declared_capabilities = capabilities(profile)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Portable manifest has an unknown build profile: {profile!r}") from exc
    if manifest.get("capabilities") != declared_capabilities:
        raise SystemExit("Portable capabilities do not match the declared build profile.")
    stem = archive_stem(__version__, profile)
    archive_path = package_dir / f"{stem}.zip"
    output = package_dir / f"{stem}-validation.json"
    checksums = package_dir / f"{stem}-SHA256SUMS.txt"
    if not archive_path.is_file():
        raise SystemExit("Portable ZIP is missing.")
    executable_sha = sha256(executable)
    if manifest.get("version") != __version__ or manifest.get("executable_sha256") != executable_sha:
        raise SystemExit("Portable executable identity does not match the release manifest.")
    for relative, expected in manifest["sources"].items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or sha256(source) != expected:
            raise SystemExit(f"Source snapshot mismatch: {relative}")

    size = archive_path.stat().st_size
    if size >= 2_000_000_000:
        raise SystemExit("Release ZIP exceeds the configured 2 GB publishing guard.")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise SystemExit("Release ZIP has duplicate entries or a CRC failure.")
        for name in names:
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or not path.parts
                or path.parts[0] != "ManiaLab"
                or any(part in (".", "..") for part in path.parts)
                or "\\" in name
            ):
                raise SystemExit("Release ZIP contains an unsafe entry path.")
        if sha256_bytes(archive.read("ManiaLab/ManiaLab.exe")) != executable_sha:
            raise SystemExit("Release ZIP executable does not match the verified build.")
        if json.loads(archive.read("ManiaLab/release-manifest.json")) != manifest:
            raise SystemExit("Release ZIP manifest does not match the verified build.")
        if profile == "core":
            try:
                verify_core_payload(names)
            except ValueError as exc:
                raise SystemExit(f"Core distribution contains an invalid payload: {exc}") from exc
        try:
            model_manifest = json.loads(
                archive.read("ManiaLab/_internal/assets/model-manifest.json")
            )
            with archive.open("ManiaLab/_internal/assets/basic-pitch.onnx") as stream:
                core_models = verify_core_model_payload(model_manifest, stream)
        except (ValueError, KeyError) as exc:
            raise SystemExit(f"Release ZIP is missing valid core model provenance: {exc}") from exc
        if manifest.get("core_models") != core_models:
            raise SystemExit("Core model identity does not match the release manifest.")

    suites = ET.parse(pytest_report).getroot()
    suite_nodes = [suites] if suites.tag == "testsuite" else list(suites.findall("testsuite"))
    totals = {
        key: sum(int(node.attrib.get(key, "0")) for node in suite_nodes)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if totals["tests"] <= 0 or totals["failures"] or totals["errors"]:
        raise SystemExit(f"Pytest report is not clean: {totals}")

    portable = json.loads(Path(portable_report).read_text(encoding="utf-8"))
    required_checks = {"release_manifest", "startup", "restart"}
    if profile == "core":
        required_checks.update({"basic_analysis", "provided_stem_analysis_export", "bad_audio_recovery"})
    else:
        required_checks.update({
            "piano_onnx_runtime",
            "deep_analysis",
            "six_stem_analysis_export",
            "cancel_and_failure",
        })
    if not isinstance(portable, list):
        raise SystemExit("Portable verification report has an unexpected format.")
    portable_checks = [item.get("check") for item in portable if isinstance(item, dict)]
    if not required_checks.issubset(portable_checks):
        raise SystemExit("Portable verification report is missing required checks.")
    portable_identities = [item for item in portable
        if isinstance(item, dict) and item.get('check') == 'release_manifest']
    if len(portable_identities) != 1:
        raise SystemExit('Portable verification must contain exactly one release_manifest identity check.')
    expected_identity = {field: manifest[field] for field in
        ('build_profile', 'capabilities', 'executable_sha256')}
    expected_identity['release_manifest_sha256'] = sha256(manifest_path)
    if any(portable_identities[0].get(field) != value for field, value in expected_identity.items()):
        raise SystemExit('Portable verification identity differs from this build; rerun EXE verification.')
    piano_model = manifest.get("piano_model")
    if profile == "core":
        if any(manifest.get(key) is not None for key in (
                "worker_sha256", "worker_sources", "worker_files", "piano_model",
                "guitar_model", "instrument_worker")):
            raise SystemExit("Core manifest declares an excluded worker or model.")
        piano_check = None
    else:
        piano_check = next(
            item for item in portable
            if isinstance(item, dict) and item.get("check") == "piano_onnx_runtime"
        )
        if (
            not isinstance(piano_model, dict)
            or piano_check.get("sha256") != piano_model.get("sha256")
            or piano_check.get("bytes") != piano_model.get("bytes")
            or "CPUExecutionProvider" not in piano_check.get("providers", [])
        ):
            raise SystemExit("Piano runtime smoke does not match the release manifest.")

    shutdown = json.loads(Path(shutdown_report).read_text(encoding="utf-8"))
    expected_shutdown_identity = dict(expected_identity, version=manifest['version'])
    if not isinstance(shutdown, dict) or any(
            shutdown.get(field) != value for field, value in expected_shutdown_identity.items()):
        raise SystemExit('Shutdown verification identity differs from this build; rerun shutdown verification.')
    if declared_capabilities["separation_four"]:
        if (
            shutdown.get("passed") is not True
            or shutdown.get("worker_expected") is not True
            or shutdown.get("application_exited") is not True
            or shutdown.get("worker_stopped_on_parent_exit") is not True
        ):
            raise SystemExit("Audio worker did not stop when the application exited.")
    elif (
        shutdown.get("passed") is not True
        or shutdown.get("worker_expected") is not False
        or shutdown.get("application_exited") is not True
        or shutdown.get("worker_processes_before_exit") != []
        or shutdown.get("worker_processes_after_exit") != []
    ):
        raise SystemExit("Core shutdown check did not verify a worker-free application exit.")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = {
        "schema": "manialab-release-validation-v1",
        "version": __version__,
        "source_commit": commit,
        "build_profile": manifest.get("build_profile"),
        "capabilities": manifest.get("capabilities"),
        "archive": {
            "filename": archive_path.name,
            "bytes": size,
            "sha256": sha256(archive_path),
        },
        "executable_sha256": executable_sha,
        "release_manifest_sha256": sha256(manifest_path),
        "source_files": len(manifest["sources"]),
        "verification": {
            "pytest": {
                **totals,
                "passed": True,
                "junit_sha256": sha256(pytest_report),
            },
            "portable": {
                "passed": True,
                "checks": portable_checks,
                "report_sha256": sha256(portable_report),
            },
            "piano_model": {
                "sha256": piano_check["sha256"],
                "bytes": piano_check["bytes"],
                "cpu_smoke_passed": True,
            } if piano_check else None,
            "core_models": manifest.get("core_models"),
            "shutdown": {
                "passed": True,
                "worker_expected": declared_capabilities["separation_four"],
                "worker_stopped_on_parent_exit": (
                    shutdown.get("worker_stopped_on_parent_exit")
                    if declared_capabilities["separation_four"] else None
                ),
                "application_exited": True,
                "report_sha256": sha256(shutdown_report),
            },
            "archive_integrity": {
                "passed": True,
                "entries": len(names),
                "crc_check": True,
            },
        },
        "limitations": {
            "musical_acceptance": False,
            "timing_acceptance": False,
            "research_weights_enabled": False,
            "portable_fixture": "synthetic audio; package/runtime checks only",
            "separation_weights_bundled": declared_capabilities["separation_four"],
        },
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksums.write_text(
        f"{report['archive']['sha256']}  {archive_path.name}\n"
        f"{sha256(output)}  {output.name}\n",
        encoding="utf-8",
    )
    return report, output, checksums


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--pytest-report", type=Path, required=True)
    parser.add_argument("--portable-report", type=Path, required=True)
    parser.add_argument("--shutdown-report", type=Path, required=True)
    args = parser.parse_args()
    report, output, checksums = collect(
        args.build_root, args.pytest_report, args.portable_report, args.shutdown_report
    )
    print(json.dumps({
        "validation": str(output),
        "checksums": str(checksums),
        "source_commit": report["source_commit"],
        "archive": report["archive"],
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
