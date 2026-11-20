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
from scripts.build_provenance import uv_release_identity
from scripts.pytest_provenance import validate_evidence


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_verification_run_id(value):
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_specialist_verification_identity(portable, expected_identity, declared_capabilities):
    specialist_checks = {
        "guitar_real_audio_http",
        "guitar_cold_cancel",
        "instrument_portable_real_audio",
    }
    required_checks = set()
    if declared_capabilities["guitar"]:
        required_checks.update({"guitar_real_audio_http", "guitar_cold_cancel"})
    if declared_capabilities["instruments"]:
        required_checks.add("instrument_portable_real_audio")

    for item in portable:
        if not isinstance(item, dict) or item.get("check") not in specialist_checks:
            continue
        if item["check"] not in required_checks:
            raise SystemExit(
                "Portable specialist evidence is present without the declared profile capability."
            )
        if not _is_verification_run_id(item.get("verification_run_id")):
            raise SystemExit("Portable specialist evidence has an invalid verification_run_id.")
        if any(item.get(field) != value for field, value in expected_identity.items()):
            raise SystemExit(
                "Portable specialist evidence identity differs from this verification run; "
                "rerun EXE verification."
            )


def _is_guitar_model_identity(value):
    if not isinstance(value, dict):
        return False
    try:
        expected = json.loads((ROOT / "assets/guitar-model.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value == expected


def _is_instrument_worker_identity(value):
    if not isinstance(value, dict):
        return False
    from manialab.instrument_bundle import (
        MODELS, MODEL_REVISION, SOURCE_COMMIT, runtime_identity, source_manifest,
    )

    models = value.get("models")
    worker_files = value.get("worker_files")
    verified_models = value.get("verified_models")
    voices = value.get("voices")
    return (
        value.get("source_commit") == SOURCE_COMMIT
        and value.get("model_revision") == MODEL_REVISION
        and value.get("license") == "MIT"
        and _is_sha256(value.get("license_sha256"))
        and value.get("source_files") == source_manifest()["source_files"]
        and models == MODELS
        and isinstance(voices, list)
        and len(voices) == 2
        and all(isinstance(voice, str) for voice in voices)
        and set(voices) == {"bass", "other"}
        and value.get("prepared_only") is False
        and value.get("portable_runtime_verified") is True
        and value.get("worker_sources") == runtime_identity(ROOT)
        and _is_sha256(value.get("worker_sha256"))
        and isinstance(worker_files, dict)
        and bool(worker_files)
        and all(
            isinstance(name, str)
            and name
            and isinstance(record, dict)
            and type(record.get("bytes")) is int
            and record["bytes"] >= 0
            and _is_sha256(record.get("sha256"))
            for name, record in worker_files.items()
        )
        and worker_files.get("ManiaLabInstrument.exe", {}).get("sha256") == value["worker_sha256"]
        and isinstance(verified_models, list)
        and len(verified_models) == len(MODELS)
        and all(isinstance(model, str) for model in verified_models)
        and set(verified_models) == set(MODELS)
        and _is_sha256(value.get("verification_sha256"))
    )


def _require_specialist_components(manifest, declared_capabilities):
    uv_runtime = manifest.get("uv_runtime")
    if declared_capabilities["mapperatorinator_install"]:
        if uv_runtime != uv_release_identity():
            raise SystemExit("Portable manifest has a missing or invalid uv_runtime identity.")
    elif "uv_runtime" not in manifest or uv_runtime is not None:
        raise SystemExit("Portable manifest must declare uv_runtime=null without the installer capability.")

    guitar_model = manifest.get("guitar_model")
    if declared_capabilities["guitar"]:
        if not _is_guitar_model_identity(guitar_model):
            raise SystemExit("Portable manifest has a missing or invalid guitar_model identity.")
    elif "guitar_model" not in manifest or guitar_model is not None:
        raise SystemExit("Portable manifest must declare guitar_model=null without guitar capability.")

    instrument_worker = manifest.get("instrument_worker")
    if declared_capabilities["instruments"]:
        if not _is_instrument_worker_identity(instrument_worker):
            raise SystemExit("Portable manifest has a missing or invalid instrument_worker identity.")
    elif "instrument_worker" not in manifest or instrument_worker is not None:
        raise SystemExit(
            "Portable manifest must declare instrument_worker=null without instruments capability."
        )

    runtime_license_files = manifest.get("instrument_runtime_license_files")
    if declared_capabilities["instruments"]:
        if not isinstance(runtime_license_files, dict) or not runtime_license_files:
            raise SystemExit(
                "Portable manifest has no instrument runtime license file inventory."
            )
    elif "instrument_runtime_license_files" not in manifest or runtime_license_files is not None:
        raise SystemExit(
            "Portable manifest must declare instrument_runtime_license_files=null without instruments capability."
        )

    runtime_license_files = manifest.get("runtime_license_files")
    if not isinstance(runtime_license_files, dict) or not runtime_license_files:
        raise SystemExit("Portable manifest has no runtime license file inventory.")
    runtime_license_inventory = manifest.get("runtime_license_inventory")
    if (
        not isinstance(runtime_license_inventory, dict)
        or type(runtime_license_inventory.get("bytes")) is not int
        or runtime_license_inventory["bytes"] < 0
        or not _is_sha256(runtime_license_inventory.get("sha256"))
    ):
        raise SystemExit("Portable manifest has no valid runtime license inventory identity.")
    license_files = manifest.get("license_files")
    if not isinstance(license_files, dict) or not license_files:
        raise SystemExit("Portable manifest has no complete license file inventory.")


def _is_safe_archive_relative_path(name):
    if not isinstance(name, str) or not name:
        return False
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and path.as_posix() == name
        and ".." not in path.parts
        and "\\" not in name
        and ":" not in name
    )


def _require_archive_file(archive, name, expected_sha256, expected_bytes=None):
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise SystemExit(f"Release ZIP is missing required component: {name}") from exc
    if expected_bytes is not None and info.file_size != expected_bytes:
        raise SystemExit(f"Release ZIP component size differs from its manifest identity: {name}")
    with archive.open(info) as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(f"Release ZIP component differs from its manifest identity: {name}")


def _require_uv_runtime_zip_payload(archive, declared_capabilities):
    if not declared_capabilities["mapperatorinator_install"]:
        return
    identity = uv_release_identity()
    _require_archive_file(
        archive,
        "ManiaLab/_internal/assets/runtime/uv.exe",
        identity["sha256"],
        identity["bytes"],
    )


def _require_archive_json(archive, name, expected):
    try:
        actual = json.loads(archive.read(name))
    except KeyError as exc:
        raise SystemExit(f"Release ZIP is missing required component: {name}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Release ZIP component is invalid JSON: {name}") from exc
    if actual != expected:
        raise SystemExit(f"Release ZIP component differs from its manifest identity: {name}")


def _require_archive_inventory(archive, prefix, expected, *, require_bytes=False):
    if not isinstance(expected, dict) or not expected:
        raise SystemExit(f"Release manifest has no component inventory for {prefix}")
    folded = set()
    for name, record in expected.items():
        if not _is_safe_archive_relative_path(name) or name.casefold() in folded:
            raise SystemExit(f"Release manifest has an unsafe component path under {prefix}")
        folded.add(name.casefold())
        if record is None:
            if require_bytes:
                raise SystemExit(f"Release manifest has an invalid component identity under {prefix}")
            continue
        if (
            not isinstance(record, dict)
            or (
                "bytes" in record
                and (type(record["bytes"]) is not int or record["bytes"] < 0)
            )
            or (require_bytes and type(record.get("bytes")) is not int)
            or not _is_sha256(record.get("sha256"))
        ):
            raise SystemExit(f"Release manifest has an invalid component identity under {prefix}")

    actual = {
        info.filename[len(prefix):]
        for info in archive.infolist()
        if info.filename.startswith(prefix) and not info.is_dir()
    }
    actual_folded = [name.casefold() for name in actual]
    if len(actual) != len(actual_folded) or len(actual_folded) != len(set(actual_folded)):
        raise SystemExit(f"Release ZIP has duplicate component paths under {prefix}")
    if actual != set(expected):
        raise SystemExit(f"Release ZIP component inventory differs from its manifest: {prefix}")
    for name, record in expected.items():
        if record is not None:
            _require_archive_file(
                archive, prefix + name, record["sha256"], expected_bytes=record.get("bytes")
            )


def _require_specialist_zip_payloads(archive, manifest, declared_capabilities):
    if declared_capabilities["guitar"]:
        identity = manifest["guitar_model"]
        assets_prefix = "ManiaLab/_internal/assets/"
        _require_archive_json(archive, assets_prefix + "guitar-model.json", identity)
        _require_archive_file(
            archive,
            assets_prefix + "deep/" + identity["filename"],
            identity["sha256"],
        )
        for path_field, hash_field in (
            ("license_file", "license_sha256"),
            ("source_license_declaration_file", "source_license_declaration_sha256"),
            ("notice_file", "notice_sha256"),
        ):
            relative = identity.get(path_field)
            if not _is_safe_archive_relative_path(relative):
                raise SystemExit("Portable manifest has an unsafe guitar attribution path.")
            _require_archive_file(
                archive, "ManiaLab/" + relative, identity.get(hash_field)
            )

    if declared_capabilities["instruments"]:
        identity = manifest["instrument_worker"]
        prefix = "ManiaLab/_internal/assets/instruments/"
        files = {
            "manifest.json": None,
            "verification.json": {"sha256": identity["verification_sha256"]},
            "LICENSE-tsumugi": {"sha256": identity["license_sha256"]},
        }
        files.update({
            f"checkpoints/best_model_{name}.pth": {"sha256": digest}
            for name, digest in identity["models"].items()
        })
        files.update({
            f"source/{name}": {"sha256": digest}
            for name, digest in identity["source_files"].items()
        })
        files.update({
            f"ManiaLabInstrument/{name}": record
            for name, record in identity["worker_files"].items()
        })
        _require_archive_inventory(archive, prefix, files)
        _require_archive_json(archive, prefix + "manifest.json", identity)
        _require_archive_inventory(
            archive,
            "ManiaLab/licenses/instrument-runtime/",
            manifest["instrument_runtime_license_files"],
            require_bytes=True,
        )


def _require_runtime_license_zip_payloads(archive, manifest):
    files = manifest["runtime_license_files"]
    inventory = manifest["runtime_license_inventory"]
    for prefix in ("ManiaLab/licenses/", "ManiaLab/_internal/licenses/"):
        _require_archive_inventory(
            archive, prefix + "runtime/", files, require_bytes=True
        )
        _require_archive_file(
            archive,
            prefix + "runtime-inventory.json",
            inventory["sha256"],
            expected_bytes=inventory["bytes"],
        )


def _require_license_zip_payloads(archive, manifest):
    files = manifest["license_files"]
    for prefix in ("ManiaLab/licenses/", "ManiaLab/_internal/licenses/"):
        _require_archive_inventory(archive, prefix, files, require_bytes=True)


_WINDOWS_INVALID_PATH_CHARS = frozenset('<>:"|?*\\')
_WINDOWS_RESERVED_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{digit}" for digit in "123456789"}
    | {f"lpt{digit}" for digit in "123456789"}
    | {f"com{digit}" for digit in "¹²³"}
    | {f"lpt{digit}" for digit in "¹²³"}
)


def _windows_archive_component_key(component):
    return component.rstrip(" .").casefold()


def _is_safe_windows_archive_component(component):
    if (
        not component
        or component.startswith(" ")
        or component.endswith((".", " "))
        or any(ord(character) < 32 or character in _WINDOWS_INVALID_PATH_CHARS
               for character in component)
    ):
        return False
    device_basename = component.split(".", 1)[0].rstrip(" .").casefold()
    return device_basename not in _WINDOWS_RESERVED_DEVICE_NAMES


def _require_archive_path_tree(infos):
    names = set()
    nodes = {}
    for info in infos:
        name = info.filename
        if name in names:
            raise SystemExit("Release ZIP has duplicate entries.")
        names.add(name)
        path = PurePosixPath(name)
        normalized_name = name[:-1] if info.is_dir() and name.endswith("/") else name
        if (
            path.is_absolute()
            or not path.parts
            or path.as_posix() != normalized_name
            or name.endswith("/") != info.is_dir()
            or path.parts[0] != "ManiaLab"
            or any(part in (".", "..") for part in path.parts)
            or "\\" in name
            or any(not _is_safe_windows_archive_component(part) for part in path.parts)
        ):
            raise SystemExit("Release ZIP contains an unsafe entry path.")

        for index, component in enumerate(path.parts):
            spelling = path.parts[:index + 1]
            key = tuple(_windows_archive_component_key(part) for part in spelling)
            kind = "dir" if index < len(path.parts) - 1 or info.is_dir() else "file"
            previous = nodes.get(key)
            if previous is not None and previous != (spelling, kind):
                raise SystemExit(
                    "Release ZIP has case-insensitive path collisions or file/directory conflicts."
                )
            nodes[key] = (spelling, kind)


def _require_mode_results(check, expected_modes, required_flags):
    modes = check.get("modes")
    if not isinstance(modes, list):
        return False
    flags = [mode.get("allow_predicted_releases") for mode in modes if isinstance(mode, dict)]
    return (
        len(flags) == len(modes) == len(expected_modes)
        and all(type(flag) is bool for flag in flags)
        and set(flags) == expected_modes
        and all(
            mode.get(name) is True
            for mode in modes
            for name in required_flags
        )
    )


def _require_guitar_evidence(portable):
    expected = {
        "guitar-00_BN1-129-Eb_comp": {False, True},
        "sakura-nova-guitar": {False},
        "youdiantian-htdemucs6-guitar": {False},
    }
    checks = [item for item in portable if isinstance(item, dict)
        and item.get("check") == "guitar_real_audio_http"]
    if any(not isinstance(item.get("case"), str) for item in checks):
        raise SystemExit("Portable verification has an invalid guitar real-audio case selector.")
    by_case = {item.get("case"): item for item in checks}
    if len(checks) != len(expected) or set(by_case) != set(expected):
        raise SystemExit("Portable verification is missing required guitar real-audio cases.")
    for case, modes in expected.items():
        result = by_case[case]
        if not _is_sha256(result.get("fixture_pcm_sha256")) or not _require_mode_results(
            result, modes, ("source_pcm_equal", "raw_trace_equal", "cache_reused_after_restart")
        ):
            raise SystemExit(f"Guitar real-audio evidence is incomplete for case {case}.")

    cancels = [item for item in portable if isinstance(item, dict)
        and item.get("check") == "guitar_cold_cancel"]
    if len(cancels) != 1 or cancels[0].get("partial_analysis_saved") is not False \
            or cancels[0].get("service_alive") is not True:
        raise SystemExit("Portable verification is missing the guitar cold-cancel check.")


def _require_instrument_evidence(portable):
    expected = {"bass": "bass_v2", "other": "other_v1_5"}
    checks = [item for item in portable if isinstance(item, dict)
        and item.get("check") == "instrument_portable_real_audio"]
    if any(not isinstance(item.get("voice"), str) for item in checks):
        raise SystemExit("Portable verification has an invalid instrument real-audio voice selector.")
    by_voice = {item.get("voice"): item for item in checks}
    if len(checks) != len(expected) or set(by_voice) != set(expected):
        raise SystemExit("Portable verification is missing required instrument real-audio voices.")
    for voice, model in expected.items():
        result = by_voice[voice]
        if (
            result.get("model") != model
            or not _is_sha256(result.get("fixture_sha256"))
            or not _require_mode_results(
                result, {False, True}, ("cache_reused_after_restart", "source_preserved")
            )
        ):
            raise SystemExit(f"Instrument real-audio evidence is incomplete for voice {voice}.")


def collect(build_root, pytest_report, portable_report, shutdown_report, *, pytest_evidence=None):
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
    _require_specialist_components(manifest, declared_capabilities)
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
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if pytest_evidence is None:
        pytest_path = Path(pytest_report)
        pytest_evidence = pytest_path.with_name(pytest_path.name + ".provenance.json")
    try:
        pytest_identity = validate_evidence(
            pytest_evidence, pytest_report, root=ROOT, expected_commit=commit
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Pytest evidence provenance is invalid: {exc}") from exc

    size = archive_path.stat().st_size
    if size >= 2_000_000_000:
        raise SystemExit("Release ZIP exceeds the configured 2 GB publishing guard.")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        _require_archive_path_tree(infos)
        if archive.testzip() is not None:
            raise SystemExit("Release ZIP has a CRC failure.")
        if sha256_bytes(archive.read("ManiaLab/ManiaLab.exe")) != executable_sha:
            raise SystemExit("Release ZIP executable does not match the verified build.")
        archive_manifest = archive.read("ManiaLab/release-manifest.json")
        if sha256_bytes(archive_manifest) != sha256(manifest_path):
            raise SystemExit("Release ZIP manifest bytes do not match the verified build.")
        if json.loads(archive_manifest) != manifest:
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
        _require_runtime_license_zip_payloads(archive, manifest)
        _require_license_zip_payloads(archive, manifest)
        _require_uv_runtime_zip_payload(archive, declared_capabilities)
        _require_specialist_zip_payloads(archive, manifest, declared_capabilities)

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
    if declared_capabilities["guitar"]:
        required_checks.update({"guitar_real_audio_http", "guitar_cold_cancel"})
    if declared_capabilities["instruments"]:
        required_checks.add("instrument_portable_real_audio")
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
    verification_run_id = portable_identities[0].get('verification_run_id')
    if not _is_verification_run_id(verification_run_id):
        raise SystemExit('Portable verification identity has an invalid verification_run_id.')
    expected_specialist_identity = dict(expected_identity, verification_run_id=verification_run_id)
    _require_specialist_verification_identity(
        portable, expected_specialist_identity, declared_capabilities
    )
    if declared_capabilities["guitar"]:
        _require_guitar_evidence(portable)
    if declared_capabilities["instruments"]:
        _require_instrument_evidence(portable)
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
                "evidence_sha256": sha256(pytest_evidence),
                "evidence_schema": pytest_identity["schema"],
                "source_commit": pytest_identity["source_commit"],
                "source_digest": pytest_identity["source_digest_before"],
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
    parser.add_argument("--pytest-evidence", type=Path)
    parser.add_argument("--portable-report", type=Path, required=True)
    parser.add_argument("--shutdown-report", type=Path, required=True)
    args = parser.parse_args()
    report, output, checksums = collect(
        args.build_root, args.pytest_report, args.portable_report, args.shutdown_report,
        pytest_evidence=args.pytest_evidence,
    )
    print(json.dumps({
        "validation": str(output),
        "checksums": str(checksums),
        "source_commit": report["source_commit"],
        "archive": report["archive"],
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
