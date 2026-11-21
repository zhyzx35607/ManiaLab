"""Validate a portable build and write sanitized, version-specific release evidence."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import sys
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT
sys.path.insert(0, str(ROOT))
from manialab import __version__
from scripts.package_assets import (
    CORE_FILES, DEEP_FILES, MAPPER_INSTALLER_FILES, MODEL_METADATA_FILES,
    archive_stem, capabilities, verify_core_model_payload, verify_core_payload,
)
from scripts.build_provenance import source_snapshot_paths, uv_release_identity
from scripts.pytest_provenance import validate_evidence


PORTABLE_CHECK_NAMES = frozenset({
    "release_manifest",
    "core_model",
    "bundled_model",
    "bundled_arrangement_profiles",
    "startup",
    "core_capability_status",
    "basic_analysis",
    "generate_export",
    "provided_stem_analysis_export",
    "unavailable_engines",
    "restart",
    "bad_audio_recovery",
    "piano_onnx_runtime",
    "deep_analysis",
    "local_recommendation",
    "stale_recommendation_rejected",
    "six_stem_analysis_export",
    "piano_cold_inference",
    "piano_manual_expression",
    "piano_restart_warm_cache",
    "guitar_real_audio_http",
    "guitar_cold_cancel",
    "instrument_portable_real_audio",
    "cancel_and_failure",
})
PORTABLE_COMMON_CHECK_NAMES = frozenset({
    "release_manifest",
    "core_model",
    "bundled_arrangement_profiles",
    "startup",
    "generate_export",
    "restart",
})
PORTABLE_CORE_CHECK_NAMES = frozenset({
    "core_capability_status",
    "basic_analysis",
    "provided_stem_analysis_export",
    "unavailable_engines",
    "bad_audio_recovery",
})
PORTABLE_NON_CORE_CHECK_NAMES = frozenset({
    "piano_onnx_runtime",
    "deep_analysis",
    "local_recommendation",
    "stale_recommendation_rejected",
    "six_stem_analysis_export",
    "cancel_and_failure",
})
PORTABLE_PIANO_AUDIO_CHECK_NAMES = frozenset({
    "piano_cold_inference",
    "piano_manual_expression",
    "piano_restart_warm_cache",
})
PORTABLE_GUITAR_CHECK_NAMES = frozenset({
    "guitar_real_audio_http",
    "guitar_cold_cancel",
})
CORE_MEDIA_SCOPE = "packaged Core EXE synthetic native-media smoke; not musical acceptance"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_source_snapshot_keys(sources, source_root, enabled):
    if not isinstance(sources, dict):
        raise SystemExit("Portable manifest has no source snapshot.")

    source_root = Path(source_root).resolve()
    expected_paths = source_snapshot_paths(source_root, enabled)
    expected_keys = tuple(path.relative_to(source_root).as_posix() for path in expected_paths)
    if len(expected_keys) != len(set(expected_keys)):
        raise SystemExit("Build source snapshot contains duplicate canonical paths.")
    if any(not _is_safe_archive_relative_path(key) for key in sources):
        raise SystemExit("Portable manifest has a non-canonical source snapshot path.")
    if set(sources) != set(expected_keys):
        raise SystemExit("Portable manifest source snapshot key inventory differs from build inputs.")
    if any(not _is_sha256(sources[key]) for key in expected_keys):
        raise SystemExit("Portable manifest has an invalid source snapshot identity.")
    return source_root, expected_keys


def _is_verification_run_id(value):
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_core_media_report(report_path, expected_identity, verification_run_id):
    try:
        report_bytes = Path(report_path).read_bytes()
        report = json.loads(report_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Core media verification report is missing or invalid: {exc}") from exc

    expected_scope = CORE_MEDIA_SCOPE
    if not isinstance(report, dict) or type(report.get("schema")) is not int \
            or report["schema"] != 1 \
            or report.get("check") != "core_media_session" \
            or report.get("status") != "passed" or report.get("scope") != expected_scope:
        raise SystemExit("Core media verification report is incomplete or has an invalid scope.")
    cleanup_failures = report.get("cleanup_failures", [])
    if not isinstance(cleanup_failures, list) or cleanup_failures:
        raise SystemExit("Core media verification report has cleanup failures or invalid cleanup_failures.")
    if report.get("build_profile") != "core" or not _matches_identity(report, expected_identity) \
            or report.get("verification_run_id") != verification_run_id:
        raise SystemExit("Core media verification identity differs from this build/run.")
    if not isinstance(report.get("run_id"), str) or not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            report["run_id"]):
        raise SystemExit("Core media verification report has an invalid run_id.")

    raw_base_url = report.get("base_url")
    if not isinstance(raw_base_url, str):
        raise SystemExit("Core media verification report has an invalid base_url.")
    try:
        base_url = urlsplit(raw_base_url)
        port = base_url.port
    except (TypeError, ValueError):
        raise SystemExit("Core media verification report has an invalid base_url.") from None
    if base_url.scheme != "http" or base_url.hostname not in ("127.0.0.1", "localhost", "::1") \
            or port is None or not 1 <= port <= 65535 or base_url.path not in ("", "/") \
            or base_url.query or base_url.fragment or base_url.username or base_url.password:
        raise SystemExit("Core media verification report has an invalid base_url.")

    expected_steps = [
        "synthetic_projects", "audio_range", "seek", "play_pause",
        "rapid_source_switch", "browser_health",
    ]
    steps = report.get("steps")
    if not isinstance(steps, list) or len(steps) != len(expected_steps) \
            or any(not isinstance(step, dict) or step.get("name") != name
                   or step.get("status") != "passed"
                   for step, name in zip(steps, expected_steps)):
        raise SystemExit("Core media verification report is missing passed smoke steps.")

    projects = steps[0].get("projects")
    if not isinstance(projects, list) or len(projects) != 2 \
            or any(not isinstance(project, dict) or not isinstance(project.get("id"), str)
                   or not project["id"] for project in projects) \
            or projects[0]["id"] == projects[1]["id"] \
            or any(type(project.get("bytes")) is not int or project["bytes"] <= 63
                   for project in projects) \
            or projects[0]["bytes"] != projects[1]["bytes"]:
        raise SystemExit("Core media verification report has invalid synthetic projects.")
    project_bytes = projects[0].get("bytes")
    content_range = steps[1].get("content_range")
    range_match = re.fullmatch(r"bytes 0-63/([0-9]+)", content_range) \
        if isinstance(content_range, str) else None
    range_total_matches_project = (
        type(project_bytes) is int
        and project_bytes > 63
        and range_match is not None
        and range_match.group(1) == str(project_bytes)
    )
    if steps[1].get("http_status") != 206 or steps[1].get("bytes") != 64 \
            or not range_total_matches_project:
        raise SystemExit("Core media verification report has invalid audio Range evidence.")

    def finite_number(value):
        return type(value) is int or (type(value) is float and math.isfinite(value))

    seek = steps[2]
    target = seek.get("target_seconds")
    actual = seek.get("actual_seconds")
    duration = seek.get("duration_seconds")
    if not finite_number(target) or target != 1.25 \
            or not finite_number(actual) or abs(actual - target) >= 0.1 \
            or not finite_number(duration) or not 5.9 <= duration <= 6.1 \
            or actual < 0 or actual >= duration:
        raise SystemExit("Core media verification report has invalid seek evidence.")

    playback = steps[3]
    started = playback.get("started_at_seconds")
    paused = playback.get("paused_at_seconds")
    after_pause = playback.get("after_pause_seconds")
    if not finite_number(started) or abs(started - actual) > 1e-6 \
            or not finite_number(paused) or paused < 0 or paused >= duration \
            or paused - started <= 0.15 \
            or not finite_number(after_pause) or after_pause < 0 or after_pause >= duration \
            or abs(after_pause - paused) > 0.04:
        raise SystemExit("Core media verification report has invalid play/pause evidence.")

    browser_health = steps[5]
    if type(browser_health.get("page_errors")) is not int \
            or browser_health["page_errors"] != 0 \
            or browser_health.get("version") != __version__:
        raise SystemExit("Core media verification report has invalid browser health evidence.")

    rapid = steps[4]
    project_ids = [project["id"] for project in projects]
    expected_order = [project_ids[1], project_ids[0]]
    pre_switch_ids = rapid.get("pre_switch_audio_project_ids")
    held_count = rapid.get("held_audio_requests")
    if rapid.get("requested_projects") != expected_order \
            or rapid.get("requested_project_ids") != expected_order \
            or rapid.get("final_project") != project_ids[0] \
            or not isinstance(pre_switch_ids, list) \
            or any(project_id != project_ids[0] for project_id in pre_switch_ids) \
            or rapid.get("unexpected_audio_project_ids") != [] \
            or type(held_count) is not int or held_count < 2:
        raise SystemExit("Core media verification report has invalid rapid-switch evidence.")
    saved_time = rapid.get("saved_seconds")
    return_start = rapid.get("return_start_seconds")
    final_time = rapid.get("final_seconds")
    if not finite_number(saved_time) or saved_time <= 1.3 or saved_time >= duration \
            or not finite_number(return_start) or not 0 <= return_start <= 0.35 \
            or return_start >= duration \
            or not finite_number(final_time) or final_time >= duration \
            or final_time - return_start <= 0.15:
        raise SystemExit("Core media verification report has invalid rapid-switch playback evidence.")

    return {
        "passed": True,
        "scope": expected_scope,
        "run_id": report["run_id"],
        "verification_run_id": verification_run_id,
        "build_profile": "core",
        "executable_sha256": expected_identity["executable_sha256"],
        "release_manifest_sha256": expected_identity["release_manifest_sha256"],
        "report_sha256": sha256_bytes(report_bytes),
    }, raw_base_url


def _validate_core_media_service_binding_report(
        report_path, expected_identity, verification_run_id, expected_base_url):
    try:
        report_bytes = Path(report_path).read_bytes()
        report = json.loads(report_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Core media service binding report is missing or invalid: {exc}"
        ) from exc

    if not isinstance(report, dict) or type(report.get("schema")) is not int \
            or report["schema"] != 1 \
            or report.get("check") != "core_media_service_process_binding" \
            or report.get("status") != "passed":
        raise SystemExit("Core media service binding report is incomplete or failed.")
    if report.get("verification_run_id") != verification_run_id \
            or report.get("base_url") != expected_base_url \
            or report.get("executable_sha256") != expected_identity["executable_sha256"]:
        raise SystemExit("Core media service binding identity differs from this build/run.")

    executable_pid = report.get("executable_pid")
    listeners_before = report.get("listeners_before")
    listeners_after = report.get("listeners_after")
    if type(executable_pid) is not int or executable_pid <= 0 \
            or not isinstance(listeners_before, list) or not listeners_before \
            or not isinstance(listeners_after, list) or not listeners_after:
        raise SystemExit("Core media service binding report has invalid process evidence.")
    try:
        service_port = urlsplit(expected_base_url).port
    except (TypeError, ValueError):
        service_port = None
    if service_port is None:
        raise SystemExit("Core media service binding report has an invalid base_url.")

    cleanup_status = report.get("cleanup_status")
    process_exited = report.get("process_exited")
    listeners_after_cleanup = report.get("listeners_after_cleanup")
    cleanup_failures = report.get("cleanup_failures")
    if cleanup_status != "passed" or not isinstance(cleanup_failures, list) \
            or cleanup_failures or not isinstance(listeners_after_cleanup, list) \
            or listeners_after_cleanup or not isinstance(process_exited, dict) \
            or "failure" not in report or report["failure"] is not None:
        raise SystemExit("Core media service binding report has incomplete cleanup evidence.")
    executable_exit = process_exited.get("executable")
    probe_exit = process_exited.get("probe")
    if not isinstance(executable_exit, dict) or type(executable_exit.get("pid")) is not int \
            or executable_exit["pid"] != executable_pid or executable_exit.get("exited") is not True \
            or not isinstance(probe_exit, dict) or type(probe_exit.get("pid")) is not int \
            or probe_exit["pid"] <= 0 or probe_exit["pid"] == executable_pid \
            or probe_exit.get("exited") is not True:
        raise SystemExit("Core media service binding report has invalid cleanup process evidence.")

    for listeners in (listeners_before, listeners_after):
        for listener in listeners:
            if not isinstance(listener, dict):
                raise SystemExit("Core media service binding report has invalid listener evidence.")
            local_address = listener.get("local_address")
            local_port = listener.get("local_port")
            owning_process = listener.get("owning_process")
            try:
                address = ipaddress.ip_address(local_address) if isinstance(local_address, str) else None
            except ValueError:
                address = None
            if address is None or not (address.is_loopback or address.is_unspecified) \
                    or type(local_port) is not int or local_port != service_port \
                    or type(owning_process) is not int or owning_process != executable_pid:
                raise SystemExit("Core media service binding report has invalid listener ownership.")

    return {
        "passed": True,
        "verification_run_id": verification_run_id,
        "base_url": expected_base_url,
        "executable_sha256": expected_identity["executable_sha256"],
        "executable_pid": executable_pid,
        "listeners_before": len(listeners_before),
        "listeners_after": len(listeners_after),
        "cleanup_status": cleanup_status,
        "process_exited": process_exited,
        "listeners_after_cleanup": len(listeners_after_cleanup),
        "cleanup_failures": cleanup_failures,
        "report_sha256": sha256_bytes(report_bytes),
    }


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
        if not _matches_identity(item, expected_identity):
            raise SystemExit(
                "Portable specialist evidence identity differs from this verification run; "
                "rerun EXE verification."
            )


def _portable_check_rows(portable, name, *, exactly_one=True):
    rows = [item for item in portable if isinstance(item, dict) and item.get("check") == name]
    if not rows or exactly_one and len(rows) != 1:
        raise SystemExit(f"Portable check payload is missing or ambiguous: {name}.")
    return rows


def _valid_count(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _valid_elapsed(value):
    return (type(value) is int and value >= 0
            or type(value) is float and math.isfinite(value) and value >= 0)


def _matches_capabilities(value, expected):
    return (
        isinstance(value, dict)
        and isinstance(expected, dict)
        and value.keys() == expected.keys()
        and all(type(capability) is bool for capability in value.values())
        and value == expected
    )


def _matches_identity(item, expected_identity):
    if not isinstance(item, dict):
        return False
    return all(
        _matches_capabilities(item.get(field), value)
        if field == "capabilities" else item.get(field) == value
        for field, value in expected_identity.items()
    )


def _validate_portable_check_payloads(portable, profile, manifest,
                                      expected_identity, verification_run_id):
    optional_piano_checks = PORTABLE_PIANO_AUDIO_CHECK_NAMES
    specialist_checks = {
        "guitar_real_audio_http",
        "guitar_cold_cancel",
        "instrument_portable_real_audio",
    }
    row_identity = dict(expected_identity, verification_run_id=verification_run_id)
    for item in portable:
        if item["check"] in specialist_checks or item["check"] == "release_manifest":
            continue
        if not _matches_identity(item, row_identity):
            raise SystemExit(f"Portable check identity differs from this run: {item['check']}.")

    def one(name):
        return _portable_check_rows(portable, name)[0]

    def require(condition, name):
        if not condition:
            raise SystemExit(f"Portable check payload is invalid: {name}.")

    startup = one("startup")
    require(startup.get("version") == manifest.get("version"), "startup")
    separator = startup.get("separator")
    require(isinstance(separator, dict) and isinstance(separator.get("profiles"), dict), "startup.separator")
    profiles = separator["profiles"]
    for profile_name, capability in (("four", "separation_four"), ("six", "separation_six")):
        row = profiles.get(profile_name)
        require(isinstance(row, dict) and type(row.get("available")) is bool
                and row["available"] is manifest["capabilities"][capability],
                f"startup.separator.{profile_name}")
    require(type(separator.get("available")) is bool
            and separator["available"] is manifest["capabilities"]["separation_four"],
            "startup.separator.available")
    require(isinstance(startup.get("prior"), dict)
            and startup["prior"].get("available") is manifest["capabilities"]["pattern_prior"],
            "startup.prior")
    require(isinstance(startup.get("engine"), dict)
            and startup["engine"].get("install_available")
            is manifest["capabilities"]["mapperatorinator_install"], "startup.engine")

    restart = one("restart")
    unicode_audio = restart.get("unicode_audio")
    require(restart.get("preserved") is True and type(restart.get("keys")) is int
            and restart["keys"] == 10 and isinstance(unicode_audio, str)
            and any(ord(character) > 127 for character in unicode_audio), "restart")

    generate_rows = _portable_check_rows(portable, "generate_export", exactly_one=False)
    by_keys = {}
    for row in generate_rows:
        keys = row.get("keys")
        if type(keys) is not int or keys in by_keys:
            raise SystemExit("Portable check payload has duplicate or invalid generate_export keys.")
        by_keys[keys] = row
    if set(by_keys) != set(range(6, 11)):
        raise SystemExit("Portable check payload is missing 6K–10K generate_export evidence.")
    for keys, row in by_keys.items():
        require(_valid_count(row.get("notes"), minimum=1)
                and _valid_count(row.get("ln")) and row["ln"] <= row["notes"]
                and _valid_count(row.get("off_grid")) and row["off_grid"] == 0,
                f"generate_export {keys}K")

    if profile == "core":
        basic = one("basic_analysis")
        require(basic.get("engine") == "basic"
                and _valid_count(basic.get("candidates"), minimum=1)
                and _valid_elapsed(basic.get("seconds")), "basic_analysis")
        provided = one("provided_stem_analysis_export")
        require(provided.get("voices") == ["other"]
                and _valid_count(provided.get("candidates"), minimum=1)
                and _valid_count(provided.get("notes"), minimum=1)
                and _valid_count(provided.get("ln"))
                and provided["ln"] <= provided["notes"]
                and _valid_count(provided.get("off_grid")) and provided["off_grid"] == 0,
                "provided_stem_analysis_export")
        bad_audio = one("bad_audio_recovery")
        require(bad_audio.get("analysis") == "error"
                and bad_audio.get("service_alive") is True, "bad_audio_recovery")
        piano_audio_rows = [item for item in portable
            if isinstance(item, dict) and item.get("check") in optional_piano_checks]
        require(not piano_audio_rows, "piano audio checks are unsupported by Core")
        return {
            "status": "not_run",
            "reason": "profile_has_no_piano_audio_capability",
            "checks": [],
            "musical_acceptance": False,
        }

    piano = one("piano_onnx_runtime")
    expected_shapes = [[1, 1001, 88]] * 4 + [[1, 1001, 1]] * 3
    output_shapes = piano.get("output_shapes")
    providers = piano.get("providers")
    require(isinstance(output_shapes, list)
            and len(output_shapes) == len(expected_shapes)
            and all(isinstance(shape, list)
                    and len(shape) == len(expected_shape)
                    and all(type(dimension) is int and dimension == expected_dimension
                            for dimension, expected_dimension in zip(shape, expected_shape))
                    for shape, expected_shape in zip(output_shapes, expected_shapes))
            and isinstance(providers, list)
            and all(isinstance(provider, str) for provider in providers)
            and "CPUExecutionProvider" in providers,
            "piano_onnx_runtime")
    deep = one("deep_analysis")
    require(deep.get("engine") == "deep" and _valid_count(deep.get("candidates"))
            and deep["candidates"] == 5
            and _valid_elapsed(deep.get("seconds")), "deep_analysis")
    six = one("six_stem_analysis_export")
    expected_voices = {"drums", "bass", "vocals", "other", "guitar", "piano"}
    voices = six.get("voices")
    require(isinstance(voices, list) and all(isinstance(voice, str) for voice in voices)
            and set(voices) == expected_voices and len(voices) == len(expected_voices)
            and _valid_count(six.get("notes")) and six["notes"] == 5
            and _valid_count(six.get("off_grid")) and six["off_grid"] == 0
            and _valid_elapsed(six.get("seconds")), "six_stem_analysis_export")
    cancelled = one("cancel_and_failure")
    require(cancelled.get("cancelled") == "cancelled"
            and cancelled.get("bad_audio") == "error"
            and cancelled.get("service_alive") is True, "cancel_and_failure")

    piano_audio_rows = [item for item in portable
        if isinstance(item, dict) and item.get("check") in optional_piano_checks]
    if piano_audio_rows:
        rows = {
            name: _portable_check_rows(portable, name)[0]
            for name in optional_piano_checks
        }
        cold = rows["piano_cold_inference"]
        manual = rows["piano_manual_expression"]
        warm = rows["piano_restart_warm_cache"]
        piano_model = manifest.get("piano_model")
        expected_model_sha = piano_model.get("sha256") if isinstance(piano_model, dict) else None
        input_sha = cold.get("input_sha256")
        candidate_count = cold.get("candidates")
        release_reasons = cold.get("release_reasons")
        allowed_release_reasons = {
            "release_regression",
            "frame_decay",
            "frame_inactive",
            "reattack",
            "duration_limit",
            "input_boundary",
        }
        require(_is_sha256(input_sha)
                and _is_sha256(cold.get("model_sha256"))
                and _is_sha256(expected_model_sha)
                and cold.get("model_sha256") == expected_model_sha
                and _valid_count(candidate_count, minimum=1)
                and _valid_elapsed(cold.get("seconds"))
                and isinstance(release_reasons, dict)
                and bool(release_reasons)
                and set(release_reasons).issubset(allowed_release_reasons)
                and all(isinstance(reason, str) and reason
                        and _valid_count(count, minimum=1)
                        for reason, count in release_reasons.items())
                and sum(release_reasons.values()) == candidate_count,
                "piano_cold_inference")

        modes = manual.get("modes")
        require(isinstance(modes, dict) and set(modes) == {"articulation", "sustain"}
                and all(isinstance(modes[mode], dict)
                        and _valid_count(modes[mode].get("notes"), minimum=1)
                        and _valid_count(modes[mode].get("ln"))
                        and modes[mode]["ln"] <= modes[mode]["notes"]
                        for mode in ("articulation", "sustain"))
                and manual.get("evidence_preserved") is True
                and isinstance(manual.get("timing"), str)
                and "not musical timing acceptance" in manual["timing"],
                "piano_manual_expression")
        require(_valid_count(warm.get("candidates"), minimum=1)
                and warm["candidates"] == candidate_count
                and warm.get("identical") is True
                and _valid_elapsed(warm.get("seconds")),
                "piano_restart_warm_cache")
    else:
        return {
            "status": "not_run",
            "reason": "optional_audio_fixture_not_supplied",
            "checks": [],
            "musical_acceptance": False,
        }
    return {
        "status": "passed",
        "checks": [
            "piano_cold_inference",
            "piano_manual_expression",
            "piano_restart_warm_cache",
        ],
        "input_sha256": input_sha,
        "model_sha256": expected_model_sha,
        "scope": "portable optional workflow evidence only; not musical acceptance",
        "musical_acceptance": False,
    }


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


def _require_source_snapshot_file(archive, manifest, relative):
    source_path = "assets/" + relative
    sources = manifest.get("sources")
    expected_sha256 = sources.get(source_path) if isinstance(sources, dict) else None
    if not _is_sha256(expected_sha256):
        raise SystemExit(f"Portable manifest has no source identity for bundled asset: {source_path}")
    source = (SOURCE_ROOT / source_path).resolve()
    if not source.is_relative_to(SOURCE_ROOT.resolve()) or not source.is_file():
        raise SystemExit(f"Portable manifest has no source identity for bundled asset: {source_path}")
    _require_archive_file(
        archive,
        "ManiaLab/_internal/assets/" + relative,
        expected_sha256,
        expected_bytes=source.stat().st_size,
    )


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
    if declared_capabilities["pattern_prior"]:
        _require_source_snapshot_file(archive, manifest, "pattern-prior.json")

    if declared_capabilities["piano"]:
        identity = manifest.get("piano_model")
        if (
            not isinstance(identity, dict)
            or identity.get("filename") != "piano-regression.onnx"
            or not _valid_count(identity.get("bytes"))
            or not _is_sha256(identity.get("sha256"))
        ):
            raise SystemExit("Portable manifest has a missing or invalid piano model identity.")
        _require_archive_file(
            archive,
            "ManiaLab/_internal/assets/deep/piano-regression.onnx",
            identity["sha256"],
            expected_bytes=identity["bytes"],
        )
        _require_archive_json(
            archive, "ManiaLab/_internal/assets/piano-model.json", identity
        )

    if declared_capabilities["separation_four"]:
        _require_archive_inventory(
            archive,
            "ManiaLab/_internal/assets/deep/ManiaLabAudio/",
            manifest.get("worker_files"),
            require_bytes=True,
        )
        worker_executable = manifest.get("worker_files", {}).get("ManiaLabAudio.exe")
        if (
            not isinstance(worker_executable, dict)
            or worker_executable.get("sha256") != manifest.get("worker_sha256")
        ):
            raise SystemExit("Portable manifest has an invalid separation worker identity.")
        try:
            worker_manifest = json.loads(
                archive.read("ManiaLab/_internal/assets/deep/manifest.json")
            )
            deep_models_bytes = archive.read("ManiaLab/_internal/assets/deep-models.json")
            deep_models = json.loads(deep_models_bytes)
        except KeyError as exc:
            raise SystemExit(f"Release ZIP is missing required component: {exc.args[0]}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit("Release ZIP has invalid separation model metadata.") from exc
        if not isinstance(worker_manifest, dict):
            raise SystemExit("Release ZIP has invalid separation worker metadata.")
        if any(
            worker_manifest.get(field) != manifest.get(field)
            for field in ("worker_sha256", "worker_sources", "worker_files")
        ):
            raise SystemExit("Release ZIP separation worker manifest differs from its release manifest.")
        _require_source_snapshot_file(archive, manifest, "deep-models.json")
        expected_models = {
            "hdemucs_high_trained.pt",
            "beat-this-final0.ckpt",
            "htdemucs_6s.th",
        }
        models = deep_models.get("models") if isinstance(deep_models, dict) else None
        if (
            not isinstance(models, list)
            or len(models) != len(expected_models)
            or any(
                not isinstance(row, dict) or not isinstance(row.get("filename"), str)
                for row in models
            )
            or {row["filename"] for row in models} != expected_models
        ):
            raise SystemExit("Release ZIP has an invalid separation model inventory.")
        for model in models:
            filename = model.get("filename")
            if (
                not _is_sha256(model.get("sha256"))
                or not _valid_count(model.get("bytes"))
                or not isinstance(filename, str)
            ):
                raise SystemExit("Release ZIP has an invalid separation model identity.")
            _require_archive_file(
                archive,
                "ManiaLab/_internal/assets/deep/" + filename,
                model["sha256"],
                expected_bytes=model["bytes"],
            )

    names = [info.filename.casefold() for info in archive.infolist()]
    if not declared_capabilities["guitar"] and any(
        name == "maniaLab/_internal/assets/deep/guitar-fl.onnx".casefold()
        for name in names
    ):
        raise SystemExit("Release ZIP contains a guitar model without guitar capability.")
    instrument_prefix = "maniaLab/_internal/assets/instruments/".casefold()
    if not declared_capabilities["instruments"] and any(
        name == instrument_prefix[:-1] or name.startswith(instrument_prefix)
        for name in names
    ):
        raise SystemExit("Release ZIP contains instrument assets without instruments capability.")

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


def _require_staged_asset_inventory(archive, manifest, declared_capabilities):
    prefix = "ManiaLab/_internal/assets/"
    if any(
        tuple(part.casefold() for part in PurePosixPath(info.filename).parts[:3])
        == ("manialab", "_internal", "models")
        for info in archive.infolist()
    ):
        raise SystemExit("Release ZIP contains an unapproved model directory.")

    expected = set(CORE_FILES)
    source_backed = set(CORE_FILES) - {"basic-pitch.onnx"}

    if declared_capabilities["pattern_prior"]:
        expected.update(MODEL_METADATA_FILES)
        source_backed.update(MODEL_METADATA_FILES)

    if declared_capabilities["mapperatorinator_install"]:
        expected.update(MAPPER_INSTALLER_FILES)
        source_backed.update(
            name for name in MAPPER_INSTALLER_FILES if name != "runtime/uv.exe"
        )

    if declared_capabilities["separation_four"]:
        expected.update(DEEP_FILES)
        worker_files = manifest.get("worker_files")
        if not isinstance(worker_files, dict):
            raise SystemExit("Portable manifest has no separation worker file inventory.")
        expected.update("deep/ManiaLabAudio/" + name for name in worker_files)

    if declared_capabilities["guitar"]:
        expected.add("deep/guitar-fl.onnx")

    if declared_capabilities["instruments"]:
        identity = manifest["instrument_worker"]
        expected.update({
            "instruments/manifest.json",
            "instruments/verification.json",
            "instruments/LICENSE-tsumugi",
        })
        expected.update(
            f"instruments/checkpoints/best_model_{name}.pth"
            for name in identity["models"]
        )
        expected.update(
            "instruments/source/" + name for name in identity["source_files"]
        )
        expected.update(
            "instruments/ManiaLabInstrument/" + name
            for name in identity["worker_files"]
        )

    for relative in source_backed:
        _require_source_snapshot_file(archive, manifest, relative)

    actual_files = {
        info.filename[len(prefix):]
        for info in archive.infolist()
        if info.filename.startswith(prefix) and not info.is_dir()
    }
    if actual_files != expected:
        raise SystemExit("Release ZIP asset inventory differs from the staged profile.")

    allowed_directories = {""}
    for relative in expected:
        parts = PurePosixPath(relative).parts[:-1]
        allowed_directories.update(
            "/".join(parts[:index]) for index in range(1, len(parts) + 1)
        )
    actual_directories = {
        info.filename[len(prefix):-1]
        for info in archive.infolist()
        if info.filename.startswith(prefix) and info.is_dir()
    }
    if not actual_directories.issubset(allowed_directories):
        raise SystemExit("Release ZIP contains an undeclared staged asset directory.")


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


def _is_safe_archive_directory_path(name):
    if not isinstance(name, str) or not name.endswith("/"):
        return False
    relative = name[:-1]
    path = PurePosixPath(relative)
    return (
        bool(relative)
        and not path.is_absolute()
        and path.as_posix() == relative
        and ".." not in path.parts
        and "\\" not in relative
        and ":" not in relative
        and path.parts[0] == "ManiaLab"
        and all(_is_safe_windows_archive_component(part) for part in path.parts)
    )


def _require_archive_payload_inventory(archive, manifest):
    payload = manifest.get("archive_payload")
    if not isinstance(payload, dict) or set(payload) != {"files", "directories"}:
        raise SystemExit("Portable manifest has no valid archive_payload inventory.")
    files = payload.get("files")
    directories = payload.get("directories")
    manifest_path = "ManiaLab/release-manifest.json"
    if not isinstance(files, dict) or not files or manifest_path in files \
            or not isinstance(directories, list) or not directories:
        raise SystemExit("Portable manifest has an incomplete archive_payload inventory.")
    if any(
        not _is_safe_archive_relative_path(name)
        or not name.startswith("ManiaLab/")
        or not _is_sha256(digest)
        for name, digest in files.items()
    ):
        raise SystemExit("Portable manifest has an unsafe archive_payload file identity.")
    if any(not _is_safe_archive_directory_path(name) for name in directories) \
            or directories != sorted(directories) \
            or len(directories) != len(set(directories)) \
            or "ManiaLab/" not in directories:
        raise SystemExit("Portable manifest has an unsafe archive_payload directory inventory.")

    directory_set = set(directories)
    for name in (*files, *directories):
        path = PurePosixPath(name[:-1] if name.endswith("/") else name)
        parent = path.parent
        while parent.parts:
            if parent.as_posix() + "/" not in directory_set:
                raise SystemExit("Portable archive_payload inventory has an incomplete directory tree.")
            parent = parent.parent

    actual_files = {info.filename for info in archive.infolist() if not info.is_dir()}
    actual_directories = {info.filename for info in archive.infolist() if info.is_dir()}
    if actual_files != set(files) | {manifest_path}:
        raise SystemExit("Release ZIP file inventory differs from its archive_payload manifest.")
    if actual_directories != directory_set:
        raise SystemExit("Release ZIP directory inventory differs from its archive_payload manifest.")

    for info in archive.infolist():
        if info.is_dir():
            if info.file_size != 0 or info.compress_size != 0 or info.CRC != 0 \
                    or info.flag_bits & 0x08:
                raise SystemExit("Release ZIP contains a directory entry with payload data.")
            stream = archive.fp
            if stream is None:
                raise SystemExit("Release ZIP has no readable local directory header.")
            position = stream.tell()
            try:
                stream.seek(info.header_offset)
                local_header = stream.read(30)
            finally:
                stream.seek(position)
            if len(local_header) != 30:
                raise SystemExit("Release ZIP has a truncated local directory header.")
            (
                signature, _, flags, compression, _, _, crc, compressed_size,
                file_size, _, _,
            ) = struct.unpack("<IHHHHHIIIHH", local_header)
            if signature != 0x04034B50 or flags != info.flag_bits \
                    or compression != info.compress_type or crc != info.CRC \
                    or compressed_size != info.compress_size or file_size != info.file_size:
                raise SystemExit("Release ZIP local directory header differs from its central entry.")
        if info.create_system == 3:
            file_type = stat.S_IFMT(info.external_attr >> 16)
            expected_type = stat.S_IFDIR if info.is_dir() else stat.S_IFREG
            if file_type not in (0, expected_type):
                raise SystemExit("Release ZIP contains a non-regular archive entry.")
    for name, expected_sha256 in files.items():
        _require_archive_file(archive, name, expected_sha256)


def _require_archive_source_documents(archive, manifest):
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise SystemExit("Portable manifest has no source snapshot for bundled documents.")
    copied_sources = {"README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"}
    copied_sources.update(name for name in sources if name.startswith("docs/"))
    if not {"README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"}.issubset(sources) \
            or not any(name.startswith("docs/") for name in copied_sources):
        raise SystemExit("Portable manifest source snapshot omits bundled documents.")
    for source_path in copied_sources:
        expected_sha256 = sources.get(source_path)
        if not _is_sha256(expected_sha256):
            raise SystemExit(f"Portable manifest has no source identity for document: {source_path}")
        _require_archive_file(archive, "ManiaLab/" + source_path, expected_sha256)


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


def collect(build_root, pytest_report, portable_report, shutdown_report, *,
            pytest_evidence=None, core_media_report=None,
            core_media_service_binding_report=None):
    build_root = Path(build_root).resolve()
    package_dir = build_root / "release"
    executable = build_root / "dist" / "ManiaLab" / "ManiaLab.exe"
    manifest_path = executable.parent / "release-manifest.json"
    if not all(path.is_file() for path in (executable, manifest_path)):
        raise SystemExit("Portable executable or manifest is missing.")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    profile = manifest.get("build_profile")
    try:
        declared_capabilities = capabilities(profile)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Portable manifest has an unknown build profile: {profile!r}") from exc
    source_root, source_keys = _require_source_snapshot_keys(
        manifest.get("sources"), SOURCE_ROOT, declared_capabilities
    )
    if not _matches_capabilities(manifest.get("capabilities"), declared_capabilities):
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
    for relative in source_keys:
        expected = manifest["sources"][relative]
        source = (source_root / PurePosixPath(relative)).resolve()
        if not source.is_relative_to(source_root) or not source.is_file() or sha256(source) != expected:
            raise SystemExit(f"Source snapshot mismatch: {relative}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if pytest_evidence is None:
        pytest_path = Path(pytest_report)
        pytest_evidence = pytest_path.with_name(pytest_path.name + ".provenance.json")
    try:
        pytest_identity = validate_evidence(
            pytest_evidence, pytest_report, root=SOURCE_ROOT, expected_commit=commit
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
        if sha256_bytes(archive_manifest) != manifest_sha256:
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
        _require_staged_asset_inventory(archive, manifest, declared_capabilities)
        _require_archive_payload_inventory(archive, manifest)
        _require_archive_source_documents(archive, manifest)

    suites = ET.parse(pytest_report).getroot()
    suite_nodes = [suites] if suites.tag == "testsuite" else list(suites.findall("testsuite"))
    totals = {
        key: sum(int(node.attrib.get(key, "0")) for node in suite_nodes)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if totals["tests"] <= 0 or totals["failures"] or totals["errors"]:
        raise SystemExit(f"Pytest report is not clean: {totals}")

    portable = json.loads(Path(portable_report).read_text(encoding="utf-8"))
    required_checks = {"release_manifest", "startup", "restart", "generate_export"}
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
    if any(not isinstance(item, dict)
           or not isinstance(item.get("check"), str)
           or not item["check"].strip()
           for item in portable):
        raise SystemExit("Portable verification report has an unexpected format.")
    if any(item["check"] not in PORTABLE_CHECK_NAMES for item in portable):
        raise SystemExit("Portable verification report contains an unsupported check name.")
    if any("capabilities" in item
           and not _matches_capabilities(item.get("capabilities"), declared_capabilities)
           for item in portable):
        raise SystemExit("Portable verification identity capabilities do not match the declared build profile.")
    portable_checks = [item["check"] for item in portable]
    if not required_checks.issubset(portable_checks):
        raise SystemExit("Portable verification report is missing required checks.")
    portable_identities = [item for item in portable
        if isinstance(item, dict) and item.get('check') == 'release_manifest']
    if len(portable_identities) != 1:
        raise SystemExit('Portable verification must contain exactly one release_manifest identity check.')
    expected_identity = {field: manifest[field] for field in
        ('build_profile', 'capabilities', 'executable_sha256')}
    expected_identity['release_manifest_sha256'] = manifest_sha256
    if not _matches_identity(portable_identities[0], expected_identity):
        raise SystemExit('Portable verification identity differs from this build; rerun EXE verification.')
    verification_run_id = portable_identities[0].get('verification_run_id')
    if not _is_verification_run_id(verification_run_id):
        raise SystemExit('Portable verification identity has an invalid verification_run_id.')
    expected_specialist_identity = dict(expected_identity, verification_run_id=verification_run_id)
    _require_specialist_verification_identity(
        portable, expected_specialist_identity, declared_capabilities
    )
    allowed_checks = set(PORTABLE_COMMON_CHECK_NAMES)
    if declared_capabilities["pattern_prior"]:
        allowed_checks.add("bundled_model")
    if profile == "core":
        allowed_checks.update(PORTABLE_CORE_CHECK_NAMES)
    else:
        allowed_checks.update(PORTABLE_NON_CORE_CHECK_NAMES)
        if declared_capabilities["piano"]:
            allowed_checks.update(PORTABLE_PIANO_AUDIO_CHECK_NAMES)
    if declared_capabilities["guitar"]:
        allowed_checks.update(PORTABLE_GUITAR_CHECK_NAMES)
    if declared_capabilities["instruments"]:
        allowed_checks.add("instrument_portable_real_audio")
    if any(item["check"] not in allowed_checks for item in portable):
        raise SystemExit(
            "Portable verification report contains a check not applicable to the declared profile or capabilities."
        )
    if profile == "core":
        if core_media_report is None:
            raise SystemExit("Core portable verification is missing the Core media report.")
        core_media_evidence, core_media_base_url = _validate_core_media_report(
            core_media_report, expected_identity, verification_run_id
        )
        if core_media_service_binding_report is None:
            raise SystemExit("Core portable verification is missing the Core media service binding report.")
        core_media_service_binding_evidence = _validate_core_media_service_binding_report(
            core_media_service_binding_report,
            expected_identity,
            verification_run_id,
            core_media_base_url,
        )
        core_media_evidence["service_binding_report_sha256"] = (
            core_media_service_binding_evidence["report_sha256"]
        )
        for field in (
            "cleanup_status", "process_exited", "listeners_after_cleanup", "cleanup_failures",
        ):
            core_media_evidence[field] = core_media_service_binding_evidence[field]
    else:
        if core_media_report is not None or core_media_service_binding_report is not None:
            raise SystemExit("Core media verification report is unsupported for this profile.")
        core_media_evidence = None
    piano_audio_evidence = _validate_portable_check_payloads(
        portable, profile, manifest, expected_identity, verification_run_id
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
            or not _valid_count(piano_model.get("bytes"))
            or not _valid_count(piano_check.get("bytes"))
            or piano_check.get("sha256") != piano_model.get("sha256")
            or piano_check.get("bytes") != piano_model.get("bytes")
            or "CPUExecutionProvider" not in piano_check.get("providers", [])
        ):
            raise SystemExit("Piano runtime smoke does not match the release manifest.")

    shutdown = json.loads(Path(shutdown_report).read_text(encoding="utf-8"))
    expected_shutdown_identity = dict(expected_identity, version=manifest['version'])
    if not _matches_identity(shutdown, expected_shutdown_identity):
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
        "release_manifest_sha256": manifest_sha256,
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
            "core_media": core_media_evidence,
            "piano_model": {
                "sha256": piano_check["sha256"],
                "bytes": piano_check["bytes"],
                "cpu_smoke_passed": True,
            } if piano_check else None,
            "piano_audio": piano_audio_evidence,
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
    parser.add_argument("--core-media-report", type=Path)
    parser.add_argument("--core-media-service-binding-report", type=Path)
    parser.add_argument("--shutdown-report", type=Path, required=True)
    args = parser.parse_args()
    report, output, checksums = collect(
        args.build_root, args.pytest_report, args.portable_report, args.shutdown_report,
        pytest_evidence=args.pytest_evidence, core_media_report=args.core_media_report,
        core_media_service_binding_report=args.core_media_service_binding_report,
    )
    print(json.dumps({
        "validation": str(output),
        "checksums": str(checksums),
        "source_commit": report["source_commit"],
        "archive": report["archive"],
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
