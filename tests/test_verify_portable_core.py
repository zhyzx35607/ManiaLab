import hashlib

import pytest

from scripts import verify_portable
from scripts.package_assets import capabilities


def test_basic_analysis_evidence_uses_stable_engine_classification():
    evidence = verify_portable.basic_analysis_evidence({
        "engine": "Basic Pitch 0.4.0 ONNX + onset refinement + phase regression",
        "candidates": [{"pitch": 60}],
        "elapsed_seconds": 0.92,
    })

    assert evidence == {
        "check": "basic_analysis",
        "engine": "basic",
        "analysis_engine": "Basic Pitch 0.4.0 ONNX + onset refinement + phase regression",
        "candidates": 1,
        "seconds": 0.92,
    }


def test_core_manifest_skips_worker_and_piano_identity_checks(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Core must not inspect excluded model or worker files")

    monkeypatch.setattr(verify_portable, "verify_worker", unexpected)
    monkeypatch.setattr(verify_portable, "verify_piano_model", unexpected)
    manifest = {
        "worker_sha256": None,
        "worker_sources": None,
        "worker_files": None,
        "piano_model": None,
        "guitar_model": None,
        "instrument_worker": None,
    }

    assert verify_portable.verify_manifest_components(
        manifest, capabilities("core"), tmp_path
    ) == (None, None)


def test_standard_manifest_keeps_worker_and_piano_checks(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    worker_path = bundle / "_internal/assets/deep/ManiaLabAudio/ManiaLabAudio.exe"
    worker_path.parent.mkdir(parents=True)
    worker_path.write_bytes(b"worker")
    worker = {"worker_sources": {"worker.py": "abc"}, "worker_files": ["worker.exe"]}
    piano = {"sha256": "def", "bytes": 3}
    calls = []

    def verify_worker(root, path):
        calls.append(("worker", root, path))
        return worker

    def verify_piano_model(path):
        calls.append(("piano", path))
        return piano

    monkeypatch.setattr(verify_portable, "verify_worker", verify_worker)
    monkeypatch.setattr(verify_portable, "verify_piano_model", verify_piano_model)
    manifest = {
        "worker_sha256": hashlib.sha256(b"worker").hexdigest(),
        "worker_sources": worker["worker_sources"],
        "worker_files": worker["worker_files"],
        "piano_model": piano,
        "guitar_model": None,
        "instrument_worker": None,
    }

    checked_worker, checked_piano = verify_portable.verify_manifest_components(
        manifest, capabilities("standard"), bundle
    )
    assert checked_worker == worker
    assert checked_piano == piano
    assert [call[0] for call in calls] == ["worker", "piano"]


def test_core_runtime_capabilities_match_manifest():
    status = {
        "separator": {
            "available": False,
            "profiles": {
                "four": {"available": False},
                "six": {"available": False},
            },
        },
        "piano": {"available": False, "default": False},
        "guitar": {"available": False},
        "instruments": {"available": False},
        "pattern_prior": {"available": False},
        "engine": {"install_available": False},
    }

    assert verify_portable.verify_runtime_capabilities(status, capabilities("core")) \
        == {"four": False, "six": False}


def test_core_runtime_capabilities_reject_unexpected_model_availability():
    status = {
        "separator": {
            "available": False,
            "profiles": {
                "four": {"available": False},
                "six": {"available": True},
            },
        },
        "piano": {"available": False, "default": False},
        "guitar": {"available": False},
        "instruments": {"available": False},
        "pattern_prior": {"available": False},
        "engine": {"install_available": False},
    }

    with pytest.raises(AssertionError, match="separation_six"):
        verify_portable.verify_runtime_capabilities(status, capabilities("core"))


@pytest.mark.parametrize("invalid", ["false", 1])
def test_standard_runtime_capabilities_reject_non_boolean_availability(invalid):
    status = {
        "separator": {
            "available": True,
            "profiles": {
                "four": {"available": invalid},
                "six": {"available": True},
            },
        },
        "piano": {"available": True, "default": False},
        "guitar": {"available": False},
        "instruments": {"available": False},
        "pattern_prior": {"available": True},
        "engine": {"install_available": True},
    }

    with pytest.raises(AssertionError, match="must be a boolean"):
        verify_portable.verify_runtime_capabilities(status, capabilities("standard"))


def test_runtime_capabilities_reject_non_boolean_manifest_value():
    status = {
        "separator": {
            "available": True,
            "profiles": {
                "four": {"available": True},
                "six": {"available": True},
            },
        },
        "piano": {"available": True, "default": False},
        "guitar": {"available": False},
        "instruments": {"available": False},
        "pattern_prior": {"available": True},
        "engine": {"install_available": True},
    }
    declared = dict(capabilities("standard"))
    declared["separation_four"] = 1

    with pytest.raises(AssertionError, match="Manifest separation_four must be a boolean"):
        verify_portable.verify_runtime_capabilities(status, declared)


@pytest.mark.parametrize(
    ("component", "field", "capability"),
    [
        ("pattern_prior", "available", "pattern_prior"),
        ("engine", "install_available", "mapperatorinator_install"),
    ],
)
@pytest.mark.parametrize("invalid", ["false", 1])
def test_runtime_capabilities_reject_non_boolean_auxiliary_status(
    component, field, capability, invalid
):
    declared = capabilities("standard")
    assert declared[capability] is True
    status = {
        "separator": {
            "available": True,
            "profiles": {
                "four": {"available": True},
                "six": {"available": True},
            },
        },
        "piano": {"available": True, "default": False},
        "guitar": {"available": False},
        "instruments": {"available": False},
        "pattern_prior": {"available": True},
        "engine": {"install_available": True},
    }
    status[component][field] = invalid

    with pytest.raises(AssertionError, match="must be a boolean"):
        verify_portable.verify_runtime_capabilities(status, declared)


def test_specialist_evidence_identity_binds_only_rows_from_current_run(tmp_path):
    executable = tmp_path / "ManiaLab.exe"
    executable.write_bytes(b"synthetic executable bytes")
    manifest_path = tmp_path / "release-manifest.json"
    manifest_path.write_bytes(b'{"profile":"extended"}\n')
    manifest = {
        "build_profile": "extended",
        "capabilities": {"guitar": True, "instruments": True},
    }
    run_id = "current-verification-run"
    identity = verify_portable.build_verification_identity(
        executable, manifest_path, manifest, run_id
    )
    expected = {
        "build_profile": "extended",
        "capabilities": {"guitar": True, "instruments": True},
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "release_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "verification_run_id": run_id,
    }
    assert identity == expected

    stale_rows = [
        {"check": "guitar_real_audio_http", "verification_run_id": "old-run"},
        {"check": "instrument_portable_real_audio", "release_manifest_sha256": "old-manifest"},
    ]
    stale_snapshot = [dict(row) for row in stale_rows]
    release_identity = {"check": "release_manifest", **identity}
    checks = stale_rows + [release_identity]
    first_new_index = len(checks)
    current_rows = [
        {"check": "guitar_real_audio_http", "case": "synthetic"},
        {"check": "guitar_cold_cancel", "partial_analysis_saved": False},
        {"check": "instrument_portable_real_audio", "voice": "bass"},
    ]
    checks.extend(current_rows)

    verify_portable.bind_new_specialist_evidence(checks, first_new_index, identity)

    assert checks[:len(stale_rows)] == stale_snapshot
    for row in [release_identity, *current_rows]:
        assert {key: row[key] for key in identity} == identity
