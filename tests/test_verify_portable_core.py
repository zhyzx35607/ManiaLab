import hashlib

import pytest

from scripts import verify_portable
from scripts.package_assets import capabilities


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
    }

    with pytest.raises(AssertionError, match="separation_six"):
        verify_portable.verify_runtime_capabilities(status, capabilities("core"))
