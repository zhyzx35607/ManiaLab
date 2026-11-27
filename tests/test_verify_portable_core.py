import hashlib
import shutil

import pytest

from scripts import verify_portable
from scripts.package_assets import (
    MAPPER_INSTALLER_FILES,
    capabilities,
    lora_adapter_inventory,
)


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


def test_workbench_lora_inventory_matches_manifest_and_only_scans_adapters(tmp_path):
    bundle = tmp_path / "bundle"
    adapter = bundle / "_internal/assets/lora/ln-focus"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r":16}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"lora weights")
    (adapter / "lora-manifest.json").write_text(
        '{"base_model":"v32-mini"}', encoding="utf-8",
    )
    manifest = {"lora_adapters": lora_adapter_inventory(bundle / "_internal/assets/lora")}

    inventory = verify_portable.verify_packaged_loras(
        manifest, capabilities("workbench"), bundle,
    )

    assert set(inventory) == {
        "ln-focus/adapter_config.json",
        "ln-focus/adapter_model.safetensors",
        "ln-focus/lora-manifest.json",
    }


def test_core_portable_inventory_requires_an_empty_lora_manifest(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "_internal/assets").mkdir(parents=True)

    assert verify_portable.verify_packaged_loras(
        {"lora_adapters": {}}, capabilities("core"), bundle,
    ) == {}


def test_profile_without_lora_support_rejects_packaged_adapter(tmp_path):
    bundle = tmp_path / "bundle"
    adapter = bundle / "_internal/assets/lora/private"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"private weights")
    manifest = {"lora_adapters": lora_adapter_inventory(bundle / "_internal/assets/lora")}

    with pytest.raises(AssertionError, match="without LoRA support"):
        verify_portable.verify_packaged_loras(
            manifest, capabilities("core"), bundle,
        )


def test_workbench_manifest_verifies_pinned_v32_installer_without_installing_it(
        tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    executable = bundle / "ManiaLab.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic workbench executable")
    assets = bundle / "_internal/assets"
    assets.mkdir(parents=True)
    source_assets = verify_portable.ROOT / "assets"
    shutil.copy2(source_assets / "model-manifest.json", assets / "model-manifest.json")
    for name in MAPPER_INSTALLER_FILES:
        if name == "runtime/uv.exe":
            continue
        target = assets / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_assets / name, target)
    (assets / "runtime").mkdir()
    (assets / "runtime/uv.exe").write_bytes(b"synthetic uv")
    identity = {
        "version": "0.12.1",
        "target": "x86_64-pc-windows-msvc",
        "archive_url": "https://example.invalid/uv.zip",
        "archive_bytes": 12,
        "archive_sha256": "a" * 64,
        "filename": "uv.exe",
        "bytes": len(b"synthetic uv"),
        "sha256": hashlib.sha256(b"synthetic uv").hexdigest(),
    }
    calls = []

    def verify_uv(path, *, check_version=True):
        calls.append((path, check_version))
        return identity

    monkeypatch.setattr(verify_portable, "verify_uv_binary", verify_uv)
    source_root = tmp_path / "source-root"
    (source_root / "assets").mkdir(parents=True)
    for name in ("model-manifest.json", "mapperatorinator-source.zip",
                 "mapperatorinator-requirements.txt"):
        shutil.copy2(source_assets / name, source_root / "assets" / name)
    monkeypatch.setattr(verify_portable, "ROOT", source_root)
    source_inventory = {
        "assets/" + name: hashlib.sha256((source_assets / name).read_bytes()).hexdigest()
        for name in MAPPER_INSTALLER_FILES
        if name != "runtime/uv.exe"
    }
    manifest = {
        "build_profile": "workbench",
        "capabilities": capabilities("workbench"),
        "sources": source_inventory,
        "uv_runtime": identity,
    }

    assert verify_portable.verify_workbench_manifest(executable, manifest) == identity
    assert calls == [(assets / "runtime/uv.exe", False)]


def test_workbench_manifest_rejects_stale_mapper_source_archive(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    executable = bundle / "ManiaLab.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic workbench executable")
    assets = bundle / "_internal/assets"
    (assets / "runtime").mkdir(parents=True)
    (assets / "runtime/uv.exe").write_bytes(b"synthetic uv")
    source_assets = verify_portable.ROOT / "assets"
    shutil.copy2(source_assets / "model-manifest.json", assets / "model-manifest.json")
    for name in MAPPER_INSTALLER_FILES:
        if name == "runtime/uv.exe":
            continue
        target = assets / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_assets / name, target)
    (assets / "mapperatorinator-source.zip").write_bytes(b"stale source archive")
    identity = {
        "version": "0.12.1",
        "target": "x86_64-pc-windows-msvc",
        "archive_url": "https://example.invalid/uv.zip",
        "archive_bytes": 12,
        "archive_sha256": "a" * 64,
        "filename": "uv.exe",
        "bytes": len(b"synthetic uv"),
        "sha256": hashlib.sha256(b"synthetic uv").hexdigest(),
    }
    monkeypatch.setattr(
        verify_portable, "verify_uv_binary", lambda *args, **kwargs: identity,
    )
    source_inventory = {
        "assets/" + name: hashlib.sha256((source_assets / name).read_bytes()).hexdigest()
        for name in MAPPER_INSTALLER_FILES
        if name != "runtime/uv.exe"
    }
    manifest = {
        "build_profile": "workbench",
        "capabilities": capabilities("workbench"),
        "sources": source_inventory,
        "uv_runtime": identity,
    }

    with pytest.raises(AssertionError, match="source archive"):
        verify_portable.verify_workbench_manifest(executable, manifest)


def test_workbench_manifest_rejects_changed_pinned_requirements(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    executable = bundle / "ManiaLab.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic workbench executable")
    assets = bundle / "_internal/assets"
    (assets / "runtime").mkdir(parents=True)
    (assets / "runtime/uv.exe").write_bytes(b"synthetic uv")
    source_assets = verify_portable.ROOT / "assets"
    shutil.copy2(source_assets / "model-manifest.json", assets / "model-manifest.json")
    for name in MAPPER_INSTALLER_FILES:
        if name == "runtime/uv.exe":
            continue
        target = assets / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_assets / name, target)
    (assets / "mapperatorinator-requirements.txt").write_bytes(b"unlocked requirements")
    identity = {
        "version": "0.12.1",
        "target": "x86_64-pc-windows-msvc",
        "archive_url": "https://example.invalid/uv.zip",
        "archive_bytes": 12,
        "archive_sha256": "a" * 64,
        "filename": "uv.exe",
        "bytes": len(b"synthetic uv"),
        "sha256": hashlib.sha256(b"synthetic uv").hexdigest(),
    }
    monkeypatch.setattr(
        verify_portable, "verify_uv_binary", lambda *args, **kwargs: identity,
    )
    manifest = {
        "build_profile": "workbench",
        "capabilities": capabilities("workbench"),
        "sources": {
            "assets/" + name: hashlib.sha256((source_assets / name).read_bytes()).hexdigest()
            for name in MAPPER_INSTALLER_FILES if name != "runtime/uv.exe"
        },
        "uv_runtime": identity,
    }

    with pytest.raises(AssertionError, match="mapperatorinator-requirements"):
        verify_portable.verify_workbench_manifest(executable, manifest)


def test_workbench_verifier_runs_bundled_ffmpeg_version_smoke(tmp_path, monkeypatch):
    binary = tmp_path / "_internal/imageio_ffmpeg/binaries/ffmpeg-win-x86_64.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic ffmpeg")

    def run(arguments, **kwargs):
        assert arguments == [str(binary), "-version"]
        return type("Result", (), {
            "returncode": 0,
            "stdout": "ffmpeg version 9.0.2-test\n",
        })()

    monkeypatch.setattr(verify_portable.subprocess, "run", run)
    identity = verify_portable.verify_packaged_ffmpeg(tmp_path)

    assert identity == {
        "path": "_internal/imageio_ffmpeg/binaries/ffmpeg-win-x86_64.exe",
        "bytes": len(b"synthetic ffmpeg"),
        "sha256": hashlib.sha256(b"synthetic ffmpeg").hexdigest(),
        "version": "ffmpeg version 9.0.2-test",
    }


def test_workbench_ffmpeg_verifier_rejects_missing_or_multiple_binaries(tmp_path):
    binary_root = tmp_path / "_internal/imageio_ffmpeg/binaries"
    binary_root.mkdir(parents=True)
    with pytest.raises(AssertionError, match="found 0"):
        verify_portable.verify_packaged_ffmpeg(tmp_path)

    (binary_root / "ffmpeg-one.exe").write_bytes(b"one")
    (binary_root / "ffmpeg-two.exe").write_bytes(b"two")
    with pytest.raises(AssertionError, match="found 2"):
        verify_portable.verify_packaged_ffmpeg(tmp_path)


def test_portable_manifest_verifies_and_smokes_bundled_torosu_cli_identity(
    tmp_path, monkeypatch,
):
    binary = tmp_path / "_internal/engines/torosu_cli/torosu-cli.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native Torosu CLI")
    identity = {
        "path": "_internal/engines/torosu_cli/torosu-cli.exe",
        "bytes": binary.stat().st_size,
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }

    calls = []

    def run(args, **kwargs):
        calls.append(args[1])
        if args[1] == "analyze":
            return type("Result", (), {
                "returncode": 0, "stderr": "", "stdout": '{"keys":10,"objects":7,"ln":1}',
            })()
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(verify_portable.subprocess, "run", run)
    assert verify_portable.verify_torosu_cli(
        tmp_path, identity, tmp_path / "smoke",
    ) == identity
    assert calls == ["convert", "analyze"]

    binary.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="size differs"):
        verify_portable.verify_torosu_cli(tmp_path, identity, tmp_path / "smoke")


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
                "four": {"available": False, "voices": ["drums", "bass", "vocals", "other"]},
                "six": {"available": False, "voices": ["drums", "bass", "vocals", "other", "guitar", "piano"]},
                "four-piano": {"available": False, "voices": ["drums", "piano", "vocals", "other"]},
            },
        },
        "piano": {"available": False, "default": False},
        "guitar": {"available": False},
        "instruments": {"available": False},
        "pattern_prior": {"available": False},
        "engine": {"install_available": False},
    }

    assert verify_portable.verify_runtime_capabilities(status, capabilities("core")) \
        == {"four": False, "six": False, "four-piano": False}


def test_core_runtime_capabilities_reject_unexpected_model_availability():
    status = {
        "separator": {
            "available": False,
            "profiles": {
                "four": {"available": False, "voices": ["drums", "bass", "vocals", "other"]},
                "six": {"available": True, "voices": ["drums", "bass", "vocals", "other", "guitar", "piano"]},
                "four-piano": {"available": False, "voices": ["drums", "piano", "vocals", "other"]},
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
                "four": {"available": invalid, "voices": ["drums", "bass", "vocals", "other"]},
                "six": {"available": True, "voices": ["drums", "bass", "vocals", "other", "guitar", "piano"]},
                "four-piano": {"available": True, "voices": ["drums", "piano", "vocals", "other"]},
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
                "four": {"available": True, "voices": ["drums", "bass", "vocals", "other"]},
                "six": {"available": True, "voices": ["drums", "bass", "vocals", "other", "guitar", "piano"]},
                "four-piano": {"available": True, "voices": ["drums", "piano", "vocals", "other"]},
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
                "four": {"available": True, "voices": ["drums", "bass", "vocals", "other"]},
                "six": {"available": True, "voices": ["drums", "bass", "vocals", "other", "guitar", "piano"]},
                "four-piano": {"available": True, "voices": ["drums", "piano", "vocals", "other"]},
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
