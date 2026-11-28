from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.package_4050_kit import (
    MODEL_DIRECTORY,
    SOURCE_COMMIT,
    WHISPER_CONFIGS,
    build_kit,
    verify_kit_manifest,
)


def _write(path: Path, content: bytes | str = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "repo"
    dataset = root / "workspace_exp" / "lora_6_10k_kit" / "dataset"
    _write(dataset / "manifest.json", json.dumps({"review_status": "human_reviewed", "summary": {"tier_chart_counts": {"tier1": 42}}}))
    _write(dataset / "sample.osu")
    model = root / "model" / MODEL_DIRECTORY
    model_files = ("gamemode=3/config.json", "gamemode=3/generation_config.json", "gamemode=3/model.safetensors", "gamemode=3/tokenizer.json")
    hashes = {}
    for filename in model_files:
        payload = filename.encode("utf-8")
        _write(model / filename, payload)
        hashes[Path(filename).name] = hashlib.sha256(payload).hexdigest()
    _write(model / "model-revision.json", json.dumps({"revision": "7807f0dc70cab671be012e1f5ddf945b0b8b7278", "sha256": hashes}))
    source = root / "source"
    _write(source / ".manialab-source.json", json.dumps({"commit": SOURCE_COMMIT}))
    _write(source / "inference.py")
    _write(source / "osuT5" / "osuT5" / "dataset" / "osu_parser.py")
    whisper = root / "whisper"
    for repo_name, revision in WHISPER_CONFIGS.items():
        _write(whisper / repo_name / "snapshots" / revision / "config.json", "{}")
        _write(whisper / repo_name / "refs" / "main", revision)
    torosu = _write(root / "engines" / "torosu_cli.exe")
    for relative in ("__init__.py", "cancellation.py", "domain.py", "task_errors.py", "torosu_bridge.py", "v32_charts.py", "v32_device.py", "v32_engine.py", "v32_process.py", "v32_runtime.py", "worker_process.py"):
        _write(root / "manialab" / relative)
    for relative in ("training/prepare_10k_dataset.py", "training/sitecustomize.py", "training/train_10k_lora.py", "training/run_train_4050.ps1", "assets/mapperatorinator-requirements.txt", "vendor/mapperatorinator/template.osu"):
        _write(root / relative)
    return {"root": root, "dataset": dataset, "model": model, "source": source, "whisper": whisper, "torosu": torosu, "output": tmp_path / "export" / "portable_4050_kit"}


def test_build_kit_copies_pinned_inputs_and_hash_manifest(tmp_path):
    paths = _fixture(tmp_path)
    result = build_kit(**paths)
    output = paths["output"]
    assert result["tier1_chart_count"] == 42
    assert result["mapperatorinator_commit"] == SOURCE_COMMIT
    assert (output / "dataset" / "sample.osu").is_file()
    assert (output / "models" / MODEL_DIRECTORY / "gamemode=3" / "model.safetensors").is_file()
    assert (output / "source" / "inference.py").is_file()
    assert (output / "assets" / "huggingface" / "models--openai--whisper-base" / "refs" / "main").is_file()
    assert (output / "assets" / "huggingface" / "models--openai--whisper-base" / "refs" / "main").read_text(encoding="ascii") == WHISPER_CONFIGS["models--openai--whisper-base"]
    assert (output / "engines" / "torosu_cli" / "torosu-cli.exe").is_file()
    assert (output / "training" / "prepare_10k_dataset.py").is_file()
    assert (output / "training" / "sitecustomize.py").is_file()
    assert (output / "4050训练说明.txt").is_file()
    manifest = verify_kit_manifest(output)
    assert manifest["total_file_count"] == len(manifest["files"])


def test_build_refuses_provisional_dataset(tmp_path):
    paths = _fixture(tmp_path)
    (paths["dataset"] / "manifest.json").write_text(json.dumps({"review_status": "provisional", "tier_chart_counts": {"tier1": 42}}), encoding="utf-8")
    with pytest.raises(ValueError, match="human_reviewed"):
        build_kit(**paths)
    assert not paths["output"].exists()


def test_build_refuses_tier1_count_other_than_42(tmp_path):
    paths = _fixture(tmp_path)
    (paths["dataset"] / "manifest.json").write_text(json.dumps({"review_status": "human_reviewed", "tier_chart_counts": {"tier1": 41}}), encoding="utf-8")
    with pytest.raises(ValueError, match="42 Tier 1"):
        build_kit(**paths)


def test_build_refuses_pinned_model_checksum_mismatch(tmp_path):
    paths = _fixture(tmp_path)
    _write(paths["model"] / "gamemode=3/config.json", "modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_kit(**paths)


def test_manifest_verification_detects_changed_package_file(tmp_path):
    paths = _fixture(tmp_path)
    build_kit(**paths)
    _write(paths["output"] / "dataset" / "sample.osu", "tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_kit_manifest(paths["output"])


def test_build_refuses_to_overwrite_existing_kit(tmp_path):
    paths = _fixture(tmp_path)
    paths["output"].mkdir(parents=True)
    (paths["output"] / "keep.txt").write_text("user file", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        build_kit(**paths)
    assert (paths["output"] / "keep.txt").read_text(encoding="utf-8") == "user file"


def test_build_refuses_output_inside_checkout_except_ignored_target(tmp_path):
    paths = _fixture(tmp_path)
    paths["output"] = paths["root"] / "somewhere" / "portable_4050_kit"
    with pytest.raises(ValueError, match="restricted to ignored"):
        build_kit(**paths)
