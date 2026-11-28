"""Assemble an offline, copyable RTX 4050 LoRA training kit.

This command only copies and verifies inputs. It never downloads dependencies or
starts training; the full kit remains in the ignored workspace_exp directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable


MODEL_DIRECTORY = "Mapperatorinator-v32-mini-7807f0d"
MODEL_FILES = (
    "gamemode=3/config.json",
    "gamemode=3/generation_config.json",
    "gamemode=3/model.safetensors",
    "gamemode=3/tokenizer.json",
    "model-revision.json",
)
SOURCE_COMMIT = "5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa"
WHISPER_CONFIGS = {
    "models--openai--whisper-base": "e37978b90ca9030d5170a5c07aadb050351a65bb",
    "models--openai--whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
}
MANIALAB_MODULES = (
    "__init__.py",
    "cancellation.py",
    "domain.py",
    "task_errors.py",
    "torosu_bridge.py",
    "v32_charts.py",
    "v32_device.py",
    "v32_engine.py",
    "v32_process.py",
    "v32_runtime.py",
    "worker_process.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths(root: Path) -> dict[str, Path]:
    workspace = root / "workspace_exp" / "lora_6_10k_kit"
    return {
        "dataset": workspace / "dataset",
        "model": root / "workspace_exp" / "lora_10k_kit" / "runtime" / "models" / MODEL_DIRECTORY,
        "source": root / "workspace_exp" / "phase06" / "v32-runtime" / "source",
        "whisper": root / "assets" / "huggingface",
        "torosu": root / "engines" / "torosu_cli" / "torosu-cli.exe",
        "output": workspace / "portable_4050_kit",
    }


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing kit content: {destination}")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_inputs(paths: dict[str, Path], root: Path) -> tuple[Path, ...]:
    dataset = _require_dir(paths["dataset"], "reviewed dataset")
    dataset_manifest = _require_file(dataset / "manifest.json", "dataset manifest")
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("review_status") != "human_reviewed":
        raise ValueError("Dataset manifest must have review_status='human_reviewed'.")
    summary = manifest.get("summary", manifest)
    if int(summary.get("tier_chart_counts", {}).get("tier1", 0)) != 42:
        raise ValueError("Dataset must contain exactly 42 Tier 1 reviewed charts.")

    model = _require_dir(paths["model"], "V32-mini model")
    model_files = tuple(_require_file(model / relative, "V32-mini model file") for relative in MODEL_FILES)
    revision = json.loads((model / "model-revision.json").read_text(encoding="utf-8"))
    if revision.get("revision") != "7807f0dc70cab671be012e1f5ddf945b0b8b7278":
        raise ValueError("V32-mini model revision does not match the pinned snapshot.")
    expected_hashes = revision.get("sha256", {})
    for relative in _MODEL_WEIGHT_FILES:
        expected = expected_hashes.get(relative, expected_hashes.get(Path(relative).name))
        if expected != _sha256(model / relative):
            raise ValueError(f"V32-mini model checksum mismatch: {relative}")

    source = _require_dir(paths["source"], "Mapperatorinator source")
    source_info = json.loads(_require_file(source / ".manialab-source.json", "source pin").read_text(encoding="utf-8"))
    if source_info.get("commit") != SOURCE_COMMIT:
        raise ValueError("Mapperatorinator source commit does not match the pinned training revision.")
    for relative in ("inference.py", "osuT5/osuT5/dataset/osu_parser.py"):
        _require_file(source / relative, "Mapperatorinator source file")

    whisper_root = _require_dir(paths["whisper"], "offline Whisper configs")
    whisper_files: list[Path] = []
    for repo, revision in WHISPER_CONFIGS.items():
        repo_dir = _require_dir(whisper_root / repo, f"Whisper cache {repo}")
        config = _require_file(repo_dir / "snapshots" / revision / "config.json", f"Whisper config {repo}")
        ref = _require_file(repo_dir / "refs" / "main", f"Whisper ref {repo}")
        if ref.read_text(encoding="utf-8").strip() != revision:
            raise ValueError(f"Whisper revision mismatch for {repo}.")
        whisper_files.extend((config, ref))

    torosu = _require_file(paths["torosu"], "Torosu CLI")
    for relative in MANIALAB_MODULES:
        _require_file(root / "manialab" / relative, "ManiaLab training module")
    _require_file(root / "vendor" / "mapperatorinator" / "template.osu", "Mapperatorinator template")
    for relative in ("training/train_10k_lora.py", "training/run_train_4050.ps1", "assets/mapperatorinator-requirements.txt"):
        _require_file(root / relative, "training package source")
    # Return the evidence files up front, so validation never relies on a partial copy.
    return (dataset_manifest, *model_files, *whisper_files, torosu)


_MODEL_WEIGHT_FILES = (
    "gamemode=3/config.json",
    "gamemode=3/generation_config.json",
    "gamemode=3/model.safetensors",
    "gamemode=3/tokenizer.json",
)


def verify_kit_manifest(kit: Path) -> dict[str, object]:
    root = kit.expanduser().resolve()
    manifest_path = _require_file(root / "KIT-MANIFEST.json", "kit manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Kit manifest files field must be a list.")
    for item in files:
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe kit manifest path: {relative}")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Kit file is missing or escaped the root: {relative}")
        if path.stat().st_size != int(item["size_bytes"]) or _sha256(path) != item["sha256"]:
            raise ValueError(f"Kit file checksum mismatch: {relative}")
    return manifest


def build_kit(
    *,
    root: Path | None = None,
    dataset: Path | None = None,
    model: Path | None = None,
    source: Path | None = None,
    whisper: Path | None = None,
    torosu: Path | None = None,
    output: Path | None = None,
) -> dict[str, object]:
    repo = (root or _repo_root()).expanduser().resolve()
    paths = _default_paths(repo)
    for key, override in (
        ("dataset", dataset),
        ("model", model),
        ("source", source),
        ("whisper", whisper),
        ("torosu", torosu),
        ("output", output),
    ):
        if override is not None:
            paths[key] = override.expanduser().resolve()
    destination = paths["output"]
    if destination.name != "portable_4050_kit":
        raise ValueError("Output directory must be named portable_4050_kit.")
    if destination.exists():
        raise FileExistsError(f"Portable kit already exists; move it aside before rebuilding: {destination}")
    if destination == repo or destination in repo.parents:
        raise ValueError("Portable kit output cannot be the Git checkout or a repository ancestor.")
    if repo in destination.parents:
        expected = (repo / "workspace_exp" / "lora_6_10k_kit" / "portable_4050_kit").resolve()
        if destination != expected:
            raise ValueError("Portable kit output inside the checkout is restricted to ignored workspace_exp/lora_6_10k_kit/portable_4050_kit.")

    evidence = _validate_inputs(paths, repo)
    dataset_root = _require_dir(paths["dataset"], "reviewed dataset")
    model_root = _require_dir(paths["model"], "V32-mini model")
    source_root = _require_dir(paths["source"], "Mapperatorinator source")
    whisper_root = _require_dir(paths["whisper"], "offline Whisper configs")
    torosu_file = _require_file(paths["torosu"], "Torosu CLI")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".portable_4050_kit-", dir=destination.parent))
    try:
        _copy_tree(dataset_root, staging / "dataset")
        _copy_tree(model_root, staging / "models" / MODEL_DIRECTORY)
        _copy_tree(source_root, staging / "source")
        for repo_name in WHISPER_CONFIGS:
            bundled_repo = staging / "assets" / "huggingface" / repo_name
            _copy_tree(whisper_root / repo_name, bundled_repo)
            (bundled_repo / "refs" / "main").write_text(
                WHISPER_CONFIGS[repo_name], encoding="ascii", newline=""
            )
        torosu_destination = staging / "engines" / "torosu_cli" / "torosu-cli.exe"
        torosu_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(torosu_file, torosu_destination)

        manialab_destination = staging / "manialab"
        manialab_destination.mkdir()
        for module in MANIALAB_MODULES:
            shutil.copy2(repo / "manialab" / module, manialab_destination / module)
        training_destination = staging / "training"
        training_destination.mkdir()
        for relative in (
            "__init__.py",
            "prepare_10k_dataset.py",
            "sitecustomize.py",
            "train_10k_lora.py",
            "run_train_4050.ps1",
            "package_4050_kit.py",
        ):
            source_file = repo / "training" / relative
            if source_file.is_file():
                shutil.copy2(source_file, training_destination / relative)
        vendor_template = staging / "vendor" / "mapperatorinator" / "template.osu"
        vendor_template.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / "vendor" / "mapperatorinator" / "template.osu", vendor_template)
        requirements = staging / "assets" / "mapperatorinator-requirements.txt"
        requirements.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / "assets" / "mapperatorinator-requirements.txt", requirements)
        (staging / "4050训练说明.txt").write_text(
            "ManiaLab RTX 4050 LoRA 训练套件\n\n"
            "1. 将整个 portable_4050_kit 文件夹复制到有 NVIDIA RTX 4050 的 Windows 电脑。\n"
            "2. 在该文件夹空白处打开 PowerShell，运行：\\training\\run_train_4050.ps1\n"
            "3. 首次安装依赖需要网络；脚本优先尝试国内镜像。模型、源码和训练数据已随包提供。\n"
            "4. 续训运行：\\training\\run_train_4050.ps1 -Resume\n"
            "5. 过程试打谱位于 training\\progress_osz；最佳/最终 LoRA 位于 training\\best_lora 和 training\\final_lora。\n\n"
            "默认参数面向 6 GB 显存：v32-mini、micro-batch 1、梯度累积 16、bf16 与 gradient checkpointing。\n"
            "本套件只包含数据与训练工具，不承诺模型音游质量；请在 osu! 中实打验收。\n",
            encoding="utf-8",
            newline="\n",
        )

        entries = []
        total_size = 0
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            relative = path.relative_to(staging).as_posix()
            size = path.stat().st_size
            total_size += size
            entries.append({"path": relative, "size_bytes": size, "sha256": _sha256(path)})
        report = {
            "format_version": 1,
            "review_status": "human_reviewed",
            "tier1_chart_count": 42,
            "model_revision": "7807f0dc70cab671be012e1f5ddf945b0b8b7278",
            "mapperatorinator_commit": SOURCE_COMMIT,
            "files": entries,
            "total_file_count": len(entries),
            "total_size_bytes": total_size,
        }
        (staging / "KIT-MANIFEST.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
        verified = verify_kit_manifest(destination)
        return {**verified, "output": str(destination), "validated_inputs": len(evidence)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    defaults = _default_paths(_repo_root())
    parser = argparse.ArgumentParser(description="Assemble a self-contained RTX 4050 training kit (no downloads/training).")
    for option, key in (
        ("--dataset", "dataset"),
        ("--model", "model"),
        ("--source", "source"),
        ("--whisper", "whisper"),
        ("--torosu", "torosu"),
        ("--output", "output"),
    ):
        parser.add_argument(option, type=Path, default=defaults[key])
    args = parser.parse_args(argv)
    result = build_kit(
        dataset=args.dataset,
        model=args.model,
        source=args.source,
        whisper=args.whisper,
        torosu=args.torosu,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
