from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
import urllib.request
import zipfile


KIT_RELATIVE = Path("workspace_exp") / "lora_10k_kit"
MAPPERATORINATOR_COMMIT = "5884d1bcb57494f09fe3d74f3a6b58ca0d2960aa"
_MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json")
_BUCKET_SAMPLE_FACTORS = {
    "10k_ln": 4.0,
    "10k_hybrid": 2.0,
    "10k_rice": 0.25,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def kit_root() -> Path:
    return repo_root() / KIT_RELATIVE


def pin_runtime_paths(root: Path | None = None) -> dict[str, Path]:
    """Keep model, package, and temporary caches beside the portable kit."""
    runtime = (root or kit_root()) / "runtime"
    cache = runtime / "cache"
    paths = {
        "TMP": runtime / "tmp",
        "TEMP": runtime / "tmp",
        "TMPDIR": runtime / "tmp",
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "TORCH_HOME": cache / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torch" / "inductor",
        "XDG_CACHE_HOME": cache / "xdg",
        "PIP_CACHE_DIR": cache / "pip",
        "UV_CACHE_DIR": cache / "uv",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path.resolve())
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a portable 10K/LN LoRA adapter using Mapperatorinator V32."
    )
    parser.add_argument("--dataset-dir", type=Path, default=kit_root() / "dataset")
    parser.add_argument("--output-dir", type=Path, default=kit_root() / "training")
    parser.add_argument("--model", choices=("v32-mini", "v32"), default="v32-mini")
    parser.add_argument("--mapperator-root", type=Path)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--torosu-cli", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--validation-samples", type=int, default=2)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=0,
        help="Maximum teacher-forced validation batches; zero validates the full split.",
    )
    parser.add_argument(
        "--generation-window-seconds",
        type=float,
        default=0,
        help="Generate full songs by default; a positive value selects the densest real excerpt.",
    )
    parser.add_argument("--seed", type=int, default=20261126)
    return parser.parse_args(argv)


def validate_options(args: argparse.Namespace) -> None:
    for name in ("epochs", "micro_batch_size", "gradient_accumulation_steps", "checkpoint_every_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if (
        args.max_steps < 0
        or args.num_workers < 0
        or args.validation_samples < 0
        or args.validation_batches < 0
    ):
        raise ValueError("step count, worker count, and validation sample count cannot be negative")
    if args.lora_r < 1 or args.lora_alpha < 1:
        raise ValueError("LoRA r and alpha must be positive integers")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    if not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if not math.isfinite(args.generation_window_seconds) or args.generation_window_seconds < 0:
        raise ValueError("generation window must be zero or a positive number of seconds")


def cuda_training_profile(vram_bytes: int, bf16_supported: bool) -> dict[str, Any]:
    vram_gib = vram_bytes / 1024**3
    if vram_gib < 5:
        raise ValueError(f"CUDA training requires at least 5 GiB; found {vram_gib:.2f} GiB")
    if not bf16_supported:
        raise ValueError("CUDA training requires native bf16 support")
    return {
        "vram_gib": round(vram_gib, 2),
        "precision": "bf16",
        "gradient_checkpointing": True,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_safe_micro_batch_size": 1 if vram_gib <= 8 else 2,
    }


def estimate_cuda_training_memory(model: Any, optimizer: Any) -> dict[str, Any]:
    total_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer_class = type(optimizer).__name__.casefold()
    optimizer_state_bytes_per_parameter = (
        2 if "8bit" in optimizer_class else 8
    )
    optimizer_state_bytes = sum(
        parameter.numel() * optimizer_state_bytes_per_parameter
        for parameter in trainable_parameters
    )
    gradient_bytes = sum(
        parameter.numel() * 4
        for parameter in trainable_parameters
    )
    workspace_allowance_bytes = int(2.5 * 1024**3)
    estimated_peak_bytes = (
        total_parameter_bytes
        + optimizer_state_bytes
        + gradient_bytes
        + workspace_allowance_bytes
    )
    return {
        "parameter_weight_mib": round(total_parameter_bytes / 1024**2, 2),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "optimizer_state_mib_estimate": round(optimizer_state_bytes / 1024**2, 2),
        "gradient_mib_estimate": round(gradient_bytes / 1024**2, 2),
        "activation_and_cuda_workspace_allowance_gib": 2.5,
        "estimated_peak_vram_gib": round(estimated_peak_bytes / 1024**3, 2),
        "estimate_is_lower_bound": True,
    }


def active_holds_before_window(
    events: list[Any],
    event_times: list[int],
    window_start_ms: float,
    event_types: Any,
) -> list[dict[str, int | None]]:
    active: dict[int, dict[str, int | None]] = {}
    current_time = None
    current_lane = None
    current_group: list[Any] = []
    for event, event_time in zip(events, event_times):
        event_time = int(event_time)
        if event_time > window_start_ms:
            break
        if event_time != current_time:
            current_time = event_time
            current_lane = None
            current_group = [event]
        else:
            current_group.append(event)
        if event_time == window_start_ms:
            if event.type == event_types.MANIA_COLUMN:
                current_lane = int(event.value)
            elif event.type == event_types.HOLD_NOTE_END and current_lane is not None:
                active.pop(current_lane, None)
            continue
        if event.type == event_types.MANIA_COLUMN:
            current_lane = int(event.value)
        elif current_lane is None:
            continue
        elif event.type == event_types.HOLD_NOTE:
            active[current_lane] = {
                "lane": current_lane,
                "start_ms": event_time,
                "last_sustain_ms": None,
                "start_events": list(current_group),
                "last_sustain_events": [],
            }
        elif event.type == event_types.HOLD_NOTE_SUSTAIN:
            if current_lane in active:
                active[current_lane]["last_sustain_ms"] = event_time
                active[current_lane]["last_sustain_events"] = list(current_group)
        elif event.type == event_types.HOLD_NOTE_END:
            active.pop(current_lane, None)
    return [active[lane] for lane in sorted(active)]


def active_hold_prefix_events(
    active_holds: list[dict[str, int | None]],
    event_types: Any,
    event_factory: Any,
    window_start_ms: int,
) -> list[Any]:
    carried = []
    for hold in active_holds:
        carried.extend(
            (
                event_factory(event_types.TIME_SHIFT, int(window_start_ms)),
                event_factory(event_types.MANIA_COLUMN, int(hold["lane"])),
                event_factory(event_types.HOLD_NOTE),
                event_factory(event_types.HOLD_NOTE_SUSTAIN),
            )
        )
    return carried


def install_cross_window_hold_state_patch(source: Path) -> None:
    source_package_root = source / "osuT5"
    if str(source_package_root) not in sys.path:
        sys.path.insert(0, str(source_package_root))
    from osuT5.dataset.data_utils import SequenceDatasetMixin
    from osuT5.event import Event, EventType

    current = SequenceDatasetMixin._create_sequences
    if getattr(current, "_manialab_active_hold_state", False):
        return

    def create_sequences_with_active_holds(
        self,
        frames,
        frame_times,
        out_context,
        in_context,
        extra_data=None,
    ):
        sequences = current(
            self,
            frames,
            frame_times,
            out_context,
            in_context,
            extra_data,
        )
        if not out_context:
            return sequences
        source_context = out_context[0]
        for sequence in sequences:
            edge = float(sequence["special"]["time"])
            active = active_holds_before_window(
                source_context["events"],
                source_context["event_times"],
                edge,
                EventType,
            )
            if active:
                sequence["pre_events"] = active_hold_prefix_events(
                    active,
                    EventType,
                    Event,
                    int(edge),
                )
            else:
                sequence.pop("pre_events", None)
        return sequences

    create_sequences_with_active_holds._manialab_active_hold_state = True
    SequenceDatasetMixin._create_sequences = create_sequences_with_active_holds


def configure_audio_decoder() -> Path:
    import imageio_ffmpeg

    executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    os.environ["PATH"] = str(executable.parent) + os.pathsep + os.environ.get("PATH", "")
    from pydub import AudioSegment

    AudioSegment.converter = str(executable)

    @classmethod
    def load_with_bundled_ffmpeg(
        cls,
        file,
        format=None,
        codec=None,
        parameters=None,
        start_second=None,
        duration=None,
        **_kwargs,
    ):
        path = os.fspath(file)
        command = [str(executable), "-hide_banner", "-loglevel", "error"]
        if start_second is not None:
            command.extend(("-ss", str(start_second)))
        command.extend(("-i", path))
        if duration is not None:
            command.extend(("-t", str(duration)))
        if parameters:
            command.extend(str(value) for value in parameters)
        command.extend(
            (
                "-vn",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                "pipe:1",
            )
        )
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=180,
            check=False,
        )
        if result.returncode:
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise ValueError(f"Bundled FFmpeg audio decode failed: {detail[-1000:]}")
        sample_width = 4
        frame_count_bytes = len(result.stdout) - len(result.stdout) % sample_width
        if frame_count_bytes <= 0:
            raise ValueError(f"Bundled FFmpeg returned no PCM audio for {path}")
        return cls(
            data=result.stdout[:frame_count_bytes],
            sample_width=2,
            frame_rate=44100,
            channels=2,
        )

    AudioSegment.from_file = load_with_bundled_ffmpeg
    return executable


def manialab_dataloader_worker_init(worker_id: int) -> None:
    source_value = os.environ.get("MANIALAB_MAPPER_REPO")
    if not source_value:
        raise RuntimeError("Mapperatorinator source path is missing in a data worker.")
    source = Path(source_value).resolve()
    source_package_root = source / "osuT5"
    if str(source_package_root) not in sys.path:
        sys.path.insert(0, str(source_package_root))
    configure_audio_decoder()
    install_cross_window_hold_state_patch(source)

    import numpy as np
    import torch

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end
    per_worker = int(np.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


def assert_cross_window_prefix_in_batch(
    batch: dict[str, Any],
    tokenizer: Any,
    event_types: Any,
    event_factory: Any,
) -> bool:
    import torch

    decoder = batch["decoder_input_ids"]
    labels = batch["labels"]
    for index in range(decoder.shape[0]):
        sos_positions = torch.nonzero(decoder[index].eq(tokenizer.sos_id), as_tuple=False)
        if not len(sos_positions):
            continue
        prefix_end = int(sos_positions[0, 0])
        prefix = decoder[index, :prefix_end]
        hold_start = tokenizer.encode(event_factory(event_types.HOLD_NOTE))
        hold_sustain = tokenizer.encode(event_factory(event_types.HOLD_NOTE_SUSTAIN))
        has_active_hold = prefix.eq(hold_start).any() or prefix.eq(hold_sustain).any()
        if has_active_hold and labels[index, :prefix_end].eq(-100).all():
            return True
    return False


def resolve_mapperator_root(explicit: Path | None = None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["MANIALAB_MAPPER_REPO"]) if os.environ.get("MANIALAB_MAPPER_REPO") else None,
        repo_root() / "assets" / "runtime" / "source",
        repo_root().parent / "Mapperatorinator",
        kit_root() / "runtime" / "source",
    ]
    managed_source = kit_root() / "runtime" / "source"
    for candidate in candidates:
        if candidate and (candidate / "inference.py").is_file():
            resolved = candidate.resolve()
            if resolved == managed_source.resolve():
                marker = resolved / ".manialab-source.json"
                if not marker.is_file():
                    raise ValueError(
                        "The existing portable Mapperatorinator source has no ManiaLab marker; "
                        f"preserving it at {resolved}."
                    )
                identity = json.loads(marker.read_text(encoding="utf-8"))
                if identity.get("commit") != MAPPERATORINATOR_COMMIT:
                    raise ValueError(
                        "The existing portable Mapperatorinator source is not the pinned revision; "
                        f"preserving it at {resolved}."
                    )
            git_dir = resolved / ".git"
            if git_dir.exists():
                revision = subprocess.run(
                    ["git", "-C", str(resolved), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if revision.returncode == 0 and revision.stdout.strip() != MAPPERATORINATOR_COMMIT:
                    raise ValueError(
                        "Mapperatorinator checkout is not the locked revision "
                        f"{MAPPERATORINATOR_COMMIT}: {revision.stdout.strip()}"
                    )
            return resolved

    destination = managed_source
    runtime_root = destination.parent
    runtime_root.mkdir(parents=True, exist_ok=True)
    archive_path = runtime_root / f"mapperatorinator-source-{MAPPERATORINATOR_COMMIT[:12]}.zip"
    archive_hash_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    archive_url = (
        "https://github.com/OliBomby/Mapperatorinator/archive/"
        f"{MAPPERATORINATOR_COMMIT}.zip"
    )
    if archive_path.is_file():
        if not archive_hash_path.is_file():
            raise FileExistsError(
                f"Unmarked upstream archive exists; preserving it at {archive_path}."
            )
        expected_hash = archive_hash_path.read_text(encoding="ascii").strip()
        actual_hash = _sha256_path(archive_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Upstream archive checksum mismatch; preserving {archive_path}."
            )
    else:
        download_path = archive_path.with_name(
            f"{archive_path.name}.download-{os.getpid()}-{time.time_ns()}"
        )
        request = urllib.request.Request(
            archive_url,
            headers={"User-Agent": "ManiaLab-10K-LoRA/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response, download_path.open("xb") as output:
                shutil.copyfileobj(response, output)
            with zipfile.ZipFile(download_path) as archive:
                if archive.testzip() is not None or not any(
                    name.replace("\\", "/").endswith("/inference.py")
                    for name in archive.namelist()
                ):
                    raise ValueError("Pinned upstream archive failed its structural validation.")
            archive_hash = _sha256_path(download_path)
            download_path.replace(archive_path)
            archive_hash_path.write_text(archive_hash + "\n", encoding="ascii")
        finally:
            download_path.unlink(missing_ok=True)

    if destination.exists():
        marker = destination / ".manialab-source.json"
        if not marker.is_file():
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"Unmarked Mapperatorinator source exists; preserving it at {destination}."
                )
            destination.rmdir()
        else:
            identity = json.loads(marker.read_text(encoding="utf-8"))
            raise FileExistsError(
                f"Managed source at {destination} cannot be replaced "
                f"(marker commit {identity.get('commit')!r}); preserving it."
            )
    staging = runtime_root / f"source-staging-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            for member in members:
                relative = Path(*Path(member.filename.replace("\\", "/")).parts[1:])
                target = (staging / relative).resolve()
                if not target.is_relative_to(staging.resolve()):
                    raise ValueError(f"Unsafe upstream archive member: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        if not (staging / "inference.py").is_file():
            raise ValueError("Pinned Mapperatorinator archive is missing inference.py")
        (staging / ".manialab-source.json").write_text(
            json.dumps(
                {
                    "commit": MAPPERATORINATOR_COMMIT,
                    "archive_sha256": _sha256_path(archive_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if destination.exists():
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"Mapperatorinator source appeared during extraction; preserving it at {destination}."
                )
            destination.rmdir()
        staging.replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def resolve_model_cache(args: argparse.Namespace, root: Path) -> Path:
    if args.model_cache:
        cache = args.model_cache.expanduser().resolve()
    elif os.environ.get("MANIALAB_V32_MODEL_CACHE"):
        cache = Path(os.environ["MANIALAB_V32_MODEL_CACHE"]).expanduser().resolve()
    else:
        cache = root / "runtime" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def ensure_model_checkpoint(model_variant: str, model_cache: Path) -> tuple[Path, dict[str, str]]:
    import hashlib
    from huggingface_hub import HfApi, snapshot_download

    from manialab.v32_engine import MODEL_SPECS

    spec = MODEL_SPECS[model_variant]
    destination = model_cache / (
        f"Mapperatorinator-{model_variant}-{spec['revision'][:7]}"
    )
    checkpoint = destination / "gamemode=3"
    if not all((checkpoint / name).is_file() for name in _MODEL_FILES):
        info = HfApi().model_info(spec["repo"], revision=spec["revision"])
        if info.sha != spec["revision"]:
            raise RuntimeError(f"Model revision mismatch for {model_variant}")
        snapshot_download(
            repo_id=spec["repo"],
            revision=spec["revision"],
            local_dir=destination,
            cache_dir=os.environ["HF_HOME"],
            allow_patterns=("gamemode=3/*",),
        )
    missing = [name for name in _MODEL_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{model_variant} model snapshot is missing: {', '.join(missing)}")
    revision_manifest = destination / "model-revision.json"
    expected_manifest = {
        "repo": spec["repo"],
        "revision": spec["revision"],
        "sha256": {
            name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            for name in _MODEL_FILES
        },
    }
    if revision_manifest.is_file():
        stored = json.loads(revision_manifest.read_text(encoding="utf-8"))
        if stored.get("repo") != spec["repo"] or stored.get("revision") != spec["revision"]:
            raise ValueError(f"{model_variant} model cache has an unexpected pinned revision")
        for name, digest in expected_manifest["sha256"].items():
            if digest != stored.get("sha256", {}).get(name):
                raise ValueError(f"{model_variant} model cache checksum mismatch: {name}")
    else:
        _atomic_write_json(revision_manifest, expected_manifest)
    return destination, spec


def load_dataset_manifest(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = dataset_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("dataset manifest must contain a records list")
    train = [record for record in records if record.get("split") == "train" and record.get("keycount") == 10]
    val = [record for record in records if record.get("split") == "val" and record.get("keycount") == 10]
    if not train or not val:
        raise ValueError("dataset needs non-empty train and validation 10K splits")
    train_audio = {record.get("audio_sha256") for record in train}
    val_audio = {record.get("audio_sha256") for record in val}
    train_sets = {str(record.get("set_id")) for record in train}
    val_sets = {str(record.get("set_id")) for record in val}
    train_groups = {str(record.get("leakage_group_id")) for record in train}
    val_groups = {str(record.get("leakage_group_id")) for record in val}
    if train_audio & val_audio or train_sets & val_sets or train_groups & val_groups:
        raise ValueError("dataset manifest leaks a song, audio hash, or mapset across splits")
    for record in train + val:
        for key in (
            "difficulty",
            "hold_note_ratio",
            "audio_sha256",
            "set_id",
            "leakage_group_id",
            "audio_relpath",
            "data_relpath",
        ):
            if record.get(key) is None:
                raise ValueError(f"dataset record is missing required field: {key}")
        if not 0 <= float(record["hold_note_ratio"]) <= 1:
            raise ValueError("dataset hold_note_ratio must be between 0 and 1")
        if not math.isfinite(float(record["difficulty"])) or float(record["difficulty"]) <= 0:
            raise ValueError("dataset difficulty must be a positive finite value")
        for key in ("audio_relpath", "data_relpath"):
            resolved = (dataset_dir / record[key]).resolve()
            if not resolved.is_relative_to(dataset_dir.resolve()) or not resolved.is_file():
                raise FileNotFoundError(f"dataset record points to a missing or unsafe file: {key}")
    return payload, train + val


def build_sample_weights(
    records: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    training_records = [
        record
        for record in records
        if record.get("split") == "train"
        and record.get("keycount") in (7, 8, 9, 10)
    ]
    if not training_records:
        raise ValueError("No train records are available for LN bucket balancing.")
    raw: dict[int, float] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in training_records:
        beatmap_id = int(record["bid"])
        bucket = str(record["bucket"])
        factor = _BUCKET_SAMPLE_FACTORS.get(bucket, 1.0 if bucket.endswith("_ln_aux") else 1.0)
        raw[beatmap_id] = factor
        grouped[bucket].append(factor)
    mean_factor = sum(raw.values()) / len(raw)
    normalized = {beatmap_id: factor / mean_factor for beatmap_id, factor in raw.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{beatmap_id},{normalized[beatmap_id]:.8f}\n" for beatmap_id in sorted(normalized)),
        encoding="ascii",
    )
    return {
        "csv": str(output_path),
        "chart_count": len(normalized),
        "raw_factor_by_bucket": {
            bucket: round(sum(values) / len(values), 4)
            for bucket, values in sorted(grouped.items())
        },
        "normalized_weight_by_bucket": {
            bucket: round(
                sum(normalized[int(record["bid"])] for record in training_records if record["bucket"] == bucket)
                / sum(record["bucket"] == bucket for record in training_records),
                4,
            )
            for bucket in sorted(grouped)
        },
        "normalized_mean": sum(normalized.values()) / len(normalized),
    }


def validate_nll(
    model: Any,
    dataloader: Any,
    accelerator: Any,
    tokenizer: Any,
    event_type: Any,
    max_batches: int = 0,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    model.eval()
    hold_start_tokens = {
        tokenizer.event_start[event_type.HOLD_NOTE],
        tokenizer.event_start[event_type.HOLD_NOTE_SUSTAIN],
    }
    cross_window_context_batches = 0
    total_nll = 0.0
    total_tokens = 0
    total_time_nll = 0.0
    total_time_tokens = 0
    total_lane_nll = 0.0
    total_lane_tokens = 0
    hold_stats: dict[str, dict[str, int]] = {
        "hold_start": {"correct": 0, "total": 0},
        "hold_release": {"correct": 0, "total": 0},
    }
    track_nll: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader, start=1):
            if max_batches and batch_index > max_batches:
                break
            labels = batch["labels"]
            if any(
                int(token) in hold_start_tokens
                for row in batch["decoder_input_ids"].tolist()
                for token in row[: next(
                    (
                        index
                        for index, value in enumerate(row)
                        if value == tokenizer.sos_id
                    ),
                    len(row),
                )]
            ):
                cross_window_context_batches += 1
            outputs = model(**batch)
            logits = outputs.logits.float()
            token_loss = functional.cross_entropy(
                logits.transpose(1, 2),
                labels,
                ignore_index=-100,
                reduction="none",
            )
            valid = labels.ne(-100)
            total_nll += float(token_loss.sum().item())
            total_tokens += int(valid.sum().item())
            time_mask = (
                labels.ge(tokenizer.event_start[event_type.TIME_SHIFT])
                & labels.lt(tokenizer.event_end[event_type.TIME_SHIFT])
            )
            lane_mask = (
                labels.ge(tokenizer.event_start[event_type.MANIA_COLUMN])
                & labels.lt(tokenizer.event_end[event_type.MANIA_COLUMN])
            )
            total_time_nll += float(token_loss[time_mask].sum().item())
            total_time_tokens += int(time_mask.sum().item())
            total_lane_nll += float(token_loss[lane_mask].sum().item())
            total_lane_tokens += int(lane_mask.sum().item())
            predictions = logits.argmax(dim=-1)
            for key, target_type in (
                ("hold_start", event_type.HOLD_NOTE),
                ("hold_release", event_type.HOLD_NOTE_END),
            ):
                lower = tokenizer.event_start[target_type]
                upper = tokenizer.event_end[target_type]
                mask = labels.ge(lower) & labels.lt(upper)
                hold_stats[key]["correct"] += int((predictions[mask] == labels[mask]).sum().item())
                hold_stats[key]["total"] += int(mask.sum().item())
            if "beatmap_idx" in batch:
                for index, sample_id in enumerate(batch["beatmap_idx"].tolist()):
                    sample_mask = valid[index]
                    sample_tokens = int(sample_mask.sum().item())
                    if not sample_tokens:
                        continue
                    sample_id = int(sample_id)
                    track_nll[sample_id][0] += float(token_loss[index][sample_mask].sum().item())
                    track_nll[sample_id][1] += sample_tokens
    track_values = [
        numerator / denominator
        for numerator, denominator in track_nll.values()
        if denominator
    ]
    model.train()
    return {
        "event_nll": total_nll / max(total_tokens, 1),
        "time_event_nll": total_time_nll / max(total_time_tokens, 1),
        "lane_nll": total_lane_nll / max(total_lane_tokens, 1),
        "track_nll": sum(track_values) / max(len(track_values), 1),
        "validated_tokens": total_tokens,
        "time_event_tokens": total_time_tokens,
        "lane_tokens": total_lane_tokens,
        "validated_tracks": len(track_values),
        "hold_start_accuracy": (
            hold_stats["hold_start"]["correct"] / max(hold_stats["hold_start"]["total"], 1)
        ),
        "hold_release_accuracy": (
            hold_stats["hold_release"]["correct"] / max(hold_stats["hold_release"]["total"], 1)
        ),
        "hold_start_count": hold_stats["hold_start"]["total"],
        "hold_release_count": hold_stats["hold_release"]["total"],
        "cross_window_hold_context_batches": cross_window_context_batches,
    }


def select_generation_window(chart_text: str, duration_ms: int) -> tuple[int, int]:
    from manialab.domain import parse_osu, validate

    chart = parse_osu(chart_text)
    if duration_ms <= 0:
        return 0, 0
    notes = chart["notes"]
    if not notes:
        return 0, duration_ms
    max_end = max(note.end for note in notes)
    duration_ms = min(duration_ms, max(max_end, duration_ms))
    candidates = range(0, max(1, max_end - duration_ms + 1), 1000)
    start_ms = max(
        candidates,
        key=lambda start: sum(start <= note.start < start + duration_ms for note in notes),
        default=0,
    )
    return start_ms, start_ms + duration_ms


def chart_generation_metrics(
    chart_text: str,
    target_hold_ratio: float,
    torosu: dict[str, Any],
) -> dict[str, Any]:
    from manialab.domain import parse_osu, validate

    chart = parse_osu(chart_text)
    difficulty = None
    in_difficulty = False
    for line in chart_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_difficulty = stripped.casefold() == "[difficulty]"
        elif in_difficulty and stripped.casefold().startswith("overalldifficulty:"):
            try:
                difficulty = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                difficulty = None
            break
    if chart["keys"] != 10:
        raise ValueError(f"generated chart must be 10K, found {chart['keys']}K")
    validation = validate(chart["notes"], chart["keys"])
    if not validation["valid"]:
        raise ValueError(f"generated chart is structurally invalid: {validation['issues'][0]}")
    notes = chart["notes"]
    hold_count = sum(note.hold for note in notes)
    actual_ratio = hold_count / max(len(notes), 1)
    sweep = []
    for note in notes:
        if note.hold and note.end > note.start:
            sweep.append((note.end, -1))
            sweep.append((note.start, 1))
    active = peak_held = 0
    for _, delta in sorted(sweep, key=lambda item: (item[0], item[1])):
        active += delta
        peak_held = max(peak_held, active)
    left_pinky = sum(note.lane == 0 for note in notes)
    right_pinky = sum(note.lane == chart["keys"] - 1 for note in notes)
    pinky_balance = (
        min(left_pinky, right_pinky) / max(left_pinky, right_pinky)
        if max(left_pinky, right_pinky)
        else 1.0
    )
    return {
        "objects": len(notes),
        "hold_notes": hold_count,
        "actual_hold_note_ratio": actual_ratio,
        "hold_ratio_error": abs(actual_ratio - float(target_hold_ratio)),
        "difficulty": difficulty,
        "peak_held": peak_held,
        "minijack": int(_torosu_value(torosu, "minijack")),
        "pill": int(_torosu_value(torosu, "pill")),
        "hand_balance": float(torosu.get("hand_balance", 1.0)),
        "left_pinky_notes": left_pinky,
        "right_pinky_notes": right_pinky,
        "pinky_balance": pinky_balance,
    }


def _torosu_value(torosu: dict[str, Any], key: str) -> Any:
    aliases = {
        "minijack": ("minijack", "minijack_count"),
        "pill": ("pill", "pill_count"),
    }
    for name in aliases.get(key, (key,)):
        if name in torosu:
            return torosu[name]
    modifiers = torosu.get("modifiers", {})
    if key in modifiers:
        return modifiers[key]
    return 0


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = Path(*Path(member.filename.replace("\\", "/")).parts[1:])
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe upstream archive member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_torosu_cli(crate_dir: Path, explicit: Path | None, runtime_dir: Path) -> Path:
    candidates = [
        explicit,
        Path(os.environ["MANIALAB_TOROSU_CLI"]) if os.environ.get("MANIALAB_TOROSU_CLI") else None,
        crate_dir / "torosu-cli.exe",
        crate_dir / "torosu-cli",
        crate_dir / "target" / "release" / "torosu-cli.exe",
        crate_dir / "target" / "release" / "torosu-cli",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()

    cargo = shutil.which("cargo")
    if not cargo:
        raise FileNotFoundError(
            "Torosu CLI is required for validation; copy the project CLI into "
            "engines/torosu_cli or install Rust Cargo."
        )
    target_root = runtime_dir / "cargo-target"
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_root)
    result = subprocess.run(
        [cargo, "build", "--release", "--locked", "--manifest-path", str(crate_dir / "Cargo.toml")],
        cwd=repo_root(),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "Torosu CLI build failed: " + (result.stderr or result.stdout)[-3000:]
        )
    built = target_root / "release" / ("torosu-cli.exe" if os.name == "nt" else "torosu-cli")
    if not built.is_file():
        raise FileNotFoundError("Cargo completed without producing torosu-cli")
    os.environ["MANIALAB_TOROSU_CLI"] = str(built.resolve())
    return built.resolve()


def _load_upstream(args: argparse.Namespace, source: Path, model_root: Path):
    source_package_root = source / "osuT5"
    if str(source_package_root) not in sys.path:
        sys.path.insert(0, str(source_package_root))
    os.environ["MANIALAB_MAPPER_REPO"] = str(source)
    os.environ["PYTHONPATH"] = (
        str(source_package_root)
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    )
    configure_audio_decoder()

    from accelerate import Accelerator
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from osuT5.event import ContextType, Event, EventType
    from osuT5.utils import get_dataloaders, get_optimizer, get_scheduler, load_model, setup_args

    install_cross_window_hold_state_patch(source)
    config_dir = source / "configs" / "train"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir.resolve())):
        config = compose(config_name=args.model)
    values = {
        "compile": False,
        "seed": args.seed,
        "device": "cpu" if args.device == "cpu" else "gpu",
        "mixed_precision": None if args.device == "cpu" else "bf16",
        "precision": "fp32" if args.device == "cpu" else "bf16",
        "attn_implementation": "sdpa",
        "enable_lora": True,
        "pretrained_path": str(model_root),
        "pretrained_gamemode": 3,
        "checkpoint_path": "",
        "data.dataset_type": "mmrs",
        "data.train_dataset_path": str(args.dataset_dir / "train"),
        "data.test_dataset_path": str(args.dataset_dir / "val"),
        "data.train_dataset_start": 0,
        "data.train_dataset_end": _count_mapsets(args.dataset_dir / "train"),
        "data.test_dataset_start": 0,
        "data.test_dataset_end": _count_mapsets(args.dataset_dir / "val"),
        "data.gamemodes": [3],
        "data.ranked_statuses": [1, 4],
        "data.min_year": None,
        "data.max_year": None,
        "data.min_difficulty": 0,
        "data.max_difficulty": 100,
        "data.per_track": True,
        "data.add_pre_tokens": False,
        "data.max_pre_token_len": -1,
        "data.add_empty_sequences": False,
        "data.add_keycount_token": True,
        "data.add_hold_note_ratio_token": True,
        "data.hold_note_ratio_dropout_prob": 0.0,
        "data.diff_dropout_prob": 0.0,
        "data.cs_dropout_prob": 0.0,
        "data.add_descriptors": False,
        "data.add_gamemode_token": True,
        "data.add_timing_points": True,
        "data.sustain_interval": 8000,
        "data.timing_random_offset": 0,
        "data.timing_random_offset_2": 0,
        "data.timing_random_offset_prob": 0.0,
        "data.snapping_random_prob": 0.0,
        "data.context_types": [{"in": [ContextType.TIMING], "out": [ContextType.MAP]}],
        "data.context_weights": [1.0],
        "data.dt_augment_prob": 0.0,
        "data.lookback_prob": 0.0,
        "data.frame_offset_augment_prob": 0.0,
        "data.flip_horizontal_prob": 0.0,
        "data.flip_vertical_prob": 0.0,
        "data.sample_weights_path": str(args.sample_weights_path),
        "dataloader.num_workers": args.num_workers,
        "dataloader.drop_last": False,
        "dataloader.balancer_buffer_size": 0,
        "optim.batch_size": args.micro_batch_size * args.gradient_accumulation_steps,
        "optim.grad_acc": args.gradient_accumulation_steps,
        "optim.total_steps": 1000000,
        "optim.base_lr": args.learning_rate,
        "optim.weight_decay": args.weight_decay,
        "optim.name": "adamw",
        "optim.warmup_steps": 0,
        "optim.sustain_steps": 0,
        "optim.lr_scheduler": "cosine",
        "optim.final_cosine": args.learning_rate * 0.1,
        "eval.every_steps": 1000000,
        "eval.steps": 1,
        "checkpoint.every_steps": args.checkpoint_every_steps,
        "logging.every_steps": 1,
        "lora.r": args.lora_r,
        "lora.lora_alpha": args.lora_alpha,
        "lora.lora_dropout": args.lora_dropout,
        "lora.target_modules": ["Wq", "Wkv", "Wqkv", "Wo", "fc1", "fc2"],
        "lora.bias": "none",
        "lora_metadata.ckpt_subfolders": ["gamemode=3"],
    }
    for key, value in values.items():
        OmegaConf.update(config, key, value, merge=False)
    train_config = OmegaConf.to_object(config)
    setup_args(train_config)
    import osuT5.utils.model_utils as model_utils

    model_utils.worker_init_fn = manialab_dataloader_worker_init

    device = "cpu" if args.device == "cpu" else None
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=None if args.device == "cpu" else "bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    model, tokenizer = load_model(
        model_root / "gamemode=3",
        train_config,
        device=accelerator.device if device is None else device,
        precision=train_config.precision,
        attn_implementation="sdpa",
        eval_mode=False,
        gamemode=3,
        auto_select_gamemode_model=False,
    )
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    if not getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError("Mapperatorinator gradient checkpointing did not activate")
    model.config.use_cache = False
    model.transformer.config.use_cache = False
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["Wq", "Wkv", "Wqkv", "Wo", "fc1", "fc2"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    train_loader, val_loader = get_dataloaders(
        tokenizer,
        train_config,
        shared_state(),
    )
    optimizer = build_optimizer(model, args)
    scheduler = get_scheduler(optimizer, train_config, accelerator)
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )
    accelerator.register_for_checkpointing(tokenizer)
    return (
        accelerator,
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        train_config,
        EventType,
        ContextType,
        Event,
    )


def shared_state():
    return SimpleNamespace(
        current_train_step=1,
        current_epoch=1,
        last_log=time.time(),
        current_loss=math.inf,
        best_loss=math.inf,
    )


def _count_mapsets(split_dir: Path) -> int:
    import pandas as pd

    metadata = pd.read_parquet(split_dir / "metadata.parquet", columns=["BeatmapSetId"])
    return int(metadata["BeatmapSetId"].nunique())


def build_optimizer(model: Any, args: argparse.Namespace):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA setup produced no trainable parameters")
    if args.device != "cpu":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                trainable,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        except (ImportError, RuntimeError, OSError):
            pass
    import torch

    return torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def _write_checkpoint(
    accelerator: Any,
    model: Any,
    checkpoint_dir: Path,
    *,
    global_step: int,
    epoch: int,
    args: argparse.Namespace,
    model_spec: dict[str, str],
    epoch_complete: bool,
    cross_window_context_seen: bool,
    training_seconds: float,
    optimizer_steps: int,
    peak_rss_bytes: int,
    validation: dict[str, Any] | None = None,
) -> Path:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        accelerator.save_state(str(checkpoint_dir), safe_serialization=False)
        adapter_dir = checkpoint_dir / "lora"
        accelerator.unwrap_model(model).save_pretrained(adapter_dir, safe_serialization=True)
        state = {
            "global_step": global_step,
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "cross_window_context_seen": cross_window_context_seen,
            "training_seconds": training_seconds,
            "optimizer_steps": optimizer_steps,
            "peak_rss_bytes": peak_rss_bytes,
            "model": args.model,
            "validation": validation,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(checkpoint_dir / "train-state.json", state)
        write_adapter_manifest(
            adapter_dir,
            model_spec=model_spec,
            args=args,
            global_step=global_step,
            epoch=epoch,
            validation=validation,
        )
    accelerator.wait_for_everyone()
    return checkpoint_dir


def write_adapter_manifest(
    adapter_dir: Path,
    *,
    model_spec: dict[str, str],
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    validation: dict[str, Any] | None,
) -> Path:
    payload = {
        "schema_version": 1,
        "base_model": {
            "variant": args.model,
            "repository": model_spec["repo"],
            "revision": model_spec["revision"],
            "gamemode": 3,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": ["Wq", "Wkv", "Wqkv", "Wo", "fc1", "fc2"],
        },
        "training": {
            "global_step": global_step,
            "epoch": epoch,
            "dataset_manifest": "workspace_exp/lora_10k_kit/dataset/manifest.json",
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
        "validation": validation,
    }
    return _atomic_write_json(adapter_dir / "lora-manifest.json", payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_checkpoint_state(checkpoint: Path) -> dict[str, Any]:
    state_path = checkpoint / "train-state.json"
    if not state_path.is_file():
        raise ValueError(f"Checkpoint is missing train-state.json: {checkpoint}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        checkpoint_root = output_dir
    for child in checkpoint_root.glob("checkpoint-*"):
        if child.is_dir() and (child / "train-state.json").is_file():
            try:
                state = _read_checkpoint_state(child)
                candidates.append(
                    (
                        int(state["global_step"]),
                        int(state["epoch"]),
                        bool(state.get("epoch_complete", True)),
                        child,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return max(
        candidates,
        default=(0, 0, False, None),
        key=lambda entry: (entry[0], entry[1], entry[2]),
    )[3]


def _resolve_validation_records(dataset_dir: Path, records: list[dict[str, Any]], count: int):
    if count < 1:
        return []
    candidates = [
        record
        for record in records
        if record.get("split") == "val"
        and record.get("keycount") == 10
        and 0 <= float(record.get("hold_note_ratio", 0)) <= 0.85
    ]
    from training.prepare_10k_dataset import leakage_group_ids

    bucket_priority = {"10k_ln": 0, "10k_hybrid": 1, "10k_rice": 2}

    def preference(record: dict[str, Any]) -> tuple[int, float, float, str, int]:
        ratio = float(record.get("hold_note_ratio", 0))
        bucket = record.get("bucket")
        if bucket not in bucket_priority:
            bucket = "10k_ln" if ratio >= 0.45 else "10k_hybrid" if ratio >= 0.15 else "10k_rice"
        return (
            bucket_priority[bucket],
            -ratio,
            -float(record.get("difficulty", 0)),
            str(record.get("audio_sha256", "")),
            int(record.get("bid", 0)),
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_id, record in zip(leakage_group_ids(candidates), candidates):
        grouped[group_id].append(record)
    representatives = [min(group, key=preference) for group in grouped.values()]
    representatives.sort(key=preference)
    return representatives[:count]


def _free_generation_evaluation(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    dataset_dir: Path,
    output_dir: Path,
    adapter_dir: Path,
    model_cache: Path,
    mapperator_root: Path,
    epoch: int,
) -> list[dict[str, Any]]:
    if args.validation_samples == 0:
        return []
    trainer_dir = str(Path(__file__).resolve().parent)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        trainer_dir
        + (os.pathsep + current_pythonpath if current_pythonpath else "")
    )
    os.environ["MANIALAB_AUDIO_FFMPEG_COMPAT"] = "1"
    from manialab.domain import parse_osu, validate
    from manialab.torosu_bridge import analyze_osu_text
    from manialab.v32_engine import V32Engine

    selected = _resolve_validation_records(dataset_dir, records, args.validation_samples)
    if not selected:
        raise ValueError("validation set has no 10K chart for free generation")
    engine = V32Engine(
        data_root=output_dir / "runtime-data",
        resources=repo_root(),
        mapper_repo=mapperator_root,
        python=sys.executable,
        model_cache=model_cache,
    )
    results = []
    for index, record in enumerate(selected, start=1):
        reference = (dataset_dir / record["data_relpath"]).resolve()
        audio = (dataset_dir / record["audio_relpath"]).resolve()
        reference_text = reference.read_text(encoding="utf-8-sig")
        start_ms = 0
        end_ms = None
        if args.generation_window_seconds > 0:
            start_ms, end_ms = select_generation_window(
                reference_text,
                round(args.generation_window_seconds * 1000),
            )
        sample_root = output_dir / "free-generation" / f"epoch-{epoch:04d}"
        sample_dir = sample_root / f"sample-{index:02d}"
        report_path = sample_dir / "run-report.json"
        report = None
        if report_path.is_file():
            try:
                cached_report = json.loads(report_path.read_text(encoding="utf-8"))
                cached_map = Path(cached_report["map"])
                if (
                    cached_report.get("model_variant") == args.model
                    and cached_report.get("lora_loaded") is True
                    and cached_report.get("lora_path")
                    and Path(cached_report["lora_path"]).resolve() == adapter_dir.resolve()
                    and cached_map.is_file()
                ):
                    report = cached_report
                    report["map"] = cached_map
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                report = None
        if report is None:
            if sample_dir.exists():
                attempt = 2
                while (sample_root / f"sample-{index:02d}-attempt-{attempt:02d}").exists():
                    attempt += 1
                sample_dir = sample_root / f"sample-{index:02d}-attempt-{attempt:02d}"
            sample_dir.parent.mkdir(parents=True, exist_ok=True)
            report = engine.generate(
                model_variant=args.model,
                audio_path=audio,
                reference_chart=reference,
                output_dir=sample_dir,
                keycount=10,
                difficulty=float(record["difficulty"]),
                hold_note_ratio=float(record["hold_note_ratio"]),
                lora_path=adapter_dir,
                device=args.device,
                progress=lambda _percent, _message: None,
                cancelled=lambda: False,
                start_ms=start_ms,
                end_ms=end_ms,
                seed=args.seed + epoch * 100 + index,
            )
        generated_path = Path(report["map"])
        generated_text = generated_path.read_text(encoding="utf-8-sig")
        model_output = parse_osu(generated_text)
        raw_validation = validate(model_output["notes"], model_output["keys"])
        repaired_path = None
        if not raw_validation["valid"]:
            repaired_path = _repair_conflicting_10k_with_torosu(generated_path)
            generated_path = repaired_path
            generated_text = generated_path.read_text(encoding="utf-8-sig")
        generated = parse_osu(generated_text)
        torosu = analyze_osu_text(generated_text, keys=10)
        inference_log_path = Path(report["log"])
        inference_log = (
            inference_log_path.read_text(encoding="utf-8", errors="replace")
            if inference_log_path.is_file()
            else ""
        )
        incomplete_ln_warnings = sum(
            "Warning: Incomplete hold note at" in line
            for line in inference_log.splitlines()
        )
        metrics = chart_generation_metrics(
            generated_text,
            float(record["hold_note_ratio"]),
            torosu,
        )
        metrics.update(
            {
                "record_set_id": record["set_id"],
                "record_bid": record["bid"],
                "target_difficulty": float(record["difficulty"]),
                "target_hold_note_ratio": float(record["hold_note_ratio"]),
                "generated_path": str(generated_path),
                "model_output_path": str(report["map"]),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "elapsed_seconds": report["elapsed_seconds"],
                "peak_rss_mib": report["peak_rss_mib"],
                "lora_loaded": report["lora_loaded"],
                "map_keys": generated["keys"],
                "incomplete_ln_warning_count": incomplete_ln_warnings,
                "raw_model_objects": len(model_output["notes"]),
                "raw_model_valid": raw_validation["valid"],
                "torosu_repair_path": (
                    str(repaired_path) if repaired_path is not None else None
                ),
                "torosu_repair_added_objects": (
                    len(generated["notes"]) - len(model_output["notes"])
                    if repaired_path is not None
                    else 0
                ),
            }
        )
        results.append(metrics)
    return results


def _repair_conflicting_10k_with_torosu(source_chart: Path) -> Path:
    binary = os.environ.get("MANIALAB_TOROSU_CLI")
    if not binary or not Path(binary).is_file():
        raise FileNotFoundError("A built Torosu CLI is required to repair the V32 smoke chart.")
    directory = source_chart.parent
    intermediate = directory / "torosu-repair-7k.osu"
    output = directory / "torosu-repaired-10k.osu"
    environment = os.environ.copy()

    commands = (
        (
            binary,
            "convert",
            "--input",
            str(source_chart),
            "--output",
            str(intermediate),
            "--keys",
            "7",
            "--density",
            "none",
            "--keep-jacks",
        ),
        (
            binary,
            "convert",
            "--input",
            str(intermediate),
            "--output",
            str(output),
            "--keys",
            "10",
            "--density",
            "dense",
            "--keep-jacks",
            "--strict",
        ),
    )
    for command in commands:
        result = subprocess.run(
            command,
            cwd=repo_root(),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "Torosu could not repair the V32 10K conflicts: "
                + (result.stderr or result.stdout)[-2000:]
            )
    from manialab.domain import parse_osu, validate

    parsed = parse_osu(output.read_text(encoding="utf-8-sig"))
    validation = validate(parsed["notes"], parsed["keys"])
    if parsed["keys"] != 10 or not validation["valid"]:
        raise ValueError("Torosu repair did not produce a legal 10K chart.")
    return output


def _copy_final_adapter(source: Path, destination: Path, global_step: int) -> Path:
    source_weights = source / "adapter_model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(f"Adapter weights are missing: {source_weights}")
    if destination.exists():
        existing_weights = destination / "adapter_model.safetensors"
        if existing_weights.is_file() and _sha256_file(existing_weights) == _sha256_file(source_weights):
            return destination
        destination = destination.with_name(f"adapter-step-{global_step:08d}")
        if destination.exists():
            existing_weights = destination / "adapter_model.safetensors"
            if existing_weights.is_file() and _sha256_file(existing_weights) == _sha256_file(source_weights):
                return destination
            raise FileExistsError(
                f"Preserving existing adapter output; choose a new output directory: {destination}"
            )
    staging = destination.with_name(
        f"{destination.name}.staging-{os.getpid()}-{time.time_ns()}"
    )
    shutil.copytree(source, staging)
    if destination.exists():
        raise FileExistsError(
            f"Adapter output appeared during export; preserving it at {destination}"
        )
    staging.replace(destination)
    return destination


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_options(args)
    root = kit_root()
    paths = pin_runtime_paths(root)
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, records = load_dataset_manifest(args.dataset_dir)
    sample_weight_report = build_sample_weights(
        records,
        args.output_dir / "sample-weights.csv",
    )
    args.sample_weights_path = sample_weight_report["csv"]
    mapperator_root = resolve_mapperator_root(args.mapperator_root)
    source_hash = hashlib.sha256()
    with (mapperator_root / "inference.py").open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            source_hash.update(block)
    source_digest = source_hash.hexdigest()
    model_cache = resolve_model_cache(args, root)
    model_root, model_spec = ensure_model_checkpoint(args.model, model_cache)
    torosu_cli = _resolve_torosu_cli(
        repo_root() / "engines" / "torosu_cli",
        args.torosu_cli,
        root / "runtime",
    )
    os.environ["MANIALAB_TOROSU_CLI"] = str(torosu_cli)
    os.environ["MANIALAB_MAPPER_PYTHON"] = sys.executable

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment")
    gpu_profile = None
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_profile = cuda_training_profile(
            props.total_memory,
            torch.cuda.is_bf16_supported(),
        )
        if gpu_profile["vram_gib"] <= 6.5 and args.micro_batch_size != 1:
            raise RuntimeError("The 6 GB VRAM profile requires --micro-batch-size 1.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    (
        accelerator,
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        _train_config,
        event_type,
        _context_type,
        event_factory,
    ) = _load_upstream(args, mapperator_root, model_root)
    if not hasattr(event_type, "HOLD_NOTE") or not hasattr(event_type, "HOLD_NOTE_END"):
        raise RuntimeError("Mapperatorinator Mania hold event types are unavailable")
    if gpu_profile is not None:
        memory_estimate = estimate_cuda_training_memory(model, optimizer)
        gpu_profile["training_memory_estimate"] = memory_estimate
        if memory_estimate["estimated_peak_vram_gib"] > gpu_profile["vram_gib"]:
            raise RuntimeError(
                "The estimated training memory exceeds available VRAM: "
                f"{memory_estimate['estimated_peak_vram_gib']:.2f} GiB > "
                f"{gpu_profile['vram_gib']:.2f} GiB. Keep micro-batch 1 and gradient "
                "checkpointing enabled, or use v32-mini."
            )

    window_width = int(round(
        (_train_config.data.src_seq_len - 1)
        * _train_config.data.hop_length
        * 1000
        / _train_config.data.sample_rate
    ))
    expected_cross_window_holds = int(
        manifest.get("summary", {}).get("cross_window_active_hold_count", 0)
    )
    resume = args.resume_from.expanduser().resolve() if args.resume_from else None
    if args.resume_latest:
        resume = find_latest_checkpoint(args.output_dir)
    global_step = 0
    starting_epoch = 1
    resumed_cross_window_context_seen = False
    total_training_seconds = 0.0
    total_optimizer_steps = 0
    peak_rss_bytes = 0
    if resume:
        if not resume.is_dir():
            raise FileNotFoundError(f"Checkpoint does not exist: {resume}")
        accelerator.load_state(str(resume))
        state = _read_checkpoint_state(resume)
        global_step = int(state["global_step"])
        resumed_cross_window_context_seen = bool(
            state.get("cross_window_context_seen", False)
        )
        total_training_seconds = float(state.get("training_seconds", 0.0))
        total_optimizer_steps = int(state.get("optimizer_steps", 0))
        peak_rss_bytes = int(state.get("peak_rss_bytes", 0))
        epoch_complete = bool(state.get("epoch_complete", True))
        starting_epoch = int(state["epoch"]) + (1 if epoch_complete else 0)

    checkpoint_root = args.output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "validation-metrics.jsonl"
    validation_history: list[dict[str, Any]] = []
    completed_epochs = starting_epoch - 1
    final_adapter_dir: Path | None = None
    resume_state = _read_checkpoint_state(resume) if resume else None
    max_steps = args.max_steps or None
    initial_started = time.perf_counter()
    latest_epoch_steps = 0
    latest_eval_seconds = 0.0
    cross_window_context_seen = resumed_cross_window_context_seen
    import psutil

    process_memory = psutil.Process()
    for epoch in range(starting_epoch, args.epochs + 1):
        model.train()
        epoch_steps = 0
        epoch_started = time.perf_counter()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in train_loader:
            with accelerator.accumulate(model):
                if max_steps is not None and global_step >= max_steps:
                    break
                peak_rss_bytes = max(peak_rss_bytes, process_memory.memory_info().rss)
                if assert_cross_window_prefix_in_batch(
                    batch,
                    tokenizer,
                    event_type,
                    event_factory,
                ):
                    cross_window_context_seen = True
                loss = model(**batch).loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("Training produced a missing or non-finite loss.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        1.0,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    global_step += 1
                    epoch_steps += 1
                    if global_step % args.checkpoint_every_steps == 0:
                        checkpoint = checkpoint_root / f"checkpoint-{global_step:08d}"
                        _write_checkpoint(
                            accelerator,
                            model,
                            checkpoint,
                            global_step=global_step,
                            epoch=epoch,
                            args=args,
                            model_spec=model_spec,
                            epoch_complete=False,
                            cross_window_context_seen=cross_window_context_seen,
                            training_seconds=(
                                total_training_seconds
                                + time.perf_counter() - epoch_started
                            ),
                            optimizer_steps=total_optimizer_steps + epoch_steps,
                            peak_rss_bytes=peak_rss_bytes,
                        )
                    if max_steps is not None and global_step >= max_steps:
                        break
        epoch_training_seconds = time.perf_counter() - epoch_started
        total_training_seconds += epoch_training_seconds
        total_optimizer_steps += epoch_steps
        latest_epoch_steps = epoch_steps
        completed_epochs = epoch
        evaluation_started = time.perf_counter()
        validation = validate_nll(
            model,
            val_loader,
            accelerator,
            tokenizer,
            event_type,
            max_batches=args.validation_batches,
        )
        adapter_dir = args.output_dir / "epoch-adapters" / f"epoch-{epoch:04d}"
        adapter_checkpoint = (
            checkpoint_root
            / f"checkpoint-{global_step:08d}"
            / "lora"
            / "adapter_model.safetensors"
        )
        existing_epoch_weights = adapter_dir / "adapter_model.safetensors"
        if (
            not existing_epoch_weights.is_file()
            or not adapter_checkpoint.is_file()
            or _sha256_file(existing_epoch_weights) != _sha256_file(adapter_checkpoint)
        ):
            if adapter_dir.exists():
                adapter_dir = args.output_dir / "epoch-adapters" / (
                    f"epoch-{epoch:04d}-step-{global_step:08d}"
                )
                if adapter_dir.exists():
                    raise FileExistsError(
                        f"Preserving existing epoch adapter at {adapter_dir}"
                    )
            accelerator.unwrap_model(model).save_pretrained(
                adapter_dir,
                safe_serialization=True,
            )
        generated = _free_generation_evaluation(
            args=args,
            records=records,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            adapter_dir=adapter_dir,
            model_cache=model_cache,
            mapperator_root=mapperator_root,
            epoch=epoch,
        )
        latest_eval_seconds = time.perf_counter() - evaluation_started
        estimated_epoch_seconds = 0.0
        if epoch_steps:
            estimated_epoch_seconds = (
                epoch_training_seconds / epoch_steps
                * (max_steps if max_steps else epoch_steps * args.epochs)
            ) + latest_eval_seconds * args.epochs
        validation["free_generation"] = generated
        validation["epoch"] = epoch
        validation["global_step"] = global_step
        validation["epoch_optimizer_steps"] = epoch_steps
        validation["device"] = accelerator.device.type
        validation["epoch_training_seconds"] = epoch_training_seconds
        validation["epoch_evaluation_seconds"] = latest_eval_seconds
        validation["peak_vram_allocated_mib"] = (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if args.device == "cuda"
            else None
        )
        validation["peak_rss_mib"] = round(peak_rss_bytes / 1024**2, 2)
        validation["estimated_total_seconds_at_observed_rate"] = estimated_epoch_seconds
        validation["train_seconds_elapsed"] = total_training_seconds
        validation_history.append(validation)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(validation, ensure_ascii=False) + "\n")
        checkpoint = checkpoint_root / f"checkpoint-{global_step:08d}-epoch-{epoch:04d}"
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        _write_checkpoint(
            accelerator,
            model,
            checkpoint,
            global_step=global_step,
            epoch=epoch,
            args=args,
            model_spec=model_spec,
            epoch_complete=True,
            cross_window_context_seen=cross_window_context_seen,
            training_seconds=total_training_seconds,
            optimizer_steps=total_optimizer_steps,
            peak_rss_bytes=peak_rss_bytes,
            validation=validation,
        )
        final_adapter_dir = _copy_final_adapter(
            adapter_dir,
            args.output_dir / "adapter",
            global_step,
        )
        write_adapter_manifest(
            final_adapter_dir,
            model_spec=model_spec,
            args=args,
            global_step=global_step,
            epoch=epoch,
            validation=validation,
        )
        if max_steps is not None and global_step >= max_steps:
            break
    if final_adapter_dir is None and resume is not None:
        checkpoint_adapter = resume / "lora"
        if not checkpoint_adapter.is_dir():
            raise FileNotFoundError(
                f"Completed checkpoint has no LoRA adapter: {checkpoint_adapter}"
            )
        validation = resume_state.get("validation") if resume_state else None
        final_adapter_dir = _copy_final_adapter(
            checkpoint_adapter,
            args.output_dir / "adapter",
            global_step,
        )
        write_adapter_manifest(
            final_adapter_dir,
            model_spec=model_spec,
            args=args,
            global_step=global_step,
            epoch=int(resume_state.get("epoch", completed_epochs)) if resume_state else completed_epochs,
            validation=validation,
        )
        if validation:
            validation_history = [validation]
            latest_eval_seconds = float(
                validation.get("epoch_evaluation_seconds", 0.0)
            )
        completed_epochs = int(
            resume_state.get("epoch", completed_epochs)
        ) if resume_state else completed_epochs
    if expected_cross_window_holds > 0 and not cross_window_context_seen:
        for batch in train_loader:
            if assert_cross_window_prefix_in_batch(
                batch,
                tokenizer,
                event_type,
                event_factory,
            ):
                cross_window_context_seen = True
                break
    if expected_cross_window_holds > 0 and not cross_window_context_seen:
        raise RuntimeError(
            "The real dataset has cross-window holds, but no encoded training batch "
            "contained carried LN start/sustain context."
        )
    result = {
        "model": args.model,
        "model_revision": model_spec["revision"],
        "source_commit": MAPPERATORINATOR_COMMIT,
        "source_path_sha256": source_digest,
        "dataset": manifest.get("summary", {}),
        "sample_weighting": sample_weight_report,
        "epochs_completed": completed_epochs,
        "global_step": global_step,
        "validation": validation_history,
        "adapter_dir": str(final_adapter_dir),
        "gpu_profile": gpu_profile,
        "training_config": {
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": True,
            "window_width_ms": window_width,
            "cross_window_active_hold_records": expected_cross_window_holds,
            "cross_window_context_seen_in_training": cross_window_context_seen,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "observed_optimizer_step_seconds": (
                total_training_seconds / max(total_optimizer_steps, 1)
            ),
            "estimated_training_seconds_at_observed_rate": (
                total_training_seconds / max(total_optimizer_steps, 1)
                * (max_steps if max_steps else latest_epoch_steps * args.epochs)
            ),
            "observed_epoch_evaluation_seconds": latest_eval_seconds,
            "peak_rss_mib": round(peak_rss_bytes / 1024**2, 2),
        },
        "cache_paths": {name: str(path) for name, path in paths.items()},
    }
    _atomic_write_json(args.output_dir / "training-report.json", result)
    accelerator.end_training()
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = train(args)
    except Exception as error:
        print(f"10K LoRA training failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
