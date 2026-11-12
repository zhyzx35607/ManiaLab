"""Separate only the frozen training split of an expression audio cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEPARATION_SCRIPT = ROOT / "scripts" / "separate_audio.py"
MODEL_SHA256 = "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"
MODEL_URL = "https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt"
SOURCES = ("drums", "bass", "other", "vocals")
INPUT_PROCESSING = {
    "revision": 2,
    "codec": "pcm_f32le",
    "sample_rate": 44100,
    "channels": 2,
    "maximum_seconds": 1201,
    "truncation": False,
}


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_filename(value: str) -> bool:
    path = Path(value)
    return bool(value) and path.name == value and value not in (".", "..")


def load_training_inputs(cohort_root: Path) -> tuple[list[dict], str, str]:
    cohort_root = Path(cohort_root).resolve()
    selection_path = cohort_root / "selection.json"
    report_path = cohort_root / "report.json"
    fetch_report_path = cohort_root / "fetch-report.json"
    selection = read_json(selection_path)
    report = read_json(report_path)
    fetch_report = read_json(fetch_report_path)
    selection_sha256 = canonical_hash(selection)
    if report.get("selection_sha256") != selection_sha256:
        raise ValueError("Frozen selection hash does not match its report")
    if report.get("selected_song_groups") != len(selection):
        raise ValueError("Frozen selection count does not match its report")
    if not isinstance(fetch_report, list):
        raise ValueError("Fetch report must be a list")

    selected_by_id = {}
    group_splits = {}
    for row in selection:
        beatmap_id = str(row.get("beatmap_id", ""))
        group = row.get("group")
        split = row.get("layout_training_split")
        if not beatmap_id.isdigit() or not group or split not in ("train", "validation", "test"):
            raise ValueError("Selection contains an invalid beatmap or split")
        if beatmap_id in selected_by_id:
            raise ValueError(f"Selection contains duplicate beatmap ID {beatmap_id}")
        if group in group_splits:
            raise ValueError(f"Selection contains duplicate song group {group}")
        selected_by_id[beatmap_id] = row
        group_splits[group] = split

    fetch_by_id = {}
    for result in fetch_report:
        reference = result.get("reference") if isinstance(result, dict) else None
        beatmap_id = str((reference or {}).get("beatmap_id", result.get("beatmap_id", "")))
        if beatmap_id in fetch_by_id:
            raise ValueError(f"Fetch report contains duplicate beatmap ID {beatmap_id}")
        fetch_by_id[beatmap_id] = result

    rows = []
    seen_groups = set()
    seen_audio = set()
    for beatmap_id, selected in selected_by_id.items():
        if selected["layout_training_split"] != "train":
            continue
        result = fetch_by_id.get(beatmap_id)
        if not result or result.get("ok") is not True:
            raise ValueError(f"Training audio is not available for beatmap {beatmap_id}")
        fetched = result["reference"]
        if (
            fetched.get("group") != selected["group"]
            or fetched.get("layout_training_split") != "train"
            or int(fetched.get("keys", 0)) != int(selected["keys"])
        ):
            raise ValueError(f"Training split identity mismatch for beatmap {beatmap_id}")

        target = cohort_root / beatmap_id
        reference_path = target / "reference.json"
        chart_path = target / "reference.osu"
        reference = read_json(reference_path)
        audio_name = reference.get("audio_filename", "")
        if not _safe_filename(audio_name):
            raise ValueError(f"Unsafe audio filename for beatmap {beatmap_id}")
        audio_path = target / audio_name
        chart_sha256 = sha256_file(chart_path)
        audio_sha256 = sha256_file(audio_path)
        expected_chart_sha256 = selected.get("sha256")
        if not (
            chart_sha256
            == expected_chart_sha256
            == reference.get("chart_sha256")
            == fetched.get("chart_sha256")
        ):
            raise ValueError(f"Training chart hash mismatch for beatmap {beatmap_id}")
        if not (
            audio_sha256
            == reference.get("audio_sha256")
            == fetched.get("audio_sha256")
        ):
            raise ValueError(f"Training audio hash mismatch for beatmap {beatmap_id}")
        if (
            reference.get("beatmap_id") != int(beatmap_id)
            or reference.get("group") != selected["group"]
            or reference.get("layout_training_split") != "train"
        ):
            raise ValueError(f"Local reference identity mismatch for beatmap {beatmap_id}")
        if selected["group"] in seen_groups or audio_sha256 in seen_audio:
            raise ValueError("Training inputs must have unique song groups and audio hashes")
        seen_groups.add(selected["group"])
        seen_audio.add(audio_sha256)
        rows.append({
            "beatmap_id": int(beatmap_id),
            "group": selected["group"],
            "keys": int(selected["keys"]),
            "ln_class": selected["ln_class"],
            "split": "train",
            "audio_filename": audio_name,
            "audio_sha256": audio_sha256,
            "chart_sha256": chart_sha256,
            "audio_path": audio_path,
        })

    if not rows:
        raise ValueError("Frozen cohort has no verified training audio")
    return rows, selection_sha256, sha256_file(fetch_report_path)


def record_identity(row: dict) -> dict:
    return {key: row[key] for key in (
        "beatmap_id", "group", "keys", "ln_class", "split",
        "audio_filename", "audio_sha256", "chart_sha256",
    )}


def read_float_stereo_wav_frames(path: Path) -> int:
    fmt = None
    data_size = None
    with Path(path).open("rb") as stream:
        if stream.read(4) != b"RIFF":
            raise ValueError("Decoded input is not a RIFF WAV")
        stream.read(4)
        if stream.read(4) != b"WAVE":
            raise ValueError("Decoded input is not a WAVE file")
        while True:
            chunk_header = stream.read(8)
            if not chunk_header:
                break
            if len(chunk_header) != 8:
                raise ValueError("Truncated WAV chunk header")
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            chunk = stream.read(chunk_size)
            if len(chunk) != chunk_size:
                raise ValueError("Truncated WAV chunk")
            if chunk_id == b"fmt ":
                if len(chunk) < 16:
                    raise ValueError("WAV format chunk is too short")
                fmt = struct.unpack("<HHIIHH", chunk[:16])
            elif chunk_id == b"data":
                data_size = chunk_size
            if chunk_size & 1:
                stream.read(1)
    if fmt is None or data_size is None:
        raise ValueError("Decoded WAV is missing a format or data chunk")
    audio_format, channels, sample_rate, _, _, bits_per_sample = fmt
    if (audio_format, channels, sample_rate, bits_per_sample) != (3, 2, 44100, 32):
        raise ValueError("Expected stereo 44100 Hz pcm_f32le input")
    bytes_per_frame = channels * bits_per_sample // 8
    if data_size % bytes_per_frame:
        raise ValueError("WAV data size is not aligned to complete frames")
    frames = data_size // bytes_per_frame
    if frames <= 0 or frames / sample_rate > INPUT_PROCESSING["maximum_seconds"]:
        raise ValueError("Audio is empty or exceeds the non-truncating duration limit")
    return frames


def validate_worker_result(directory: Path, model_sha256: str, input_frames: int) -> dict:
    directory = Path(directory)
    report_path = directory / "separation.json"
    report = read_json(report_path)
    expected_frames = (input_frames + 1) // 2
    if report.get("model_sha256") != model_sha256:
        raise ValueError("Worker used a different separation model")
    if report.get("sample_rate") != 22050 or report.get("frames") != expected_frames:
        raise ValueError("Worker output sample rate or frame count is inconsistent")
    sources = report.get("sources")
    expected_sources = {name: f"{name}.npy" for name in SOURCES}
    if sources != expected_sources:
        raise ValueError("Worker output does not contain the expected four stems")
    reconstruction_rmse = report.get("reconstruction_rmse")
    if (
        not isinstance(reconstruction_rmse, (int, float))
        or not math.isfinite(reconstruction_rmse)
        or reconstruction_rmse < 0
    ):
        raise ValueError("Worker reconstruction diagnostic is invalid")

    stems = {}
    for name in SOURCES:
        path = directory / expected_sources[name]
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (expected_frames,):
            raise ValueError(f"Unexpected shape or dtype for {name} stem")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite samples in {name} stem")
        stems[name] = {
            "filename": path.name,
            "sha256": sha256_file(path),
            "frames": int(values.shape[0]),
            "dtype": str(values.dtype),
        }
    return {
        "separation_report_sha256": sha256_file(report_path),
        "frames": expected_frames,
        "sample_rate": 22050,
        "reconstruction_rmse": reconstruction_rmse,
        "stems": stems,
    }


def verify_completed(
    directory: Path,
    row: dict,
    model_sha256: str,
    run_sha256: str,
) -> dict:
    complete = read_json(Path(directory) / "complete.json")
    if (
        complete.get("input") != record_identity(row)
        or complete.get("model_sha256") != model_sha256
        or complete.get("run_identity_sha256") != run_sha256
    ):
        raise ValueError(f"Completed output identity mismatch for {row['beatmap_id']}")
    result = validate_worker_result(
        directory, model_sha256, int(complete.get("input_frames_44100", 0))
    )
    if complete.get("validation") != result:
        raise ValueError(f"Completed output verification mismatch for {row['beatmap_id']}")
    return complete


def _worker_environment(python: Path) -> dict:
    code = (
        "import json,sys,torch,torchaudio;"
        "torch.set_num_threads(2);torch.set_num_interop_threads(1);"
        "print(json.dumps({'python':sys.version.split()[0],"
        "'torch':torch.__version__,'torchaudio':torchaudio.__version__,"
        "'cuda_available':torch.cuda.is_available(),"
        "'threads':torch.get_num_threads(),"
        "'interop_threads':torch.get_num_interop_threads()}))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Audio worker environment is unavailable: {result.stderr.strip()}")
    return json.loads(result.stdout.strip())


def _ffmpeg_version(ffmpeg: Path) -> str:
    result = subprocess.run(
        [str(ffmpeg), "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode or not result.stdout:
        raise ValueError("Unable to identify the configured FFmpeg binary")
    return result.stdout.splitlines()[0].strip()


def build_run_identity(
    rows: list[dict],
    selection_sha256: str,
    fetch_report_sha256: str,
    model_sha256: str,
    worker_environment: dict,
    ffmpeg_version: str,
) -> dict:
    return {
        "schema": "manialab-expression-audio-stems-v1",
        "split": "train",
        "selection_sha256": selection_sha256,
        "fetch_report_sha256": fetch_report_sha256,
        "model_url": MODEL_URL,
        "model_sha256": model_sha256,
        "profile": "four",
        "sources": list(SOURCES),
        "input_processing": INPUT_PROCESSING,
        "threads": 2,
        "interop_threads": 1,
        "worker_environment": worker_environment,
        "ffmpeg_version": ffmpeg_version,
        "batch_script_sha256": sha256_file(Path(__file__)),
        "separation_script_sha256": sha256_file(SEPARATION_SCRIPT),
        "records": [record_identity(row) for row in rows],
    }


def _prepare_output(output: Path, run_identity: dict) -> Path:
    test_data = (ROOT / "test-data").resolve()
    output = Path(output).resolve()
    if output == test_data or not output.is_relative_to(test_data):
        raise ValueError("Output must be a new ignored directory under test-data")
    run_path = output / "run.json"
    if output.exists():
        if not output.is_dir() or not run_path.is_file():
            raise FileExistsError("Output exists without a matching run manifest; refusing to reuse it")
        if read_json(run_path) != run_identity:
            raise ValueError("Existing output belongs to a different frozen run")
    else:
        output.mkdir(parents=True)
        write_json_atomic(run_path, run_identity)
    return output


def _decode_audio(ffmpeg: Path, source: Path, destination: Path) -> int:
    command = [
        str(ffmpeg), "-v", "error", "-nostdin", "-y", "-i", str(source),
        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", str(destination),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        (destination.parent / "decode.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        raise ValueError(f"FFmpeg could not decode {source.name}")
    return read_float_stereo_wav_frames(destination)


def _run_worker(
    python: Path,
    stage: Path,
    wav_path: Path,
    model: Path,
    item_number: int,
    item_count: int,
) -> None:
    command = [
        str(python), str(SEPARATION_SCRIPT), str(wav_path), str(stage),
        "--model", str(model), "--threads", "2",
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    log_path = stage / "separation.log"
    last_progress = -10
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                percent = int(event.get("progress", last_progress))
                if percent >= last_progress + 10 or percent == 100:
                    print(
                        f"[{item_number}/{item_count}] "
                        f"{stage.name}: {percent}% {event.get('message', '')}",
                        flush=True,
                    )
                    last_progress = percent
            return_code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise
    if return_code:
        raise ValueError(f"Audio worker failed for {stage.name}; partial stage preserved")


def run(
    cohort_root: Path,
    output: Path,
    model: Path,
    worker_python: Path,
    ffmpeg: Path,
    *,
    plan_only: bool = False,
) -> dict:
    rows, selection_sha256, fetch_report_sha256 = load_training_inputs(cohort_root)
    model = Path(model).resolve()
    worker_python = Path(worker_python).resolve()
    ffmpeg = Path(ffmpeg).resolve()
    if not model.is_file() or sha256_file(model) != MODEL_SHA256:
        raise ValueError("Hybrid Demucs checkpoint is missing or has the wrong SHA-256")
    if not worker_python.is_file() or not ffmpeg.is_file():
        raise ValueError("Audio worker Python or FFmpeg executable is missing")
    worker_environment = _worker_environment(worker_python)
    ffmpeg_version = _ffmpeg_version(ffmpeg)
    identity = build_run_identity(
        rows, selection_sha256, fetch_report_sha256, MODEL_SHA256,
        worker_environment, ffmpeg_version,
    )
    run_sha256 = canonical_hash(identity)
    if plan_only:
        return {
            "run_identity_sha256": run_sha256,
            "selected_training_groups": len(rows),
            "keys": {
                str(key): sum(row["keys"] == key for row in rows)
                for key in sorted({row["keys"] for row in rows})
            },
            "worker_environment": worker_environment,
            "ffmpeg_version": ffmpeg_version,
            "model_sha256": MODEL_SHA256,
            "output": str(Path(output).resolve()),
            "inference_started": False,
        }

    output = _prepare_output(output, identity)
    staging_root = output / ".staging"
    staging_root.mkdir(exist_ok=True)
    completed = []
    for index, row in enumerate(rows, start=1):
        target = output / str(row["beatmap_id"])
        if target.exists():
            complete = verify_completed(target, row, MODEL_SHA256, run_sha256)
            completed.append(complete)
            print(f"[{index}/{len(rows)}] verified existing {row['beatmap_id']}", flush=True)
            continue

        stage = staging_root / f"{row['beatmap_id']}-{uuid.uuid4().hex}"
        stage.mkdir()
        wav_path = stage / "input.wav"
        print(f"[{index}/{len(rows)}] decoding {row['beatmap_id']}", flush=True)
        input_frames = _decode_audio(ffmpeg, row["audio_path"], wav_path)
        _run_worker(worker_python, stage, wav_path, model, index, len(rows))
        validation = validate_worker_result(stage, MODEL_SHA256, input_frames)
        wav_path.unlink()
        complete = {
            "schema": "manialab-expression-audio-stem-record-v1",
            "run_identity_sha256": run_sha256,
            "input": record_identity(row),
            "model_sha256": MODEL_SHA256,
            "input_frames_44100": input_frames,
            "input_seconds": input_frames / 44100,
            "validation": validation,
        }
        write_json_atomic(stage / "complete.json", complete)
        if target.exists():
            raise FileExistsError(f"Output appeared while processing {row['beatmap_id']}")
        stage.rename(target)
        complete = verify_completed(target, row, MODEL_SHA256, run_sha256)
        completed.append(complete)
        write_json_atomic(output / "summary.json", {
            "schema": "manialab-expression-audio-stems-summary-v1",
            "run_identity_sha256": run_sha256,
            "expected_training_groups": len(rows),
            "completed_training_groups": len(completed),
            "completed_beatmap_ids": [
                item["input"]["beatmap_id"] for item in completed
            ],
        })

    summary = {
        "schema": "manialab-expression-audio-stems-summary-v1",
        "run_identity_sha256": run_sha256,
        "expected_training_groups": len(rows),
        "completed_training_groups": len(completed),
        "completed_beatmap_ids": [item["input"]["beatmap_id"] for item in completed],
        "inference_started": True,
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("test-data/expression-audio-cohort-v3"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test-data/expression-audio-cohort-v3-separated-train-v1"),
    )
    parser.add_argument("--model", type=Path, default=Path("assets/deep/hdemucs_high_trained.pt"))
    parser.add_argument("--worker-python", type=Path, default=Path(".venv-audio/Scripts/python.exe"))
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.ffmpeg is None:
        from imageio_ffmpeg import get_ffmpeg_exe
        args.ffmpeg = Path(get_ffmpeg_exe())
    result = run(
        args.cohort,
        args.output,
        args.model,
        args.worker_python,
        args.ffmpeg,
        plan_only=args.plan_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
