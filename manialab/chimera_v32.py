"""CPU-first Mapperatorinator v32-mini comparison pipeline.

All model caches and experiment artifacts stay outside tracked source. The
Mapperatorinator checkout is treated as a read-only inference dependency.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
from scipy.io import wavfile

from .domain import export_osu, note_from, parse_osu, sections, validate


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "workspace_exp" / "ab_v32"
MAPPER_REPO = Path(os.environ.get("MANIALAB_MAPPER_REPO", r"D:\project\Mapperatorinator"))
MAPPER_PYTHON = Path(os.environ.get(
    "MANIALAB_MAPPER_PYTHON",
    r"D:\.venvs\mapperatorinator-v32mini\Scripts\python.exe",
))
MODEL_REPO = "OliBomby/Mapperatorinator-v32-mini"
MODEL_REVISION = "7807f0dc70cab671be012e1f5ddf945b0b8b7278"
MODEL_CACHE = Path(os.environ.get(
    "MANIALAB_V32_MODEL_DIR",
    rf"D:\.cache\huggingface\manialab\Mapperatorinator-v32-mini-{MODEL_REVISION[:7]}",
))
_CHART_LABELS = (
    "[1-Old-BasicPitch-MILP]",
    "[2-V32Mini-Native10K-FullMix]",
    "[3-V32Mini-7K-TorosuDense10K-FullMix]",
    "[4-V32Mini-7K-TorosuDense10K-NoVocal]",
)
_NO_VOCAL_GAINS = {
    "vocals": 0.0,
    "drums": 1.1,
    "piano": 1.2,
    "bass": 1.0,
    "other": 1.0,
    # HTDemucs six-source separates guitar from "other"; retain it as part of
    # the accompaniment rather than silently dropping that instrument.
    "guitar": 1.0,
}


def configure_d_cache_environment(
    env: dict[str, str] | None = None,
    *,
    cache_root: str | Path = r"D:\.cache",
    temp_root: str | Path = r"D:\.tmp",
) -> dict[str, str]:
    """Return an environment that pins every supported model/temp cache to D:."""
    result = dict(os.environ if env is None else env)
    cache = Path(cache_root)
    temp = Path(temp_root)
    if os.name == "nt" and (cache.drive.upper() != "D:" or temp.drive.upper() != "D:"):
        raise ValueError("Mapperatorinator、PyTorch 与临时缓存必须全部位于 D: 盘")
    hf_home = cache / "huggingface"
    paths = {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_home / "hub",
        "HUGGINGFACE_HUB_CACHE": hf_home / "hub",
        "TRANSFORMERS_CACHE": hf_home / "transformers",
        "HF_ASSETS_CACHE": hf_home / "assets",
        "TORCH_HOME": cache / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torch" / "inductor",
        "XDG_CACHE_HOME": cache / "xdg",
        "UV_CACHE_DIR": cache / "uv",
        "PIP_CACHE_DIR": cache / "pip",
        "TMP": temp,
        "TEMP": temp,
        "TMPDIR": temp,
    }
    for key, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        result[key] = str(value)
    return result


def configure_separator_runtime() -> Path:
    """Use the current D-drive Python worker when the bundled EXE is stale."""
    env = configure_d_cache_environment()
    os.environ.update(env)
    configured = os.environ.get("MANIALAB_DEEP_PYTHON")
    python = Path(configured) if configured else ROOT / ".venv-audio" / "Scripts" / "python.exe"
    if not python.is_file():
        raise FileNotFoundError(f"HTDemucs Python worker 不存在：{python}")
    if os.name == "nt" and python.drive.upper() != "D:":
        raise ValueError("HTDemucs Python worker 必须位于 D: 盘")
    os.environ["MANIALAB_DEEP_PYTHON"] = str(python)
    return python


def ensure_v32_model_snapshot(
    *,
    model_dir: str | Path = MODEL_CACHE,
    revision: str = MODEL_REVISION,
    download: bool = True,
) -> Path:
    """Fetch only the pinned mania checkpoint into the D-drive local model dir."""
    if revision != MODEL_REVISION:
        raise ValueError("仅允许使用本工单固定的 v32-mini revision")
    root = Path(model_dir)
    model = root / "gamemode=3"
    required = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json")
    if all((model / name).is_file() for name in required):
        manifest_path = root / "model-revision.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("revision") != revision or manifest.get("repo") != MODEL_REPO:
                raise ValueError("D 盘模型缓存清单与固定 v32-mini revision 不一致")
            for name, expected in manifest.get("sha256", {}).items():
                if _sha256_file(model / name) != expected:
                    raise ValueError(f"固定 revision 模型文件校验失败：{name}")
            return model
    if not download:
        raise FileNotFoundError(f"缺少固定 revision 的 v32-mini 权重：{model}")
    if os.name == "nt" and root.drive.upper() != "D:":
        raise ValueError("v32-mini 权重目录必须位于 D: 盘")

    env = configure_d_cache_environment()
    os.environ.update(env)
    loader = (
        "import hashlib,json,sys\n"
        "from pathlib import Path\n"
        "from huggingface_hub import HfApi,snapshot_download\n"
        "repo,revision,root,cache=sys.argv[1:]\n"
        "root=Path(root); model=root/'gamemode=3'\n"
        "info=HfApi().model_info(repo,revision=revision)\n"
        "if info.sha != revision: raise RuntimeError(f'revision mismatch: {info.sha}')\n"
        "snapshot_download(repo_id=repo,revision=revision,local_dir=str(root),"
        "cache_dir=cache,allow_patterns=('gamemode=3/*',))\n"
        "required=('config.json','generation_config.json','model.safetensors','tokenizer.json')\n"
        "if not all((model/name).is_file() for name in required): "
        "raise FileNotFoundError('missing v32-mini gamemode=3 files')\n"
        "def sha(path):\n"
        " h=hashlib.sha256()\n"
        " with path.open('rb') as f:\n"
        "  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)\n"
        " return h.hexdigest()\n"
        "(root/'model-revision.json').write_text(json.dumps("
        "{'repo':repo,'revision':revision,'sha256':{n:sha(model/n) for n in required}},"
        "indent=2),encoding='utf-8')\n"
    )
    result = subprocess.run(
        [str(MAPPER_PYTHON), "-c", loader, MODEL_REPO, revision, str(root), env["HF_HOME"]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=1800,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(
            "无法验证/下载固定 v32-mini checkpoint："
            + (result.stderr or result.stdout).strip()[-3000:]
        )
    if not all((model / name).is_file() for name in required):
        raise FileNotFoundError("固定 revision 下载后缺少 v32-mini gamemode=3 文件")
    manifest = json.loads((root / "model-revision.json").read_text(encoding="utf-8"))
    if manifest.get("repo") != MODEL_REPO or manifest.get("revision") != revision:
        raise ValueError("模型 D 盘 revision manifest 无效")
    return model


def _sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_identity(
    *,
    audio_path: str | Path,
    reference_chart: str | Path,
    keycount: int,
    bpm: float,
    offset_ms: float,
    hold_note_ratio: float,
    threads: int,
    precision: str,
    model_dir: str | Path,
    start_ms: int,
    end_ms: int,
) -> dict:
    """Bind a generated map to every input that can change its events."""
    return {
        "audio_sha256": _sha256_file(audio_path),
        "reference_chart_sha256": _sha256_file(reference_chart),
        "keycount": keycount,
        "bpm": float(bpm),
        "offset_ms": float(offset_ms),
        "hold_note_ratio": float(hold_note_ratio),
        "threads": threads,
        "precision": precision,
        "model_revision": MODEL_REVISION,
        "model_directory": str(Path(model_dir).resolve()),
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


def validate_reusable_inference_report(report: dict, expected: dict) -> None:
    actual = report.get("input_identity")
    if not isinstance(actual, dict):
        raise ValueError("续跑报告缺少完整输入身份；拒绝复用旧模型谱")
    if actual != expected:
        changed = sorted(
            key for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise ValueError("续跑模型输入与当前请求不一致：" + ", ".join(changed))
    if report.get("model_revision") != MODEL_REVISION:
        raise ValueError("续跑模型 revision 不匹配")


def _start_rss_monitor(target_pid: int, stop_path: Path, env: dict[str, str]) -> subprocess.Popen:
    monitor_source = (
        "import pathlib, psutil, sys, time\n"
        "pid, stop = int(sys.argv[1]), pathlib.Path(sys.argv[2])\n"
        "peak = 0\n"
        "while True:\n"
        "    try:\n"
        "        root = psutil.Process(pid)\n"
        "        nodes = [root, *root.children(recursive=True)]\n"
        "        current = 0\n"
        "        for node in nodes:\n"
        "            try: current += node.memory_info().rss\n"
        "            except psutil.Error: pass\n"
        "        peak = max(peak, current)\n"
        "    except psutil.Error: pass\n"
        "    if stop.exists(): break\n"
        "    time.sleep(0.1)\n"
        "print(peak)\n"
    )
    monitor_python = MAPPER_PYTHON if MAPPER_PYTHON.is_file() else Path(sys.executable)
    stop_path.unlink(missing_ok=True)
    return subprocess.Popen(
        [str(monitor_python), "-c", monitor_source, str(target_pid), str(stop_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _run_measured(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: float = 1200,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = None
    peak_rss = 0
    stop_path = log_path.with_suffix(".rss-stop")
    monitor = None
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            monitor = _start_rss_monitor(process.pid, stop_path, env)
            while process.poll() is None:
                if time.perf_counter() - started > timeout_seconds:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TimeoutError(f"Mapperatorinator 超过 {timeout_seconds:g} 秒")
                time.sleep(0.2)
            process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if monitor is not None:
                stop_path.touch()
                monitor_stdout, monitor_stderr = monitor.communicate(timeout=15)
                if monitor.returncode and monitor_stderr.strip():
                    raise RuntimeError("D 盘内存监控失败：" + monitor_stderr.strip()[-1000:])
                try:
                    peak_rss = int(monitor_stdout.strip() or "0")
                except ValueError:
                    peak_rss = 0
                stop_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    if process.returncode:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(
            f"Mapperatorinator 退出码 {process.returncode}；日志 {log_path}\n{detail}"
        )
    return {
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "peak_rss_mib": peak_rss / 2**20,
        "log": str(log_path),
    }


def build_mapper_command(
    *,
    audio_path: str | Path,
    reference_chart: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    keycount: int,
    duration_ms: int,
    bpm: float,
    offset_ms: float,
    hold_note_ratio: float,
    threads: int = 6,
    precision: str = "fp32",
    start_ms: int = 0,
    end_ms: int | None = None,
) -> list[str]:
    """Build the upstream Hydra command without importing its global runtime."""
    if keycount not in (7, 10):
        raise ValueError("v32-mini 比较仅支持 7K 与 10K")
    if type(duration_ms) is not int or duration_ms < 1000:
        raise ValueError("推理音频时长必须至少为 1 秒")
    if end_ms is None:
        end_ms = duration_ms
    if type(start_ms) is not int or type(end_ms) is not int or not 0 <= start_ms < end_ms <= duration_ms:
        raise ValueError("Mapperatorinator 推理窗口必须位于音频时长内")
    if type(threads) is not int or not 1 <= threads <= 6:
        raise ValueError("CPU 线程数必须为 1–6")
    if precision not in ("fp32", "bf16"):
        raise ValueError("精度仅支持 fp32 或 bf16")
    if not math.isfinite(hold_note_ratio) or not 0 <= hold_note_ratio <= 1:
        raise ValueError("hold_note_ratio 必须在 0–1 之间")

    def hydra_path(value: str | Path) -> str:
        return json.dumps(Path(value).resolve().as_posix())

    return [
        str(MAPPER_PYTHON),
        str(MAPPER_REPO / "inference.py"),
        "--config-name",
        "v32-mini",
        f"model_path={hydra_path(model_dir)}",
        "auto_select_gamemode_model=false",
        f"audio_path={hydra_path(audio_path)}",
        f"beatmap_path={hydra_path(reference_chart)}",
        f"output_path={hydra_path(output_dir)}",
        "device=cpu",
        f"precision={precision}",
        "fast_decoder_loop=true",
        "use_server=false",
        "gamemode=3",
        "max_batch_size=2",
        f"keycount={keycount}",
        f"hold_note_ratio={hold_note_ratio:.4f}",
        # Upstream InferenceConfig types these metadata fields as integers.
        # Exact fractional timing is carried by the reference chart above.
        f"bpm={int(round(bpm))}",
        f"offset={int(round(offset_ms))}",
        f"start_time={start_ms}",
        f"end_time={end_ms}",
        "in_context=[TIMING]",
        "output_type=[MAP,SV]",
        "generate_positions=false",
        "seed=20261126",
        "hitsounded=false",
    ]


def run_mapperatorinator(
    *,
    audio_path: str | Path,
    reference_chart: str | Path,
    output_dir: str | Path,
    keycount: int,
    bpm: float,
    offset_ms: float,
    hold_note_ratio: float = 0.426,
    threads: int = 6,
    precision: str = "fp32",
    model_dir: str | Path | None = None,
    timeout_seconds: float = 1200,
    start_ms: int = 0,
    end_ms: int | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(model_dir) if model_dir else ensure_v32_model_snapshot()
    command = build_mapper_command(
        audio_path=audio_path,
        reference_chart=reference_chart,
        output_dir=output,
        model_dir=checkpoint,
        keycount=keycount,
        duration_ms=_audio_duration_ms(audio_path),
        bpm=bpm,
        offset_ms=offset_ms,
        hold_note_ratio=hold_note_ratio,
        threads=threads,
        precision=precision,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    actual_end_ms = end_ms if end_ms is not None else _audio_duration_ms(audio_path)
    input_identity = inference_identity(
        audio_path=audio_path,
        reference_chart=reference_chart,
        keycount=keycount,
        bpm=bpm,
        offset_ms=offset_ms,
        hold_note_ratio=hold_note_ratio,
        threads=threads,
        precision=precision,
        model_dir=checkpoint,
        start_ms=start_ms,
        end_ms=actual_end_ms,
    )
    env = configure_d_cache_environment()
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    measured = _run_measured(
        command,
        cwd=MAPPER_REPO,
        env=env,
        log_path=output / "inference.log",
        timeout_seconds=timeout_seconds,
    )
    maps = sorted(output.glob("*.osu"))
    if len(maps) != 1:
        raise RuntimeError(f"Mapperatorinator 应生成 1 张谱面，实际找到 {len(maps)} 张")
    text = maps[0].read_text(encoding="utf-8-sig")
    text, sanitation = sanitize_zero_length_model_holds(text)
    maps[0].write_text(text, encoding="utf-8", newline="\n")
    parsed = parse_osu(text)
    structure = validate(parsed["notes"], keycount)
    if parsed["keys"] != keycount:
        raise ValueError("Mapperatorinator 输出谱面键数错误")
    log = (output / "inference.log").read_text(encoding="utf-8", errors="replace")
    measured.update({
        "map": str(maps[0]),
        "keycount": keycount,
        "objects": len(parsed["notes"]),
        "ln": sum(note.hold for note in parsed["notes"]),
        "structure_valid_before_layout": structure["valid"],
        "structure_issues_before_layout": structure["issues"],
        "invalid_lns_downgraded_to_taps": sanitation["downgraded"],
        "sanitation": sanitation,
        "audio_duration_ms": _audio_duration_ms(audio_path),
        "start_ms": start_ms,
        "end_ms": end_ms if end_ms is not None else _audio_duration_ms(audio_path),
        "bpm": bpm,
        "offset_ms": offset_ms,
        "model_revision": MODEL_REVISION,
        "input_identity": input_identity,
        "fast_decoder_loop_requested": True,
        "fast_decoder_loop_active": "disabling on 'cpu'" not in log,
        "precision": precision,
        "threads": threads,
    })
    (output / "run-report.json").write_text(
        json.dumps(measured, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return measured


def sanitize_zero_length_model_holds(osu_text: str) -> tuple[str, dict]:
    """Keep an invalid v32 hold's attack as a tap instead of dropping it."""
    content = sections(osu_text)
    lines = content.get("HitObjects")
    if lines is None:
        raise ValueError("Mapperatorinator 输出缺少 HitObjects")
    downgraded = []
    repaired = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split(",")
        if len(fields) < 6:
            repaired.append(line)
            continue
        try:
            kind = int(fields[3])
            start = int(fields[2])
            end = int(fields[5].split(":", 1)[0])
        except ValueError:
            repaired.append(line)
            continue
        if not kind & 128 or end > start:
            repaired.append(line)
            continue
        # The model emitted a HOLD_NOTE without positive duration. Preserve its
        # onset and sample settings as a tap; never invent a release timestamp.
        fields[3] = str((kind & ~128) | 1)
        tail = fields[5].split(":")
        sample = ":".join(tail[1:]) if len(tail) > 1 else "0:0:0:0:"
        fields[5] = sample or "0:0:0:0:"
        repaired.append(",".join(fields))
        downgraded.append({"line": line_number, "start_ms": start, "invalid_end_ms": end})
    content["HitObjects"] = repaired
    result = _format_sections(content)
    parse_osu(result)
    return result, {
        "downgraded": len(downgraded),
        "preserved_attacks": len(downgraded),
        "objects_before": len(lines),
        "objects_after": len(repaired),
        "details": downgraded,
    }


def benchmark_full_song_seconds(report: dict, source_duration_seconds: float) -> float:
    """Estimate full-track work from a timed short inference, excluding reloads."""
    window = max(0.001, (float(report["end_ms"]) - float(report["start_ms"])) / 1000)
    generation_seconds = float(report.get("generate_s", report.get("inference_seconds", 0)))
    if generation_seconds <= 0:
        generation_seconds = float(report["elapsed_s"])
    fixed = float(report.get("setup_s", 0)) + float(report.get("load_s", 0))
    return fixed + generation_seconds * float(source_duration_seconds) / window


def _read_benchmark_report(path: str | Path, source_duration_s: float) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("model_revision") not in (None, MODEL_REVISION):
        raise ValueError("30 秒基准报告的模型 revision 与固定 v32-mini 不一致")
    clip_seconds = (float(report["end_ms"]) - float(report["start_ms"])) / 1000
    if clip_seconds < 30 or float(report["source_duration_s"]) <= 0:
        raise ValueError("基准报告必须包含至少 30 秒真实音频计时")
    if not math.isclose(float(report["source_duration_s"]), source_duration_s, rel_tol=0.01):
        raise ValueError("基准报告歌曲时长与当前输入不匹配")
    report["clip_duration_s"] = clip_seconds
    report["projected_full_song_seconds"] = benchmark_full_song_seconds(
        report, source_duration_s,
    )
    return report


def _find_separation_preflight(audio_path: Path) -> dict | None:
    candidates = sorted(
        EXPERIMENTS.glob("separation_*preflight*/report.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if Path(report.get("audio", "")).resolve() == audio_path.resolve():
            return report
    return None


def _audio_duration_ms(path: str | Path) -> int:
    source = Path(path)
    if source.suffix.casefold() == ".wav":
        rate, data = wavfile.read(source, mmap=True)
        return round(len(data) * 1000 / rate)
    import imageio_ffmpeg

    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-v", "info", "-i", str(source),
         "-f", "null", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    duration = None
    for line in (result.stderr or "").splitlines():
        if "Duration:" in line:
            token = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
            hours, minutes, seconds = token.split(":")
            duration = (int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000
            break
    if result.returncode or duration is None:
        raise ValueError(f"无法读取音频时长：{source}")
    return round(duration)


def _decode_for_energy(path: str | Path, work_dir: Path) -> tuple[np.ndarray, int]:
    from .audio_io import decode
    from .audio_config import SR

    work_dir.mkdir(parents=True, exist_ok=True)
    samples = decode(path, work_dir)
    # audio_io.decode returns normalized float32 samples from FFmpeg f32le.
    return samples.astype(np.float32), SR


def choose_climax_window(
    samples: np.ndarray,
    sample_rate: int,
    duration_seconds: float = 45,
    *,
    step_seconds: float = 1,
) -> tuple[int, float]:
    """Return the loudest fixed-length excerpt start (ms) by rolling RMS."""
    mono = np.asarray(samples, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if mono.ndim != 1 or sample_rate <= 0 or not np.isfinite(mono).all():
        raise ValueError("音频能量输入无效")
    window = min(len(mono), max(1, round(duration_seconds * sample_rate)))
    step = max(1, round(step_seconds * sample_rate))
    if window == len(mono):
        return 0, float(np.sqrt(np.mean(np.square(mono))))
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(mono), dtype=np.float64)))
    starts = np.arange(0, len(mono) - window + 1, step, dtype=np.int64)
    if starts[-1] != len(mono) - window:
        starts = np.append(starts, len(mono) - window)
    energies = (cumulative[starts + window] - cumulative[starts]) / window
    index = int(np.argmax(energies))
    return round(starts[index] * 1000 / sample_rate), float(np.sqrt(energies[index]))


def extract_audio_excerpt(
    source: str | Path,
    output: str | Path,
    *,
    start_ms: int,
    duration_ms: int,
) -> Path:
    """Write an accurate-seek stereo PCM16 excerpt without touching the source."""
    import imageio_ffmpeg

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-y",
         "-i", str(source), "-ss", f"{start_ms / 1000:.3f}",
         "-t", f"{duration_ms / 1000:.3f}", "-vn", "-ac", "2", "-ar", "44100",
         "-c:a", "pcm_s16le", str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if result.returncode:
        raise ValueError("无法从真实音频提取高潮片段：" + (result.stderr or "").strip()[-1000:])
    rate, data = wavfile.read(target, mmap=True)
    if rate != 44100 or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("高潮片段必须为 44.1 kHz stereo WAV")
    if abs(len(data) * 1000 / rate - duration_ms) > 100:
        raise ValueError("高潮片段时长与请求不一致")
    return target


def remix_no_vocal(
    stem_dir: str | Path,
    output_wav: str | Path,
    *,
    gains: dict[str, float] | None = None,
) -> dict:
    """Mix the separated accompaniment stems using the phase's default gains."""
    weights = dict(_NO_VOCAL_GAINS if gains is None else gains)
    if weights.get("vocals") != 0:
        raise ValueError("no_vocal 重混必须将 vocals 增益设为 0")
    if any(not math.isfinite(float(gain)) or not 0 <= gain <= 2 for gain in weights.values()):
        raise ValueError("声部增益必须在 0–2 之间")
    root = Path(stem_dir)
    missing = [voice for voice, gain in weights.items() if gain and not (root / f"{voice}.npy").is_file()]
    if missing:
        raise FileNotFoundError("缺少重混声部：" + ", ".join(missing))
    loaded = {
        voice: np.load(root / f"{voice}.npy", allow_pickle=False).astype(np.float32)
        for voice, gain in weights.items()
        if gain
    }
    if not loaded or any(samples.ndim != 1 for samples in loaded.values()):
        raise ValueError("分轨必须是非空的一维单声道波形")
    length = min(map(len, loaded.values()))
    if length <= 0:
        raise ValueError("分轨没有有效采样")
    mixed = np.zeros(length, dtype=np.float32)
    for voice, samples in loaded.items():
        mixed += samples[:length] * float(weights[voice])
    if not np.isfinite(mixed).all():
        raise ValueError("重混波形包含非有限值")
    peak = float(np.max(np.abs(mixed)))
    scale = min(1.0, 0.98 / peak) if peak > 0 else 1.0
    mixed *= scale
    stereo = np.column_stack((mixed, mixed))
    output = Path(output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output, 22050, np.round(stereo * 32767).astype(np.int16))
    return {
        "path": str(output),
        "sample_rate": 22050,
        "samples": length,
        "duration_seconds": length / 22050,
        "gains": weights,
        "peak_before_normalization": peak,
        "normalization_gain": scale,
    }


def benchmark_open_source_separation(
    audio_path: str | Path,
    *,
    work_dir: str | Path,
    start_ms: int = 51181,
    duration_ms: int = 30000,
) -> dict:
    """Time the six-source CPU separator on a bounded real-song clip."""
    from .separation import Separator

    if type(duration_ms) is not int or not 15000 <= duration_ms <= 30000:
        raise ValueError("分轨预检必须限定在 15–30 秒真实音频")
    configure_separator_runtime()
    source = Path(audio_path).resolve()
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    clip = extract_audio_excerpt(
        source,
        root / "real-song-separation-bench.wav",
        start_ms=start_ms,
        duration_ms=duration_ms,
    )
    separator = Separator(ROOT, profile="six")
    if not separator.status()["available"]:
        raise RuntimeError("HTDemucs 六声部分离模型或 CPU worker 不可用")
    separation_work = root / "separation"
    separation_work.mkdir(parents=True, exist_ok=True)
    (stem_dir, worker_report), measurement = _measured_call(lambda: separator.separate(
        clip,
        separation_work,
        progress=lambda *_: None,
        cancelled=lambda: False,
        include_beats=False,
    ))
    source_samples, source_rate = _decode_for_energy(source, root / "duration")
    source_duration_s = len(source_samples) / source_rate
    report = {
        "audio": str(source),
        "audio_sha256": worker_report.get("audio_sha256"),
        "clip": str(clip),
        "clip_start_ms": start_ms,
        "clip_duration_ms": duration_ms,
        "source_duration_seconds": source_duration_s,
        "model": worker_report.get("model"),
        "model_sha256": worker_report.get("model_sha256"),
        "profile": worker_report.get("profile"),
        "elapsed_seconds": measurement["elapsed_seconds"],
        "worker_elapsed_seconds": worker_report.get("elapsed_seconds"),
        "peak_rss_mib": measurement["peak_rss_mib"],
        "estimated_full_song_seconds": (
            measurement["elapsed_seconds"] * source_duration_s / (duration_ms / 1000)
        ),
        "stem_directory": str(stem_dir),
    }
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def trim_air_ln_tails(
    osu_text: str,
    audio: np.ndarray,
    sample_rate: int,
    *,
    minimum_silence_ms: int = 150,
    relative_threshold: float = 0.01,
    window_ms: int = 25,
    hop_ms: int = 10,
    grid_bpm: float | None = None,
    grid_offset_ms: float = 0,
    grid_division: int = 16,
) -> tuple[str, dict]:
    """Trim only LN tails followed by a sustained >150ms near-silent region."""
    if minimum_silence_ms < 150:
        raise ValueError("空气长条门槛不得低于 150ms")
    wave = np.asarray(audio, dtype=np.float64)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if wave.ndim != 1 or sample_rate <= 0 or not np.isfinite(wave).all():
        raise ValueError("LN 能量包络输入无效")
    if not 0 < relative_threshold < 1:
        raise ValueError("静音相对阈值无效")
    parsed = parse_osu(osu_text)
    window = max(1, round(sample_rate * window_ms / 1000))
    hop = max(1, round(sample_rate * hop_ms / 1000))
    starts = np.arange(0, max(1, len(wave)), hop, dtype=np.int64)
    ends = np.minimum(len(wave), starts + window)
    prefix = np.concatenate(([0.0], np.cumsum(np.square(wave), dtype=np.float64)))
    envelope = np.sqrt(np.maximum(0, (prefix[ends] - prefix[starts]) / np.maximum(1, ends - starts)))
    threshold = max(1e-5, float(np.percentile(envelope, 95)) * relative_threshold)
    period = 60000 / grid_bpm / grid_division if grid_bpm and grid_bpm > 0 else None

    replacements: dict[str, str] = {}
    trimmed = []
    for note in parsed["notes"]:
        if not note.hold or note.end <= note.start:
            continue
        left = max(0, int(note.start * sample_rate / 1000))
        right = min(len(wave), int(note.end * sample_rate / 1000))
        if right <= left:
            continue
        mask = (starts < right) & (ends > left) & (envelope >= threshold)
        active = np.flatnonzero(mask)
        if not len(active):
            continue
        active_end = round(ends[active[-1]] * 1000 / sample_rate)
        silence = note.end - active_end
        if silence < minimum_silence_ms:
            continue
        new_end = active_end
        if period:
            new_end = round((new_end - grid_offset_ms) / period) * period + grid_offset_ms
            new_end = int(round(new_end))
        if new_end <= note.start + 20 or new_end >= note.end:
            continue
        fields = note.raw.split(",")
        tail = fields[5].split(":")
        tail[0] = str(new_end)
        fields[5] = ":".join(tail)
        replacements.setdefault(note.raw, ",".join(fields))
        trimmed.append({
            "start_ms": note.start,
            "old_end_ms": note.end,
            "new_end_ms": new_end,
            "quiet_tail_ms": silence,
        })

    if not replacements:
        return osu_text, {
            "examined_lns": sum(note.hold for note in parsed["notes"]),
            "trimmed_lns": 0,
            "threshold_rms": threshold,
            "trims": [],
        }
    counts = Counter()
    output_lines = []
    in_objects = False
    for line in osu_text.splitlines():
        if line.strip() == "[HitObjects]":
            in_objects = True
            output_lines.append(line)
            continue
        if line.startswith("[") and line.endswith("]") and line.strip() != "[HitObjects]":
            in_objects = False
        if in_objects and line in replacements:
            index = line
            used = counts[index]
            counts[index] += 1
            if used == 0:
                line = replacements[index]
        output_lines.append(line)
    result = "\n".join(output_lines) + ("\n" if osu_text.endswith("\n") else "")
    verify = parse_osu(result)
    if len(verify["notes"]) != len(parsed["notes"]):
        raise ValueError("LN 能量裁切意外改变了物件数")
    return result, {
        "examined_lns": sum(note.hold for note in parsed["notes"]),
        "trimmed_lns": len(trimmed),
        "threshold_rms": threshold,
        "trims": trimmed,
    }


def validate_map_grid(
    osu_text: str,
    *,
    bpm: float,
    offset_ms: float,
    division: int = 48,
) -> dict:
    """Require all generated starts and LN releases to land on the clip clock."""
    from .timing import grid_report

    parsed = parse_osu(osu_text)
    report = grid_report(
        parsed["notes"],
        {"bpm": float(bpm), "offset": float(offset_ms), "division": division},
    )
    if report["off_grid"]:
        raise ValueError(
            f"{parsed['keys']}K 谱存在 {report['off_grid']} 个离线起点/LN 尾端，"
            f"最大偏差 {report['max_error_ms']}ms"
        )
    return report


def snap_chart_to_grid(
    osu_text: str,
    *,
    bpm: float,
    offset_ms: float,
    division: int = 48,
) -> tuple[str, dict]:
    """Snap generated starts and LN releases to the confirmed timing clock."""
    from dataclasses import replace

    parsed = parse_osu(osu_text)
    tick = 60000 / float(bpm) / division
    if not math.isfinite(tick) or tick <= 0:
        raise ValueError("拍格周期无效")
    notes = []
    moved = []
    for note in parsed["notes"]:
        start = round(offset_ms + round((note.start - offset_ms) / tick) * tick)
        end = (
            round(offset_ms + round((note.end - offset_ms) / tick) * tick)
            if note.hold else start
        )
        if note.hold and end <= start:
            end = start + max(1, round(tick))
        if start != note.start or end != note.end:
            moved.append({
                "old_start_ms": note.start,
                "new_start_ms": start,
                "old_end_ms": note.end,
                "new_end_ms": end,
            })
        fields = note.raw.split(",")
        fields[2] = str(start)
        if note.hold:
            tail = fields[5].split(":")
            tail[0] = str(end)
            fields[5] = ":".join(tail)
        notes.append(replace(note, start=start, end=end, raw=",".join(fields)))
    content = sections(osu_text)
    content["HitObjects"] = [
        note.raw for note in sorted(notes, key=lambda item: (item.start, item.lane))
    ]
    result = _format_sections(content)
    parse_osu(result)
    return result, {
        "division": division,
        "tick_ms": tick,
        "adjusted_objects": len(moved),
        "max_endpoint_shift_ms": max(
            (max(abs(row["old_start_ms"] - row["new_start_ms"]),
                 abs(row["old_end_ms"] - row["new_end_ms"])) for row in moved),
            default=0,
        ),
        "adjustments": moved,
    }


def repair_native_10k_collisions(osu_text: str) -> tuple[str, dict]:
    """Reassign only lanes when native 10K has same-track collisions.

    Torosu's same-key convert is a strict no-op, so this narrowly scoped
    fallback preserves every object field except the lane before Torosu
    validates the repaired 10K chart.
    """
    from .arrange import assign_lanes

    parsed = parse_osu(osu_text)
    before = validate(parsed["notes"], 10)
    if before["valid"]:
        return osu_text, {
            "reassigned": False,
            "objects_before": len(parsed["notes"]),
            "objects_after": len(parsed["notes"]),
            "issues_before": [],
        }
    original_payload = Counter(
        tuple(field.strip() for field in note.raw.split(",")[1:])
        for note in parsed["notes"]
    )
    arranged = assign_lanes(
        parsed["notes"],
        {"keys": 10, "layout_engine": "beam"},
        timing_points=parsed["timing"],
    )
    result = export_osu(
        arranged,
        10,
        None,
        original=osu_text,
        version=parsed["metadata"].get("Version", "V32-mini native 10K"),
    )
    verified = parse_osu(result)
    after = validate(verified["notes"], 10)
    actual_payload = Counter(
        tuple(field.strip() for field in note.raw.split(",")[1:])
        for note in verified["notes"]
    )
    if not after["valid"] or actual_payload != original_payload:
        raise ValueError(
            "native 10K 冲突重排未保留全部物件字段："
            + (after["issues"][0] if after["issues"] else "物件数量/字段发生变化")
        )
    return result, {
        "reassigned": True,
        "objects_before": len(parsed["notes"]),
        "objects_after": len(verified["notes"]),
        "issues_before": before["issues"],
        "issues_after": after["issues"],
    }


def _format_sections(content: dict[str, list[str]]) -> str:
    return "\n".join(
        ["\n".join(content.get("", ["osu file format v14"]))]
        + [f"[{name}]\n" + "\n".join(body) for name, body in content.items() if name]
    ) + "\n"


def shift_reference_timing(
    source_text: str,
    *,
    segment_start_ms: int,
    segment_duration_ms: int | None = None,
    bpm: float | None = None,
    offset_ms: float | None = None,
    audio_filename: str | None = None,
) -> tuple[str, float, float]:
    """Build a clip-relative reference map and return its base BPM/offset."""
    content = sections(source_text)
    timing = content.get("TimingPoints", [])
    if bpm is not None:
        if not math.isfinite(bpm) or not 10 <= bpm <= 1000:
            raise ValueError("BPM 必须在 10–1000 之间")
        source_offset = float(offset_ms if offset_ms is not None else 0)
        relative_offset = source_offset - segment_start_ms
        timing = [f"{relative_offset:.3f},{60000 / bpm:.8f},4,2,0,70,1,0"]
    else:
        redlines = []
        for line in timing:
            fields = line.split(",")
            if len(fields) < 2:
                continue
            try:
                beat = float(fields[1])
                position = float(fields[0])
            except ValueError:
                continue
            if beat > 0:
                redlines.append((position, beat))
        if not redlines:
            raise ValueError("参考谱面没有有效的红线 BPM")
        position, beat = redlines[0]
        bpm = 60000 / beat
        source_offset = position if offset_ms is None else float(offset_ms)
        relative_offset = source_offset - segment_start_ms
        timing = [
            ",".join([f"{float(line.split(',')[0]) - segment_start_ms:.3f}", *line.split(",")[1:]])
            if line.strip() and "," in line else line
            for line in timing
        ]
    content["TimingPoints"] = timing
    editor = content.get("Editor", [])
    for line in editor:
        if line.split(":", 1)[0].strip().casefold() != "bookmarks":
            continue
        try:
            values = [
                round(float(value.strip()) - segment_start_ms)
                for value in line.split(":", 1)[1].split(",")
                if value.strip()
            ]
        except ValueError:
            continue
        values = [
            value for value in values
            if value >= 0 and (segment_duration_ms is None or value < segment_duration_ms)
        ]
        content["Editor"] = _set_field(editor, "Bookmarks", ",".join(map(str, values)))
        break
    general = content.get("General", [])
    preview = next(
        (line.split(":", 1)[1].strip() for line in general
         if line.split(":", 1)[0].strip().casefold() == "previewtime"),
        None,
    )
    if preview is not None:
        try:
            preview_ms = round(float(preview) - segment_start_ms)
        except ValueError:
            preview_ms = 0
        if preview_ms < 0 or (
            segment_duration_ms is not None and preview_ms >= segment_duration_ms
        ):
            preview_ms = 0
        content["General"] = _set_field(general, "PreviewTime", str(preview_ms))
    if audio_filename is not None:
        general = content.setdefault("General", [])
        content["General"] = _set_field(general, "AudioFilename", audio_filename)
    return _format_sections(content), float(bpm), float(relative_offset)


def _set_field(lines: list[str], key: str, value: str) -> list[str]:
    result = []
    found = False
    for line in lines:
        if line.split(":", 1)[0].strip().casefold() == key.casefold():
            if not found:
                result.append(f"{key}:{value}")
                found = True
        else:
            result.append(line)
    if not found:
        result.append(f"{key}:{value}")
    return result


def _set_chart_metadata(text: str, *, version: str, audio_filename: str) -> str:
    content = sections(text)
    content["General"] = _set_field(content.get("General", []), "AudioFilename", audio_filename)
    content["General"] = _set_field(content["General"], "Mode", "3")
    metadata = content.get("Metadata", [])
    for key, value in (
        ("Version", version),
        ("BeatmapID", "0"),
        ("BeatmapSetID", "-1"),
    ):
        metadata = _set_field(metadata, key, value)
    content["Metadata"] = metadata
    return _format_sections(content)


def _copy_template_media(
    source_path: Path,
    chart_member: str,
    source_text: str,
    output: zipfile.ZipFile,
    *,
    source_is_archive: bool,
) -> set[str]:
    from .ab_export import _media_references, _resolve_asset, _safe_member

    parsed = parse_osu(source_text)
    source_audio = _resolve_asset(chart_member, parsed["general"].get("AudioFilename", ""))
    references = {
        _safe_member(name)
        for name in _media_references(source_text, chart_member)
        if _safe_member(name).casefold() != source_audio.casefold()
    }
    copied = set()
    if source_is_archive:
        with zipfile.ZipFile(source_path, "r") as archive:
            members = {}
            for info in archive.infolist():
                if info.is_dir() or info.filename.lower().endswith(".osu"):
                    continue
                name = _safe_member(info.filename)
                members[name.casefold()] = (name, info)
            for member in sorted(references, key=str.casefold):
                item = members.get(member.casefold())
                if item is None:
                    raise ValueError(f"参考 .osz 缺少媒体文件：{member}")
                name, info = item
                output.writestr(name, archive.read(info))
                copied.add(name.casefold())
    else:
        root = source_path.parent.resolve()
        for member in sorted(references, key=str.casefold):
            candidate = (root / Path(*PurePosixPath(member).parts)).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise ValueError(f"媒体路径越出参考谱面目录：{member}") from error
            if not candidate.is_file():
                raise ValueError(f"参考谱面引用的媒体不存在：{candidate}")
            output.write(candidate, member)
            copied.add(member.casefold())
    return copied


def package_four_difficulties(
    *,
    output_path: str | Path,
    source_chart_path: str | Path,
    source_chart_member: str | None,
    chart_texts: dict[str, str],
    audio_by_label: dict[str, str | Path],
    duration_ms: int,
) -> dict:
    """Package four fully validated charts and their shared/remixed audio."""
    from .ab_export import _load_source, _media_references, _resolve_asset, _safe_member

    if tuple(chart_texts) != _CHART_LABELS:
        raise ValueError("四难度顺序或标签与验收约定不一致")
    if set(audio_by_label) != set(_CHART_LABELS):
        raise ValueError("每个难度都必须显式绑定自己的播放音频")
    source = _load_source(Path(source_chart_path), source_chart_member)
    chart_parent = PurePosixPath(source.chart_member).parent
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有实验谱包：{output}")
    fd, temporary = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".tmp", dir=output.parent)
    os.close(fd)
    chart_paths = []
    metrics = {}
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            copied = _copy_template_media(
                source.path, source.chart_member, source.text, archive,
                source_is_archive=source.is_archive,
            )
            audio_members = {}
            for label, audio_path in audio_by_label.items():
                audio = Path(audio_path)
                member = _safe_member((chart_parent / audio.name).as_posix())
                if member.casefold() not in copied:
                    archive.write(audio, member)
                    copied.add(member.casefold())
                audio_members[label] = member
            for label in _CHART_LABELS:
                audio_member = audio_members[label]
                relative_audio = PurePosixPath(audio_member).relative_to(chart_parent).as_posix()
                text = _set_chart_metadata(
                    chart_texts[label], version=label, audio_filename=relative_audio,
                )
                parsed = parse_osu(text)
                report = validate(parsed["notes"], 10)
                if parsed["keys"] != 10 or not report["valid"]:
                    raise ValueError(f"{label} 存在键数错误、同轨冲突或非法 LN")
                if any(note.end > duration_ms + 100 for note in parsed["notes"]):
                    raise ValueError(f"{label} 含超出音频片段的物件")
                _resolve_asset((chart_parent / f"{label}.osu").as_posix(), relative_audio)
                chart_path = _safe_member((chart_parent / f"{label}.osu").as_posix())
                archive.writestr(chart_path, text.encode("utf-8"))
                chart_paths.append(chart_path)
                metrics[label] = {
                    "objects": len(parsed["notes"]),
                    "ln": report["ln"],
                    "peak_held": report["max_held"],
                    "lane_counts": report["lane_counts"],
                }
        with zipfile.ZipFile(temporary, "r") as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f".osz CRC 校验失败：{bad}")
            names = {name.casefold() for name in archive.namelist()}
            for chart_path in chart_paths:
                chart_text = archive.read(chart_path).decode("utf-8")
                parsed = parse_osu(chart_text)
                audio_ref = _resolve_asset(chart_path, parsed["general"]["AudioFilename"])
                if audio_ref.casefold() not in names:
                    raise ValueError(f"{chart_path} 引用的音频不在 .osz 中")
                missing = [
                    member for member in _media_references(chart_text, chart_path)
                    if member.casefold() not in names
                ]
                if missing:
                    raise ValueError(
                        f"{chart_path} 引用的媒体不在 .osz 中：" + ", ".join(missing[:5])
                    )
                if not validate(parsed["notes"], 10)["valid"]:
                    raise ValueError(f"{chart_path} 回读后结构无效")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"output": output, "chart_paths": tuple(chart_paths), "metrics": metrics}


def _measured_call(function):
    future_started = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor

    stop_path = EXPERIMENTS / f".rss-{os.getpid()}-{time.time_ns()}.stop"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    env = configure_d_cache_environment()
    monitor = _start_rss_monitor(os.getpid(), stop_path, env)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(function)
        try:
            while not future.done():
                time.sleep(0.1)
            result = future.result()
        finally:
            stop_path.touch()
            stdout, stderr = monitor.communicate(timeout=15)
            if monitor.returncode and stderr.strip():
                raise RuntimeError("D 盘内存监控失败：" + stderr.strip()[-1000:])
            try:
                peak = int(stdout.strip() or "0")
            except ValueError:
                peak = 0
            stop_path.unlink(missing_ok=True)
    return result, {
        "elapsed_seconds": time.perf_counter() - future_started,
        "peak_rss_bytes": peak,
        "peak_rss_mib": peak / 2**20,
    }


def generate_old_baseline(
    *,
    audio_path: str | Path,
    reference_text: str,
    work_dir: str | Path,
    bpm: float,
    offset_ms: float,
    hold_note_ratio: float = 0.55,
) -> tuple[str, dict]:
    """Use ManiaLab's actual Basic Pitch evidence + MILP selector + beam layout."""
    from . import audio
    from .arrange_config import parameters
    from .generation import generate_variant

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    settings = parameters({
        "keys": 10,
        "layout_engine": "beam",
        "density": 7,
        "ln_ratio": hold_note_ratio,
    })
    analysis, analysis_timing = _measured_call(lambda: audio.analyze(
        audio_path,
        work,
        ROOT / "assets" / "basic-pitch.onnx",
        settings,
    ))
    tempo = dict(analysis.get("tempo") or {})
    tempo.update({
        "bpm": float(bpm),
        "offset": round(float(offset_ms)),
        "division": int(tempo.get("division", 4)),
        "audio_shift_ms": 0,
        "stable": True,
    })
    state = {
        "analysis": analysis,
        "plan": {},
        "expression": None,
        "variants": [],
        "active": -1,
        "settings": settings,
        "input_notes": [],
        "input_keys": 7,
        "input_timing": [],
        "audio_shift_ms": 0,
        "mode": "audio",
    }
    report = generate_variant(
        state,
        {"mode": "audio", "snap": True, "timing_confirmed": True},
        settings,
        tempo,
        lambda *_: None,
        lambda: False,
    )
    variant = state["variants"][state["active"]]
    notes = [note_from(note) for note in variant["notes"]]
    chart = export_osu(
        notes,
        10,
        None,
        original=reference_text,
        version=_CHART_LABELS[0],
    )
    return chart, _summarize_baseline_timing({
        "elapsed_seconds": analysis_timing["elapsed_seconds"] + variant["elapsed_seconds"],
        "analysis_seconds": analysis_timing["elapsed_seconds"],
        "generation_seconds": variant["elapsed_seconds"],
        "peak_rss_mib": analysis_timing["peak_rss_mib"],
        "objects": len(notes),
        "ln": sum(note.hold for note in notes),
        "selection": variant.get("selection", {}),
    })


def _summarize_baseline_timing(report: dict) -> dict:
    selection = report.get("selection") or {}
    if not isinstance(selection, dict):
        selection = {}
    solver = selection.get("selection_solver")
    if isinstance(solver, dict):
        solver = {
            key: solver.get(key)
            for key in ("name", "status", "objective", "elapsed_seconds", "fallback")
            if key in solver
        }
    alignment = selection.get("alignment")
    if isinstance(alignment, dict):
        alignment = {
            key: alignment.get(key)
            for key in (
                "enabled", "moved", "tails_moved", "large_attack_shifts",
                "max_shift_ms", "collapsed_holds",
            )
            if key in alignment
        }
    result = {key: value for key, value in report.items() if key != "selection"}
    result["selection"] = {
        "selected": selection.get("selected"),
        "candidates": selection.get("candidates"),
        "omitted": selection.get("omitted"),
        "attack_selection": selection.get("attack_selection"),
        "attack_optimal": selection.get("attack_optimal"),
        "selection_solver": solver,
        "actual_ln_ratio": selection.get("actual_ln_ratio"),
        "selected_voices": selection.get("selected_voices"),
        "omission_reasons": selection.get("omission_reasons"),
        "selected_large_attack_shifts": selection.get("selected_large_attack_shifts"),
        "alignment": alignment,
    }
    return result


def run_experiment(
    *,
    audio_path: str | Path,
    chart_path: str | Path,
    output_path: str | Path,
    work_dir: str | Path = EXPERIMENTS,
    chart_member: str | None = None,
    benchmark_report: str | Path | None = None,
    segment_start_ms: int | None = None,
    segment_duration_ms: int | None = None,
    bpm: float | None = None,
    offset_ms: float | None = None,
    hold_note_ratio: float = 0.426,
    threads: int = 6,
    model_dir: str | Path | None = None,
    reuse_run_dir: str | Path | None = None,
) -> dict:
    """Run the four-way A-route experiment, first enforcing the 30s CPU gate."""
    from .ab_export import _load_source
    from .separation import Separator
    from .torosu_bridge import (
        TorosuBridgeError,
        analyze_osu_text,
        convert_osu_text,
    )

    source_audio = Path(audio_path).resolve()
    source_chart = Path(chart_path).resolve()
    out = Path(output_path).resolve()
    experiment_root = Path(work_dir).resolve()
    try:
        out.relative_to((ROOT / "workspace_exp").resolve())
    except ValueError as error:
        raise ValueError("实验 .osz 必须输出到 workspace_exp/") from error
    if out.exists() or out.with_suffix(".report.json").exists():
        raise FileExistsError(f"拒绝覆盖已有实验产物：{out}")
    experiment_root.mkdir(parents=True, exist_ok=True)
    if reuse_run_dir is None:
        run_root = experiment_root / f"run-{time.strftime('%Y%m%d-%H%M%S')}"
        run_root.mkdir(parents=True, exist_ok=False)
    else:
        run_root = Path(reuse_run_dir).resolve()
        try:
            run_root.relative_to(experiment_root)
        except ValueError as error:
            raise ValueError("续跑目录必须位于 workspace_exp/ab_v32/") from error
        if not run_root.is_dir():
            raise FileNotFoundError(f"续跑目录不存在：{run_root}")
    env = configure_d_cache_environment()
    os.environ.update(env)
    configure_separator_runtime()
    source = _load_source(source_chart, chart_member)
    source_data = _decode_for_energy(source_audio, run_root / "duration")
    samples, sample_rate = source_data
    duration_s = len(samples) / sample_rate
    chart_data = parse_osu(source.text)
    redlines = []
    for line in chart_data["timing"]:
        fields = line.split(",")
        if len(fields) >= 2:
            try:
                beat = float(fields[1])
                point = float(fields[0])
            except ValueError:
                continue
            if beat > 0:
                redlines.append((point, 60000 / beat))
    if not redlines and bpm is None:
        raise ValueError("参考谱面缺少有效 BPM；请显式传入 --bpm")
    source_offset = float(offset_ms if offset_ms is not None else (redlines[0][0] if redlines else 0))
    bpm_value = float(bpm if bpm is not None else redlines[0][1])

    if benchmark_report:
        benchmark = _read_benchmark_report(benchmark_report, duration_s)
    else:
        benchmark_start, _ = choose_climax_window(samples, sample_rate, 30)
        benchmark = _run_short_benchmark(
            audio_path=source_audio,
            source_text=source.text,
            segment_start_ms=benchmark_start,
            source_duration_s=duration_s,
            bpm=bpm_value,
            offset_ms=source_offset,
            hold_note_ratio=hold_note_ratio,
            threads=threads,
            output_dir=run_root / "benchmark",
            model_dir=model_dir,
        )
    estimate = float(benchmark["projected_full_song_seconds"])
    clip_mode = estimate > 300
    if segment_duration_ms is not None:
        if type(segment_duration_ms) is not int or not 30000 <= segment_duration_ms <= 60000:
            raise ValueError("高潮片段时长必须为 30–60 秒")
        clip_mode = True
    duration_ms = (segment_duration_ms if segment_duration_ms is not None
                   else round(duration_s * 1000) if not clip_mode else 45000)
    if duration_ms > round(duration_s * 1000):
        raise ValueError("片段时长超过源音频长度")
    if segment_start_ms is None:
        if clip_mode:
            segment_start_ms, peak_rms = choose_climax_window(
                samples, sample_rate, duration_ms / 1000,
            )
        else:
            segment_start_ms, peak_rms = 0, float(np.sqrt(np.mean(np.square(samples))))
    else:
        if type(segment_start_ms) is not int or segment_start_ms < 0:
            raise ValueError("片段起点必须是非负毫秒整数")
        clip_start_sample = round(segment_start_ms * sample_rate / 1000)
        clip_end_sample = min(
            len(samples),
            clip_start_sample + round(duration_ms * sample_rate / 1000),
        )
        clip_wave = samples[clip_start_sample:clip_end_sample]
        peak_rms = float(np.sqrt(np.mean(np.square(clip_wave)))) if len(clip_wave) else 0.0
    if segment_start_ms + duration_ms > round(duration_s * 1000):
        raise ValueError("片段超出源音频长度")

    run_root.mkdir(parents=True, exist_ok=True)
    if clip_mode:
        full_mix_path = run_root / "tan_gu_zhi_full_mix.wav"
        full_mix_audio = (
            full_mix_path if full_mix_path.is_file() else extract_audio_excerpt(
                source_audio,
                full_mix_path,
                start_ms=segment_start_ms,
                duration_ms=duration_ms,
            )
        )
        full_mix_model_audio = full_mix_audio
    else:
        full_mix_audio = source_audio
        model_input_path = run_root / "tan_gu_zhi_full_model_input.wav"
        full_mix_model_audio = (
            model_input_path if model_input_path.is_file() else extract_audio_excerpt(
                source_audio,
                model_input_path,
                start_ms=0,
                duration_ms=duration_ms,
            )
        )
    reference_text, clip_bpm, clip_offset = shift_reference_timing(
        source.text,
        segment_start_ms=segment_start_ms,
        segment_duration_ms=duration_ms,
        bpm=bpm,
        offset_ms=source_offset,
        audio_filename=full_mix_model_audio.name,
    )
    reference_path = run_root / "reference_clip.osu"
    reference_path.write_text(reference_text, encoding="utf-8")
    duration_clip_ms = _audio_duration_ms(full_mix_model_audio)
    if duration_clip_ms > duration_ms + 100:
        raise ValueError("全混音输入时长超出目标片段")

    no_vocal_audio = run_root / "tan_gu_zhi_no_vocal.wav"
    separator = Separator(ROOT, profile="six")
    separation_preflight = _find_separation_preflight(source_audio)
    separation_work = run_root / "separation"
    separation_work.mkdir(parents=True, exist_ok=True)
    separation_result, separation_timing = _measured_call(lambda: separator.separate(
        full_mix_audio,
        separation_work,
        progress=lambda *_: None,
        cancelled=lambda: False,
        include_beats=False,
    ))
    stem_dir, separation_manifest = separation_result
    remix_report, remix_timing = _measured_call(lambda: remix_no_vocal(
        stem_dir, no_vocal_audio,
    ))
    energy = {
        "full_mix": _decode_for_energy(full_mix_audio, run_root / "energy-full")[0],
        "no_vocal": _decode_for_energy(no_vocal_audio, run_root / "energy-no-vocal")[0],
    }
    timings = {
        "benchmark": benchmark,
        "separation": separation_manifest | {
            "worker_elapsed_seconds": separation_manifest.get("elapsed_seconds"),
            "cache_lookup_seconds": separation_timing["elapsed_seconds"],
            "cache_lookup_peak_rss_mib": separation_timing["peak_rss_mib"],
        },
        "separation_30s_real_audio_benchmark": separation_preflight,
        "remix": remix_report | remix_timing,
    }

    outputs = {}
    chart_texts = {}
    audio_by_label = {}
    for label, keycount, audio_input, energy_key, density in (
        (_CHART_LABELS[1], 10, full_mix_model_audio, "full_mix", "none"),
        (_CHART_LABELS[2], 7, full_mix_model_audio, "full_mix", "dense"),
        (_CHART_LABELS[3], 7, no_vocal_audio, "no_vocal", "dense"),
    ):
        print(f"生成 {label} · {keycount}K · {'no_vocal' if energy_key == 'no_vocal' else 'full_mix'}")
        inference_dir = run_root / f"mapper-{keycount}k-{energy_key}"
        run_report_path = inference_dir / "run-report.json"
        if run_report_path.is_file():
            inference = json.loads(run_report_path.read_text(encoding="utf-8"))
            expected_identity = inference_identity(
                audio_path=audio_input,
                reference_chart=reference_path,
                keycount=keycount,
                bpm=clip_bpm,
                offset_ms=clip_offset,
                hold_note_ratio=hold_note_ratio,
                threads=threads,
                precision="fp32",
                model_dir=model_dir or MODEL_CACHE / "gamemode=3",
                start_ms=0,
                end_ms=_audio_duration_ms(audio_input),
            )
            validate_reusable_inference_report(inference, expected_identity)
        else:
            inference = run_mapperatorinator(
                audio_path=audio_input,
                reference_chart=reference_path,
                output_dir=inference_dir,
                keycount=keycount,
                bpm=clip_bpm,
                offset_ms=clip_offset,
                hold_note_ratio=hold_note_ratio,
                threads=threads,
                model_dir=model_dir,
            )
        generated = Path(inference["map"]).read_text(encoding="utf-8-sig")
        gridded, grid_snap = snap_chart_to_grid(
            generated,
            bpm=clip_bpm,
            offset_ms=clip_offset,
        )
        trim_started = time.perf_counter()
        trimmed, trim_report = trim_air_ln_tails(
            gridded,
            energy[energy_key],
            22050,
            grid_bpm=clip_bpm,
            grid_offset_ms=clip_offset,
        )
        trim_seconds = time.perf_counter() - trim_started
        torosu_started = time.perf_counter()
        if keycount == 7:
            arranged = convert_osu_text(
                trimmed,
                10,
                playstyle="5+5",
                density="dense",
                keep_jacks=True,
                strict=True,
            )
        else:
            # Same-key Torosu conversion is intentionally strict/no-density:
            # it validates the native 10K map without changing its thickness.
            native_repair = {
                "reassigned": False,
                "objects_before": len(parse_osu(trimmed)["notes"]),
                "objects_after": len(parse_osu(trimmed)["notes"]),
                "issues_before": [],
            }
            try:
                arranged = convert_osu_text(
                    trimmed,
                    10,
                    playstyle="5+5",
                    density="none",
                    keep_jacks=True,
                    strict=True,
                )
            except TorosuBridgeError as error:
                detail = str(error).casefold()
                if not any(marker in detail for marker in (
                    "冲突", "share lane", "same lane", "lane overlap",
                )):
                    raise
                repaired, native_repair = repair_native_10k_collisions(trimmed)
                arranged = convert_osu_text(
                    repaired,
                    10,
                    playstyle="5+5",
                    density="none",
                    keep_jacks=True,
                    strict=True,
                )
        torosu_seconds = time.perf_counter() - torosu_started
        notes = parse_osu(arranged)["notes"]
        chart_texts[label] = export_osu(
            notes,
            10,
            None,
            original=reference_text,
            version=label,
        )
        grid = validate_map_grid(
            chart_texts[label],
            bpm=clip_bpm,
            offset_ms=clip_offset,
        )
        outputs[label] = {
            **inference,
            "model_input_audio": str(audio_input),
            "playback_audio": str(full_mix_audio),
            "torosu_density": density,
            "ln_energy_trim": trim_report,
            "grid_snap": grid_snap,
            "grid": grid,
            "ln_check_seconds": trim_seconds,
            "torosu_seconds": torosu_seconds,
        }
        if keycount == 10:
            outputs[label]["native_10k_repair"] = native_repair
        # Keep all four in-game comparisons on the same real-song playback;
        # only the model's inference input changes for the no-vocal route.
        audio_by_label[label] = full_mix_audio
        timings[label] = outputs[label]

    old_work = run_root / "old-basicpitch"
    baseline_cache = old_work / "baseline.osu"
    baseline_report_cache = old_work / "baseline-report.json"
    if baseline_cache.is_file() and baseline_report_cache.is_file():
        baseline = baseline_cache.read_text(encoding="utf-8")
        baseline_timing = _summarize_baseline_timing(
            json.loads(baseline_report_cache.read_text(encoding="utf-8"))
        )
    else:
        baseline, baseline_timing = generate_old_baseline(
            audio_path=full_mix_audio,
            reference_text=reference_text,
            work_dir=old_work,
            bpm=clip_bpm,
            offset_ms=clip_offset,
        )
        old_work.mkdir(parents=True, exist_ok=True)
        baseline_cache.write_text(baseline, encoding="utf-8")
        baseline_report_cache.write_text(
            json.dumps(baseline_timing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    chart_texts[_CHART_LABELS[0]] = _set_chart_metadata(
        baseline,
        version=_CHART_LABELS[0],
        audio_filename=full_mix_audio.name,
    )
    baseline_grid = validate_map_grid(
        chart_texts[_CHART_LABELS[0]],
        bpm=clip_bpm,
        offset_ms=clip_offset,
    )
    audio_by_label[_CHART_LABELS[0]] = full_mix_audio
    timings[_CHART_LABELS[0]] = baseline_timing | {"grid": baseline_grid}

    # Preserve the externally specified difficulty order in the archive.
    chart_texts = {label: chart_texts[label] for label in _CHART_LABELS}
    audio_by_label = {label: audio_by_label[label] for label in _CHART_LABELS}
    packaged = package_four_difficulties(
        output_path=out,
        source_chart_path=source_chart,
        source_chart_member=chart_member,
        chart_texts=chart_texts,
        audio_by_label=audio_by_label,
        duration_ms=duration_ms,
    )

    from .ab_export import _metrics
    metric_rows = {}
    for label, text in chart_texts.items():
        parsed = parse_osu(text)
        metric_rows[label] = _metrics(
            analyze_osu_text(text, keys=10, playstyle="5+5"),
            parsed,
        )
    report = {
        "source_audio": str(source_audio),
        "source_chart": str(source_chart),
        "source_duration_seconds": duration_s,
        "segment": {
            "start_ms": segment_start_ms,
            "duration_ms": duration_ms,
            "climax_window_rms": peak_rms,
            "limited_to_excerpt": clip_mode,
        },
        "benchmark": benchmark,
        "projected_full_song_seconds": estimate,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_directory": str(model_dir or MODEL_CACHE / "gamemode=3"),
        "cache_environment": {
            key: env[key] for key in (
                "HF_HOME", "TORCH_HOME", "TMP", "TEMP", "UV_CACHE_DIR",
            )
        },
        "timings": timings,
        "torosu_metrics": metric_rows,
        "osz": str(packaged["output"]),
        "chart_paths": list(packaged["chart_paths"]),
        "archive_metrics": packaged["metrics"],
    }
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_metrics(metric_rows)
    return report


def _run_short_benchmark(
    *,
    audio_path: Path,
    source_text: str,
    segment_start_ms: int,
    source_duration_s: float,
    bpm: float,
    offset_ms: float,
    hold_note_ratio: float,
    threads: int,
    output_dir: Path,
    model_dir: str | Path | None,
) -> dict:
    start = round(segment_start_ms)
    end = start + 30000
    model = Path(model_dir) if model_dir else ensure_v32_model_snapshot()
    benchmark_audio = extract_audio_excerpt(
        audio_path,
        output_dir.parent / "benchmark-30s.wav",
        start_ms=start,
        duration_ms=30000,
    )
    reference_text, benchmark_bpm, benchmark_offset = shift_reference_timing(
        source_text,
        segment_start_ms=start,
        segment_duration_ms=30000,
        bpm=bpm,
        offset_ms=offset_ms,
        audio_filename=benchmark_audio.name,
    )
    reference_path = output_dir.parent / "benchmark-reference.osu"
    reference_path.write_text(reference_text, encoding="utf-8")
    result = run_mapperatorinator(
        audio_path=benchmark_audio,
        reference_chart=reference_path,
        output_dir=output_dir,
        keycount=7,
        bpm=benchmark_bpm,
        offset_ms=benchmark_offset,
        hold_note_ratio=hold_note_ratio,
        threads=threads,
        model_dir=model,
        start_ms=0,
        end_ms=30000,
    )
    result.update({
        "start_ms": start,
        "end_ms": end,
        "source_duration_s": source_duration_s,
        "generate_s": result["elapsed_seconds"],
        "elapsed_s": result["elapsed_seconds"],
        "source_chart_timing": str(reference_path),
        "model_revision": MODEL_REVISION,
        "fast_decoder_loop_requested": True,
        "fast_decoder_loop_active": False,
    })
    result["projected_full_song_seconds"] = benchmark_full_song_seconds(result, source_duration_s)
    return result


def _print_metrics(rows: dict[str, dict]) -> None:
    columns = ("难度", "物件", "LN", "minijack", "pill", "双手均衡")
    print(" | ".join(columns))
    print("-+-".join("-" * len(value) for value in columns))
    for label, values in rows.items():
        balance = values.get("hand_balance")
        print(" | ".join(map(str, (
            label,
            values["objects"],
            values["ln"],
            values["minijack"],
            values["pill"],
            "n/a" if balance is None else f"{float(balance):.3f}",
        ))))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 v32-mini + Torosu 的真实歌曲四难度 CPU 对比",
    )
    parser.add_argument("--audio", required=True, help="真实歌曲音频路径")
    parser.add_argument("--chart", required=True, help="同曲 .osu 或 .osz 参考谱")
    parser.add_argument(
        "--output",
        default=str(EXPERIMENTS / "tan_gu_zhi_four_way.osz"),
        help="输出到 workspace_exp/ 内的 .osz",
    )
    parser.add_argument("--chart-member", help=".osz 内指定参考 .osu 路径")
    parser.add_argument("--benchmark-report", help="已完成的 30 秒基准 JSON")
    parser.add_argument("--segment-start-ms", type=int)
    parser.add_argument("--segment-duration-ms", type=int)
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--offset", type=float, dest="offset_ms")
    parser.add_argument("--hold-note-ratio", type=float, default=0.426)
    parser.add_argument("--threads", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--model-dir")
    parser.add_argument("--reuse-run-dir", help="从指定 workspace_exp/ab_v32/run-* 续跑")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_experiment(
            audio_path=args.audio,
            chart_path=args.chart,
            output_path=args.output,
            chart_member=args.chart_member,
            benchmark_report=args.benchmark_report,
            segment_start_ms=args.segment_start_ms,
            segment_duration_ms=args.segment_duration_ms,
            bpm=args.bpm,
            offset_ms=args.offset_ms,
            hold_note_ratio=args.hold_note_ratio,
            threads=args.threads,
            model_dir=args.model_dir,
            reuse_run_dir=args.reuse_run_dir,
        )
    except (OSError, ValueError, RuntimeError, TimeoutError, zipfile.BadZipFile) as error:
        print(f"Chimera v32 失败：{error}", file=sys.stderr)
        return 2
    print(f"已生成：{report['osz']}")
    print(f"实测整曲外推：{report['projected_full_song_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
