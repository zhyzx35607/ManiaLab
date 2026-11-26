"""Portable audio utilities used by the V32 generation pipeline."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import wavfile


def _audio_duration_ms(path: str | Path) -> int:
    source = Path(path)
    if source.suffix.casefold() == ".wav":
        rate, data = wavfile.read(source, mmap=True)
        return round(len(data) * 1000 / rate)
    import imageio_ffmpeg

    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-v",
            "info", "-i", str(source), "-f", "null", "-",
        ],
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
            duration = (
                int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            ) * 1000
            break
    if result.returncode or duration is None:
        raise ValueError(f"无法读取音频时长：{source}")
    return round(duration)


def _decode_for_energy(path: str | Path, work_dir: Path) -> tuple[np.ndarray, int]:
    from .audio_io import decode
    from .audio_config import SR

    work_dir.mkdir(parents=True, exist_ok=True)
    samples = decode(path, work_dir)
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
    if not len(mono) or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("高潮片段长度或音频为空")
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

    if type(start_ms) is not int or start_ms < 0:
        raise ValueError("片段起点必须为非负毫秒整数")
    if type(duration_ms) is not int or duration_ms <= 0:
        raise ValueError("片段时长必须为正毫秒整数")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-y",
            "-i", str(source), "-ss", f"{start_ms / 1000:.3f}",
            "-t", f"{duration_ms / 1000:.3f}", "-vn", "-ac", "2", "-ar",
            "44100", "-c:a", "pcm_s16le", str(target),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if result.returncode:
        raise ValueError("无法从真实音频提取片段：" + (result.stderr or "").strip()[-1000:])
    rate, data = wavfile.read(target, mmap=True)
    if rate != 44100 or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("音频片段必须为 44.1 kHz stereo WAV")
    if abs(len(data) * 1000 / rate - duration_ms) > 100:
        raise ValueError("音频片段时长与请求不一致")
    return target
