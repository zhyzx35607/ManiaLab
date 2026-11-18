"""Shared FFmpeg-backed audio decoding without model-runtime dependencies."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile

from .audio_config import SR


def run_hidden(args, **kwargs):
    return subprocess.run(args, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, **kwargs)


def decode_preview_window(path, first_sample, frames):
    # Trim after resampling so the excerpt uses the same sample clock as preview.wav.
    audio_filter = (f"aformat=sample_fmts=flt:sample_rates={SR}:channel_layouts=mono,"
                    f"atrim=start_sample={first_sample}:end_sample={first_sample + frames}")
    result = run_hidden([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
                         "-vn", "-af", audio_filter, "-f", "f32le", "pipe:1"], capture_output=True, timeout=180)
    if result.returncode:
        raise ValueError("音频无法解码，请检查导入的音频")
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if not np.all(np.isfinite(samples)):
        raise ValueError("音频含无效采样值")
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16)


def decode(path, work, *, runner=None):
    runner = run_hidden if runner is None else runner
    result = runner([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
                     "-vn", "-ac", "1", "-ar", str(SR), "-t", "1201", "-f", "f32le", "pipe:1"],
                    capture_output=True, timeout=180)
    if result.returncode:
        raise ValueError("音频无法解码，请换用 WAV / MP3 / FLAC / OGG")
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(samples) < SR // 4:
        raise ValueError("音频短于 0.25 秒")
    if len(samples) > SR * 1200:
        raise ValueError("首版支持不超过 20 分钟的音频")
    if not np.all(np.isfinite(samples)):
        raise ValueError("音频含无效采样值")
    wavfile.write(Path(work) / "preview.wav", SR, (np.clip(samples, -1, 1) * 32767).astype(np.int16))
    return samples
