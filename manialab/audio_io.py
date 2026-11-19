"""Shared FFmpeg-backed audio decoding without model-runtime dependencies."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import time

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile

from .audio_config import SR
from .cancellation import report_process_cleanup_failure, timeout_for


def run_hidden(args, **kwargs):
    return subprocess.run(args, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, **kwargs)


def run_hidden_cancellable(args, cancelled, *, timeout):
    if cancelled():
        raise InterruptedError("已取消，原有谱面已保留")
    process = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    def kill_and_reap():
        if process.poll() is None:
            try:
                process.kill()
            except OSError as error:
                if process.poll() is None:
                    message = "FFmpeg process could not be killed during cleanup"
                    report_process_cleanup_failure(cancelled, message)
                    raise ChildProcessError(message) from error
        try:
            process.communicate(timeout=timeout_for(cancelled, 5))
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                message = "FFmpeg process did not exit before its cleanup deadline"
                report_process_cleanup_failure(
                    cancelled,
                    message,
                )
                raise ChildProcessError(message)

    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancelled():
                raise InterruptedError("已取消，原有谱面已保留")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(.2, remaining))
                return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    except BaseException as error:
        try:
            kill_and_reap()
        except ChildProcessError as cleanup_error:
            raise cleanup_error from error
        raise


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


def decode(path, work, *, runner=None, cancelled=None):
    runner = run_hidden if runner is None else runner
    args = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(SR), "-t", "1201", "-f", "f32le", "pipe:1"]
    if cancelled is None:
        result = runner(args, capture_output=True, timeout=180)
    else:
        result = run_hidden_cancellable(args, cancelled, timeout=180)
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
