"""Shared FFmpeg-backed audio decoding without model-runtime dependencies."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
import time

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile

from .audio_config import SR
from .cancellation import report_process_cleanup_failure, timeout_for

_PREVIEW_REPLACE_RETRY_DELAYS = (.01, .02, .04, .08, .16)


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
        cleanup_failures = []
        query_errors = []
        signal_errors = []
        cleanup_budget = timeout_for(cancelled, 5)
        cleanup_deadline = time.monotonic() + cleanup_budget
        reap_reserve = min(.2, cleanup_budget * .1)
        reader_join_reserve = min(.1, cleanup_budget * .05)

        def remaining_cleanup_time():
            return min(max(0.0, cleanup_deadline - time.monotonic()), timeout_for(cancelled, 5))

        def remember_cleanup_failure(message, error=None):
            cleanup_failures.append((message, error))

        def close_pipes():
            if active_readers():
                return False
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = getattr(process, stream_name, None)
                if stream is None or getattr(stream, "closed", False):
                    continue
                try:
                    stream.close()
                except OSError as error:
                    remember_cleanup_failure(
                        "FFmpeg process pipes could not be closed during cleanup", error
                    )
            return True

        def active_readers():
            return [
                reader for name in ("stdout", "stderr")
                if (reader := getattr(process, f"{name}_thread", None)) is not None
                and reader.is_alive()
            ]

        def join_readers():
            readers = active_readers()
            join_deadline = min(
                cleanup_deadline,
                time.monotonic() + min(reader_join_reserve, remaining_cleanup_time()),
            )
            for index, reader in enumerate(readers):
                remaining = min(
                    max(0.0, join_deadline - time.monotonic()),
                    remaining_cleanup_time(),
                )
                active_count = sum(candidate.is_alive() for candidate in readers[index:])
                if remaining <= 0 or active_count <= 0:
                    break
                reader.join(timeout=remaining / active_count)
            return not active_readers()

        def poll():
            try:
                return process.poll()
            except OSError as error:
                query_errors.append(error)
                return None

        running = poll() is None
        if running:
            try:
                process.kill()
            except OSError as error:
                if poll() is None:
                    signal_errors.append(("FFmpeg process could not be killed during cleanup", error))

        def remember_unresolved_signal_errors():
            for message, error in signal_errors:
                remember_cleanup_failure(message, error)

        communicate_timeout = max(
            0.0, remaining_cleanup_time() - reap_reserve - reader_join_reserve
        )
        try:
            process.communicate(timeout=communicate_timeout)
        except subprocess.TimeoutExpired:
            running = poll() is None
            if running:
                try:
                    process.wait(timeout=max(0.0, remaining_cleanup_time() - reader_join_reserve))
                except subprocess.TimeoutExpired as wait_error:
                    if poll() is None:
                        remember_unresolved_signal_errors()
                        remember_cleanup_failure(
                            "FFmpeg process did not exit before its cleanup deadline", wait_error
                        )
                except OSError as wait_error:
                    if poll() is None:
                        remember_unresolved_signal_errors()
                        remember_cleanup_failure(
                            "FFmpeg process could not be reaped during cleanup", wait_error
                        )
            join_readers()
            if not close_pipes():
                remember_cleanup_failure(
                    "FFmpeg process pipes remain open while communication threads are active"
                )
        except OSError as error:
            try:
                process.wait(timeout=max(0.0, remaining_cleanup_time() - reader_join_reserve))
            except subprocess.TimeoutExpired as wait_error:
                if poll() is None:
                    remember_unresolved_signal_errors()
                    remember_cleanup_failure(
                        "FFmpeg process did not exit before its cleanup deadline", wait_error
                    )
                    remember_cleanup_failure("FFmpeg cleanup communicate failed", error)
            except OSError as wait_error:
                if poll() is None:
                    remember_unresolved_signal_errors()
                    remember_cleanup_failure("FFmpeg process could not be reaped during cleanup", wait_error)
                    remember_cleanup_failure("FFmpeg cleanup communicate failed", error)
            finally:
                join_readers()
                if not close_pipes():
                    remember_cleanup_failure(
                        "FFmpeg process pipes remain open while communication threads are active"
                    )

        if cleanup_failures:
            message, error = cleanup_failures[0]
            cleanup_error = ChildProcessError(message)
            report_process_cleanup_failure(cancelled, message)
            if error is not None:
                cleanup_error.add_note(f"Cleanup detail: {error!r}")
            for detail, extra_error in cleanup_failures[1:]:
                cleanup_error.add_note(
                    f"{detail}: {extra_error!r}" if extra_error is not None else detail
                )
            for query_error in query_errors:
                cleanup_error.add_note(f"Process state query detail: {query_error!r}")
            raise cleanup_error

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
            if cleanup_error.__cause__ is not None:
                cleanup_error.add_note(f"Cleanup detail: {cleanup_error.__cause__!r}")
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


def _replace_preview(source, destination):
    if os.name == "nt":
        for delay in _PREVIEW_REPLACE_RETRY_DELAYS:
            try:
                os.replace(source, destination)
                return
            except PermissionError as error:
                # A playback response can briefly keep the old preview open.
                if getattr(error, "winerror", None) not in (5, 32):
                    raise
                time.sleep(delay)
    os.replace(source, destination)


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
    preview_path = Path(work) / "preview.wav"
    with tempfile.NamedTemporaryFile(prefix=".preview-", suffix=".wav", dir=preview_path.parent,
                                     delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        wavfile.write(temporary_path, SR, (np.clip(samples, -1, 1) * 32767).astype(np.int16))
        _replace_preview(temporary_path, preview_path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return samples
