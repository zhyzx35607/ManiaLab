"""Owned child-process execution with cancellation and optional RSS sampling."""
from __future__ import annotations

from pathlib import Path
import subprocess
import time

from .cancellation import stop_process
from .task_errors import UserVisibleTaskError, UserVisibleTimeoutError
from .worker_process import start_worker


_RSS_MONITOR = (
    "import pathlib,psutil,sys,time\n"
    "pid,stop=int(sys.argv[1]),pathlib.Path(sys.argv[2]);peak=0\n"
    "while True:\n"
    " try:\n"
    "  root=psutil.Process(pid);nodes=[root,*root.children(recursive=True)];"
    "current=sum((node.memory_info().rss for node in nodes),0);peak=max(peak,current)\n"
    " except psutil.Error: pass\n"
    " if stop.exists(): break\n"
    " time.sleep(.1)\n"
    "print(peak)\n"
)


def run_owned_command(
    command: list[str],
    *,
    log_path: str | Path,
    cwd: str | Path,
    env: dict[str, str],
    progress,
    cancelled,
    message: str,
    timeout_seconds: float,
    monitor_python: str | Path | None = None,
    global_failure: bool = False,
) -> dict:
    if cancelled():
        raise InterruptedError("已取消，原有谱面已保留")
    log_path = Path(log_path)
    cwd = Path(cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path = log_path.with_suffix(log_path.suffix + ".rss-stop")
    stop_path.unlink(missing_ok=True)
    monitor = None
    peak_rss_bytes = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = start_worker(
            command,
            cancelled=cancelled,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if monitor_python is not None and Path(monitor_python).is_file():
            try:
                monitor = subprocess.Popen(
                    [
                        str(monitor_python), "-c", _RSS_MONITOR,
                        str(process.pid), str(stop_path),
                    ],
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError:
                monitor = None
        cleanup_attempted = False
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if cancelled():
                    cleanup_attempted = True
                    stop_process(
                        process, cancelled, terminate_timeout=15, kill_timeout=15,
                    )
                    raise InterruptedError("已取消，原有谱面已保留")
                if elapsed > timeout_seconds:
                    cleanup_attempted = True
                    stop_process(
                        process, cancelled, terminate_timeout=15, kill_timeout=15,
                    )
                    raise UserVisibleTimeoutError(
                        f"{message}超过 {timeout_seconds:g} 秒，已停止",
                        log_path=log_path,
                        global_failure=global_failure,
                    )
                progress(
                    min(89, 10 + int(elapsed / max(1, timeout_seconds) * 79)),
                    f"{message} · 已运行 {int(elapsed)} 秒",
                )
                time.sleep(0.4)
            if process.returncode:
                raise UserVisibleTaskError(
                    f"{message}失败（退出码 {process.returncode}）",
                    log_path=log_path,
                    global_failure=global_failure,
                )
        finally:
            try:
                if not cleanup_attempted:
                    stop_process(
                        process, cancelled, terminate_timeout=15, kill_timeout=15,
                    )
            finally:
                if monitor is not None:
                    stop_path.touch()
                    try:
                        stdout, _ = monitor.communicate(timeout=15)
                        if monitor.returncode == 0:
                            peak_rss_bytes = int(stdout.strip() or "0")
                    except (OSError, subprocess.TimeoutExpired, ValueError):
                        try:
                            monitor.kill()
                            monitor.communicate(timeout=5)
                        except (OSError, subprocess.TimeoutExpired):
                            pass
                stop_path.unlink(missing_ok=True)
    return {
        "log": log_path.read_text(encoding="utf-8", errors="replace"),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_mib": None if peak_rss_bytes is None else peak_rss_bytes / 2**20,
    }
