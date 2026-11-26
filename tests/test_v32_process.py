from __future__ import annotations

from itertools import repeat
from pathlib import Path

import pytest

from manialab import v32_process


class FakeProcess:
    pid = 12345

    def __init__(self, polls=(None, 0), returncode=0):
        self.polls = iter(polls)
        self.returncode = returncode

    def poll(self):
        try:
            value = next(self.polls)
        except StopIteration:
            value = self.returncode
        if value is not None:
            self.returncode = value
        return value


def test_owned_command_reports_elapsed_and_keeps_worker_log_open(tmp_path, monkeypatch):
    process = FakeProcess()
    captured = {}

    def start(command, **kwargs):
        captured["stdout"] = kwargs["stdout"]
        captured["stdout"].write("worker output\n")
        captured["stdout"].flush()
        return process

    monkeypatch.setattr(v32_process, "start_worker", start)
    monkeypatch.setattr(v32_process, "stop_process", lambda *_args, **_kwargs: None)
    progress = []
    log = tmp_path / "worker.log"

    result = v32_process.run_owned_command(
        ["worker"],
        log_path=log,
        cwd=tmp_path,
        env={},
        progress=lambda percent, message: progress.append((percent, message)),
        cancelled=lambda: False,
        message="fixture",
        timeout_seconds=10,
    )

    assert log.read_text(encoding="utf-8") == "worker output\n"
    assert captured["stdout"].closed
    assert result["elapsed_seconds"] >= 0
    assert result["peak_rss_bytes"] is None
    assert progress


def test_owned_command_cancellation_reaps_the_worker(tmp_path, monkeypatch):
    process = FakeProcess(polls=repeat(None))
    stopped = []

    def stop(child, *_args, **_kwargs):
        stopped.append(child)
        child.returncode = 1

    monkeypatch.setattr(v32_process, "start_worker", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(v32_process, "stop_process", stop)
    checks = iter((False, False, True))

    with pytest.raises(InterruptedError, match="已取消"):
        v32_process.run_owned_command(
            ["worker"],
            log_path=tmp_path / "worker.log",
            cwd=tmp_path,
            env={},
            progress=lambda *_: None,
            cancelled=lambda: next(checks),
            message="fixture",
            timeout_seconds=10,
        )

    assert stopped == [process]
