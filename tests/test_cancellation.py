import subprocess
import time

import pytest

from manialab import cancellation


def test_cancellation_deadline_is_shared_and_can_only_shrink(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    context = cancellation.CancellationContext()

    assert cancellation.timeout_for(context, 8) == 8
    context.cancel(deadline=105.0)
    context.cancel(deadline=106.0)
    assert context()
    assert context.deadline == 105.0
    assert cancellation.timeout_for(context, 8) == 5

    now[0] = 106.0
    assert cancellation.timeout_for(context, 8) == 0


def test_child_cleanup_failure_is_reported_by_parent():
    parent = cancellation.CancellationContext()
    child = cancellation.CancellationContext(parent=parent)

    child.report_cleanup_failure("synthetic cleanup failure")

    assert parent.cleanup_failures == ("synthetic cleanup failure",)
    assert child.cleanup_failures == ("synthetic cleanup failure",)


def test_process_escalation_does_not_reset_shutdown_budget(monkeypatch, caplog):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    context = cancellation.CancellationContext()
    context.cancel(deadline=103.0)

    class Process:
        def __init__(self):
            self.alive = True
            self.wait_timeouts = []
            self.killed = False

        def poll(self):
            return None if self.alive else -9

        def terminate(self):
            pass

        def wait(self, timeout):
            self.wait_timeouts.append(timeout)
            now[0] += timeout
            if self.alive:
                raise subprocess.TimeoutExpired("worker", timeout)

        def kill(self):
            self.killed = True

    process = Process()
    with pytest.raises(ChildProcessError, match="cleanup deadline"):
        cancellation.stop_process(
            process,
            context,
            terminate_timeout=10,
            kill_timeout=10,
        )

    assert process.killed
    assert process.wait_timeouts == [3.0, 0.0]
    assert now[0] == 103.0
    assert "did not exit before its cleanup deadline" in caplog.text
    assert context.cleanup_failures


def test_process_signal_failure_is_recorded_when_child_remains_alive(caplog):
    context = cancellation.CancellationContext()

    class Process:
        def poll(self):
            return None

        def terminate(self):
            raise PermissionError("synthetic access denied")

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("worker", timeout)

        def kill(self):
            raise PermissionError("synthetic kill denied")

    with pytest.raises(ChildProcessError, match="could not be killed"):
        cancellation.stop_process(
            Process(),
            context,
            terminate_timeout=3,
            kill_timeout=3,
        )

    assert context.cleanup_failures == (
        "Child process could not be killed during cleanup",
    )
    assert "could not be killed" in caplog.text


def test_process_kill_escalation_runs_after_terminate_failure():
    context = cancellation.CancellationContext()

    class Process:
        def __init__(self):
            self.alive = True
            self.kill_calls = 0

        def poll(self):
            return None if self.alive else -9

        def terminate(self):
            raise PermissionError("synthetic terminate denial")

        def wait(self, timeout):
            if self.alive:
                raise subprocess.TimeoutExpired("worker", timeout)
            return -9

        def kill(self):
            self.kill_calls += 1
            self.alive = False

    process = Process()

    assert cancellation.stop_process(
        process,
        context,
        terminate_timeout=0,
        kill_timeout=1,
    )

    assert process.kill_calls == 1
    assert not process.alive
    assert not context.cleanup_failures
