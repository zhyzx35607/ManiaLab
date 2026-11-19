"""Cooperative cancellation and shared shutdown deadlines."""
from __future__ import annotations

import threading
import time
import subprocess
import logging


class CancellationContext:
    def __init__(self, parent=None):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._deadline = None
        self._cleanup_failures = []
        self._process_cleanup_failed = False
        self._parent = parent

    def __call__(self):
        return self._event.is_set() or (self._parent is not None and self._parent())

    @property
    def deadline(self):
        with self._lock:
            deadline = self._deadline
        if self._parent is not None:
            parent_deadline = self._parent.deadline
            if deadline is None or (parent_deadline is not None and parent_deadline < deadline):
                deadline = parent_deadline
        return deadline

    def cancel(self, *, deadline=None):
        if deadline is not None:
            with self._lock:
                if self._deadline is None or deadline < self._deadline:
                    self._deadline = deadline
        self._event.set()

    @property
    def cleanup_failures(self):
        with self._lock:
            failures = tuple(self._cleanup_failures)
        if self._parent is not None:
            failures += self._parent.cleanup_failures
        return failures

    def report_cleanup_failure(self, message):
        with self._lock:
            self._cleanup_failures.append(message)
        if self._parent is not None:
            self._parent.report_cleanup_failure(message)

    @property
    def process_cleanup_failed(self):
        with self._lock:
            failed = self._process_cleanup_failed
        return failed or (self._parent is not None and self._parent.process_cleanup_failed)

    def _mark_process_cleanup_failed(self):
        with self._lock:
            self._process_cleanup_failed = True
        if self._parent is not None:
            self._parent._mark_process_cleanup_failed()

    def report_process_cleanup_failure(self, message):
        self._mark_process_cleanup_failed()
        self.report_cleanup_failure(message)


def remaining_until(deadline):
    return max(0.0, deadline - time.monotonic())


def timeout_for(cancelled, maximum):
    deadline = getattr(cancelled, "deadline", None)
    if deadline is None:
        return maximum
    return min(maximum, remaining_until(deadline))


def report_process_cleanup_failure(cancelled, message):
    logging.getLogger(__name__).warning(message)
    report = getattr(cancelled, "report_process_cleanup_failure", None)
    if report is None:
        report = getattr(cancelled, "report_cleanup_failure", None)
    if report is not None:
        report(message)


def stop_process(process, cancelled, *, terminate_timeout, kill_timeout):
    if process.poll() is not None:
        return True
    try:
        process.terminate()
    except OSError as error:
        if process.poll() is not None:
            return True
        message = "Child process could not be terminated during cleanup"
        report_process_cleanup_failure(cancelled, message)
        raise ChildProcessError(message) from error
    try:
        process.wait(timeout=timeout_for(cancelled, terminate_timeout))
        return True
    except subprocess.TimeoutExpired:
        if process.poll() is not None:
            return True
    try:
        process.kill()
    except OSError as error:
        if process.poll() is not None:
            return True
        message = "Child process could not be killed during cleanup"
        report_process_cleanup_failure(cancelled, message)
        raise ChildProcessError(message) from error
    try:
        process.wait(timeout=timeout_for(cancelled, kill_timeout))
        return True
    except subprocess.TimeoutExpired:
        stopped = process.poll() is not None
        if not stopped:
            report_process_cleanup_failure(
                cancelled,
                "Child process did not exit before its cleanup deadline",
            )
            raise ChildProcessError(
                "Child process did not exit before its cleanup deadline"
            )
        return stopped
