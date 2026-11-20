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
        if self._parent is not None:
            self._parent.report_cleanup_failure(message)
            return
        with self._lock:
            self._cleanup_failures.append(message)

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


def poll_process_for_cleanup(process, cancelled, *, report=True):
    try:
        return process.poll()
    except OSError as error:
        message = "Child process state could not be checked during cleanup"
        if report:
            report_process_cleanup_failure(cancelled, message)
        raise ChildProcessError(message) from error


def wait_process_for_cleanup(process, cancelled, *, timeout, report=True):
    try:
        return process.wait(timeout=timeout)
    except OSError as error:
        message = "Child process could not be reaped during cleanup"
        if report:
            report_process_cleanup_failure(cancelled, message)
        raise ChildProcessError(message) from error


def stop_process(process, cancelled, *, terminate_timeout, kill_timeout):
    owned_tree = getattr(process, '_manialab_stop_tree', None)
    if owned_tree is not None:
        return owned_tree(cancelled, terminate_timeout + kill_timeout)

    query_error = None

    def poll():
        nonlocal query_error
        try:
            return poll_process_for_cleanup(process, cancelled, report=False)
        except ChildProcessError as error:
            if query_error is None:
                query_error = error
            else:
                query_error.add_note(f"Additional process state query failure: {error.__cause__!r}")
            return None

    def wait(timeout):
        nonlocal query_error
        try:
            wait_process_for_cleanup(process, cancelled, timeout=timeout, report=False)
            return True
        except subprocess.TimeoutExpired:
            return False
        except ChildProcessError as error:
            if query_error is None:
                query_error = error
            else:
                query_error.add_note(f"Additional process wait failure: {error.__cause__!r}")
            return False

    def finish(stopped):
        if stopped:
            return True
        deadline_message = "Child process did not exit before its cleanup deadline"
        if query_error is not None:
            query_error.add_note(deadline_message)
            if signal_errors:
                query_error.add_note(
                    "; ".join(message for message, _ in signal_errors)
                )
            report_process_cleanup_failure(cancelled, str(query_error))
            raise query_error
        if signal_errors:
            message, error = signal_errors[0]
            report_process_cleanup_failure(cancelled, message)
            failure = ChildProcessError(message)
            raise failure from error
        report_process_cleanup_failure(cancelled, deadline_message)
        raise ChildProcessError(deadline_message)

    if poll() is not None:
        return True

    signal_errors = []
    try:
        process.terminate()
    except OSError as error:
        message = "Child process could not be terminated during cleanup"
        if poll() is None:
            if query_error is None:
                report_process_cleanup_failure(cancelled, message)
                raise ChildProcessError(message) from error
            signal_errors.append((message, error))
    if wait(timeout_for(cancelled, terminate_timeout)):
        return finish(True)
    if poll() is not None:
        return finish(True)

    try:
        process.kill()
    except OSError as error:
        message = "Child process could not be killed during cleanup"
        if poll() is None:
            if query_error is None:
                report_process_cleanup_failure(cancelled, message)
                raise ChildProcessError(message) from error
            signal_errors.append((message, error))
    stopped = wait(timeout_for(cancelled, kill_timeout))
    if not stopped:
        stopped = poll() is not None

    return finish(stopped)
