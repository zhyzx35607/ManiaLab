"""Owned Windows worker jobs retain subprocess behavior and verified tree cleanup."""
import ctypes
from ctypes import wintypes
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from manialab import worker_process
from manialab.cancellation import CancellationContext, stop_process


WINDOWS = pytest.mark.skipif(os.name != 'nt', reason='Windows owned worker jobs')


def _await_pid(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            return int(path.read_text())
        except (OSError, ValueError):
            time.sleep(.005)
    raise AssertionError('owned synthetic payload did not start')


def _kernel():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.DuplicateHandle.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    return kernel


def _finish_owned(process, handle=None):
    try:
        try:
            stop_process(process, CancellationContext(), terminate_timeout=2, kill_timeout=3)
        except ChildProcessError:
            if not process._tree_cleanup_failed:
                raise
    finally:
        kernel = _kernel()
        try:
            if handle:
                try:
                    if kernel.WaitForSingleObject(handle, 0) != 0:
                        assert kernel.TerminateProcess(handle, 1), ctypes.get_last_error()
                    assert kernel.WaitForSingleObject(handle, 5000) == 0
                finally:
                    kernel.CloseHandle(handle)
        finally:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_non_windows_worker_factory_preserves_popen_arguments(monkeypatch):
    sentinel, calls = object(), []
    monkeypatch.setattr(worker_process, '_IS_WINDOWS', False)
    monkeypatch.setattr(worker_process.subprocess, 'Popen',
        lambda *args, **kwargs: calls.append((args, kwargs)) or sentinel)
    assert worker_process.start_worker(['synthetic', 'worker'], cancelled=lambda: False,
        cwd='synthetic-work', env={'MARKER': 'value'}, stdout=subprocess.PIPE) is sentinel
    assert calls == [((['synthetic', 'worker'],),
        dict(cwd='synthetic-work', env={'MARKER': 'value'}, stdout=subprocess.PIPE))]


def test_worker_destructor_skips_base_poll_when_root_handle_is_already_closed(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess.Popen, '__del__', lambda process: calls.append(process))
    process = object.__new__(worker_process._WindowsWorkerPopen)
    process._worker_job = None
    process._handle = SimpleNamespace(closed=True)
    worker_process._WindowsWorkerPopen.__del__(process)
    assert calls == []


@WINDOWS
@pytest.mark.parametrize('kind', ['direct', 'venv', 'root-exited'])
def test_owned_tree_reaps_payload_even_when_venv_or_root_exits(tmp_path, kind):
    base = getattr(sys, '_base_executable', sys.executable)
    marker = tmp_path / 'payload.pid'
    sleeper = ('import os,sys,time; from pathlib import Path; '
        'p=Path(sys.argv[1]); q=p.with_suffix(".pending"); '
        'q.write_text(str(os.getpid())); q.replace(p); time.sleep(15)')
    executable = sys.executable if kind == 'venv' else base
    program = sleeper if kind != 'root-exited' else (
        f'import subprocess,sys; subprocess.Popen([{base!r},"-c",{sleeper!r},sys.argv[1]])')
    context = CancellationContext()
    process = worker_process.start_worker([executable, '-c', program, str(marker)],
        cancelled=context, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    handle = None
    try:
        payload_pid = _await_pid(marker)
        kernel = _kernel()
        handle = kernel.OpenProcess(0x101401, False, payload_pid)
        assert handle, ctypes.get_last_error()
        if kind == 'root-exited':
            assert process.wait(timeout=5) == 0
            assert kernel.WaitForSingleObject(handle, 0) == 258
        if kind == 'venv' and Path(sys.executable) != Path(base):
            assert process.pid != payload_pid
        context.cancel()
        assert stop_process(process, context, terminate_timeout=2, kill_timeout=3)
        assert kernel.WaitForSingleObject(handle, 0) == 0
        assert process.poll() is not None
        assert not context.process_cleanup_failed
        assert stop_process(process, context, terminate_timeout=2, kill_timeout=3)
    finally:
        _finish_owned(process, handle)


@WINDOWS
def test_worker_preserves_stdio_environment_unicode_cwd_hidden_and_timeout(tmp_path):
    folder = tmp_path / '含空格 目录'
    folder.mkdir()
    program = ('import ctypes,json,os,sys; '
        'print(json.dumps([os.getcwd(),os.environ["MANIALAB_PROCESS_PROBE"],'
        'ctypes.windll.kernel32.GetConsoleWindow()])); sys.stderr.write("stderr marker")')
    context = CancellationContext()
    process = worker_process.start_worker([sys.executable, '-c', program], cancelled=context,
        cwd=folder, env=os.environ | {'MANIALAB_PROCESS_PROBE': 'env marker'},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output, error = process.communicate(timeout=5)
        assert json.loads(output) == [str(folder), 'env marker', 0]
        assert error == b'stderr marker' and process.returncode == 0
        assert stop_process(process, context, terminate_timeout=2, kill_timeout=3)
    finally:
        _finish_owned(process)

    with (folder / 'worker.log').open('wb') as log:
        process = worker_process.start_worker([getattr(sys, '_base_executable', sys.executable),
            '-c', 'import time; time.sleep(15)'], cancelled=context,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=.01)
            assert stop_process(process, context, terminate_timeout=2, kill_timeout=3)
            assert not log.closed
        finally:
            _finish_owned(process)


@WINDOWS
@pytest.mark.parametrize('failure', ['assign', 'initial_thread', 'resume'])
def test_startup_verification_failure_never_executes_payload_or_leaks_root(tmp_path, monkeypatch, failure):
    kernel = _kernel()
    captured = []
    execute = worker_process._WindowsWorkerPopen._execute_child

    def capture(self, *args, **kwargs):
        execute(self, *args, **kwargs)
        duplicate = wintypes.HANDLE()
        current = kernel.GetCurrentProcess()
        assert kernel.DuplicateHandle(current, int(self._handle), current,
            ctypes.byref(duplicate), 0, False, 2)
        captured.append(duplicate)

    def denied(*args, **kwargs):
        raise PermissionError('synthetic startup verification denied')

    monkeypatch.setattr(worker_process._WindowsWorkerPopen, '_execute_child', capture)
    monkeypatch.setattr(worker_process._WindowsAPI, failure, denied)
    marker = tmp_path / 'must-not-run'
    context = CancellationContext()
    try:
        with pytest.raises(PermissionError, match='startup verification denied'):
            worker_process.start_worker([getattr(sys, '_base_executable', sys.executable), '-c',
                'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("ran")', str(marker)],
                cancelled=context, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert not marker.exists() and len(captured) == 1
        assert kernel.WaitForSingleObject(captured[0], 0) == 0
        assert not context.process_cleanup_failed
    finally:
        for handle in captured:
            if kernel.WaitForSingleObject(handle, 0) != 0:
                kernel.TerminateProcess(handle, 1)
                assert kernel.WaitForSingleObject(handle, 5000) == 0
            kernel.CloseHandle(handle)


@WINDOWS
@pytest.mark.parametrize('failure', ['members', 'terminate_job', 'wait', 'active_count'])
def test_cleanup_verification_failure_is_reported_to_process_gate(tmp_path, monkeypatch, failure):
    context = CancellationContext()
    process = worker_process.start_worker([getattr(sys, '_base_executable', sys.executable),
        '-c', 'import time; time.sleep(15)'], cancelled=context,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def denied(*args, **kwargs):
        raise PermissionError('synthetic cleanup verification denied')

    with monkeypatch.context() as fault:
        fault.setattr(worker_process._WindowsAPI, failure, denied)
        with pytest.raises(ChildProcessError, match='worker process tree'):
            stop_process(process, context, terminate_timeout=.2, kill_timeout=.2)
    try:
        assert context.process_cleanup_failed
        assert context.cleanup_failures
    finally:
        _finish_owned(process)


@WINDOWS
def test_worker_job_and_startup_failure_handles_do_not_grow(monkeypatch):
    kernel = _kernel()

    def count():
        gc.collect()
        value = wintypes.DWORD()
        assert kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(value))
        return value.value

    def once():
        context = CancellationContext()
        process = worker_process.start_worker([getattr(sys, '_base_executable', sys.executable), '-c', 'pass'],
            cancelled=context, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            process.communicate(timeout=5)
            stop_process(process, context, terminate_timeout=2, kill_timeout=3)
        finally:
            _finish_owned(process)

    once()
    before = count()
    for _ in range(4):
        once()
    assert count() == before
    with monkeypatch.context() as fault:
        fault.setattr(worker_process._WindowsAPI, 'initial_thread',
            lambda *args: (_ for _ in ()).throw(PermissionError('synthetic initial thread denied')))
        for _ in range(4):
            with pytest.raises(PermissionError):
                once()
    assert count() == before


def test_owned_tree_cleanup_shares_one_deadline_across_member_waits(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(worker_process.time, 'monotonic', lambda: now[0])
    context = CancellationContext()
    context.cancel(deadline=100.05)
    waits, closed = [], []

    class API:
        def members(self, job):
            now[0] += .01
            return [3, 4]

        def terminate_job(self, job):
            now[0] += .01

        def wait(self, handle, seconds):
            waits.append((handle, seconds))
            now[0] += seconds
            return False

        def active_count(self, job):
            return 0

        def close(self, handle):
            closed.append(handle)

    job = object.__new__(worker_process._WindowsJob)
    job.api, job.handle, job.root, job.assigned = API(), 1, 2, True
    job.cancelled = context
    with pytest.raises(ChildProcessError, match='worker process tree'):
        job.stop(context, 10)
    assert [handle for handle, _ in waits] == [2, 3, 4]
    assert [seconds for _, seconds in waits] == pytest.approx([.03, 0, 0])
    assert closed == [3, 4]
    assert now[0] == pytest.approx(100.05)
    assert context.process_cleanup_failed


def test_member_handle_close_failure_does_not_skip_other_handles_or_hide_gate():
    context = CancellationContext()
    closed = []

    class API:
        def members(self, job):
            return [3, 4]

        def terminate_job(self, job):
            pass

        def wait(self, handle, seconds):
            return True

        def active_count(self, job):
            return 0

        def close(self, handle):
            closed.append(handle)
            if handle == 3:
                raise OSError('synthetic member handle close failed')

    job = object.__new__(worker_process._WindowsJob)
    job.api, job.handle, job.root, job.assigned = API(), 1, 2, True
    job.cancelled = context
    with pytest.raises(ChildProcessError, match='handles could not be closed'):
        job.stop(context, 1)
    assert closed == [3, 4]
    assert context.process_cleanup_failed


@WINDOWS
def test_startup_cleanup_that_cannot_be_verified_sets_process_gate(tmp_path, monkeypatch):
    kernel = _kernel()
    handles = []
    execute = worker_process._WindowsWorkerPopen._execute_child

    def capture(self, *args, **kwargs):
        execute(self, *args, **kwargs)
        duplicate = wintypes.HANDLE()
        current = kernel.GetCurrentProcess()
        assert kernel.DuplicateHandle(current, int(self._handle), current,
            ctypes.byref(duplicate), 0, False, 2)
        handles.append(duplicate)

    monkeypatch.setattr(worker_process._WindowsWorkerPopen, '_execute_child', capture)
    monkeypatch.setattr(worker_process._WindowsAPI, 'initial_thread',
        lambda *args: (_ for _ in ()).throw(PermissionError('synthetic initial thread denied')))
    monkeypatch.setattr(worker_process._WindowsAPI, 'wait', lambda *args: False)
    marker = tmp_path / 'must-not-run'
    context = CancellationContext()
    try:
        with pytest.raises(ChildProcessError, match='worker process tree'):
            worker_process.start_worker([getattr(sys, '_base_executable', sys.executable), '-c',
                'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("ran")', str(marker)],
                cancelled=context, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert not marker.exists() and context.process_cleanup_failed
        assert len(handles) == 1 and kernel.WaitForSingleObject(handles[0], 5000) == 0
    finally:
        for handle in handles:
            try:
                if kernel.WaitForSingleObject(handle, 0) != 0:
                    assert kernel.TerminateProcess(handle, 1), ctypes.get_last_error()
                    assert kernel.WaitForSingleObject(handle, 5000) == 0
            finally:
                kernel.CloseHandle(handle)


@WINDOWS
def test_snapshot_close_failure_does_not_lose_selected_initial_thread_handle():
    closed = []

    class Kernel:
        def CreateToolhelp32Snapshot(self, *args):
            return 101

        def Thread32First(self, snapshot, pointer):
            entry = ctypes.cast(pointer, ctypes.POINTER(worker_process._ThreadEntry)).contents
            entry.owner_pid, entry.thread_id = 42, 201
            return True

        def Thread32Next(self, *args):
            ctypes.set_last_error(18)
            return False

        def OpenThread(self, *args):
            return 201

        def GetProcessIdOfThread(self, *args):
            return 42

        def WaitForSingleObject(self, *args):
            return 258

        def CloseHandle(self, handle):
            closed.append(handle)
            if handle == 101:
                raise OSError('synthetic snapshot close failed')
            return True

    api = object.__new__(worker_process._WindowsAPI)
    api.kernel = Kernel()
    with pytest.raises(OSError, match='snapshot close failed'):
        api.initial_thread(999, 42)
    assert closed == [101, 201]


@WINDOWS
def test_member_lookup_failure_attempts_to_close_every_already_owned_handle():
    closed = []

    class Kernel:
        def QueryInformationJobObject(self, job, kind, buffer, length, returned):
            headers = (wintypes.DWORD * 2).from_buffer(buffer)
            headers[:] = [3, 3]
            identifiers = (ctypes.c_size_t * 3).from_buffer(buffer, 8)
            identifiers[:] = [1, 2, 3]
            return True

        def OpenProcess(self, access, inherit, pid):
            if pid == 3:
                raise PermissionError('synthetic member lookup denied')
            return pid + 200

        def IsProcessInJob(self, handle, job, pointer):
            ctypes.cast(pointer, ctypes.POINTER(wintypes.BOOL)).contents.value = True
            return True

        def CloseHandle(self, handle):
            closed.append(handle)
            if handle == 201:
                raise OSError('synthetic member close failed')
            return True

    api = object.__new__(worker_process._WindowsAPI)
    api.kernel = Kernel()
    with pytest.raises(OSError, match='member close failed'):
        api.members(999)
    assert closed == [201, 202]


@WINDOWS
def test_stale_member_pid_outside_owned_job_is_never_terminated(monkeypatch):
    base = getattr(sys, '_base_executable', sys.executable)
    foreign = subprocess.Popen([base, '-c', 'import time; time.sleep(15)'],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW)
    context = CancellationContext()
    owned = None
    try:
        owned = worker_process.start_worker([base, '-c', 'import time; time.sleep(15)'],
            cancelled=context, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        kernel = owned._worker_job.api.kernel
        query = kernel.QueryInformationJobObject

        def stale_member(job, kind, buffer, length, returned):
            if kind != 3:
                return query(job, kind, buffer, length, returned)
            (wintypes.DWORD * 2).from_buffer(buffer)[:] = [1, 1]
            (ctypes.c_size_t * 1).from_buffer(buffer, 8)[0] = foreign.pid
            return True

        with monkeypatch.context() as fault:
            fault.setattr(kernel, 'QueryInformationJobObject', stale_member)
            assert stop_process(owned, context, terminate_timeout=2, kill_timeout=3)
        assert foreign.poll() is None
        assert not context.process_cleanup_failed
    finally:
        try:
            if owned is not None:
                _finish_owned(owned)
        finally:
            if foreign.poll() is None:
                foreign.kill()
            foreign.wait(timeout=5)
