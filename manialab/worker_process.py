"""Start Windows workers in an owned job before their initial thread can run.

Popen still owns its root handle and pipes. The job owns only its job handle;
temporary thread/member handles are closed by the operation that opens them.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes as w
import os
import subprocess
import time

from .cancellation import poll_process_for_cleanup, report_process_cleanup_failure, timeout_for


_IS_WINDOWS = os.name == 'nt'
_CREATE_SUSPENDED = 4
_KILL_ON_JOB_CLOSE = 0x2000


class _BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
        ('flags', w.DWORD), ('minimum_working_set', ctypes.c_size_t),
        ('maximum_working_set', ctypes.c_size_t), ('active_limit', w.DWORD),
        ('affinity', ctypes.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', _BasicLimits), ('io_counters', ctypes.c_ulonglong * 6),
        ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
        ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [('times', ctypes.c_longlong * 4), ('page_faults', w.DWORD),
        ('total_processes', w.DWORD), ('active_processes', w.DWORD), ('terminated', w.DWORD)]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [('size', w.DWORD), ('usage', w.DWORD), ('thread_id', w.DWORD),
        ('owner_pid', w.DWORD), ('priority', w.LONG), ('priority_delta', w.LONG), ('flags', w.DWORD)]


class _WindowsAPI:
    def __init__(self):
        self.kernel = kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            'SetInformationJobObject': ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            'QueryInformationJobObject': ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD,
                ctypes.POINTER(w.DWORD)], w.BOOL),
            'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
            'TerminateJobObject': ([w.HANDLE, w.UINT], w.BOOL),
            'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
            'TerminateProcess': ([w.HANDLE, w.UINT], w.BOOL),
            'OpenProcess': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'IsProcessInJob': ([w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
            'CreateToolhelp32Snapshot': ([w.DWORD, w.DWORD], w.HANDLE),
            'Thread32First': ([w.HANDLE, ctypes.POINTER(_ThreadEntry)], w.BOOL),
            'Thread32Next': ([w.HANDLE, ctypes.POINTER(_ThreadEntry)], w.BOOL),
            'OpenThread': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'GetProcessIdOfThread': ([w.HANDLE], w.DWORD),
            'ResumeThread': ([w.HANDLE], w.DWORD),
            'CloseHandle': ([w.HANDLE], w.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes, function.restype = arguments, result

    @staticmethod
    def _check(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def close(self, handle):
        self._check(self.kernel.CloseHandle(handle))

    def new_job(self):
        job = self._check(self.kernel.CreateJobObjectW(None, None))
        limits = _ExtendedLimits()
        limits.basic.flags = _KILL_ON_JOB_CLOSE  # Never allow worker breakaway.
        try:
            self._check(self.kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
            return job
        except BaseException as error:
            try:
                self.close(job)
            except BaseException as close_error:
                error.add_note(f'Could not close the unconfigured worker job handle: {close_error!r}')
            raise

    def assign(self, job, root):
        self._check(self.kernel.AssignProcessToJobObject(job, root))

    def initial_thread(self, root, pid):
        snapshot = self.kernel.CreateToolhelp32Snapshot(4, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

        handles = []

        def close_owned(owned):
            errors = []
            for description, handle in owned:
                try:
                    self.close(handle)
                except BaseException as error:
                    errors.append((description, error))
            return errors

        def note_close_errors(primary, errors):
            for description, error in errors:
                primary.add_note(f'Could not close {description}: {error!r}')

        try:
            entry = _ThreadEntry(size=ctypes.sizeof(_ThreadEntry))
            found = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.owner_pid == pid:
                    handle = self._check(self.kernel.OpenThread(0x0802, False, entry.thread_id))
                    handles.append(handle)
                    if self.kernel.GetProcessIdOfThread(handle) != pid:
                        raise ChildProcessError('Suspended worker thread identity changed')
                found = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            if ctypes.get_last_error() != 18:
                raise ctypes.WinError(ctypes.get_last_error())
            if self.wait(root, 0) or len(handles) != 1:
                raise ChildProcessError('Cannot identify one suspended worker thread')
        except BaseException as error:
            close_errors = close_owned([('thread snapshot', snapshot),
                *(('worker thread', handle) for handle in handles)])
            note_close_errors(error, close_errors)
            raise

        try:
            self.close(snapshot)
        except BaseException as error:
            close_errors = close_owned([('worker thread', handle) for handle in handles])
            note_close_errors(error, close_errors)
            raise
        return handles.pop()

    def resume(self, thread):
        if self.kernel.ResumeThread(thread) != 1:
            raise ChildProcessError('Worker initial thread suspension could not be verified')

    def wait(self, handle, seconds):
        status = self.kernel.WaitForSingleObject(handle, max(0, round(seconds * 1000)))
        if status == 0:
            return True
        if status == 258:
            return False
        raise ctypes.WinError(ctypes.get_last_error())

    def terminate_job(self, job):
        self._check(self.kernel.TerminateJobObject(job, 1))

    def terminate_root(self, root):
        if not self.wait(root, 0):
            self._check(self.kernel.TerminateProcess(root, 1))

    def active_count(self, job):
        accounting = _Accounting()
        self._check(self.kernel.QueryInformationJobObject(job, 1,
            ctypes.byref(accounting), ctypes.sizeof(accounting), None))
        return accounting.active_processes

    def members(self, job):
        count = 32
        while True:
            buffer = ctypes.create_string_buffer(8 + count * ctypes.sizeof(ctypes.c_size_t))
            if self.kernel.QueryInformationJobObject(job, 3, buffer, len(buffer), None):
                break
            if ctypes.get_last_error() != 234 or count >= 4096:
                raise ctypes.WinError(ctypes.get_last_error())
            count *= 2
        _, listed = (w.DWORD * 2).from_buffer(buffer)
        identifiers = (ctypes.c_size_t * listed).from_buffer(buffer, 8)
        handles = []
        try:
            for pid in identifiers:
                handle = self.kernel.OpenProcess(0x101400, False, pid)
                if not handle:
                    if ctypes.get_last_error() == 87:
                        continue  # The former member already exited.
                    raise ctypes.WinError(ctypes.get_last_error())
                handles.append(handle)
                owned = w.BOOL()
                self._check(self.kernel.IsProcessInJob(handle, job, ctypes.byref(owned)))
                if not owned.value:
                    self.close(handle)  # Reused outside this job; never terminate it.
                    handles.pop()
            return handles
        except BaseException as error:
            close_errors = []
            for handle in handles:
                try:
                    self.close(handle)
                except OSError as close_error:
                    close_errors.append(close_error)
            if close_errors:
                error.add_note(
                    "Could not close every owned worker process handle: "
                    + "; ".join(str(close_error) for close_error in close_errors)
                )
            raise


class _WindowsJob:
    def __init__(self, cancelled):
        self.api = _WindowsAPI()
        self.handle = self.api.new_job()
        self.root = None  # Borrowed from Popen; Popen alone closes this handle.
        self.cancelled = cancelled
        self.assigned = False

    def launch(self, process):
        self.root = int(process._handle)
        self.api.assign(self.handle, self.root)
        self.assigned = True
        thread = self.api.initial_thread(self.root, process.pid)
        try:
            self.api.resume(thread)
        except BaseException as error:
            try:
                self.api.close(thread)
            except BaseException as close_error:
                error.add_note(f'Could not close the suspended worker thread handle: {close_error!r}')
            raise
        else:
            self.api.close(thread)

    def stop(self, cancelled, maximum):
        # Job membership contains later descendants too. Exact waits cover the
        # root and captured members; the count is not a historical-handle ledger.
        deadline = time.monotonic() + timeout_for(cancelled, maximum)
        handles, errors, job_terminated = [], [], False

        def close_members():
            close_errors = []
            for handle in handles:
                try:
                    self.api.close(handle)
                except OSError as error:
                    close_errors.append(error)
            return close_errors

        try:
            try:
                handles = self.api.members(self.handle)
            except OSError as error:
                errors.append(error)
            try:
                self.api.terminate_job(self.handle)
                job_terminated = True
            except OSError as error:
                errors.append(error)
            if self.root is not None:
                try:
                    if not self.assigned or not job_terminated:
                        self.api.terminate_root(self.root)  # Failed job assignment has no tree membership.
                except OSError as error:
                    errors.append(error)
                for handle in [self.root, *handles]:
                    try:
                        if not self.api.wait(handle, max(0, deadline - time.monotonic())):
                            raise TimeoutError('Worker process handle did not signal before cleanup deadline')
                    except OSError as error:
                        errors.append(error)
            try:
                while self.api.active_count(self.handle):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Worker job still has active processes at cleanup deadline')
                    time.sleep(min(.005, max(0, deadline - time.monotonic())))
            except OSError as error:
                errors.append(error)
            if errors:
                raise ChildProcessError('Owned worker process tree cleanup could not be verified') from errors[0]
        except BaseException as error:
            report_process_cleanup_failure(cancelled, 'Owned worker process tree cleanup could not be verified')
            close_errors = close_members()
            if close_errors:
                report_process_cleanup_failure(cancelled, 'Owned worker process tree handles could not be closed')
                error.add_note(
                    'Owned worker process tree handles could not be closed: '
                    + '; '.join(str(close_error) for close_error in close_errors)
                )
            raise
        close_errors = close_members()
        if close_errors:
            report_process_cleanup_failure(cancelled, 'Owned worker process tree handles could not be closed')
            raise ChildProcessError('Owned worker process tree handles could not be closed') from close_errors[0]
        return True

    def close(self):
        if self.handle is not None:
            try:
                self.api.close(self.handle)
            except OSError as error:
                report_process_cleanup_failure(self.cancelled, 'Owned worker process tree job handle could not be closed')
                raise ChildProcessError('Owned worker process tree job handle could not be closed') from error
            self.handle = None


class _WindowsWorkerPopen(subprocess.Popen):
    def __init__(self, args, *, cancelled, **kwargs):
        self._worker_job = None
        self._tree_cleanup_failed = False
        if kwargs.get('shell'):
            raise ValueError('Worker commands must execute directly')
        if cancelled():
            raise InterruptedError('已取消，原有谱面已保留')
        job = _WindowsJob(cancelled)
        try:
            kwargs['creationflags'] = kwargs.get('creationflags', 0) | subprocess.CREATE_NO_WINDOW | _CREATE_SUSPENDED
            super().__init__(args, **kwargs)
            self._worker_job = job
            job.launch(self)
        except BaseException as original:
            try:
                if getattr(self, '_child_created', False):
                    job.root = int(self._handle)
                    job.stop(cancelled, 5)
            except BaseException as cleanup_error:
                detail = f'Worker startup cleanup also failed: {cleanup_error!r}'
                if cleanup_error.__cause__ is not None:
                    detail += f' (cause: {cleanup_error.__cause__!r})'
                original.add_note(detail)
            job_close_failed = False
            try:
                job.close()
            except BaseException as close_error:
                job_close_failed = True
                report_process_cleanup_failure(cancelled, 'Owned worker process tree job handle could not be closed')
                original.add_note(f'Worker startup job handle close also failed: {close_error!r}')
            self._worker_job = job if job_close_failed else None
            for name in ('stdin', 'stdout', 'stderr'):
                stream = getattr(self, name, None)
                if stream is not None:
                    try:
                        stream.close()
                    except BaseException as close_error:
                        original.add_note(f'Worker startup {name} stream close also failed: {close_error!r}')
            raise

    def _manialab_stop_tree(self, cancelled, maximum):
        if self._tree_cleanup_failed:
            raise ChildProcessError('Owned worker process tree cleanup remains unverified')
        job = self._worker_job
        if job is None:
            try:
                return poll_process_for_cleanup(self, cancelled) is not None
            except ChildProcessError:
                self._tree_cleanup_failed = True
                raise
        stop_error = None
        try:
            result = job.stop(cancelled, maximum)
        except BaseException as error:
            self._tree_cleanup_failed = True
            stop_error = error
        try:
            job.close()
        except BaseException as close_error:
            self._tree_cleanup_failed = True
            report_process_cleanup_failure(cancelled, 'Owned worker process tree job handle could not be closed')
            if stop_error is None:
                raise
            stop_error.add_note(f'Owned worker process tree job handle close also failed: {close_error!r}')
        else:
            self._worker_job = None
        if stop_error is not None:
            raise stop_error
        return result

    def __del__(self):
        try:
            job = getattr(self, '_worker_job', None)
            if job is not None:
                try:
                    job.close()  # Kill-on-close is a fallback, not a verified wait result.
                except OSError:
                    pass  # close() already records the unresolved cleanup failure.
                finally:
                    self._worker_job = None
        finally:
            # A failed suspended launch can be finalized after Popen's Handle
            # object has already closed. Its base destructor would poll that
            # invalid handle and emit an unraisable WinError 6.
            handle = getattr(self, '_handle', None)
            if handle is None or not getattr(handle, 'closed', False):
                super().__del__()


def start_worker(args, *, cancelled=lambda: False, **kwargs):
    """Preserve Popen arguments; Windows workers stay in one non-breakaway job."""
    if not _IS_WINDOWS:
        return subprocess.Popen(args, **kwargs)
    return _WindowsWorkerPopen(args, cancelled=cancelled, **kwargs)
