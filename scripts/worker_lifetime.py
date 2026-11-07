"""End a Windows audio worker if the process that launched it disappears."""
from __future__ import annotations

import os


def watch_parent():
    if os.name != 'nt':
        return
    import ctypes
    from ctypes import wintypes
    import threading

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Retain a handle to the actual process, so PID reuse cannot change the
    # object we wait for. The handle is private to this worker, not inherited.
    handle = kernel.OpenProcess(0x00100000, False, os.getppid())
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    def monitor():
        result = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        kernel.CloseHandle(handle)
        if result == 0:
            # A normal Python exit waits for model/native threads. A vanished
            # parent cannot receive output, so release this worker immediately.
            os._exit(1)

    threading.Thread(target=monitor, name='parent-lifetime', daemon=True).start()
