"""A worker must stop after abrupt parent termination, even during model work."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows process lifetime')
def test_worker_exits_after_its_parent_is_terminated(tmp_path):
    root=Path(__file__).resolve().parents[1]
    # Avoid the venv redirector: the test needs the actual parent PID, as in
    # the packaged application and its one-directory frozen audio worker.
    executable=getattr(sys,'_base_executable',sys.executable)
    marker=tmp_path/'worker.pid'
    child=("from scripts.worker_lifetime import watch_parent; "
           "import os,time; from pathlib import Path; "
           "watch_parent(); Path("+repr(str(marker))+").write_text(str(os.getpid())); time.sleep(120)")
    parent=("import subprocess,time; subprocess.Popen(["+repr(executable)+",'-c',"+repr(child)+"]); time.sleep(120)")
    process=subprocess.Popen([executable,'-c',parent],cwd=root,creationflags=subprocess.CREATE_NO_WINDOW)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=None
    try:
        deadline=time.monotonic()+10
        while not marker.exists():
            assert process.poll() is None
            assert time.monotonic()<deadline,'Worker failed to become ready'
            time.sleep(.05)
        handle=kernel.OpenProcess(0x00100001,False,int(marker.read_text()))
        assert handle,ctypes.get_last_error()
        assert kernel.WaitForSingleObject(handle,0)==258
        process.terminate();process.wait(timeout=5)
        assert kernel.WaitForSingleObject(handle,5000)==0,'Worker outlived its parent'
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=5)
        if handle:
            if kernel.WaitForSingleObject(handle,0)==258:kernel.TerminateProcess(handle,1)
            kernel.CloseHandle(handle)
