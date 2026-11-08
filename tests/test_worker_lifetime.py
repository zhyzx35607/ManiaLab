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
@pytest.mark.parametrize('worker_kind',['audio','mapper'])
def test_worker_exits_after_its_parent_is_terminated(tmp_path,worker_kind):
    root=Path(__file__).resolve().parents[1]
    # Avoid the venv redirector: the test needs the actual parent PID, as in
    # the packaged application and its one-directory frozen audio worker.
    executable=getattr(sys,'_base_executable',sys.executable)
    marker=tmp_path/'worker.pid'
    child=("from scripts.worker_lifetime import watch_parent; "
           "import os,time; from pathlib import Path; "
           "watch_parent(); marker=Path("+repr(str(marker))+"); "
           "pending=marker.with_suffix('.pending'); pending.write_text(str(os.getpid())); pending.replace(marker); time.sleep(120)")
    parent=("import subprocess,time; subprocess.Popen(["+repr(executable)+",'-c',"+repr(child)+"]); time.sleep(120)")
    if worker_kind=='mapper':
        script=tmp_path/'inference.py'
        script.write_text("import os,time;from pathlib import Path;marker=Path("+repr(str(marker))+");"
            "pending=marker.with_suffix('.pending');pending.write_text(str(os.getpid()));pending.replace(marker);time.sleep(120)")
        # Exercise the venv redirector: the runner must watch the explicit
        # desktop owner, rather than its intermediate Python parent.
        parent=("import subprocess,os,time;subprocess.Popen(["+repr(sys.executable)+","+
            repr(str(root/'assets/run_mapper.py'))+",str(os.getpid()),"+repr(str(script))+"]);time.sleep(120)")
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
