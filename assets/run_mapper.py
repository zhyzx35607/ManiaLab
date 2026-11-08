"""Run the optional model in its own environment, tied to the desktop's lifetime."""
import os
from pathlib import Path
import runpy
import sys


def watch_owner(pid):
    if os.name != 'nt':
        return
    import ctypes
    from ctypes import wintypes
    import threading
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    def monitor():
        result=kernel.WaitForSingleObject(handle,0xFFFFFFFF)
        kernel.CloseHandle(handle)
        if result==0:
            os._exit(1)
    threading.Thread(target=monitor,name='desktop-lifetime',daemon=True).start()


if __name__=='__main__':
    owner=int(sys.argv[1]);script=Path(sys.argv[2]).resolve()
    watch_owner(owner)
    sys.argv=[str(script),*sys.argv[3:]]
    sys.path.insert(0,str(script.parent))
    codec_flags=[arg for arg in sys.argv[1:] if arg.startswith('--mania-sketch-context=')]
    if codec_flags:
        from mapper_sketch_context import CODEC,install_runtime
        if codec_flags!=['--mania-sketch-context='+CODEC]:
            raise ValueError('Unsupported experimental sketch codec')
        sys.argv.remove(codec_flags[0])
        install_runtime()
    runpy.run_path(str(script),run_name='__main__')
