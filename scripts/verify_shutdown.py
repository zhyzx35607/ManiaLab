"""Verify closing the actual application also stops its active audio worker."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_portable import synthetic


def run(executable, work):
    work=work.resolve();work.mkdir(parents=True,exist_ok=True)
    data=work/'shutdown-data';data.mkdir(exist_ok=True)
    source=work/'shutdown.wav';synthetic(source,60)
    marker=data/'current-url.txt';marker.unlink(missing_ok=True)
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=None;worker_pid=None;result={}
    with (work/'launch.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([str(executable.resolve()),'--no-browser','--data-dir',str(data)],
            stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline=time.monotonic()+60
            while not marker.exists():
                assert process.poll() is None,'Application exited during startup'
                if time.monotonic()>deadline:raise TimeoutError('Startup marker')
                time.sleep(.1)
            url=marker.read_text(encoding='utf-8').strip()
            while True:
                try:html=http.open(url,timeout=2).read().decode();break
                except OSError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.1)
            token=re.search(r'name="manialab-token" content="([^"]+)"',html).group(1)
            def post(path,body,kind='application/json'):
                request=urllib.request.Request(url+path,data=body,headers={'Content-Type':kind,'X-ManiaLab-Token':token})
                return json.load(http.open(request,timeout=30))
            boundary='ManiaLabShutdownCheck'
            content=(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="shutdown.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()+source.read_bytes()+f'\r\n--{boundary}--\r\n'.encode())
            project=post('/api/import',content,'multipart/form-data; boundary='+boundary)
            post('/api/projects/'+project['id']+'/analyze',json.dumps({'engine':'deep','settings':{}}).encode())
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                command=f"Get-CimInstance Win32_Process -Filter \"Name='ManiaLabAudio.exe' AND ParentProcessId={process.pid}\" | Select-Object -ExpandProperty ProcessId"
                found=subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],creationflags=subprocess.CREATE_NO_WINDOW,text=True).strip()
                if found:
                    worker_pid=int(found.splitlines()[0]);handle=kernel.OpenProcess(0x00100001,False,worker_pid)
                    if not handle:raise ctypes.WinError(ctypes.get_last_error())
                    break
                time.sleep(.1)
            if handle is None:raise TimeoutError('No active separation process observed')
            # Allow the worker's lifetime guard to initialize after bootloading.
            time.sleep(2)
            assert kernel.WaitForSingleObject(handle,0)==258,'Worker already exited before close test'
            process.terminate();process.wait(timeout=15)
            stopped=kernel.WaitForSingleObject(handle,10000)==0
            result=dict(application_pid=process.pid,worker_pid=worker_pid,worker_stopped_on_parent_exit=stopped)
            assert stopped,'Audio worker remained alive after the application exited'
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=15)
            if handle:
                if kernel.WaitForSingleObject(handle,0)==258:
                    kernel.TerminateProcess(handle,1);kernel.WaitForSingleObject(handle,5000)
                kernel.CloseHandle(handle)
            (work/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args()
    if os.name!='nt':raise SystemExit('Windows verification only')
    run(args.executable,args.work)
