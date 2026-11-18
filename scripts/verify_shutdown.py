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
from scripts.package_assets import capabilities
from scripts.verify_portable import synthetic


def verify_core_shutdown(status, version, worker_processes_before_exit,
                         worker_processes_after_exit, application_exited):
    if status.get('version') != version:
        raise ValueError('Core application status version differs from the release manifest')
    separator = status.get('separator', {})
    profiles = separator.get('profiles', {})
    if separator.get('available') or any(
            profiles.get(name, {}).get('available') for name in ('four', 'six')):
        raise ValueError('Core runtime unexpectedly reports a separation worker')
    if worker_processes_before_exit or worker_processes_after_exit:
        raise ValueError('Core runtime spawned an unexpected audio worker process')
    if not application_exited:
        raise ValueError('Core application did not exit')
    return dict(
        passed=True,
        build_profile='core',
        worker_expected=False,
        worker_processes_before_exit=worker_processes_before_exit,
        worker_processes_after_exit=worker_processes_after_exit,
        application_exited=True,
        worker_stopped_on_parent_exit=None,
    )


def run(executable, work):
    executable=Path(executable).resolve()
    release_manifest=json.loads((executable.parent/'release-manifest.json').read_text(encoding='utf-8'))
    build_profile=release_manifest.get('build_profile')
    declared=capabilities(build_profile)
    expects_worker=declared['separation_four']
    work=work.resolve();work.mkdir(parents=True,exist_ok=True)
    data=work/'shutdown-data';data.mkdir(exist_ok=True)
    source=work/'shutdown.wav'
    if expects_worker:
        synthetic(source,60)
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

    def worker_processes(parent_pid):
        command=(f"Get-CimInstance Win32_Process -Filter "
                 f"\"Name='ManiaLabAudio.exe' AND ParentProcessId={int(parent_pid)}\" "
                 "| Select-Object -ExpandProperty ProcessId")
        found=subprocess.check_output(
            ['powershell.exe','-NoProfile','-Command',command],
            creationflags=subprocess.CREATE_NO_WINDOW,text=True,
        ).strip()
        return [int(line) for line in found.splitlines() if line.strip()]

    with (work/'launch.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([str(executable),'--no-browser','--data-dir',str(data)],
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
            if not expects_worker:
                status_request=urllib.request.Request(
                    url+'/api/status',headers={'X-ManiaLab-Token':token}
                )
                status=json.load(http.open(status_request,timeout=30))
                before=worker_processes(process.pid)
                process.terminate();process.wait(timeout=15)
                after=worker_processes(process.pid)
                result=verify_core_shutdown(
                    status,release_manifest['version'],before,after,
                    process.poll() is not None,
                )
                result['application_pid']=process.pid
            else:
                boundary='ManiaLabShutdownCheck'
                content=(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="shutdown.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()+source.read_bytes()+f'\r\n--{boundary}--\r\n'.encode())
                project=post('/api/import',content,'multipart/form-data; boundary='+boundary)
                post('/api/projects/'+project['id']+'/analyze',json.dumps({'engine':'deep','settings':{}}).encode())
                deadline=time.monotonic()+60
                while time.monotonic()<deadline:
                    found=worker_processes(process.pid)
                    if found:
                        worker_pid=found[0];handle=kernel.OpenProcess(0x00100001,False,worker_pid)
                        if not handle:raise ctypes.WinError(ctypes.get_last_error())
                        break
                    time.sleep(.1)
                if handle is None:raise TimeoutError('No active separation process observed')
                time.sleep(2)
                assert kernel.WaitForSingleObject(handle,0)==258,'Worker already exited before close test'
                process.terminate();process.wait(timeout=15)
                stopped=kernel.WaitForSingleObject(handle,10000)==0
                result=dict(passed=stopped,build_profile=build_profile,worker_expected=True,
                            application_exited=process.poll() is not None,application_pid=process.pid,
                            worker_pid=worker_pid,worker_stopped_on_parent_exit=stopped)
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
