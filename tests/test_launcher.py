"""Startup must work with a legacy English console and Chinese project path."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from manialab import __version__
from launcher import _run_server


def test_launcher_shares_one_shutdown_deadline_across_cleanup(monkeypatch):
    calls = []
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    class Dispatcher:
        def shutdown(self, *, cancel_pending, timeout):
            calls.append(("dispatcher-shutdown", cancel_pending, timeout))
            clock[0] += 4

    class Server:
        task_dispatcher = Dispatcher()

        def run(self):
            calls.append("run")
            raise RuntimeError("synthetic server failure")

        def close(self):
            calls.append("close")
            clock[0] += 2

    class TaskService:
        def shutdown(self, *, wait, deadline):
            calls.append(("shutdown", wait, deadline))
            return True

    with pytest.raises(RuntimeError, match="synthetic server failure"):
        _run_server(Server(), TaskService())

    assert calls == [
        "run",
        ("shutdown", False, 115.0),
        "close",
        ("dispatcher-shutdown", True, 13.0),
        ("shutdown", True, 115.0),
    ]


def test_launcher_surfaces_shutdown_deadline_failure_without_hiding_server_error():
    class Server:
        task_dispatcher = None

        def run(self):
            raise RuntimeError("synthetic server failure")

        def close(self):
            pass

    class TaskService:
        def shutdown(self, *, wait, deadline):
            return False if wait else True

    with pytest.raises(ChildProcessError, match="shutdown deadline") as error:
        _run_server(Server(), TaskService())

    assert isinstance(error.value.__context__, RuntimeError)


def test_launcher_survives_cp1252_output_and_unicode_data_directory(tmp_path):
    root=Path(__file__).resolve().parents[1]
    data=tmp_path/'启动验证'
    environment=os.environ|{'PYTHONUTF8':'0','PYTHONIOENCODING':'cp1252:strict'}
    log=tmp_path/'startup.log'
    local_http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with log.open('wb') as output:
        process=subprocess.Popen([sys.executable,str(root/'launcher.py'),'--no-browser','--data-dir',str(data)],
            cwd=root,env=environment,stdout=output,stderr=output,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            deadline=time.monotonic()+20
            while True:
                assert process.poll() is None, log.read_bytes().decode('utf-8',errors='replace')
                marker=data/'current-url.txt'
                if marker.exists():
                    try:
                        with local_http.open(marker.read_text(encoding='utf-8')+'/api/status',timeout=1) as response:
                            assert json.load(response)['version']==__version__
                        break
                    except urllib.error.URLError:
                        pass
                assert time.monotonic()<deadline, 'Launcher did not become ready'
                time.sleep(.1)
            assert '项目数据' in log.read_bytes().decode('utf-8')
        finally:
            process.terminate()
            process.wait(timeout=15)
