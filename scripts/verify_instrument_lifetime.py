"""Verify the real portable instrument worker stops on cancel and parent exit.

Each case imports a fresh 60-second synthetic recording as a provided other
stem. An explicitly synthetic saved analysis is seeded only as a preservation
sentinel; transcription, process creation, cancellation and restart use the EXE.
No neural inference runs until this script is invoked with an executable.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_portable import synthetic


def checked(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


class Processes:
    """Retain Windows handles so cleanup cannot act on a reused PID."""
    def __init__(self):
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        declarations = {
            'CreateToolhelp32Snapshot': ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
            'Process32FirstW': ([wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
            'Process32NextW': ([wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
            'OpenProcess': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            'QueryFullProcessImageNameW': ([wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                           ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            'WaitForSingleObject': ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
            'TerminateProcess': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in declarations.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result

    def children(self, parent_pid):
        class Entry(ctypes.Structure):
            _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                        ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
                        ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
                        ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG),
                        ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR * 260)]
        snapshot = self.kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        result = []
        try:
            entry = Entry()
            entry.dwSize = ctypes.sizeof(entry)
            present = self.kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while present:
                if entry.th32ParentProcessID == parent_pid and entry.szExeFile.lower() == 'manialabinstrument.exe':
                    result.append(entry.th32ProcessID)
                present = self.kernel.Process32NextW(snapshot, ctypes.byref(entry))
            error = ctypes.get_last_error()
            checked(error == 18, 'Process enumeration failed: ' + str(error))  # ERROR_NO_MORE_FILES
        finally:
            self.kernel.CloseHandle(snapshot)
        return result

    def own(self, pid, expected_executable):
        handle = self.kernel.OpenProcess(0x00101001, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            length = wintypes.DWORD(32768)
            name = ctypes.create_unicode_buffer(length.value)
            if not self.kernel.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(length)):
                raise ctypes.WinError(ctypes.get_last_error())
            checked(Path(name.value).resolve() == expected_executable,
                    'Observed child does not match the packaged instrument executable')
            return handle
        except BaseException:
            self.kernel.CloseHandle(handle)
            raise

    def exited(self, handle, timeout_ms=0):
        value = self.kernel.WaitForSingleObject(handle, timeout_ms)
        checked(value in (0, 258), 'Waiting on retained process handle failed: ' + str(value))
        return value == 0

    def terminate(self, handle):
        if not self.exited(handle):
            if not self.kernel.TerminateProcess(handle, 1) and not self.exited(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            checked(self.exited(handle, 10000), 'Owned worker survived cleanup termination')


class App:
    def __init__(self, executable, folder, environment):
        self.executable, self.folder, self.environment = executable, folder, environment
        self.data = folder / '独立项目'
        self.data.mkdir(parents=True)
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.process = self.log = None
        self.launches = 0
        self.url = self.token = ''

    def start(self):
        checked(self.process is None or self.process.poll() is not None, 'Application already running')
        marker = self.data / 'current-url.txt'
        marker.unlink(missing_ok=True)
        self.log = (self.folder / f'launch-{self.launches}.log').open('x', encoding='utf-8')
        self.launches += 1
        self.process = subprocess.Popen([str(self.executable), '--no-browser', '--data-dir', str(self.data)],
                                        cwd=self.folder, env=self.environment, stdout=self.log, stderr=self.log,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + 60
        while not marker.exists():
            checked(self.process.poll() is None, 'Application exited during startup')
            checked(time.monotonic() < deadline, 'Application startup marker timed out')
            time.sleep(.1)
        self.url = marker.read_text(encoding='utf-8').strip()
        while True:
            checked(self.process.poll() is None, 'Application exited before HTTP readiness')
            try:
                with self.http.open(self.url, timeout=2) as response:
                    html = response.read().decode('utf-8')
                break
            except (urllib.error.URLError, TimeoutError):
                checked(time.monotonic() < deadline, 'Application HTTP startup timed out')
                time.sleep(.1)
        match = re.search(r'name="manialab-token" content="([^"]+)"', html)
        checked(match is not None, 'Application session token missing')
        self.token = match.group(1)

    def stop(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        finally:
            if self.log:
                self.log.close()

    def request(self, path, payload=None, upload=None):
        headers, body = {}, None
        if upload:
            field, name, content = upload
            boundary = 'ManiaLabInstrumentLifetimeBoundary'
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
                    'Content-Type: audio/wav\r\n\r\n').encode() + content + f'\r\n--{boundary}--\r\n'.encode()
            headers['Content-Type'] = 'multipart/form-data; boundary=' + boundary
        elif payload is not None:
            body = json.dumps(payload).encode()
            headers['Content-Type'] = 'application/json'
        if body is not None:
            headers['X-ManiaLab-Token'] = self.token
        with self.http.open(urllib.request.Request(self.url + path, data=body, headers=headers), timeout=30) as response:
            content = response.read()
            return json.loads(content) if 'application/json' in response.headers.get('Content-Type', '') else content

    def wait(self, job, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.request('/api/jobs/' + job['id'])
            if result['status'] in ('done', 'error', 'cancelled'):
                return result
            time.sleep(.1)
        raise TimeoutError('Application job did not finish: ' + job['id'])


def case(executable, worker, folder, environment, source, mode):
    folder.mkdir()
    app, processes = App(executable, folder, environment), Processes()
    owned = {}
    report = dict(mode=mode, passed=False, neural_inference=False, previous_analysis_synthetic=True)
    try:
        app.start()
        status = app.request('/api/status')
        checked(status['instruments']['available'], 'Packaged specialist is unavailable')
        audio = source.read_bytes()
        state = app.request('/api/import', upload=('files', '生命周期原声.wav', audio))
        pid, base = state['id'], '/api/projects/' + state['id']
        imported = app.wait(app.request(base + '/provided-stems', upload=('other', '其他伴奏.wav', audio)))
        checked(imported['status'] == 'done', 'Provided-stem import failed: ' + str(imported))
        state = app.request(base)
        project = app.data / 'projects' / pid
        state_path = project / 'state.json'
        # This sentinel is not a model result. The target analysis always runs
        # through the unchanged application's real HTTP and EXE worker path.
        state['analysis'] = dict(revision=8, duration=60, candidates=[], waveform=[],
                                tempo=dict(bpm=120, offset=0),
                                lifetime_sentinel='synthetic previous analysis; preserve byte-for-byte')
        state['analysis_options'] = dict(engine='basic', piano_engine='general',
                                         instrument_engine='general', allow_predicted_releases=False)
        pending = state_path.with_name('lifetime-sentinel.tmp')
        write_new(pending, state)
        pending.replace(state_path)
        checked(app.request(base) == state, 'Server did not read the saved sentinel')
        before_hash = digest(state_path)
        report.update(application_pid=app.process.pid, project=pid, previous_state_sha256=before_hash,
                      input_sha256=hashlib.sha256(audio).hexdigest())
        options = dict(engine='provided', instrument_engine='tsumugi', allow_predicted_releases=False)
        job = app.request(base + '/analyze', options)
        report['job_id'] = job['id']
        deadline = time.monotonic() + 240
        first_window = log_path = worker_pid = handle = None
        while time.monotonic() < deadline:
            checked(app.process.poll() is None, 'Application exited before the intervention')
            current = app.request('/api/jobs/' + job['id'])
            checked(current['status'] not in ('done', 'error', 'cancelled'),
                    'Job finished before a real worker window was observed: ' + str(current))
            for child in processes.children(app.process.pid):
                if child not in owned:
                    owned[child] = processes.own(child, worker)
            checked(len(owned) <= 1, 'Unexpected extra instrument child for a one-stem job')
            if owned:
                worker_pid, handle = next(iter(owned.items()))
                checked(not processes.exited(handle), 'Instrument worker exited before intervention')
                logs = list(project.glob('stems-provided/*/instrument-other/*/attempt-*/worker.log'))
                checked(len(logs) <= 1, 'Fresh project unexpectedly has multiple instrument attempts')
                if logs:
                    log_path = logs[0]
                    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
                        try:
                            item = json.loads(line)
                        except ValueError:
                            continue
                        if (isinstance(item, dict) and item.get('stage') == 'window'
                                and item.get('start_sample') == 0 and item.get('windows') == 1):
                            first_window = item
                            break
                if first_window is not None:
                    break
            time.sleep(.1)
        checked(first_window is not None and handle is not None, 'No real first-window callback observed within 240s')
        checked(not processes.exited(handle), 'Worker completed before cancellation/parent termination')
        report.update(worker_pid=worker_pid, worker_executable=str(worker), window_evidence=first_window,
                      worker_log=str(log_path), neural_inference=True)
        print(json.dumps(dict(stage='window_observed', mode=mode, application_pid=app.process.pid,
                              worker_pid=worker_pid, window=first_window)), flush=True)
        intervention = time.monotonic()
        if mode == 'api_cancel':
            app.request('/api/jobs/' + job['id'] + '/cancel', {})
            stopped = processes.exited(handle, 15000)
            checked(stopped, 'Instrument worker survived API cancellation')
            outcome = app.wait(job, timeout=30)
            checked(outcome['status'] == 'cancelled', 'API job was not cancelled: ' + str(outcome))
            checked(app.request(base) == state, 'Cancellation changed the previous project')
            checked(app.request('/api/status')['version'] == status['version'], 'Service unavailable after cancellation')
            later = app.request('/api/import', upload=('files', '取消后可用.wav', audio))
            checked(app.request('/api/projects/' + later['id'])['id'] == later['id'],
                    'Service could not create a project after cancellation')
            report.update(job_status=outcome['status'], service_usable_after_cancel=True,
                          post_cancel_import=later['id'])
        else:
            app.stop()
            checked(processes.exited(handle, 10000), 'Instrument worker outlived its terminated parent')
            report['parent_returncode'] = app.process.returncode
        report['intervention_to_verification_seconds'] = round(time.monotonic() - intervention, 3)
        checked(digest(state_path) == before_hash, 'Previous state bytes changed during interruption')
        checked(not list(project.glob('stems-provided/*/instrument-other/*/current.json')),
                'Interrupted analysis published a completed cache marker')
        report.update(worker_exited=True, previous_state_preserved=True, completed_cache_published=False,
                      worker_log_sha256=digest(log_path))
        if mode == 'parent_exit':
            app.start()
            checked(app.request(base) == state, 'Restart did not restore the previous project')
            checked(app.request('/api/status')['version'] == status['version'], 'Restarted service unavailable')
            report.update(restart_preserved=True, restarted_application_pid=app.process.pid)
        report['passed'] = True
        return report
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        cleanup_errors = []
        # Acquire only still-live direct children of this test's live parent.
        # Cleanup never searches by image name globally or kills a bare PID.
        if app.process is not None and app.process.poll() is None:
            try:
                for child in processes.children(app.process.pid):
                    if child not in owned:
                        owned[child] = processes.own(child, worker)
            except Exception as error:
                cleanup_errors.append(str(error))
        try:
            app.stop()
        except Exception as error:
            cleanup_errors.append(str(error))
        for child, child_handle in owned.items():
            try:
                processes.terminate(child_handle)
            except Exception as error:
                cleanup_errors.append(f'Owned worker {child}: {error}')
            finally:
                processes.kernel.CloseHandle(child_handle)
        report['cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            report['passed'] = False
        write_new(folder / 'report.json', report)
        if cleanup_errors:
            raise RuntimeError('Lifetime-test cleanup failed: ' + '; '.join(cleanup_errors))


def run(executable, work):
    checked(os.name == 'nt', 'This verification requires Windows process handles')
    executable, work = executable.resolve(), work.resolve()
    checked(executable.is_file(), 'Packaged application executable is missing')
    checked(not work.exists() and work.is_relative_to((ROOT / 'test-data').resolve()),
            'Use a new directory under this repository test-data; previous evidence is preserved')
    work.mkdir(parents=True)
    worker = (executable.parent / '_internal/assets/instruments/ManiaLabInstrument/ManiaLabInstrument.exe').resolve()
    manifest = read(executable.parent / 'release-manifest.json')
    executable_hash, worker_hash = digest(executable), digest(worker)
    checked(manifest['executable_sha256'] == executable_hash, 'Application differs from release manifest')
    checked(manifest['instrument_worker']['worker_sha256'] == worker_hash, 'Worker differs from release manifest')
    environment = {k: v for k, v in os.environ.items()
                   if k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')
                   and not k.startswith(('MANIALAB_DEEP_', 'MANIALAB_INSTRUMENT_'))}
    environment['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32')
    environment.update(HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9',
                       NO_PROXY='127.0.0.1,localhost', PYTHONUTF8='1')
    source = work / '60秒合成音.wav'
    synthetic(source, seconds=60)
    with wave.open(str(source), 'rb') as audio:
        checked(audio.getnframes() == audio.getframerate() * 60, 'Fixture is not 60 seconds')
        checked(any(audio.readframes(audio.getframerate() * 8)), 'First window must contain audible signal')
    report = dict(passed=False, executable=str(executable), executable_sha256=executable_hash,
                  worker_sha256=worker_hash, verifier_sha256=digest(__file__),
                  input_sha256=digest(source), source_seconds=60, cases=[], musical_acceptance=False,
                  scope='Real portable HTTP and other worker; first non-silent window before interruption; '
                        'retained Windows process handles; synthetic old-state sentinel; clean PATH and '
                        'unreachable proxies, not a firewall')
    try:
        for mode in ('api_cancel', 'parent_exit'):
            report['cases'].append(case(executable, worker, work / mode, environment, source, mode))
        checked(digest(executable) == executable_hash and digest(worker) == worker_hash,
                'Application or worker executable changed during verification')
        report['passed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        write_new(work / 'report.json', report)
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    run(args.executable, args.work)
