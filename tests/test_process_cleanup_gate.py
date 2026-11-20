"""Unresolved child processes block computation without confusing file failures."""
import io
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from manialab.cancellation import CancellationContext, report_process_cleanup_failure, stop_process
from manialab.server import create_app
from manialab.storage import Projects
from manialab.task_service import ProjectTasks


ROOT = Path(__file__).resolve().parents[1]


def _finished(service, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service.get(job_id)
        if job['status'] in ('done', 'error', 'cancelled'):
            return job
        time.sleep(.005)
    raise AssertionError('synthetic task did not finish')


@pytest.fixture
def task_store(tmp_path):
    projects = Projects(tmp_path / 'projects')
    pid, _ = projects.create()
    projects.save({'id': pid, 'value': 'original'})
    service = ProjectTasks(projects)
    try:
        yield projects, service, pid
    finally:
        service.shutdown()


def _assert_next_task_blocked(service, pid):
    jobs, cancellations = dict(service.jobs), dict(service._cancellations)
    operation_calls = []
    with pytest.raises(ValueError, match='确认.*退出.*重启'):
        service.submit(pid, lambda *args: operation_calls.append(True))
    assert service.jobs == jobs
    assert service._cancellations == cancellations
    assert not operation_calls
    assert service._idle.is_set()


def test_real_child_cleanup_failure_blocks_next_task_even_after_child_eventually_exits(
        task_store, monkeypatch):
    projects, service, pid = task_store
    executable = getattr(sys, '_base_executable', sys.executable)
    process = subprocess.Popen([executable, '-c', 'import time; time.sleep(60)'],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    kill = process.kill

    def denied():
        raise PermissionError('synthetic child termination denied')

    monkeypatch.setattr(process, 'terminate', denied)

    def operation(state, progress, cancelled):
        state['value'] = 'must not commit'
        cancelled.cancel()
        stop_process(process, cancelled, terminate_timeout=.01, kill_timeout=.01)

    try:
        job = service.submit(pid, operation)
        assert _finished(service, job['id'])['status'] == 'error'
        assert process.poll() is None
        assert service._shutdown_cancellation.process_cleanup_failed
        assert projects.load(pid)['value'] == 'original'
        _assert_next_task_blocked(service, pid)
        kill()
        process.wait(timeout=5)
        # Root exit alone cannot prove that every possible descendant was reaped.
        _assert_next_task_blocked(service, pid)
        assert service.shutdown() is False
    finally:
        if process.poll() is None:
            kill()
        process.wait(timeout=5)


def test_successful_process_reaping_keeps_next_task_available(task_store):
    _, service, pid = task_store

    class StoppedProcess:
        alive = True

        def poll(self):
            return None if self.alive else -15

        def terminate(self):
            self.alive = False

        def wait(self, timeout):
            return -15

    job = service.submit(pid, lambda state, progress, cancelled:
        {'reaped': stop_process(StoppedProcess(), cancelled, terminate_timeout=.01, kill_timeout=.01)})
    assert _finished(service, job['id'])['result'] == {'reaped': True}
    assert not service._shutdown_cancellation.process_cleanup_failed
    next_job = service.submit(pid, lambda *args: {'worker_reused': True})
    assert _finished(service, next_job['id'])['status'] == 'done'
    assert service.shutdown() is True


def test_file_cleanup_failure_does_not_trigger_process_gate(task_store):
    _, service, pid = task_store

    def cleanup(committed):
        raise PermissionError('synthetic leftover file is locked')

    job = service.submit(pid, lambda *args: {'saved': True}, cleanup=cleanup)
    assert _finished(service, job['id'])['cleanup_failed'] is True
    assert service._shutdown_cancellation.cleanup_failures
    assert not service._shutdown_cancellation.process_cleanup_failed
    next_job = service.submit(pid, lambda *args: {'worker_reused': True})
    assert _finished(service, next_job['id'])['status'] == 'done'
    assert service.shutdown() is False


def test_process_failure_is_sticky_across_contexts_without_requesting_cancellation():
    parent = CancellationContext()
    first = CancellationContext(parent=parent)
    report_process_cleanup_failure(first, 'synthetic child still alive')
    second = CancellationContext(parent=parent)
    assert first.process_cleanup_failed
    assert parent.process_cleanup_failed
    assert second.process_cleanup_failed
    assert not parent() and not second()


def test_process_failure_reporter_keeps_legacy_callable_diagnostics():
    failures = []

    def cancelled():
        return False

    cancelled.report_cleanup_failure = failures.append
    report_process_cleanup_failure(cancelled, 'synthetic child still alive')
    assert failures == ['synthetic child still alive']


@pytest.mark.skipif(os.name != 'nt', reason='Windows descendant cleanup')
def test_mapper_descendant_failure_blocks_tasks_after_root_has_exited(task_store, monkeypatch):
    from manialab import upstream

    _, service, pid = task_store

    class ApiCall:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    class Kernel:
        OpenProcess = ApiCall(lambda *args: 42)
        TerminateProcess = ApiCall(lambda *args: True)
        WaitForSingleObject = ApiCall(lambda *args: 258)
        CloseHandle = ApiCall(lambda *args: True)

    class FinishedParent:
        pid = 123

        def poll(self):
            return 0

    monkeypatch.setattr(upstream.ctypes, 'WinDLL', lambda *args, **kwargs: Kernel())
    monkeypatch.setattr(upstream.MapperEngine, '_windows_descendants', lambda pid: [456])
    job = service.submit(pid, lambda state, progress, cancelled:
        upstream.MapperEngine.stop_process(FinishedParent(), cancelled))
    assert _finished(service, job['id'])['status'] == 'error'
    assert service._shutdown_cancellation.process_cleanup_failed
    _assert_next_task_blocked(service, pid)
    assert service.shutdown() is False


@pytest.mark.skipif(os.name != 'nt', reason='Windows descendant cleanup')
def test_mapper_enumeration_failure_blocks_tasks_after_root_has_exited(task_store, monkeypatch):
    from manialab import upstream

    _, service, pid = task_store
    child_calls = []

    class ApiCall:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    class Kernel:
        OpenProcess = ApiCall(lambda *args: 42)
        TerminateProcess = ApiCall(lambda *args: child_calls.append(args) or True)
        WaitForSingleObject = ApiCall(lambda *args: 0)
        CloseHandle = ApiCall(lambda *args: True)

    class FinishedParent:
        pid = 123

        def poll(self):
            return 0

    enumeration_calls = []

    def enumerate_descendants(root_pid):
        enumeration_calls.append(root_pid)
        if len(enumeration_calls) == 1:
            raise OSError('synthetic process snapshot failure')
        return [456]

    monkeypatch.setattr(upstream.ctypes, 'WinDLL', lambda *args, **kwargs: Kernel())
    monkeypatch.setattr(upstream.MapperEngine, '_windows_descendants', enumerate_descendants)
    job = service.submit(pid, lambda state, progress, cancelled:
        upstream.MapperEngine.stop_process(FinishedParent(), cancelled))
    assert _finished(service, job['id'])['status'] == 'error'
    assert enumeration_calls == [123, 123]
    assert child_calls == [(42, 1)]
    assert service._shutdown_cancellation.process_cleanup_failed
    _assert_next_task_blocked(service, pid)
    assert service.shutdown() is False


def test_process_gate_rejects_stem_upload_before_allocating_resources(tmp_path):
    app = create_app(ROOT, tmp_path / 'data')
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    service = app.extensions['task_service']
    projects = app.extensions['projects']
    try:
        imported = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(b'synthetic audio placeholder'), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        before = set(projects.path(pid).rglob('*'))
        report_process_cleanup_failure(service._shutdown_cancellation, 'synthetic unresolved child')
        blocked = client.post(f'/api/projects/{pid}/provided-stems', headers=headers,
            data={'piano': (io.BytesIO(b'synthetic stem placeholder'), 'piano.wav')})
        assert blocked.status_code == 400
        assert '确认' in blocked.json['error'] and '重启' in blocked.json['error']
        assert set(projects.path(pid).rglob('*')) == before
        assert not service.jobs and service._idle.is_set()
        assert client.get(f'/api/projects/{pid}').status_code == 200
    finally:
        service.shutdown()
