"""Unresolved child processes block computation without confusing file failures."""
import io
import os
from pathlib import Path
import subprocess
import sys
import threading
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


@pytest.mark.parametrize('failing_query', ['poll', 'wait'])
def test_transient_process_query_failure_recovers_without_blocking_next_task(
        task_store, failing_query):
    _, service, pid = task_store

    class TransientQueryFailure:
        def __init__(self):
            self.alive = True
            self.poll_calls = 0
            self.wait_calls = 0
            self.kill_calls = 0

        def poll(self):
            self.poll_calls += 1
            if failing_query == 'poll' and self.poll_calls == 1:
                raise OSError('synthetic one-time process status query failure')
            return None if self.alive else -15

        def terminate(self):
            self.alive = False

        def wait(self, timeout):
            self.wait_calls += 1
            if failing_query == 'wait' and self.wait_calls == 1:
                raise OSError('synthetic one-time process wait failure')
            return -15

        def kill(self):
            self.kill_calls += 1
            self.alive = False

    process = TransientQueryFailure()
    job = service.submit(pid, lambda state, progress, cancelled:
        {'reaped': stop_process(process, cancelled, terminate_timeout=.01, kill_timeout=.01)})
    result = _finished(service, job['id'])

    assert result['status'] == 'done'
    assert result['result'] == {'reaped': True}
    assert not service._shutdown_cancellation.process_cleanup_failed
    assert not service._shutdown_cancellation.cleanup_failures
    next_job = service.submit(pid, lambda *args: {'worker_reused': True})
    assert _finished(service, next_job['id'])['status'] == 'done'
    assert service.shutdown() is True


@pytest.mark.parametrize('failing_query', ['poll', 'wait'])
def test_process_query_failure_during_cleanup_blocks_next_task(task_store, failing_query):
    _, service, pid = task_store

    class QueryFailure:
        def __init__(self):
            self.kill_calls = 0

        def poll(self):
            if failing_query == 'poll':
                raise OSError('synthetic process status query failure')
            return None

        def terminate(self):
            pass

        def wait(self, timeout):
            raise OSError('synthetic process wait failure')

        def kill(self):
            self.kill_calls += 1

    process = QueryFailure()
    job = service.submit(pid, lambda state, progress, cancelled:
        stop_process(process, cancelled, terminate_timeout=.01, kill_timeout=.01))
    assert _finished(service, job['id'])['status'] == 'error'
    assert service._shutdown_cancellation.process_cleanup_failed
    assert service._shutdown_cancellation.cleanup_failures
    assert process.kill_calls == 1
    _assert_next_task_blocked(service, pid)
    assert service.shutdown() is False


@pytest.mark.parametrize('failing_query', ['poll', 'wait'])
def test_mapper_process_query_failure_during_cleanup_blocks_next_task(
        task_store, monkeypatch, failing_query):
    from manialab.upstream import MapperEngine
    from types import SimpleNamespace

    _, service, pid = task_store
    monkeypatch.setattr('manialab.upstream.os', SimpleNamespace(name='posix'))

    class QueryFailure:
        def __init__(self):
            self.kill_calls = 0

        def poll(self):
            if failing_query == 'poll':
                raise OSError('synthetic mapper process status query failure')
            return None

        def terminate(self):
            pass

        def kill(self):
            self.kill_calls += 1

        def wait(self, timeout):
            raise OSError('synthetic mapper process wait failure')

    process = QueryFailure()
    job = service.submit(pid, lambda state, progress, cancelled:
        MapperEngine.stop_process(process, cancelled))
    assert _finished(service, job['id'])['status'] == 'error'
    assert service._shutdown_cancellation.process_cleanup_failed
    assert service._shutdown_cancellation.cleanup_failures
    assert process.kill_calls >= 1
    _assert_next_task_blocked(service, pid)
    assert service.shutdown() is False


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


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object cleanup')
def test_mapper_rejects_unowned_windows_process_after_root_exits(task_store):
    from manialab import upstream

    _, service, pid = task_store

    class FinishedParent:
        pid = 123

        def poll(self):
            return 0

    job = service.submit(pid, lambda state, progress, cancelled:
        upstream.MapperEngine.stop_process(FinishedParent(), cancelled))
    assert _finished(service, job['id'])['status'] == 'error'
    assert service._shutdown_cancellation.process_cleanup_failed
    _assert_next_task_blocked(service, pid)
    assert service.shutdown() is False


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object cleanup')
def test_mapper_owned_job_cleanup_failure_blocks_later_tasks(task_store):
    from manialab import upstream
    from manialab.cancellation import report_process_cleanup_failure

    _, service, pid = task_store
    cleanup_calls = []

    class OwnedJobProcess:
        def _manialab_stop_tree(self, cancelled, maximum):
            cleanup_calls.append(maximum)
            report_process_cleanup_failure(cancelled, 'synthetic owned worker job still has active processes')
            raise ChildProcessError('Owned worker process tree cleanup could not be verified')

    job = service.submit(pid, lambda state, progress, cancelled:
        upstream.MapperEngine.stop_process(OwnedJobProcess(), cancelled))
    assert _finished(service, job['id'])['status'] == 'error'
    assert cleanup_calls == [30]
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


def test_cancel_api_process_cleanup_failure_blocks_later_stem_upload(
        tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path / 'data')
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    service = app.extensions['task_service']
    projects = app.extensions['projects']
    started, release = threading.Event(), threading.Event()
    process = subprocess.Popen(
        [getattr(sys, '_base_executable', sys.executable), '-c', 'import time; time.sleep(60)'],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        def denied():
            raise PermissionError('synthetic child termination denied')

        monkeypatch.setattr(process, 'terminate', denied)

        def fail_to_reap(mix, sources, work, progress, cancelled):
            started.set()
            assert release.wait(5)
            stop_process(process, cancelled, terminate_timeout=.01, kill_timeout=.01)

        monkeypatch.setattr(app.extensions['analysis_services'], 'prepare_provided_stems', fail_to_reap)

        imported = client.post('/api/import', headers=headers,
            data={'files': (io.BytesIO(b'synthetic audio placeholder'), 'mix.wav')})
        assert imported.status_code == 200
        pid = imported.json['id']
        project_path = projects.path(pid)
        original_state = projects.load(pid)
        original_files = {
            path.relative_to(project_path): path.read_bytes()
            for path in project_path.rglob('*') if path.is_file()
        }

        submitted = client.post(f'/api/projects/{pid}/provided-stems', headers=headers,
            data={'piano': (io.BytesIO(b'synthetic stem placeholder'), 'piano.wav')})
        assert submitted.status_code == 200, submitted.json
        job_id = submitted.json['id']
        assert started.wait(5)
        cancelled = client.post(f'/api/jobs/{job_id}/cancel', headers=headers)
        assert cancelled.status_code == 200 and cancelled.json == {'ok': True}
        assert client.get(f'/api/jobs/{job_id}').json['cancel'] is True
        release.set()

        job = _finished(service, job_id)
        assert job['status'] == 'error'
        assert job['cancel'] is True
        assert process.poll() is None
        assert service._shutdown_cancellation.process_cleanup_failed
        assert projects.load(pid) == original_state
        assert {
            path.relative_to(project_path): path.read_bytes()
            for path in project_path.rglob('*') if path.is_file()
        } == original_files

        after_cleanup = set(project_path.rglob('*'))
        blocked = client.post(f'/api/projects/{pid}/provided-stems', headers=headers,
            data={'piano': (io.BytesIO(b'second synthetic stem'), 'piano.wav')})
        assert blocked.status_code == 400
        assert '确认' in blocked.json['error'] and '重启' in blocked.json['error']
        assert set(project_path.rglob('*')) == after_cleanup
        assert not service.has_active() and service._idle.is_set()
    finally:
        release.set()
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        finally:
            service.shutdown()
