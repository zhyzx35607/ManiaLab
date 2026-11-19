"""Cleanup failures remain visible without changing the state commit boundary."""
import time

import pytest

from manialab.storage import Projects
from manialab.task_service import ProjectTasks


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
    pid, folder = projects.create()
    projects.save({'id': pid, 'value': 'original'})
    service = ProjectTasks(projects)
    try:
        yield projects, service, pid, folder
    finally:
        service.shutdown()


@pytest.mark.parametrize('stage', ['cancelled', 'save-failed', 'committed'])
def test_cleanup_failure_preserves_commit_fact_and_does_not_hold_worker(task_store, monkeypatch, stage):
    projects, service, pid, folder = task_store
    save = projects.save
    marker = folder / 'published.synthetic'
    cleanup_calls = []

    def operation(state, progress, cancelled):
        marker.write_text('must remain visible when cleanup fails', encoding='utf-8')
        state['value'] = 'updated'
        if stage == 'cancelled':
            cancelled.cancel()
        return {'updated': True}

    def cleanup(committed):
        cleanup_calls.append(committed)
        raise PermissionError('synthetic rollback resource is locked')

    if stage == 'save-failed':
        def fail_save(state):
            raise OSError('synthetic final state save failure')
        monkeypatch.setattr(projects, 'save', fail_save)
    job = service.submit(pid, operation, cleanup=cleanup)
    result = _finished(service, job['id'])
    committed = stage == 'committed'

    assert result['status'] == ('done' if committed else 'error')
    assert result['cleanup_failed'] is True
    assert '资源清理失败' in result['message']
    assert '原有谱面已保留' not in result['message']
    assert cleanup_calls == [committed]
    assert projects.load(pid)['value'] == ('updated' if committed else 'original')
    assert marker.is_file()
    assert service._idle.is_set()
    assert service._shutdown_cancellation.cleanup_failures
    if committed:
        assert result['result'] == {'updated': True}
        assert result['cancel'] is False
        assert '已保存' in result['message']
    elif stage == 'cancelled':
        assert result['cancel'] is True

    monkeypatch.setattr(projects, 'save', save)
    recovered = service.submit(pid, lambda *args: {'worker_reused': True})
    recovery = _finished(service, recovered['id'])
    assert recovery['status'] == 'done'
    assert recovery['result'] == {'worker_reused': True}
    assert not recovery.get('cleanup_failed', False)
    assert service.shutdown() is False


def test_worker_start_failure_also_records_failed_resource_cleanup(task_store, monkeypatch):
    projects, service, pid, folder = task_store
    marker = folder / 'queued-upload.synthetic'
    marker.write_text('allocated before worker submission', encoding='utf-8')
    operation_calls, cleanup_calls = [], []

    def fail_submit(*args):
        raise RuntimeError('synthetic executor submit failure')

    def cleanup(committed):
        cleanup_calls.append(committed)
        raise PermissionError('synthetic upload directory is locked')

    monkeypatch.setattr(service.executor, 'submit', fail_submit)
    job = service.submit(pid, lambda *args: operation_calls.append(True), cleanup=cleanup)

    assert job['status'] == 'error'
    assert job['cleanup_failed'] is True
    assert '资源清理失败' in job['message']
    assert not operation_calls
    assert cleanup_calls == [False]
    assert projects.load(pid)['value'] == 'original'
    assert marker.is_file()
    assert service._idle.is_set()
    assert service.shutdown() is False
    with pytest.raises(ValueError, match='任务服务已关闭'):
        service.submit(pid, lambda *args: None)
