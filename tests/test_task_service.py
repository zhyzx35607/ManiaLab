import logging
import threading
import time

import pytest

from manialab.storage import Projects
from manialab.task_service import ProjectTasks


@pytest.fixture
def task_store(tmp_path):
    projects = Projects(tmp_path / "projects")
    service = ProjectTasks(projects, logging.getLogger("test.task_service"))
    try:
        yield projects, service
    finally:
        service.executor.shutdown(wait=True)


def wait_for_job(service, job_id):
    for _ in range(500):
        job = service.get(job_id)
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError("task did not finish")


def create_project(projects):
    project_id, _ = projects.create()
    projects.save({"id": project_id, "value": "original"})
    return project_id


def test_successful_task_commits_snapshot_before_cleanup(task_store):
    projects, service = task_store
    project_id = create_project(projects)
    cleanup = []

    def operation(state, progress, cancelled):
        state["value"] = "updated"
        progress(50, "halfway")
        return {"updated": True}

    job = service.submit(project_id, operation, cleanup=cleanup.append)
    result = wait_for_job(service, job["id"])

    assert result["status"] == "done"
    assert result["result"] == {"updated": True}
    assert projects.load(project_id)["value"] == "updated"
    assert cleanup == [True]


def test_failed_task_preserves_snapshot_and_runs_cleanup(task_store):
    projects, service = task_store
    project_id = create_project(projects)
    cleanup = []

    def operation(state, progress, cancelled):
        state["value"] = "partial"
        raise ValueError("synthetic failure")

    job = service.submit(project_id, operation, cleanup=cleanup.append)
    result = wait_for_job(service, job["id"])

    assert result["status"] == "error"
    assert result["message"] == "synthetic failure"
    assert projects.load(project_id)["value"] == "original"
    assert cleanup == [False]


def test_cancelled_task_blocks_other_work_and_does_not_commit(task_store):
    projects, service = task_store
    project_id = create_project(projects)
    other_id = create_project(projects)
    started, release = threading.Event(), threading.Event()
    cleanup = []

    def operation(state, progress, cancelled):
        state["value"] = "partial"
        started.set()
        assert release.wait(5)
        return {"cancelled_during_work": cancelled()}

    try:
        job = service.submit(project_id, operation, cleanup=cleanup.append)
        assert started.wait(5)
        with pytest.raises(ValueError, match="该项目已有任务运行"):
            service.submit(project_id, lambda *args: None)
        with pytest.raises(ValueError, match="已有计算任务运行"):
            service.submit(other_id, lambda *args: None)
        assert service.cancel(job["id"])
        release.set()
        result = wait_for_job(service, job["id"])
    finally:
        release.set()

    assert result["status"] == "cancelled"
    assert projects.load(project_id)["value"] == "original"
    assert cleanup == [False]
