import json
import logging
from pathlib import Path
import queue
import threading
import time

import pytest

from manialab.storage import Projects
from manialab.task_errors import UserVisibleTaskError, UserVisibleTimeoutError
from manialab.task_service import ProjectTasks


@pytest.fixture
def task_store(tmp_path):
    projects = Projects(tmp_path / "projects")
    service = ProjectTasks(projects, logging.getLogger("test.task_service"))
    try:
        yield projects, service
    finally:
        service.shutdown()


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
    assert service.cancel(job["id"])
    assert not service.get(job["id"])["cancel"]


def test_submit_returns_a_stable_job_snapshot(task_store, monkeypatch):
    projects, service = task_store
    project_id = create_project(projects)
    submitted = []
    monkeypatch.setattr(service.executor, "submit", lambda operation: submitted.append(operation))

    job = service.submit(project_id, lambda *args: {"completed": True})
    assert job["status"] == "queued"
    submitted[0]()

    assert service.get(job["id"])["status"] == "done"
    assert job["status"] == "queued"


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


def test_base_exception_finishes_job_and_allows_later_work(task_store):
    projects, service = task_store
    project_id = create_project(projects)
    cleanup = []

    def operation(state, progress, cancelled):
        raise SystemExit("synthetic worker exit")

    job = service.submit(project_id, operation, cleanup=cleanup.append)
    result = wait_for_job(service, job["id"])

    assert result["status"] == "error"
    assert result["message"] == "任务失败，请查看本地日志"
    assert cleanup == [False]
    recovered = service.submit(project_id, lambda *args: {"recovered": True})
    assert wait_for_job(service, recovered["id"])["status"] == "done"


@pytest.mark.parametrize(
    "error_type",
    [UserVisibleTaskError, UserVisibleTimeoutError],
)
def test_task_exposes_safe_runtime_summary_and_log_path(task_store, error_type):
    projects, service = task_store
    project_id = create_project(projects)
    log_path = Path("D:/ManiaLab/logs/v32-install.log")

    def operation(_state, _progress, _cancelled):
        raise error_type(
            "安装 CUDA PyTorch 超时\nprivate subprocess log tail",
            log_path=log_path,
        )

    submitted = service.submit(project_id, operation)
    result = wait_for_job(service, submitted["id"])

    assert result["status"] == "error"
    assert result["message"] == (
        "安装 CUDA PyTorch 超时；日志：D:\\ManiaLab\\logs\\v32-install.log"
    )
    assert "private subprocess log tail" not in result["message"]


def test_worker_start_failure_cancels_queued_work_and_closes_service(task_store, monkeypatch):
    projects, service = task_store
    project_id = create_project(projects)
    cleanup = []

    class TrackingQueue(queue.SimpleQueue):
        def __init__(self):
            super().__init__()
            self.items = []

        def put(self, item, *args, **kwargs):
            self.items.append(item)
            return super().put(item, *args, **kwargs)

    work_queue = TrackingQueue()
    monkeypatch.setattr(service.executor, "_work_queue", work_queue)

    def fail_worker_start():
        raise RuntimeError("synthetic executor failure")

    monkeypatch.setattr(service.executor, "_adjust_thread_count", fail_worker_start)
    job = service.submit(project_id, lambda *args: None, cleanup=cleanup.append)

    assert job["status"] == "error"
    assert cleanup == [False]
    assert not service.has_active()
    assert service._closed
    assert len(work_queue.items) == 2
    assert work_queue.items[0].future.cancelled()
    assert work_queue.items[1] is None
    with pytest.raises(ValueError, match="任务服务已关闭"):
        service.submit(project_id, lambda *args: None)


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


def test_shutdown_cancels_active_work_waits_for_cleanup_and_rejects_submits(task_store):
    projects, service = task_store
    project_id = create_project(projects)
    started = threading.Event()
    cleanup = []

    def operation(state, progress, cancelled):
        started.set()
        while not cancelled():
            time.sleep(0.005)
        return {"cancelled": True}

    job = service.submit(project_id, operation, cleanup=cleanup.append)
    assert started.wait(5)

    service.shutdown()

    result = service.get(job["id"])
    assert result["status"] == "cancelled"
    assert cleanup == [False]
    with pytest.raises(ValueError, match="任务服务已关闭"):
        service.submit(project_id, lambda *args: None)


def test_shutdown_deadline_bounds_wait_and_stays_out_of_job_json(task_store):
    projects, service = task_store
    project_id = create_project(projects)
    started, release = threading.Event(), threading.Event()
    received = []

    def operation(state, progress, cancelled):
        received.append(cancelled)
        started.set()
        assert release.wait(5)

    job = service.submit(project_id, operation)
    assert started.wait(5)
    deadline = time.monotonic() + 0.05
    try:
        assert service.shutdown(wait=True, deadline=deadline) is False
        assert received[0]()
        assert received[0].deadline == deadline
        assert service.get(job["id"])["cancel"] is True
        assert "deadline" not in service.get(job["id"])
        with pytest.raises(ValueError, match="任务服务已关闭"):
            service.submit(project_id, lambda *args: None)
    finally:
        release.set()
        service.shutdown()

    assert service.get(job["id"])["status"] == "cancelled"


def test_shutdown_does_not_wait_past_deadline_for_task_lock(task_store):
    projects, service = task_store
    project_id = create_project(projects)
    locked, release = threading.Event(), threading.Event()

    def hold_task_lock():
        with service.lock:
            locked.set()
            release.wait(5)

    holder = threading.Thread(target=hold_task_lock)
    holder.start()
    assert locked.wait(5)
    deadline = time.monotonic() + 0.05
    try:
        started = time.monotonic()
        assert service.shutdown(wait=True, deadline=deadline) is False
        assert time.monotonic() - started < 0.5
        assert service._shutdown_requested.is_set()
        with pytest.raises(ValueError, match="任务服务已关闭"):
            service.submit(project_id, lambda *args: None)
    finally:
        release.set()
        holder.join(timeout=5)
        service.shutdown()

    assert not holder.is_alive()


def test_shutdown_lock_deadline_reports_active_job_as_cancelled(task_store):
    projects, service = task_store
    project_id = create_project(projects)
    started, release_operation = threading.Event(), threading.Event()
    locked, release_lock = threading.Event(), threading.Event()

    def operation(state, progress, cancelled):
        started.set()
        assert release_operation.wait(5)

    job = service.submit(project_id, operation)
    assert started.wait(5)

    def hold_task_lock():
        with service.lock:
            locked.set()
            release_lock.wait(5)

    holder = threading.Thread(target=hold_task_lock)
    holder.start()
    assert locked.wait(5)
    try:
        assert service.shutdown(wait=True, deadline=time.monotonic() + 0.05) is False
    finally:
        release_lock.set()
        holder.join(timeout=5)

    try:
        assert not holder.is_alive()
        assert service.get(job["id"])["status"] == "running"
        assert service.get(job["id"])["cancel"] is True
    finally:
        release_operation.set()

    assert wait_for_job(service, job["id"])["status"] == "cancelled"


def test_shutdown_during_project_load_skips_operation(task_store, monkeypatch):
    projects, service = task_store
    project_id = create_project(projects)
    loading, release = threading.Event(), threading.Event()
    operation_calls = []
    original_load = projects.load

    def delayed_load(value):
        loading.set()
        assert release.wait(5)
        return original_load(value)

    monkeypatch.setattr(projects, "load", delayed_load)
    job = service.submit(
        project_id,
        lambda *args: operation_calls.append(True),
    )
    assert loading.wait(5)
    assert service.shutdown(wait=False) is False
    release.set()

    assert service._idle.wait(5)
    assert service.get(job["id"])["status"] == "cancelled"
    assert service.get(job["id"])["cancel"] is True
    assert not operation_calls


def test_shutdown_reports_child_cleanup_failure(task_store):
    projects, service = task_store
    project_id = create_project(projects)
    started = threading.Event()
    child_failures = []

    def operation(state, progress, cancelled):
        started.set()
        while not cancelled():
            time.sleep(0.001)
        cancelled.report_cleanup_failure("synthetic child did not stop")
        child_failures.extend(cancelled.cleanup_failures)
        raise ChildProcessError("synthetic child did not stop")

    job = service.submit(project_id, operation)
    assert started.wait(5)

    assert service.shutdown(wait=True, deadline=time.monotonic() + 2) is False
    assert service.get(job["id"])["status"] == "error"
    assert child_failures == ["synthetic child did not stop"]
    assert service._shutdown_cancellation.cleanup_failures == (
        "synthetic child did not stop",
    )


def test_completed_task_clears_shutdown_request_during_commit(task_store, monkeypatch):
    projects, service = task_store
    project_id = create_project(projects)
    saving, release = threading.Event(), threading.Event()
    original_save = projects.save

    def delayed_save(state):
        saving.set()
        assert release.wait(5)
        return original_save(state)

    monkeypatch.setattr(projects, "save", delayed_save)
    job = service.submit(project_id, lambda *args: {"saved": True})
    assert saving.wait(5)
    assert service.shutdown(wait=False) is False
    release.set()

    result = wait_for_job(service, job["id"])
    assert result["status"] == "done"
    assert result["cancel"] is False


def test_shutdown_deadline_during_project_commit_preserves_done_result(task_store, monkeypatch):
    projects, service = task_store
    project_id = create_project(projects)
    saving, release = threading.Event(), threading.Event()
    original_save = projects.save

    def delayed_save(state):
        saving.set()
        assert release.wait(5)
        return original_save(state)

    monkeypatch.setattr(projects, "save", delayed_save)

    def operation(state, progress, cancelled):
        state["value"] = "committed"
        return {"committed": True}

    job = service.submit(project_id, operation)
    assert saving.wait(5)
    deadline = time.monotonic() + 0.05

    try:
        started = time.monotonic()
        assert service.shutdown(wait=True, deadline=deadline) is False
        assert time.monotonic() - started < 0.5
        with pytest.raises(ValueError, match="任务服务已关闭"):
            service.submit(project_id, lambda *args: None)
    finally:
        release.set()

    result = wait_for_job(service, job["id"])
    assert result["status"] == "done"
    assert result["result"] == {"committed": True}
    assert result["cancel"] is False
    assert projects.load(project_id)["value"] == "committed"


@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["cleanup-ok", "cleanup-error"])
def test_shutdown_deadline_during_post_commit_cleanup_preserves_done_result(
        task_store, cleanup_fails):
    projects, service = task_store
    project_id = create_project(projects)
    cleanup_started, release_cleanup = threading.Event(), threading.Event()
    cleanup_calls = []

    def operation(state, progress, cancelled):
        state["value"] = "committed"
        return {"committed": True}

    def cleanup(committed):
        cleanup_calls.append(committed)
        cleanup_started.set()
        assert release_cleanup.wait(5)
        if cleanup_fails:
            raise OSError("synthetic post-commit cleanup failure")

    job = service.submit(project_id, operation, cleanup=cleanup)
    assert cleanup_started.wait(5)
    deadline = time.monotonic() + 0.05

    try:
        started = time.monotonic()
        assert service.shutdown(wait=True, deadline=deadline) is False
        assert time.monotonic() - started < 0.5
        assert service.get(job["id"])["status"] == "running"
        with pytest.raises(ValueError, match="任务服务已关闭"):
            service.submit(project_id, lambda *args: None)
    finally:
        release_cleanup.set()

    result = wait_for_job(service, job["id"])
    assert result["status"] == "done"
    assert result["result"] == {"committed": True}
    assert result["cancel"] is False
    assert cleanup_calls == [True]
    assert projects.load(project_id)["value"] == "committed"
    assert service._idle.is_set()
    assert result.get("cleanup_failed", False) is cleanup_fails
    if cleanup_fails:
        assert "资源清理失败" in result["message"]
        assert service._shutdown_cancellation.cleanup_failures
    else:
        assert not service._shutdown_cancellation.cleanup_failures


def test_completed_job_history_is_bounded_without_pruning_active_job(task_store):
    projects, _ = task_store
    project_id = create_project(projects)
    service = ProjectTasks(
        projects,
        logging.getLogger("test.task_service.retention"),
        completed_job_limit=2,
    )
    started, release = threading.Event(), threading.Event()
    job_ids = []
    try:
        for value in range(3):
            job = service.submit(
                project_id,
                lambda state, progress, cancelled, value=value: {"value": value},
            )
            job_ids.append(job["id"])
            assert wait_for_job(service, job["id"])["status"] == "done"

        assert service.get(job_ids[0]) is None
        assert service.cancel(job_ids[0]) is False
        assert service.get(job_ids[1])["result"] == {"value": 1}
        assert service.get(job_ids[2])["result"] == {"value": 2}

        def active_operation(state, progress, cancelled):
            started.set()
            assert release.wait(5)
            return {"active": True}

        active = service.submit(project_id, active_operation)
        assert started.wait(5)
        assert service.get(job_ids[1]) is None
        assert service.get(active["id"])["status"] == "running"
        assert service.get(job_ids[2])["status"] == "done"
        release.set()

        result = wait_for_job(service, active["id"])
        assert result["status"] == "done"
        assert result["result"] == {"active": True}
        assert len(service.jobs) == 2
    finally:
        release.set()
        service.shutdown()


def test_batch_tasks_run_with_bounded_parallelism_and_isolate_item_failures(task_store):
    projects, service = task_store
    project_ids = [create_project(projects) for _ in range(10)]
    gate = threading.Lock()
    active = 0
    peak = 0

    def make_operation(index):
        def operation(state, progress, cancelled):
            nonlocal active, peak
            with gate:
                active += 1
                peak = max(peak, active)
            try:
                state["value"] = f"updated-{index}"
                progress(40, "分析中")
                time.sleep(0.04)
                if index == 2:
                    raise ValueError("synthetic item failure")
                return {"index": index}
            finally:
                with gate:
                    active -= 1
        return operation

    entries = [
        {"project": project_id, "title": f"song-{index}",
         "operation": make_operation(index)}
        for index, project_id in enumerate(project_ids)
    ]
    submitted = service.submit_batch(entries)
    result = wait_for_job(service, submitted["id"])

    assert result["status"] == "error"
    assert result["progress"] == 100
    assert result["result"] == {"done": 9, "error": 1, "cancelled": 0}
    assert peak == 2
    items = {item["project"]: item for item in result["items"]}
    assert [items[project_id]["status"] for project_id in project_ids] == [
        "done", "done", "error", *(["done"] * 7),
    ]
    assert items[project_ids[2]]["message"] == "synthetic item failure"
    assert items[project_ids[0]]["result"] == {"index": 0}
    assert items[project_ids[-1]]["result"] == {"index": 9}
    assert projects.load(project_ids[0])["value"] == "updated-0"
    assert projects.load(project_ids[2])["value"] == "original"
    assert projects.load(project_ids[-1])["value"] == "updated-9"
    assert service._idle.is_set()


def test_batch_project_members_are_locked_and_cancel_cancels_queued_items(task_store):
    projects, service = task_store
    project_ids = [create_project(projects) for _ in range(3)]
    started, release = threading.Event(), threading.Event()

    def operation(state, progress, cancelled):
        started.set()
        assert release.wait(5)
        return {"cancelled": cancelled()}

    submitted = service.submit_batch([
        {"project": project_id, "operation": operation}
        for project_id in project_ids
    ], max_parallel=1)
    assert started.wait(5)
    with pytest.raises(ValueError, match="该项目已有任务运行"):
        service.ensure_project_idle(project_ids[-1])
    assert service.cancel(submitted["id"])
    release.set()
    result = wait_for_job(service, submitted["id"])

    assert result["status"] == "cancelled"
    assert result["result"] == {"done": 0, "error": 0, "cancelled": 3}
    assert all(item["status"] == "cancelled" for item in result["items"])
    assert all(projects.load(project_id)["value"] == "original" for project_id in project_ids)
    assert service._idle.is_set()


def test_batch_shutdown_deadline_cancels_nested_workers_and_finishes_after_release(task_store):
    projects, service = task_store
    project_ids = [create_project(projects) for _ in range(2)]
    started = [threading.Event(), threading.Event()]
    release = threading.Event()

    def make_operation(index):
        def operation(state, progress, cancelled):
            started[index].set()
            assert release.wait(5)
            return {"cancelled": cancelled()}
        return operation

    submitted = service.submit_batch([
        {"project": project_id, "operation": make_operation(index)}
        for index, project_id in enumerate(project_ids)
    ])
    try:
        assert all(event.wait(5) for event in started)
        deadline = time.monotonic() + 0.05
        began = time.monotonic()
        assert service.shutdown(wait=True, deadline=deadline) is False
        assert time.monotonic() - began < 0.5
        assert service.get(submitted["id"])["cancel"] is True
    finally:
        release.set()
        service.shutdown()

    result = service.get(submitted["id"])
    assert result["status"] == "cancelled"
    assert all(item["status"] == "cancelled" for item in result["items"])
    assert service._idle.is_set()


@pytest.mark.parametrize("max_parallel", [0, 3, True])
def test_batch_rejects_unbounded_or_invalid_parallelism(task_store, max_parallel):
    projects, service = task_store
    project_ids = [create_project(projects) for _ in range(2)]
    with pytest.raises(ValueError, match="批量并行数"):
        service.submit_batch([
            {"project": project_id, "operation": lambda *args: {}}
            for project_id in project_ids
        ], max_parallel=max_parallel)


def test_batch_worker_start_failure_marks_all_items_terminal(task_store, monkeypatch):
    projects, service = task_store
    project_ids = [create_project(projects) for _ in range(2)]

    def fail_submit(operation):
        raise RuntimeError("synthetic batch executor failure")

    monkeypatch.setattr(service.executor, "submit", fail_submit)
    result = service.submit_batch([
        {"project": project_id, "operation": lambda *args: {}}
        for project_id in project_ids
    ])

    assert result["status"] == "error"
    assert result["progress"] == 100
    assert all(item["status"] == "error" for item in result["items"])
    assert service._idle.is_set()
    assert service._closed


def test_batch_tasks_accept_twenty_items_and_reject_twenty_one(task_store):
    projects, service = task_store
    project_ids = [create_project(projects) for _ in range(21)]
    entries = [
        {"project": project_id, "operation": lambda state, progress, cancelled: {"ok": True}}
        for project_id in project_ids
    ]

    with pytest.raises(ValueError, match="1–20"):
        service.submit_batch(entries, min_items=1)

    submitted = service.submit_batch(entries[:20], max_parallel=1, min_items=1)
    result = wait_for_job(service, submitted["id"])

    assert result["status"] == "done"
    assert len(result["items"]) == 20
    assert all(item["status"] == "done" for item in result["items"])


def test_persistent_batch_recovers_completed_items_and_marks_live_items_interrupted(tmp_path):
    projects = Projects(tmp_path / "projects")
    service = ProjectTasks(projects, logging.getLogger("test.task_service.persistence"))
    project_ids = [create_project(projects) for _ in range(3)]
    persistent_spec = {
        "kind": "generation",
        "payload": {"projects": project_ids, "preset": "hybrid_adaptive", "keys": 10},
    }
    submitted = service.submit_batch(
        [
            {
                "project": project_id,
                "operation": lambda state, progress, cancelled, index=index: {
                    "candidate_id": f"{index:032x}",
                },
            }
            for index, project_id in enumerate(project_ids)
        ],
        max_parallel=1,
        min_items=1,
        persistent_spec=persistent_spec,
    )
    finished = wait_for_job(service, submitted["id"])
    assert finished["status"] == "done"
    assert service.get(submitted["id"])["items"][0]["result"]["candidate_id"] == f"{0:032x}"

    record_path = projects.root.parent / "batch-jobs" / f"{submitted['id']}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    items = record["job"]["items"]
    items[0]["status"] = "done"
    items[0]["candidate_id"] = f"{0:032x}"
    items[1]["status"] = "running"
    items[2]["status"] = "queued"
    record["job"]["status"] = "running"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    service.shutdown()

    recovered = ProjectTasks(projects, logging.getLogger("test.task_service.recovery"))
    try:
        snapshot = recovered.get(submitted["id"])
        assert snapshot["status"] == "interrupted"
        assert snapshot["resume_available"] is True
        assert [item["status"] for item in snapshot["items"]] == [
            "done", "interrupted", "interrupted",
        ]
        assert snapshot["items"][0]["candidate_id"] == f"{0:032x}"
        assert recovered.batch_resume_spec(submitted["id"]) == persistent_spec
        assert recovered.recent_batches()[0]["id"] == submitted["id"]
    finally:
        recovered.shutdown()


def test_persistent_batch_journal_retries_transient_windows_replace_lock(
        tmp_path, monkeypatch):
    import manialab.task_service as task_service_module

    projects = Projects(tmp_path / "projects")
    service = ProjectTasks(projects, logging.getLogger("test.task_service.replace_retry"))
    project_id = create_project(projects)
    real_replace = task_service_module.os.replace
    attempts = []

    def flaky_replace(source, destination):
        if Path(destination).parent.name == "batch-jobs" and not attempts:
            attempts.append("locked")
            raise PermissionError("transient target lock")
        return real_replace(source, destination)

    monkeypatch.setattr(task_service_module.os, "replace", flaky_replace)
    monkeypatch.setattr(task_service_module, "_BATCH_REPLACE_RETRY_DELAYS", (0, 0))
    try:
        submitted = service.submit_batch(
            [{"project": project_id, "operation": lambda *_: {"candidate_id": "a" * 32}}],
            min_items=1,
            persistent_spec={"kind": "generation", "payload": {"projects": [project_id]}},
        )
        result = wait_for_job(service, submitted["id"])
        assert attempts == ["locked"]
        assert result["status"] == "done"
        assert (projects.root.parent / "batch-jobs" / f"{submitted['id']}.json").is_file()
    finally:
        service.shutdown()
