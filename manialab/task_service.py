from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import secrets
import threading
import time

from .cancellation import CancellationContext, remaining_until


TERMINAL_STATUSES = ("done", "error", "cancelled")
MAX_BATCH_SIZE = 10
MAX_BATCH_PARALLELISM = 2
_USER_RECOVERY_NOTE_PREFIX = "ManiaLab恢复提示："


def _user_recovery_hints(error):
    hints = []
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for note in getattr(current, "__notes__", ()):
            if note.startswith(_USER_RECOVERY_NOTE_PREFIX):
                hints.append(note[len(_USER_RECOVERY_NOTE_PREFIX):].strip())
        pending.extend(getattr(current, "exceptions", ()))
    return list(dict.fromkeys(hint for hint in hints if hint))


class ProjectTasks:
    """Serialize project work and commit its snapshot only after success."""

    def __init__(self, projects, logger=None, *, completed_job_limit=64):
        if type(completed_job_limit) is not int or completed_job_limit < 1:
            raise ValueError("completed_job_limit must be a positive integer")
        self.projects = projects
        self.logger = logger or logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.RLock()
        self._idle = threading.Event()
        self._idle.set()
        self.jobs = {}
        self._cancellations = {}
        self.completed_job_limit = completed_job_limit
        self._closed = False
        self._shutdown_requested = threading.Event()
        self._shutdown_cancellation = CancellationContext()

    def _prune_completed_for_submission(self):
        completed = [
            job_id for job_id, job in self.jobs.items()
            if job["status"] in TERMINAL_STATUSES
        ]
        excess = max(0, len(completed) - self.completed_job_limit + 1)
        for job_id in completed[:excess]:
            del self.jobs[job_id]

    def has_active(self):
        with self.lock:
            return any(job["status"] in ("queued", "running") for job in self.jobs.values())

    def has_active_for(self, project_id):
        with self.lock:
            return any((job.get("project") == project_id
                        or project_id in job.get("projects", ()))
                       and job["status"] in ("queued", "running")
                       for job in self.jobs.values())

    def ensure_project_idle(self, project_id):
        if self.has_active_for(project_id):
            raise ValueError("该项目已有任务运行，请等待或取消")

    def ensure_idle(self, message):
        with self.lock:
            if self._shutdown_cancellation.process_cleanup_failed:
                raise ValueError("子进程回收未完成，已暂停新任务；请确认相关进程已退出后重启程序")
            if self.has_active():
                raise ValueError(message)

    def submit(self, project_id, operation, *, identity=None, cleanup=None):
        submission_error = None
        if self._shutdown_requested.is_set():
            raise ValueError("任务服务已关闭")
        with self.lock:
            if self._closed or self._shutdown_requested.is_set():
                raise ValueError("任务服务已关闭")
            self.ensure_project_idle(project_id)
            self.ensure_idle("已有计算任务运行，请完成后再启动新任务")
            self._prune_completed_for_submission()
            job_id = secrets.token_hex(10)
            job = dict(id=job_id, project=project_id, status="queued", progress=0,
                       message="等待运行", cancel=False)
            cancellation = CancellationContext(parent=self._shutdown_cancellation)
            self._cancellations[job_id] = cancellation
            if identity is not None:
                job["identity"] = identity
            self.jobs[job_id] = job
            self._idle.clear()

            def progress(value, message):
                with self.lock:
                    job.update(progress=value, message=message)

            def record_cleanup_failure(outcome, committed, error):
                self.logger.exception(
                    "Project task cleanup failed: project=%s job=%s", project_id, job_id)
                cancellation.report_cleanup_failure(
                    f"Project task cleanup failed: project={project_id} job={job_id}")
                outcome["cleanup_failed"] = True
                if committed:
                    outcome["message"] = "结果已保存，但资源清理失败，请查看本地日志并重新打开项目核对"
                else:
                    message = ("任务已取消" if outcome["status"] == "cancelled"
                               else outcome["message"])
                    outcome.update(status="error",
                        message=message + "；资源清理失败，请重新打开项目核对")
                hints = _user_recovery_hints(error)
                if hints:
                    outcome["message"] += "；" + "；".join(hints)

            def run():
                started = time.perf_counter()
                with self.lock:
                    if job["status"] != "queued":
                        self._cancellations.pop(job_id, None)
                        if not any(item["status"] in ("queued", "running") for item in self.jobs.values()):
                            self._idle.set()
                        return
                    job["status"] = "running"
                committed = False
                try:
                    if cancellation():
                        raise InterruptedError("已取消，原有谱面已保留")
                    state = self.projects.load(project_id)
                    if cancellation():
                        raise InterruptedError("已取消，原有谱面已保留")
                    result = operation(state, progress, cancellation)
                    with self.lock:
                        if cancellation():
                            raise InterruptedError("已取消，原有谱面已保留")
                        self.projects.save(state)
                        committed = True
                    outcome = dict(status="done", progress=100, message="完成", result=result,
                                   elapsed_seconds=round(time.perf_counter() - started, 2))
                except InterruptedError as error:
                    outcome = dict(status="cancelled", message=str(error))
                    recovery_hints = _user_recovery_hints(error)
                    if recovery_hints:
                        cancellation.report_cleanup_failure(
                            f"Project task cleanup failed: project={project_id} job={job_id}")
                        outcome.update(
                            status="error",
                            cleanup_failed=True,
                            message="；".join([str(error), "取消后资源清理失败", *recovery_hints]),
                        )
                except BaseException as error:
                    self.logger.exception("Project task failed: project=%s job=%s", project_id, job_id)
                    message = str(error) if isinstance(error, ValueError) else "任务失败，请查看本地日志"
                    outcome = dict(status="error", message=message)
                    recovery_hints = _user_recovery_hints(error)
                    if recovery_hints:
                        cancellation.report_cleanup_failure(
                            f"Project task cleanup failed: project={project_id} job={job_id}")
                        outcome.update(cleanup_failed=True,
                                       message="；".join([message, "资源清理失败", *recovery_hints]))
                finally:
                    if cleanup is not None:
                        try:
                            cleanup(committed)
                        except BaseException as cleanup_error:
                            record_cleanup_failure(outcome, committed, cleanup_error)
                    with self.lock:
                        job.update(outcome)
                        if outcome["status"] == "done":
                            job["cancel"] = False
                        elif cancellation():
                            job["cancel"] = True
                        self._cancellations.pop(job_id, None)
                        if not any(item["status"] in ("queued", "running") for item in self.jobs.values()):
                            self._idle.set()

            try:
                self.executor.submit(run)
            except BaseException as error:
                self.logger.exception("Project task could not be submitted: project=%s job=%s",
                                      project_id, job_id)
                self._closed = True
                job.update(status="error", message="任务无法启动，请查看本地日志", cancel=True)
                cancellation.cancel()
                self._cancellations.pop(job_id, None)
                if not any(item["status"] in ("queued", "running") for item in self.jobs.values()):
                    self._idle.set()
                self.executor.shutdown(wait=False, cancel_futures=True)
                submission_error = error
                if cleanup is not None:
                    try:
                        cleanup(False)
                    except BaseException as cleanup_error:
                        record_cleanup_failure(job, False, cleanup_error)

        if submission_error is not None:
            if not isinstance(submission_error, Exception):
                raise submission_error
        with self.lock:
            return dict(job)

    def submit_batch(self, entries, *, max_parallel=MAX_BATCH_PARALLELISM,
                     identity=None, min_items=2):
        """Run independent project operations under one bounded batch job."""
        if type(min_items) is not int or not 1 <= min_items <= MAX_BATCH_SIZE:
            raise ValueError("批量任务最少项目数无效")
        if (not isinstance(entries, (list, tuple))
                or not min_items <= len(entries) <= MAX_BATCH_SIZE):
            raise ValueError(f"批量任务需要 {min_items}–{MAX_BATCH_SIZE} 首歌曲")
        if type(max_parallel) is not int or not 1 <= max_parallel <= MAX_BATCH_PARALLELISM:
            raise ValueError(f"批量并行数必须是 1–{MAX_BATCH_PARALLELISM}")
        normalized = []
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("批量任务项目无效")
            project_id, operation = entry.get("project"), entry.get("operation")
            if not isinstance(project_id, str) or not callable(operation):
                raise ValueError("批量任务项目或分析操作无效")
            if project_id in seen:
                raise ValueError("批量任务不能包含重复项目")
            seen.add(project_id)
            normalized.append({
                "project": project_id,
                "title": str(entry.get("title") or project_id),
                "operation": operation,
            })

        if self._shutdown_requested.is_set():
            raise ValueError("任务服务已关闭")
        submission_error = None
        with self.lock:
            if self._closed or self._shutdown_requested.is_set():
                raise ValueError("任务服务已关闭")
            for entry in normalized:
                self.ensure_project_idle(entry["project"])
            self.ensure_idle("已有计算任务运行，请完成后再启动批量采音")
            self._prune_completed_for_submission()
            job_id = secrets.token_hex(10)
            items = [dict(project=entry["project"], title=entry["title"],
                          status="queued", progress=0, message="等待运行")
                     for entry in normalized]
            job = dict(id=job_id, project=normalized[0]["project"],
                       projects=[entry["project"] for entry in normalized],
                       batch=True, max_parallel=max_parallel, status="queued", progress=0,
                       message="等待运行", cancel=False, items=items)
            if identity is not None:
                job["identity"] = identity
            cancellation = CancellationContext(parent=self._shutdown_cancellation)
            self._cancellations[job_id] = cancellation
            self.jobs[job_id] = job
            self._idle.clear()

            def update_item(item_index, **values):
                with self.lock:
                    items[item_index].update(values)
                    terminal = sum(item["status"] in TERMINAL_STATUSES for item in items)
                    if terminal == len(items):
                        job["progress"] = 100
                    else:
                        job["progress"] = round(sum(item["progress"] for item in items) / len(items))
                    if job["status"] == "running":
                        job["message"] = f"已完成 {terminal}/{len(items)} 首歌曲"

            def run_entry(index, entry):
                started = time.perf_counter()
                try:
                    if cancellation():
                        raise InterruptedError("已取消")
                    update_item(index, status="running", message="正在分析")
                    state = self.projects.load(entry["project"])
                    if cancellation():
                        raise InterruptedError("已取消，原有谱面已保留")

                    def progress(value, message):
                        value = max(0, min(100, int(value)))
                        update_item(index, progress=value, message=str(message))

                    result = entry["operation"](state, progress, cancellation)
                    with self.lock:
                        if cancellation():
                            raise InterruptedError("已取消，原有谱面已保留")
                        self.projects.save(state)
                    outcome = dict(status="done", progress=100, message="完成", result=result,
                                   elapsed_seconds=round(time.perf_counter() - started, 2))
                except InterruptedError as error:
                    outcome = dict(status="cancelled", progress=100,
                                   message=str(error) or "已取消")
                except BaseException as error:
                    self.logger.exception("Batch project task failed: project=%s batch=%s",
                                          entry["project"], job_id)
                    message = str(error) if isinstance(error, ValueError) else "任务失败，请查看本地日志"
                    outcome = dict(status="error", progress=100, message=message)
                finally:
                    update_item(index, **outcome)

            def run():
                with self.lock:
                    if job["status"] != "queued":
                        self._cancellations.pop(job_id, None)
                        if not any(item["status"] in ("queued", "running")
                                   for item in self.jobs.values()):
                            self._idle.set()
                        return
                    job["status"] = "running"
                try:
                    if cancellation():
                        for index in range(len(items)):
                            update_item(index, status="cancelled", progress=100, message="已取消")
                    else:
                        with ThreadPoolExecutor(max_workers=max_parallel,
                                                thread_name_prefix="manialab-batch") as batch_executor:
                            futures = [batch_executor.submit(run_entry, index, entry)
                                       for index, entry in enumerate(normalized)]
                            for future in as_completed(futures):
                                future.result()
                    statuses = [item["status"] for item in items]
                    counts = {status: statuses.count(status) for status in TERMINAL_STATUSES}
                    if counts["error"]:
                        status = "error"
                    elif counts["cancelled"] and cancellation():
                        status = "cancelled"
                    elif counts["cancelled"] == len(items):
                        status = "cancelled"
                    else:
                        status = "done"
                    outcome = dict(
                        status=status,
                        progress=100,
                        message=(f"批量处理结束：{counts['done']} 首成功，"
                                 f"{counts['error']} 首失败，{counts['cancelled']} 首取消"),
                        result={"done": counts["done"], "error": counts["error"],
                                "cancelled": counts["cancelled"]},
                        elapsed_seconds=round(time.perf_counter() - batch_started, 2),
                    )
                except BaseException as error:
                    self.logger.exception("Batch task failed: batch=%s", job_id)
                    outcome = dict(status="error", progress=100,
                                   message=(str(error) if isinstance(error, ValueError)
                                            else "批量任务失败，请查看本地日志"))
                    for index, item in enumerate(items):
                        if item["status"] not in TERMINAL_STATUSES:
                            update_item(index, status="error", progress=100,
                                        message=outcome["message"])
                finally:
                    with self.lock:
                        job.update(outcome)
                        if outcome["status"] == "done":
                            job["cancel"] = False
                        elif cancellation():
                            job["cancel"] = True
                        self._cancellations.pop(job_id, None)
                        if not any(item["status"] in ("queued", "running")
                                   for item in self.jobs.values()):
                            self._idle.set()

            batch_started = time.perf_counter()
            try:
                self.executor.submit(run)
            except BaseException as error:
                self.logger.exception("Batch task could not be submitted: batch=%s", job_id)
                self._closed = True
                message = "批量任务无法启动，请查看本地日志"
                job.update(status="error", progress=100, message=message, cancel=True)
                for item in items:
                    item.update(status="error", progress=100, message=message)
                cancellation.cancel()
                self._cancellations.pop(job_id, None)
                self._idle.set()
                self.executor.shutdown(wait=False, cancel_futures=True)
                submission_error = error

        if submission_error is not None and not isinstance(submission_error, Exception):
            raise submission_error
        with self.lock:
            return {**job, "items": [dict(item) for item in job["items"]]}

    def shutdown(self, *, wait=True, deadline=None):
        self._shutdown_requested.set()
        self._shutdown_cancellation.cancel(deadline=deadline)
        self._closed = True

        if deadline is None and wait:
            acquired = self.lock.acquire()
        elif deadline is None:
            acquired = self.lock.acquire(blocking=False)
        else:
            acquired = self.lock.acquire(timeout=remaining_until(deadline))
        if not acquired:
            self.executor.shutdown(wait=False)
            return False
        try:
            for job_id, job in self.jobs.items():
                if job["status"] in ("queued", "running"):
                    job["cancel"] = True
                    cancellation = self._cancellations.get(job_id)
                    if cancellation is not None:
                        cancellation.cancel(deadline=deadline)
            active = any(job["status"] in ("queued", "running") for job in self.jobs.values())
        finally:
            self.lock.release()

        if wait and deadline is None:
            self.executor.shutdown(wait=True)
            return not self._shutdown_cancellation.cleanup_failures

        self.executor.shutdown(wait=False)
        if not wait:
            return not active and not self._shutdown_cancellation.cleanup_failures
        if not active:
            return not self._shutdown_cancellation.cleanup_failures
        idle = self._idle.wait(remaining_until(deadline))
        return idle and not self._shutdown_cancellation.cleanup_failures

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            snapshot = dict(job)
            if "items" in snapshot:
                snapshot["items"] = [dict(item) for item in job["items"]]
            if (snapshot["status"] in ("queued", "running")
                    and self._shutdown_requested.is_set()):
                snapshot["cancel"] = True
            return snapshot

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if job["status"] not in TERMINAL_STATUSES:
                job["cancel"] = True
                cancellation = self._cancellations.get(job_id)
                if cancellation is not None:
                    cancellation.cancel()
            return True
