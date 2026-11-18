from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import secrets
import threading
import time


TERMINAL_STATUSES = ("done", "error", "cancelled")


class ProjectTasks:
    """Serialize project work and commit its snapshot only after success."""

    def __init__(self, projects, logger=None, *, completed_job_limit=64):
        if type(completed_job_limit) is not int or completed_job_limit < 1:
            raise ValueError("completed_job_limit must be a positive integer")
        self.projects = projects
        self.logger = logger or logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.RLock()
        self.jobs = {}
        self.completed_job_limit = completed_job_limit
        self._closed = False

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
            return any(job["project"] == project_id and job["status"] in ("queued", "running")
                       for job in self.jobs.values())

    def ensure_project_idle(self, project_id):
        if self.has_active_for(project_id):
            raise ValueError("该项目已有任务运行，请等待或取消")

    def ensure_idle(self, message):
        if self.has_active():
            raise ValueError(message)

    def submit(self, project_id, operation, *, identity=None, cleanup=None):
        submission_error = None
        with self.lock:
            if self._closed:
                raise ValueError("任务服务已关闭")
            self.ensure_project_idle(project_id)
            self.ensure_idle("已有计算任务运行，请完成后再启动新任务")
            self._prune_completed_for_submission()
            job_id = secrets.token_hex(10)
            job = dict(id=job_id, project=project_id, status="queued", progress=0,
                       message="等待运行", cancel=False)
            if identity is not None:
                job["identity"] = identity
            self.jobs[job_id] = job

            def progress(value, message):
                with self.lock:
                    job.update(progress=value, message=message)

            def run():
                started = time.perf_counter()
                with self.lock:
                    if job["status"] != "queued":
                        return
                    job["status"] = "running"
                committed = False
                try:
                    if job["cancel"]:
                        raise InterruptedError("已取消，原有谱面已保留")
                    state = self.projects.load(project_id)
                    result = operation(state, progress, lambda: job["cancel"])
                    with self.lock:
                        if job["cancel"]:
                            raise InterruptedError("已取消，原有谱面已保留")
                        self.projects.save(state)
                        committed = True
                    outcome = dict(status="done", progress=100, message="完成", result=result,
                                   elapsed_seconds=round(time.perf_counter() - started, 2))
                except InterruptedError as error:
                    outcome = dict(status="cancelled", message=str(error))
                except BaseException as error:
                    self.logger.exception("Project task failed: project=%s job=%s", project_id, job_id)
                    message = str(error) if isinstance(error, ValueError) else "任务失败，请查看本地日志"
                    outcome = dict(status="error", message=message)
                finally:
                    if cleanup is not None:
                        try:
                            cleanup(committed)
                        except BaseException:
                            self.logger.exception("Project task cleanup failed: %s", project_id)
                    with self.lock:
                        job.update(outcome)

            try:
                self.executor.submit(run)
            except BaseException as error:
                self.logger.exception("Project task could not be submitted: project=%s job=%s",
                                      project_id, job_id)
                self._closed = True
                job.update(status="error", message="任务无法启动，请查看本地日志", cancel=True)
                self.executor.shutdown(wait=False, cancel_futures=True)
                submission_error = error
                if cleanup is not None:
                    try:
                        cleanup(False)
                    except BaseException:
                        self.logger.exception(
                            "Project task cleanup after submit failure failed: %s",
                            project_id,
                        )

        if submission_error is not None:
            if not isinstance(submission_error, Exception):
                raise submission_error
        with self.lock:
            return dict(job)

    def shutdown(self, *, wait=True):
        with self.lock:
            self._closed = True
            for job in self.jobs.values():
                if job["status"] in ("queued", "running"):
                    job["cancel"] = True
        self.executor.shutdown(wait=wait)

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job is not None else None

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            if job["status"] not in TERMINAL_STATUSES:
                job["cancel"] = True
            return True
