"""Errors that may be summarized safely in task status responses."""
from __future__ import annotations


class UserVisibleTaskError(RuntimeError):
    def __init__(self, message: str, *, log_path=None, global_failure=False):
        super().__init__(message)
        self.user_message = message
        self.log_path = str(log_path) if log_path is not None else None
        self.global_failure = bool(global_failure)


class UserVisibleTimeoutError(TimeoutError):
    def __init__(self, message: str, *, log_path=None, global_failure=False):
        super().__init__(message)
        self.user_message = message
        self.log_path = str(log_path) if log_path is not None else None
        self.global_failure = bool(global_failure)


def task_error_message(error) -> str:
    """Return only an explicitly safe summary, never a subprocess log tail."""
    if isinstance(error, (UserVisibleTaskError, UserVisibleTimeoutError)):
        message = error.user_message.splitlines()[0].strip()
        if error.log_path:
            message += f"；日志：{error.log_path}"
        return message or "任务失败，请查看本地日志"
    if isinstance(error, TimeoutError):
        message = str(error).splitlines()[0].strip()
        log_path = getattr(error, "log_path", None)
        if log_path:
            message += f"；日志：{log_path}"
        return message or "任务超时，请查看本地日志"
    if isinstance(error, ValueError):
        return str(error)
    return "任务失败，请查看本地日志"
