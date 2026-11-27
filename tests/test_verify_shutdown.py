import pytest

from scripts.verify_shutdown import verify_core_shutdown


def test_core_shutdown_requires_worker_free_status_and_app_exit():
    status = {
        "version": "test",
        "separator": {
            "available": False,
            "profiles": {
                "four": {"available": False},
                "six": {"available": False},
            },
        },
    }

    report = verify_core_shutdown(status, "test", [], [], True)

    assert report == {
        "passed": True,
        "build_profile": "core",
        "worker_expected": False,
        "worker_processes_before_exit": [],
        "worker_processes_after_exit": [],
        "application_exited": True,
        "worker_stopped_on_parent_exit": None,
    }


@pytest.mark.parametrize(
    ("before", "after", "exited"),
    [([123], [], True), ([], [123], True), ([], [], False)],
)
def test_core_shutdown_rejects_workers_or_a_live_parent(before, after, exited):
    status = {
        "version": "test",
        "separator": {
            "available": False,
            "profiles": {
                "four": {"available": False},
                "six": {"available": False},
            },
        },
    }

    with pytest.raises(ValueError):
        verify_core_shutdown(status, "test", before, after, exited)


def test_core_shutdown_rejects_unexpectedly_available_separator():
    status = {
        "version": "test",
        "separator": {
            "available": True,
            "profiles": {
                "four": {"available": True},
                "six": {"available": False},
            },
        },
    }

    with pytest.raises(ValueError, match="unexpectedly reports"):
        verify_core_shutdown(status, "test", [], [], True)


def test_workbench_shutdown_report_keeps_its_declared_profile():
    status = {
        "version": "test",
        "separator": {
            "available": False,
            "profiles": {
                "four": {"available": False},
                "six": {"available": False},
            },
        },
    }

    report = verify_core_shutdown(
        status, "test", [], [], True, build_profile="workbench",
    )

    assert report["build_profile"] == "workbench"
    assert report["passed"] is True
    assert report["worker_expected"] is False
