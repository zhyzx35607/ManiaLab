import json
import os
from pathlib import Path
import threading

import pytest

from manialab import storage
from manialab.storage import Projects


def project_with_saved_state(tmp_path):
    projects = Projects(tmp_path / "projects")
    project_id, _ = projects.create()
    projects.save({"id": project_id, "title": "before", "mode": "audio"})
    state_path = projects.path(project_id) / "state.json"
    return projects, project_id, state_path, state_path.with_suffix(".tmp")


def test_save_removes_partial_temporary_file_after_write_failure(tmp_path, monkeypatch):
    projects, project_id, state_path, temporary = project_with_saved_state(tmp_path)
    previous = state_path.read_bytes()
    write_text = Path.write_text

    def fail_after_partial_write(path, text, *args, **kwargs):
        if path == temporary:
            with path.open("w", encoding=kwargs.get("encoding", "utf-8")) as output:
                output.write(text[:8])
                output.flush()
            raise OSError("simulated state write failure")
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated state write failure"):
        projects.save({"id": project_id, "title": "after", "mode": "audio"})

    assert state_path.read_bytes() == previous
    assert not temporary.exists()


def test_save_removes_temporary_file_after_replace_failure(tmp_path, monkeypatch):
    projects, project_id, state_path, temporary = project_with_saved_state(tmp_path)
    previous = state_path.read_bytes()
    replace = storage.os.replace

    def fail_replace(source, destination):
        if Path(source) == temporary:
            raise PermissionError("simulated state replace failure")
        return replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", fail_replace)

    with pytest.raises(PermissionError, match="simulated state replace failure"):
        projects.save({"id": project_id, "title": "after", "mode": "audio"})

    assert state_path.read_bytes() == previous
    assert not temporary.exists()


def test_save_does_not_clean_temporary_path_after_successful_replace(tmp_path, monkeypatch):
    projects, project_id, state_path, temporary = project_with_saved_state(tmp_path)
    unlink = Path.unlink

    def fail_if_unlinked(path, *args, **kwargs):
        if path == temporary:
            raise OSError("simulated post-replace temporary cleanup failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_if_unlinked)

    projects.save({"id": project_id, "title": "after", "mode": "audio"})

    assert projects.load(project_id)["title"] == "after"
    assert not temporary.exists()
    assert state_path.is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows readers can block replace of an open state file")
def test_save_retries_after_concurrent_state_reader_closes(tmp_path, monkeypatch):
    projects, project_id, state_path, _ = project_with_saved_state(tmp_path)
    reader = state_path.open("rb")
    replace_blocked = threading.Event()
    lock_errors = []
    save_errors = []
    replace = storage.os.replace

    def observe_replace(source, destination):
        if Path(destination) == state_path:
            try:
                return replace(source, destination)
            except PermissionError as error:
                lock_errors.append(error)
                replace_blocked.set()
                raise
        return replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", observe_replace)

    def save():
        try:
            projects.save({"id": project_id, "title": "after", "mode": "audio"})
        except BaseException as error:
            save_errors.append(error)

    worker = threading.Thread(target=save)
    worker.start()
    try:
        assert replace_blocked.wait(5)
        assert getattr(lock_errors[0], "winerror", None) == 5
    finally:
        reader.close()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert not save_errors
    assert projects.load(project_id)["title"] == "after"


def test_recent_returns_only_the_most_recent_thirty_projects(tmp_path):
    projects = Projects(tmp_path / "projects")
    project_ids = []
    for index in range(31):
        project_id, _ = projects.create()
        projects.save({"id": project_id, "title": str(index), "mode": "audio"})
        os.utime(projects.path(project_id) / "state.json", (1000 + index, 1000 + index))
        project_ids.append(project_id)

    recent = projects.recent()

    assert len(recent) == 30
    assert [project["id"] for project in recent] == list(reversed(project_ids[1:]))


def test_legacy_candidates_receive_distinct_persistent_ids(tmp_path):
    projects, project_id, state_path, _ = project_with_saved_state(tmp_path)
    legacy = json.loads(state_path.read_text(encoding="utf-8"))
    legacy.update(variants=[{"created": 1.0}, {"created": 1.0}], active=1)
    state_path.write_text(json.dumps(legacy), encoding="utf-8")

    state = projects.load(project_id)
    ids = [variant["candidate_id"] for variant in state["variants"]]
    assert len(set(ids)) == 2
    assert [variant["candidate_id"] for variant in
            json.loads(state_path.read_text(encoding="utf-8"))["variants"]] == ids
    assert Projects(projects.root).load(project_id) == state
