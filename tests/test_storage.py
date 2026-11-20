import os
from pathlib import Path

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
