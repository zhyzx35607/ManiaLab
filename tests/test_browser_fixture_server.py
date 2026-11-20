import json
import os
from pathlib import Path
import runpy

import pytest

from manialab.storage import Projects


include_fixture_projects_in_recent = runpy.run_path(
    str(Path(__file__).parent / "browser" / "fixture_server.py")
)["include_fixture_projects_in_recent"]


def add_project(projects, title):
    project_id, _ = projects.create()
    projects.save({"id": project_id, "title": title, "mode": "audio"})
    return project_id


def test_recent_wrapper_without_preloaded_fixtures_keeps_dynamic_projects(tmp_path):
    projects = Projects(tmp_path / "projects")
    include_fixture_projects_in_recent(projects)
    project_id = add_project(projects, "dynamic")

    assert projects.recent() == [{"id": project_id, "title": "dynamic", "mode": "audio"}]


def test_recent_wrapper_merges_fixtures_and_sorts_by_state_mtime(tmp_path):
    projects = Projects(tmp_path / "projects")
    fixture_id = add_project(projects, "fixture")
    include_fixture_projects_in_recent(projects)
    dynamic_id = add_project(projects, "dynamic")
    fixture_path = projects.path(fixture_id) / "state.json"
    dynamic_path = projects.path(dynamic_id) / "state.json"

    os.utime(fixture_path, (1000, 1000))
    os.utime(dynamic_path, (2000, 2000))
    assert [project["id"] for project in projects.recent()] == [dynamic_id, fixture_id]

    os.utime(fixture_path, (3000, 3000))
    assert [project["id"] for project in projects.recent()] == [fixture_id, dynamic_id]


@pytest.mark.parametrize("damaged_state", [
    "invalid-json",
    "non-object-json",
    "invalid-id",
    "mismatched-id",
    "missing-mode",
])
def test_recent_wrapper_skips_damaged_fixture_state(tmp_path, damaged_state):
    projects = Projects(tmp_path / "projects")
    fixture_id = add_project(projects, "fixture")
    include_fixture_projects_in_recent(projects)
    dynamic_id = add_project(projects, "dynamic")
    fixture_path = projects.path(fixture_id) / "state.json"
    if damaged_state == "invalid-json":
        fixture_path.write_text("{broken", encoding="utf-8")
    elif damaged_state == "non-object-json":
        fixture_path.write_text("[]", encoding="utf-8")
    elif damaged_state == "invalid-id":
        fixture_path.write_text(json.dumps({"id": "../bad", "title": "fixture", "mode": "audio"}), encoding="utf-8")
    elif damaged_state == "mismatched-id":
        fixture_path.write_text(json.dumps({"id": dynamic_id, "title": "fixture", "mode": "audio"}), encoding="utf-8")
    else:
        fixture_path.write_text(json.dumps({"id": fixture_id, "title": "fixture"}), encoding="utf-8")

    recent = projects.recent()

    assert [project["id"] for project in recent] == [dynamic_id]
    assert all(project["title"] != "fixture" for project in recent)
