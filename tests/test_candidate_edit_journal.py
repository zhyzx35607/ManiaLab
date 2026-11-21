"""Candidate edit journals keep full audit history with linear disk growth."""
import io
import hashlib
import json
from pathlib import Path
import time
import wave

import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.server import create_app
from manialab.storage import Projects


ROOT = Path(__file__).resolve().parents[1]


def _wait_job(client, response):
    assert response.status_code == 200, response.json
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + response.json["id"]).json
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(.01)
    pytest.fail("Timed out waiting for candidate operation")


def _persisted(value):
    return json.loads(json.dumps(value))


def _project(tmp_path, *, analysis=None, notes=None):
    audio = io.BytesIO()
    with wave.open(audio, "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x00" * 40000)
    audio_bytes = audio.getvalue()
    audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    imported = client.post("/api/import", headers=headers,
        data={"files": (io.BytesIO(audio_bytes), "song.wav")})
    assert imported.status_code == 200, imported.json
    state = imported.json
    state["analysis"] = dict(analysis or {}, audio_sha256=audio_sha256)
    if notes is not None:
        state["variants"] = [dict(mode="audio", timing_revision=3, created=1.0,
            settings=parameters({"ln_ratio": 0, "phrase_ln": 0}),
            notes=[note.data() for note in notes], report={}, selection={},
            bpm=120, offset=0, division=4, audio_shift_ms=0, meter=4,
            source_audio_sha256=audio_sha256, region=None)]
        state["active"] = 0
    projects = app.extensions["projects"]
    projects.save(state)
    base = "/api/projects/" + state["id"]
    return app, client, headers, projects, state["id"], base


def test_repeated_edits_store_each_event_once_and_keep_full_projection(tmp_path):
    app, client, headers, projects, project_id, base = _project(
        tmp_path, analysis={"duration": 5}, notes=[Note(1000, 1000, 0, id="tap")])
    try:
        size_after_eight = None
        for index in range(40):
            state = client.get(base).json
            parent = state["variants"][state["active"]]
            response = client.post(base + "/edit", headers=headers, json={
                "base_created": parent["created"], "op": "update", "note_id": "tap",
                "values": {"lane": index % parent["settings"]["keys"]},
            })
            assert _wait_job(client, response)["status"] == "done"
            if index == 7:
                size_after_eight = (projects.path(project_id) / "state.json").stat().st_size

        state = client.get(base).json
        active = state["variants"][state["active"]]
        assert len(active["edits"]) == 40
        assert active["edits"][0]["parent_created"] == 1.0
        assert active["edits"][-1]["after"]["id"] == "tap"
        assert len(state["variants"]) == 8

        persisted = json.loads((projects.path(project_id) / "state.json").read_text(encoding="utf-8"))
        assert all("edits" not in variant for variant in persisted["variants"])
        assert len(persisted["_candidate_edit_journal"]) == 40
        assert (projects.path(project_id) / "state.json").stat().st_size < size_after_eight * 7

        exported = client.get(base + "/export/json").json["variant"]
        assert exported["edits"] == active["edits"]
        assert "_edit_journal_head" not in exported
    finally:
        app.extensions["executor"].shutdown()

    reopened = create_app(ROOT, tmp_path)
    try:
        client = reopened.test_client()
        restored = client.get(base).json
        assert restored["variants"][restored["active"]]["edits"] == active["edits"]
    finally:
        reopened.extensions["executor"].shutdown()


def test_edit_journal_keeps_branch_histories_separate(tmp_path):
    _, _, _, projects, project_id, _ = _project(
        tmp_path, analysis={"duration": 5}, notes=[Note(1000, 1000, 0, id="tap")])
    state = projects.load(project_id)
    parent = state["variants"][0]
    first_event = _persisted(dict(parent_created=parent["created"], op="update",
        before=parent["notes"][0], after=Note(1000, 1000, 1, id="tap").data()))
    first = dict(parent, created=2.0, edits=[first_event])
    state["variants"].append(first)
    state["active"] = 1
    projects.save(state)

    state = projects.load(project_id)
    parent = state["variants"][0]
    branch_event = _persisted(dict(parent_created=parent["created"], op="update",
        before=parent["notes"][0], after=Note(1000, 1000, 2, id="tap").data()))
    branch = dict(parent, created=3.0, edits=[branch_event])
    state["variants"].append(branch)
    state["active"] = 2
    projects.save(state)

    state = projects.load(project_id)
    first = next(item for item in state["variants"] if item["created"] == 2.0)
    second_event = _persisted(dict(parent_created=first["created"], op="update",
        before=first["notes"][0], after=Note(1000, 1000, 3, id="tap").data()))
    descendant = dict(first, created=4.0, edits=[*first["edits"], second_event])
    state["variants"].append(descendant)
    state["active"] = 3
    projects.save(state)

    restored = projects.load(project_id)
    histories = {item["created"]: item.get("edits", []) for item in restored["variants"]}
    assert histories[1.0] == []
    assert histories[2.0] == [first_event]
    assert histories[3.0] == [branch_event]
    assert histories[4.0] == [first_event, second_event]
    persisted = json.loads((projects.path(project_id) / "state.json").read_text(encoding="utf-8"))
    assert len(persisted["_candidate_edit_journal"]) == 3


def test_legacy_project_edit_history_loads_and_migrates_to_journal(tmp_path):
    projects = Projects(tmp_path / "projects")
    project_id, project_path = projects.create()
    edits = [dict(op="update", before={"lane": 0}, after={"lane": 1})]
    legacy_state = dict(id=project_id, title="legacy", mode="audio", variants=[
        dict(created=1.0, edits=edits),
    ])
    state_path = project_path / "state.json"
    state_path.write_text(json.dumps(legacy_state), encoding="utf-8")

    loaded = projects.load(project_id)
    assert loaded["variants"][0]["edits"] == edits
    projects.save(loaded)

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert "edits" not in persisted["variants"][0]
    assert len(persisted["_candidate_edit_journal"]) == 1
    assert projects.load(project_id)["variants"][0]["edits"] == edits


def test_load_rejects_edit_content_changed_without_updating_its_hash(tmp_path):
    projects = Projects(tmp_path / "projects")
    project_id, project_path = projects.create()
    projects.save({"id": project_id, "title": "synthetic", "mode": "audio", "variants": [
        {"created": 1.0, "edits": [
            {"op": "update", "before": {"lane": 0}, "after": {"lane": 1}},
        ]},
    ]})

    state_path = project_path / "state.json"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    node = next(iter(persisted["_candidate_edit_journal"].values()))
    node["event"]["after"]["lane"] = 5
    state_path.write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(ValueError, match="候选编辑历史链损坏"):
        projects.load(project_id)


def test_save_rejects_changed_journal_event_without_replacing_state(tmp_path):
    projects = Projects(tmp_path / "projects")
    project_id, project_path = projects.create()
    projects.save({"id": project_id, "title": "synthetic", "mode": "audio", "variants": [
        {"created": 1.0, "edits": [
            {"op": "update", "before": {"lane": 0}, "after": {"lane": 1}},
        ]},
    ]})

    state_path = project_path / "state.json"
    original = state_path.read_bytes()
    loaded = projects.load(project_id)
    loaded["variants"][0]["edits"][0]["after"]["lane"] = 5

    with pytest.raises(ValueError, match="候选编辑历史链损坏"):
        projects.save(loaded)

    assert state_path.read_bytes() == original
    assert not state_path.with_suffix(".tmp").exists()


def test_regional_child_retains_deleted_outside_edit_in_json_and_after_reopen(tmp_path):
    import hashlib

    audio = io.BytesIO()
    with wave.open(audio, "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        raw_audio = b"\x00\x00" * 40000
        stream.writeframes(raw_audio)
    # The imported WAV is the source identity used by the synthetic analysis.
    import_wave = io.BytesIO()
    with wave.open(import_wave, "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(raw_audio)
    audio_bytes = import_wave.getvalue()
    audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    imported = client.post("/api/import", headers=headers,
        data={"files": (io.BytesIO(audio_bytes), "song.wav")})
    assert imported.status_code == 200, imported.json
    state = imported.json
    state["analysis"] = dict(revision=3, duration=5, audio_sha256=audio_sha256,
        candidates=[Note(3500, 3500, id="new", source="transient").data()],
        phrase_candidates=[], timing_attacks=[])
    state["variants"] = [dict(mode="audio", timing_revision=3, created=1.0,
        settings=parameters({"keys": 6, "ln_ratio": 0, "phrase_ln": 0}),
        notes=[Note(500, 500, 0, id="locked").data(),
               Note(2000, 2000, 1, id="to-delete").data()],
        report={}, selection={}, bpm=120, offset=0, division=4,
        audio_shift_ms=0, meter=4, source_audio_sha256=audio_sha256, region=None)]
    state["active"] = 0
    projects = app.extensions["projects"]
    projects.save(state)
    base = "/api/projects/" + state["id"]
    try:
        parent = client.get(base).json["variants"][0]
        deleted_response = client.post(base + "/edit", headers=headers, json={
            "base_created": parent["created"], "op": "delete", "note_id": "to-delete",
        })
        assert _wait_job(client, deleted_response)["status"] == "done"
        deleted = client.get(base).json["variants"][1]
        assert deleted["edits"][-1]["op"] == "delete"
        assert deleted["edits"][-1]["after"] is None

        generated = client.post(base + "/generate", headers=headers, json={
            "mode": "audio", "settings": {"keys": 6, "ln_ratio": 0, "phrase_ln": 0},
            "bpm": 120, "offset": 0, "division": 4,
            "audio_shift_ms": 0, "timing_confirmed": True, "region": [3000, 4000],
        })
        assert _wait_job(client, generated)["status"] == "done"
        child = client.get(base).json["variants"][-1]
        assert child["notes"][0] == _persisted(Note(500, 500, 0, id="locked").data())
        assert child["notes"][1]["id"] == "new"
        assert child["notes"][1]["start"] == 3500
        assert child["edits"] == deleted["edits"]

        exported = client.get(base + "/export/json").json["variant"]
        assert exported["edits"] == child["edits"]
        assert exported["edits"][-1]["before"]["id"] == "to-delete"
        assert not any(note["id"] == "to-delete" for note in exported["notes"])
    finally:
        app.extensions["executor"].shutdown()

    reopened = create_app(ROOT, tmp_path)
    try:
        client = reopened.test_client()
        restored = client.get(base).json
        child = restored["variants"][restored["active"]]
        assert child["edits"][-1]["op"] == "delete"
        assert child["edits"][-1]["before"]["id"] == "to-delete"
        assert client.get(base + "/export/json").json["variant"] == child
    finally:
        reopened.extensions["executor"].shutdown()
