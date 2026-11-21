import io
from copy import deepcopy
from pathlib import Path
import time
import wave
import zipfile

import pytest

from manialab.arrange import parameters
from manialab.domain import Note, parse_osu, sections, values
from manialab.server import create_app
from manialab.storage import Projects


ROOT = Path(__file__).resolve().parents[1]


def _silent_audio():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 40000)
    buffer.seek(0)
    return buffer


def _edit(client, base, headers, parent_created, **changes):
    job = client.post(base + "/edit", json={"base_created": parent_created, **changes}, headers=headers)
    assert job.status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get("/api/jobs/" + job.json["id"]).json
        if status["status"] in ("done", "error", "cancelled"):
            assert status["status"] == "done", status["message"]
            return client.get(base).json
        time.sleep(.01)
    pytest.fail("Timed out editing the candidate")


def _generate(client, base, headers):
    response = client.post(base + "/generate", json={
        "mode": "audio",
        "settings": {"keys": 7},
        "bpm": 120,
        "offset": 0,
        "division": 4,
        "audio_shift_ms": 0,
        "timing_confirmed": True,
    }, headers=headers)
    assert response.status_code == 200, response.json
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get("/api/jobs/" + response.json["id"]).json
        if status["status"] in ("done", "error", "cancelled"):
            assert status["status"] == "done", status["message"]
            return client.get(base).json
        time.sleep(.01)
    pytest.fail("Timed out generating the candidate")


def _exported_notes(client, base, expected_bookmarks):
    chart = client.get(base + "/export/osu")
    bundle = client.get(base + "/export/osz")
    assert chart.status_code == bundle.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        assert archive.read("ManiaLab.osu") == chart.data
    text = chart.get_data(as_text=True)
    assert values(sections(text)["Editor"])["Bookmarks"] == expected_bookmarks
    return [(note.start, note.end, note.lane) for note in parse_osu(text)["notes"]]


def _export_payloads(client, base):
    chart = client.get(base + "/export/osu")
    bundle = client.get(base + "/export/osz")
    assert chart.status_code == bundle.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    assert contents["ManiaLab.osu"] == chart.data
    return chart.data, contents


@pytest.mark.parametrize("source", ["full", "regional", "legacy"])
def test_successive_edits_export_the_saved_candidates_section_bookmarks(tmp_path, source):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    try:
        project = client.post("/api/import", data={"files": (_silent_audio(), "silence.wav")}, headers=headers).json
        base = "/api/projects/" + project["id"]
        parent = dict(notes=[Note(1000, 2000, 0, id="hold").data(), Note(1500, 1500, 1, id="tap").data()],
                      mode="audio", created=1., settings=parameters(), report={}, timing_revision=3,
                      bpm=120, offset=0, division=4, audio_shift_ms=20, timing_confirmed=True)
        context = {}
        bookmarks = ""
        if source != "legacy":
            context = dict(structure={"boundaries": [0, 1500, 3000, 5000]}, expression={"source": "manual"})
            parent["selection"] = dict(context, selected=2, decisions=[{"id": "tap"}])
            parent["source_evidence"] = {"hold": {"detected": {"start": 1020, "end": 2020}}}
            bookmarks = "1480,2980"
        if source == "regional":
            parent["region"] = [0, 2000]
            parent["selection"].update(scope="regional_output", proposal={"scope": "full_song_proposal"})
        project.update(variants=[parent], active=0,
                       analysis={"duration": 5, "structure": {"boundaries": [0, 4000, 5000]}})
        Projects(tmp_path / "projects").save(project)
        parent = client.get(base).json["variants"][0]

        assert _exported_notes(client, base, bookmarks) == [(1000, 2000, 0), (1500, 1500, 1)]
        moved = _edit(client, base, headers, parent["created"], op="update", note_id="hold", values={"lane": 2})
        assert _exported_notes(client, base, bookmarks) == [(1000, 2000, 2), (1500, 1500, 1)]
        saved = _edit(client, base, headers, moved["variants"][moved["active"]]["created"], op="delete", note_id="tap")
        assert saved["variants"][0] == parent
        for edited in saved["variants"][1:]:
            selection = dict(edited["selection"])
            assert selection.pop("message")
            assert selection == context
            assert edited.get("source_evidence") == parent.get("source_evidence")
        assert saved["variants"][saved["active"]]["report"]["notes"] == 1
    finally:
        app.extensions["executor"].shutdown()

    reopened = create_app(ROOT, tmp_path)
    try:
        client = reopened.test_client()
        assert client.get(base).json["variants"] == saved["variants"]
        assert _exported_notes(client, base, bookmarks) == [(1000, 2000, 2)]
    finally:
        reopened.extensions["executor"].shutdown()


def test_late_edit_for_deselected_candidate_does_not_change_active_exports(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    try:
        project = client.post("/api/import", data={"files": (_silent_audio(), "silence.wav")}, headers=headers).json
        base = "/api/projects/" + project["id"]
        project.update(analysis={
            "revision": 13,
            "duration": 5,
            "candidates": [
                Note(1000, 1000, id="first").data(),
                Note(3000, 3000, id="second").data(),
            ],
            "phrase_candidates": [],
            "timing_attacks": [],
        })
        Projects(tmp_path / "projects").save(project)

        first_state = _generate(client, base, headers)
        first = deepcopy(first_state["variants"][first_state["active"]])
        second_state = _generate(client, base, headers)
        second = deepcopy(second_state["variants"][second_state["active"]])
        assert len(second_state["variants"]) == 2
        assert second_state["active"] == 1

        edited_state = _edit(client, base, headers, second["created"], op="update",
            note_id="first", values={"lane": (second["notes"][0]["lane"] + 1) % 7})
        assert len(edited_state["variants"]) == 3
        assert edited_state["active"] == 2
        assert edited_state["variants"][0] == first
        assert edited_state["variants"][1] == second

        selected = client.post(base + "/variant", json={"index": 0}, headers=headers)
        assert selected.status_code == 200, selected.json
        before = client.get(base).json
        assert before["active"] == 0
        assert before["variants"] == edited_state["variants"]
        before_exports = _export_payloads(client, base)

        delayed = client.post(base + "/edit", json={
            "base_created": second["created"],
            "op": "update",
            "note_id": "first",
            "values": {"lane": (second["notes"][0]["lane"] + 2) % 7},
        }, headers=headers)
        assert delayed.status_code == 200, delayed.json
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = client.get("/api/jobs/" + delayed.json["id"]).json
            if status["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(.01)
        assert status["status"] == "error"
        assert "候选版本已经改变" in status["message"]

        after = client.get(base).json
        assert after["variants"] == before["variants"]
        assert after["active"] == before["active"] == 0
        assert _export_payloads(client, base) == before_exports
    finally:
        app.extensions["executor"].shutdown()
