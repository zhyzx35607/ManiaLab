import io
from pathlib import Path
import time
import zipfile

import pytest

from manialab.domain import Note, export_osu, parse_osu
from manialab.server import create_app
from manialab.storage import Projects


ROOT = Path(__file__).parents[1]


def test_project_routes_keep_endpoints_and_chart_variant_state(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    try:
        assert {"import_files", "get_project", "choose_chart", "choose_variant"} <= set(app.view_functions)
        charts = [
            ("a.osu", export_osu([Note(1000, 1000, 0)], 6, ROOT / "vendor/mapperatorinator/template.osu",
                                  audio="song.wav")),
            ("b.osu", export_osu([Note(2000, 2000, 1)], 7, ROOT / "vendor/mapperatorinator/template.osu",
                                  audio="other.wav")),
        ]
        imported = client.post("/api/import", data={"files": [
            (io.BytesIO(content.encode()), name) for name, content in charts
        ] + [(io.BytesIO(b"audio"), "song.wav"), (io.BytesIO(b"other audio"), "other.wav")]}, headers=headers)
        assert imported.status_code == 200, imported.json
        project_id = imported.json["id"]
        projects = Projects(tmp_path / "projects")
        state = projects.load(project_id)
        state["analysis"] = {"sentinel": True}
        state["expression"] = {"sentinel": True}
        state["plan"] = {"sentinel": True}
        state["provided_stems"] = {"sentinel": True}
        projects.save(state)
        (projects.path(project_id) / "activations.npz").write_bytes(b"stale")
        (projects.path(project_id) / "preview.wav").write_bytes(b"stale")

        selected = client.post(f"/api/projects/{project_id}/chart", json={"path": "b.osu"}, headers=headers)
        assert selected.status_code == 200, selected.json
        assert selected.json["original"] == "b.osu"
        assert selected.json["input_keys"] == 7
        assert selected.json["audio"] == "other.wav"
        assert selected.json["variants"] == []
        assert selected.json["active"] == -1
        assert "analysis" not in selected.json
        assert "expression" not in selected.json
        assert "plan" not in selected.json
        assert "provided_stems" not in selected.json
        assert (projects.path(project_id) / "activations.npz").read_bytes() == b"stale"
        assert client.get(f"/api/projects/{project_id}/audio").data == b"other audio"

        state = projects.load(project_id)
        state["variants"] = [
            {"settings": {"keys": 6}, "notes": []},
            {"settings": {"keys": 8}, "notes": []},
        ]
        projects.save(state)
        selected = client.post(f"/api/projects/{project_id}/variant", json={"index": 1}, headers=headers)
        assert selected.status_code == 200, selected.json
        assert selected.json["active"] == 1
        assert selected.json["settings"]["keys"] == 8
        assert "expression" not in selected.json

        restored = client.post(f"/api/projects/{project_id}/chart", json={"path": "a.osu"}, headers=headers)
        assert restored.status_code == 200
        for field in ("analysis", "expression", "plan", "provided_stems"):
            assert restored.json[field] == {"sentinel": True}
        assert client.get(f"/api/projects/{project_id}/audio").data == b"stale"
        again = client.post(f"/api/projects/{project_id}/chart", json={"path": "b.osu"}, headers=headers)
        assert again.json == selected.json
    finally:
        app.extensions["executor"].shutdown()


@pytest.mark.parametrize(
    ("route", "payload"),
    [
        ("chart", {"path": "missing.osu"}),
        ("variant", {"index": None}),
        ("variant", {"index": -1}),
        ("variant", {"index": 0}),
        ("variant", {"index": True}),
    ],
)
def test_project_routes_reject_unknown_chart_and_invalid_variant(tmp_path, route, payload):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    try:
        chart = export_osu([Note(1000, 1000, 0)], 6, ROOT / "vendor/mapperatorinator/template.osu",
                           audio="song.wav")
        imported = client.post("/api/import", data={"files": [
            (io.BytesIO(chart.encode()), "chart.osu"), (io.BytesIO(b"audio"), "song.wav")
        ]}, headers=headers)
        assert imported.status_code == 200, imported.json
        project_id = imported.json["id"]
        before = Projects(tmp_path / "projects").load(project_id)

        response = client.post(f"/api/projects/{project_id}/{route}", json=payload, headers=headers)

        assert response.status_code == 400
        assert Projects(tmp_path / "projects").load(project_id) == before
    finally:
        app.extensions["executor"].shutdown()


def test_get_project_backfills_legacy_input_timing(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    try:
        chart = export_osu([Note(1000, 1000, 0)], 6, ROOT / "vendor/mapperatorinator/template.osu",
                           audio="song.wav")
        imported = client.post("/api/import", data={"files": [
            (io.BytesIO(chart.encode()), "chart.osu"), (io.BytesIO(b"audio"), "song.wav")
        ]}, headers=headers)
        assert imported.status_code == 200, imported.json
        project_id = imported.json["id"]
        projects = Projects(tmp_path / "projects")
        state = projects.load(project_id)
        state.pop("input_timing")
        projects.save(state)

        response = client.get(f"/api/projects/{project_id}", headers=headers)

        assert response.status_code == 200
        assert response.json["input_timing"] == parse_osu(chart)["timing"]
    finally:
        app.extensions["executor"].shutdown()


def _finish(client, base, headers, action, payload):
    response = client.post(f"{base}/{action}", json=payload, headers=headers)
    assert response.status_code == 200, response.json
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + response.json["id"]).json
        if job["status"] not in ("queued", "running"):
            assert job["status"] == "done", job
            return client.get(base).json
        time.sleep(.01)
    pytest.fail(f"{action} timed out")


@pytest.mark.parametrize("same_audio", [True, False])
def test_chart_switch_preserves_saved_edits_and_uses_its_own_audio_across_restarts(tmp_path, same_audio):
    from manialab.chart_workspace import audio_work

    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    archive = io.BytesIO()
    audio_b = "mix.wav" if same_audio else "other.wav"
    sources = {"a.osu": (6, 1003, "mix.wav"), "b.osu": (9, 2100, audio_b)}
    with zipfile.ZipFile(archive, "w") as out:
        for name, (keys, start, audio) in sources.items():
            chart = export_osu([Note(start, start + 750, 1)], keys, ROOT / "vendor/mapperatorinator/template.osu",
                               title=name, audio=audio)
            out.writestr(name, chart)
        for name in {audio_b, "mix.wav"}:
            out.writestr(name, name.encode())
    archive.seek(0)
    try:
        state = client.post("/api/import", data={"files": (archive, "two.osz")}, headers=headers).json
        base = "/api/projects/" + state["id"]
        projects = app.extensions["projects"]
        project = projects.path(state["id"])
        # Older versions stored the active difficulty and its derived media at the project root.
        state.pop("cache_origin_audio", None)
        projects.save(state)
        history = _finish(client, base, headers, "generate", {"mode": "remap", "settings": {"keys": 8}})
        version = history["variants"][0]
        changed_lane = (version["notes"][0]["lane"] + 1) % 8
        edited = _finish(client, base, headers, "edit", {
            "op": "update", "base_created": version["created"], "note_id": version["notes"][0]["id"],
            "values": {"lane": changed_lane},
        })
        preview_a = project / "preview.wav"
        preview_a.write_bytes(b"preview A")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(projects, "save", lambda *_: pytest.fail("same-source selection performed a write"))
            assert client.post(base + "/chart", json={"path": "a.osu"}, headers=headers).json == edited
        other = client.post(base + "/chart", json={"path": "b.osu"}, headers=headers).json
        assert other["variants"] == [] and other["active"] == -1
        assert other["settings"]["keys"] == 9 and other["input_notes"][0]["start"] == 2100
        work_b = audio_work(project, other)
        assert (work_b == project) == same_audio
        assert client.get(base + "/audio").data == (b"preview A" if same_audio else b"other.wav")
        if not same_audio:
            (work_b / "preview.wav").write_bytes(b"preview B")
        _finish(client, base, headers, "generate", {"mode": "remap", "settings": {"keys": 9}})
    finally:
        app.extensions["executor"].shutdown()

    reopened = create_app(ROOT, tmp_path)
    try:
        client = reopened.test_client()
        headers = {"X-ManiaLab-Token": reopened.config["TOKEN"]}
        restored = client.post(base + "/chart", json={"path": "a.osu"}, headers=headers).json
        assert restored["variants"] == edited["variants"] and restored["active"] == edited["active"]
        assert restored["settings"]["keys"] == 8
        assert client.get(base + "/audio").data == b"preview A"
        with zipfile.ZipFile(io.BytesIO(client.get(base + "/export/osz").data)) as output:
            contents = parse_osu(output.read("ManiaLab.osu").decode())
            assert [(n.start, n.end, n.lane) for n in contents["notes"]] == [(1003, 1753, changed_lane)]
            assert output.read(contents["general"]["AudioFilename"]) == b"mix.wav"
        other = client.post(base + "/chart", json={"path": "b.osu"}, headers=headers).json
        assert len(other["variants"]) == 1 and other["active"] == 0
        assert client.get(base + "/audio").data == (b"preview A" if same_audio else b"preview B")
    finally:
        reopened.extensions["executor"].shutdown()


def test_switch_to_missing_audio_cannot_serve_previous_tracks(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    try:
        files = []
        for name, audio in [("a.osu", "song.wav"), ("b.osu", "missing.wav")]:
            chart = export_osu([Note(1000, 1500, 1)], 7, ROOT / "vendor/mapperatorinator/template.osu", audio=audio)
            files.append((io.BytesIO(chart.encode()), name))
        state = client.post("/api/import", data={"files": files + [(io.BytesIO(b"song"), "song.wav")]},
                            headers=headers).json
        base = "/api/projects/" + state["id"]
        work = app.extensions["projects"].path(state["id"])
        (work / "preview.wav").write_bytes(b"preview")
        (work / "stems").mkdir()
        (work / "stems/drums.wav").write_bytes(b"other source")
        switched = client.post(base + "/chart", json={"path": "b.osu"}, headers=headers).json
        assert switched["audio"] is None
        assert client.get(base + "/audio").status_code == 400
        assert client.get(base + "/stem/drums").status_code == 400
        assert client.get(base + "/waveform").status_code == 400
    finally:
        app.extensions["executor"].shutdown()
