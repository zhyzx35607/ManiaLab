import io
from pathlib import Path

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
        assert not (projects.path(project_id) / "activations.npz").exists()
        assert not (projects.path(project_id) / "preview.wav").exists()

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
