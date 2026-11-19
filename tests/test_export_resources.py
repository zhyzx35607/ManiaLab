import io
from pathlib import Path
import zipfile

from manialab.domain import Note
from manialab.server import create_app


ROOT = Path(__file__).resolve().parents[1]


def export_project(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    imported = client.post("/api/import", data={
        "files": (io.BytesIO(b"synthetic audio"), "song.wav"),
    }, headers=headers)
    assert imported.status_code == 200, imported.json
    project_id = imported.json["id"]
    projects = app.extensions["projects"]
    state = projects.load(project_id)
    state["variants"] = [{
        "mode": "audio",
        "timing_revision": 3,
        "settings": {"keys": 6, "seed": 1},
        "notes": [Note(1000, 1500, 1).data()],
        "bpm": 120,
        "offset": 0,
        "division": 4,
        "meter": 4,
    }]
    state["active"] = 0
    projects.save(state)
    return app, client, f"/api/projects/{project_id}/export/osz"


def test_osz_export_uses_temporary_file_and_closes_after_response(tmp_path, monkeypatch):
    from manialab import export_routes

    app, client, path = export_project(tmp_path)
    real_temporary_file = export_routes.tempfile.TemporaryFile
    files = []

    def tracked_temporary_file(*args, **kwargs):
        file = real_temporary_file(*args, **kwargs)
        files.append(file)
        return file

    monkeypatch.setattr(export_routes.tempfile, "TemporaryFile", tracked_temporary_file)
    try:
        response = client.get(path, buffered=False)
        assert response.status_code == 200
        assert len(files) == 1
        assert not files[0].closed

        payload = b"".join(response.response)
        response.close()

        assert files[0].closed
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            assert archive.testzip() is None
            assert set(archive.namelist()) == {"ManiaLab.osu", "song.wav"}
            assert archive.read("song.wav") == b"synthetic audio"
            assert b"128,192,1000,128,0,1500:0:0:0:0:" in archive.read("ManiaLab.osu")
    finally:
        app.extensions["executor"].shutdown()


def test_osz_export_closes_temporary_file_when_archive_build_fails(tmp_path, monkeypatch):
    from manialab import export_routes

    app, client, path = export_project(tmp_path)
    real_temporary_file = export_routes.tempfile.TemporaryFile
    real_writestr = export_routes.zipfile.ZipFile.writestr
    files = []

    def tracked_temporary_file(*args, **kwargs):
        file = real_temporary_file(*args, **kwargs)
        files.append(file)
        return file

    def fail_after_write(archive, *args, **kwargs):
        real_writestr(archive, *args, **kwargs)
        raise OSError("synthetic archive failure")

    monkeypatch.setattr(export_routes.tempfile, "TemporaryFile", tracked_temporary_file)
    monkeypatch.setattr(export_routes.zipfile.ZipFile, "writestr", fail_after_write)
    try:
        response = client.get(path)
        assert response.status_code == 500
        assert len(files) == 1
        assert files[0].closed
    finally:
        app.extensions["executor"].shutdown()
