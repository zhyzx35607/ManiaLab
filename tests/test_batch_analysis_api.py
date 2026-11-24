"""Batch analysis accepts imported projects and reports per-song outcomes."""
import io
from pathlib import Path
import threading
import time

import pytest

from manialab.server import create_app


ROOT = Path(__file__).resolve().parents[1]


def _wait_job(client, job_id):
    for _ in range(500):
        job = client.get(f"/api/jobs/{job_id}").json
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.01)
    pytest.fail(f"batch job {job_id} did not finish")


def _import_song(client, headers, name):
    response = client.post(
        "/api/import", headers=headers,
        data={"files": (io.BytesIO(b"synthetic placeholder audio"), name)},
    )
    assert response.status_code == 200, response.json
    return response.json


def test_batch_analysis_reports_independent_results_with_max_two_active(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    songs = [
        _import_song(client, headers, "first.wav"),
        _import_song(client, headers, "broken.wav"),
        _import_song(client, headers, "third.wav"),
    ]
    gate = threading.Lock()
    active = 0
    peak = 0

    def analyze(source, *args):
        nonlocal active, peak
        with gate:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)
            if source.name == "broken.wav":
                raise ValueError("synthetic decode error")
            return {
                "revision": 1,
                "candidates": [],
                "timing_attacks": [],
                "tempo": {"bpm": 120, "offset": 0},
            }
        finally:
            with gate:
                active -= 1

    monkeypatch.setattr(app.extensions["analysis_services"], "analyze", analyze)
    try:
        response = client.post(
            "/api/batch/analyze", headers=headers,
            json={"projects": [song["id"] for song in songs]},
        )
        assert response.status_code == 200, response.json
        queued = response.json
        assert queued["batch"] is True
        assert queued["max_parallel"] == 2
        assert [item["title"] for item in queued["items"]] == [
            "first", "broken", "third",
        ]
        result = _wait_job(client, queued["id"])
        assert result["status"] == "error"
        assert result["result"] == {"done": 2, "error": 1, "cancelled": 0}
        assert [item["status"] for item in result["items"]] == [
            "done", "error", "done",
        ]
        assert result["items"][1]["message"] == "synthetic decode error"
        assert result["items"][0]["result"] == {"candidates": 0}
        assert 1 < peak <= 2
        projects = app.extensions["projects"]
        assert projects.load(songs[0]["id"])["analysis"]["revision"] == 1
        assert "analysis" not in projects.load(songs[1]["id"])
        assert projects.load(songs[2]["id"])["analysis"]["revision"] == 1
    finally:
        app.extensions["task_service"].shutdown()


@pytest.mark.parametrize(
    ("projects", "max_parallel", "message"),
    [
        (["a"], 2, "2–5 首歌曲"),
        (["a", "a"], 2, "不能重复选择"),
        (["a", "b"], 3, "只能为 1 或 2"),
    ],
)
def test_batch_analysis_rejects_invalid_selection_and_parallelism(
        tmp_path, projects, max_parallel, message):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    try:
        response = client.post(
            "/api/batch/analyze", headers=headers,
            json={"projects": projects, "max_parallel": max_parallel},
        )
        assert response.status_code == 400
        assert message in response.json["error"]
    finally:
        app.extensions["task_service"].shutdown()
