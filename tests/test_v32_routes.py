from __future__ import annotations

import hashlib
import io
from pathlib import Path
import threading
import time
import zipfile

import numpy as np
import pytest
from flask import Flask, jsonify

from manialab.audio_cache import sha256_file
from manialab.domain import Note, export_osu, parse_osu
from manialab.export_routes import register_export_routes
from manialab.separation import PROFILES
from manialab.storage import Projects as ProjectStore
from manialab.task_service import ProjectTasks
from manialab.v32_engine import MODEL_SPECS, V32Engine
from manialab.v32_routes import register_v32_routes


ROOT = Path(__file__).parents[1]


def _osu(keys: int, *, audio: str = "song.wav", notes=None) -> str:
    notes = notes or [Note(500, 1500, 0, id="hold"), Note(1000, 1000, 1, id="tap")]
    return export_osu(
        notes,
        keys,
        ROOT / "vendor" / "mapperatorinator" / "template.osu",
        title="V32 fixture",
        artist="Test",
        audio=audio,
        bpm=120,
        offset=0,
        grid_division=4,
    )


class FakeEngine:
    available = True
    install_available = False
    generate_started = None
    wait_for_cancel = False
    calls = []

    @staticmethod
    def validate_options(**options):
        return V32Engine.validate_options(**options)

    def status(self):
        return {
            "available": self.available,
            "install_available": self.install_available,
        }

    def install_runtime(self, *, progress, cancelled):
        assert not cancelled()
        progress(100, "V32 推理环境已就绪")
        return {"device": "cuda", "gpu": "RTX 4050"}

    def generate(self, **kwargs):
        self.calls.append(kwargs.copy())
        if self.generate_started is not None:
            self.generate_started.set()
        while self.wait_for_cancel:
            if kwargs["cancelled"]():
                raise InterruptedError("已取消，原有候选已保留")
            time.sleep(0.005)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        chart_path = output / "generated.osu"
        chart_path.write_text(_osu(kwargs["keycount"]), encoding="utf-8")
        variant = kwargs["model_variant"]
        return {
            "map": chart_path,
            "model_variant": variant,
            "model_revision": MODEL_SPECS[variant]["revision"],
            "device": {
                "device": "cpu",
                "name": "CPU",
                "memory_gib": None,
                "precision": "fp32",
                "fast_decoder_loop": False,
                "max_batch_size": 1,
                "cpu_threads": 2,
            },
            "fast_decoder_loop_active": False,
            "lora_loaded": False,
            "lora_path": None,
            "elapsed_seconds": 0.01,
            "peak_rss_bytes": 1024,
        }


class FakeSeparator:
    def __init__(self, _resources, profile):
        assert profile == "six"

    def status(self):
        return {"available": True}

    def separate(self, audio, work, progress, cancelled, *, include_beats):
        assert include_beats is False
        assert not cancelled()
        directory = Path(work) / "stems-six"
        directory.mkdir(parents=True, exist_ok=True)
        for voice in PROFILES["six"]["voices"]:
            np.save(directory / f"{voice}.npy", np.full(90000, 0.2, dtype=np.float32))
        return directory, {
            "audio_sha256": sha256_file(audio),
            "model": "fake-six-source",
            "model_sha256": "a" * 64,
            "profile": "six",
        }


@pytest.fixture
def route_app(tmp_path, monkeypatch):
    import manialab.v32_routes as routes

    monkeypatch.setattr(routes, "Separator", FakeSeparator)
    monkeypatch.setattr(routes, "convert_osu_text", _fake_torosu)

    def fake_decode(_audio, work, *, cancelled):
        assert not cancelled()
        path = Path(work) / "preview.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"decoded test preview")
        return np.zeros(90000, dtype=np.float32)

    monkeypatch.setattr(routes, "decode", fake_decode)
    app = Flask(__name__)
    projects = ProjectStore(tmp_path / "projects")
    tasks = ProjectTasks(projects)
    project_id, project_path = projects.create()
    audio_path = project_path / "files" / "song.wav"
    audio_path.write_bytes(b"test audio bytes")
    audio_sha = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    existing = {
        "candidate_id": "a" * 32,
        "mode": "audio",
        "notes": [Note(500, 500, 0, id="old").data()],
        "settings": {"keys": 10, "key_layout": "10k_5_5"},
        "bpm": 120,
        "offset": 0,
        "division": 4,
        "timing_revision": 3,
        "source_audio_sha256": audio_sha,
    }
    projects.save({
        "id": project_id,
        "mode": "audio",
        "title": "Test song",
        "artist": "Test artist",
        "audio": "song.wav",
        "original": None,
        "charts": [],
        "input_notes": [],
        "settings": {"keys": 10, "key_layout": "10k_5_5"},
        "analysis": {
            "revision": 3,
            "audio_sha256": audio_sha,
            "duration": 10,
            "tempo": {"bpm": 120, "offset": 0, "division": 4, "stable": True},
        },
        "variants": [existing],
        "active": 0,
    })
    engine = FakeEngine()
    engine.calls = []

    def current_audio(state):
        return projects.path(state["id"]) / "files" / state["audio"]

    def requested_tempo(payload, state):
        return {
            "bpm": float(payload.get("bpm", 120)),
            "offset": float(payload.get("offset", 0)),
            "division": int(payload.get("division", 4)),
            "audio_shift_ms": 0.0,
            "meter": 4,
            "stable": state["analysis"]["tempo"]["stable"],
        }

    def submit(pid, operation, *, cleanup=None):
        return jsonify(tasks.submit(pid, operation, cleanup=cleanup))

    @app.errorhandler(Exception)
    def error(err):
        from werkzeug.exceptions import HTTPException

        if isinstance(err, HTTPException):
            return jsonify(error=err.description), err.code
        if isinstance(err, (ValueError, KeyError, FileNotFoundError)):
            return jsonify(error=str(err)), 400
        raise err

    register_v32_routes(
        app,
        engine=engine,
        projects=projects,
        submit=submit,
        current_audio=current_audio,
        requested_tempo=requested_tempo,
        resources=ROOT,
    )
    register_export_routes(
        app,
        projects=projects,
        resources=ROOT,
        current_audio=current_audio,
    )

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        job = tasks.get(job_id)
        return (jsonify(job), 200) if job else (jsonify(error="任务不存在"), 404)

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id):
        return (jsonify(ok=True), 200) if tasks.cancel(job_id) else (
            jsonify(error="任务不存在"), 404,
        )

    try:
        yield {
            "app": app,
            "client": app.test_client(),
            "projects": projects,
            "tasks": tasks,
            "project_id": project_id,
            "engine": engine,
            "audio_sha": audio_sha,
        }
    finally:
        tasks.shutdown()


def _fake_torosu(
    osu_text,
    keys,
    *,
    density,
    cancelled,
    **_kwargs,
):
    if cancelled():
        raise InterruptedError("Torosu 定轨已取消")
    parsed = parse_osu(osu_text)
    notes = [
        Note(
            note.start,
            note.end,
            min(keys - 1, note.lane + max(0, keys - parsed["keys"])),
            note.pitch,
            note.strength,
            note.source,
            note.id,
            note.raw,
            note.voice,
        )
        for note in parsed["notes"]
    ]
    return export_osu(notes, keys, None, original=osu_text, version=f"fake {density}")


def _wait_job(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def test_v32_candidate_uses_torus_task_pipeline_and_existing_osz_export(route_app):
    client = route_app["client"]
    project_id = route_app["project_id"]
    response = client.post(
        f"/api/projects/{project_id}/v32-generate",
        json={
            "model_variant": "v32-mini",
            "generation_strategy": "7k_to_torosu_10k",
            "torosu_density": "dense",
            "difficulty": 5.5,
            "hold_note_ratio": 0.55,
            "mania_column_temperature": 1.0,
            "lora_path": "",
            "timing_confirmed": False,
        },
    )
    assert response.status_code == 200, response.json
    job = _wait_job(client, response.json["id"])
    assert job["status"] == "done", job
    assert route_app["engine"].calls[0]["keycount"] == 7

    saved = route_app["projects"].load(project_id)
    assert saved["active"] == 1
    assert len(saved["variants"]) == 2
    candidate = saved["variants"][1]
    assert candidate["mode"] == "audio"
    assert candidate["settings"]["keys"] == 10
    assert candidate["timing_revision"] == 3
    assert candidate["source_audio_sha256"] == route_app["audio_sha"]
    assert candidate["report"]["v32"]["separation"]["profile"] == "six"
    assert candidate["report"]["grid"]["off_grid"] == 0

    osu_response = client.get(f"/api/projects/{project_id}/export/osu")
    assert osu_response.status_code == 200, osu_response.json
    parsed = parse_osu(osu_response.data.decode("utf-8"))
    assert parsed["keys"] == 10
    osz_response = client.get(f"/api/projects/{project_id}/export/osz")
    assert osz_response.status_code == 200, osz_response.json
    with zipfile.ZipFile(io.BytesIO(osz_response.data)) as archive:
        assert archive.testzip() is None
        chart = parse_osu(archive.read("ManiaLab.osu").decode("utf-8"))
        assert chart["general"]["AudioFilename"] == "song.wav"
        assert "song.wav" in archive.namelist()


def test_v32_runtime_install_uses_shared_task_queue(route_app):
    engine = route_app["engine"]
    engine.install_available = True
    submitted = route_app["client"].post(
        f"/api/projects/{route_app['project_id']}/v32-install",
        json={},
    )
    assert submitted.status_code == 200
    job = _wait_job(route_app["client"], submitted.json["id"])
    assert job["status"] == "done", job
    assert job["result"] == {"device": "cuda", "gpu": "RTX 4050"}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"model_variant": "v31"},
        {"generation_strategy": "native"},
        {"torosu_density": "sparse"},
        {"difficulty": True},
        {"hold_note_ratio": 0.9},
        {"mania_column_temperature": 2.1},
        {"timing_confirmed": 1},
        {"unexpected": True},
    ],
)
def test_v32_rejects_invalid_requests_before_queueing(route_app, payload):
    response = route_app["client"].post(
        f"/api/projects/{route_app['project_id']}/v32-generate",
        json=payload,
    )
    assert response.status_code == 400
    assert not route_app["tasks"].has_active()
    assert route_app["engine"].calls == []


def test_v32_rejects_unavailable_runtime_and_unconfirmed_unstable_timing(route_app):
    engine = route_app["engine"]
    engine.available = False
    url = f"/api/projects/{route_app['project_id']}/v32-generate"
    unavailable = route_app["client"].post(url, json={})
    assert unavailable.status_code == 400
    assert "未就绪" in unavailable.json["error"]

    engine.available = True
    state = route_app["projects"].load(route_app["project_id"])
    state["analysis"]["tempo"]["stable"] = False
    route_app["projects"].save(state)
    unstable = route_app["client"].post(url, json={})
    assert unstable.status_code == 200
    failed = _wait_job(route_app["client"], unstable.json["id"])
    assert failed["status"] == "error"
    assert "Timing" in failed["message"]
    assert route_app["projects"].load(route_app["project_id"])["active"] == 0


def test_v32_rejects_stale_audio_identity(route_app):
    audio = route_app["projects"].path(route_app["project_id"]) / "files" / "song.wav"
    audio.write_bytes(b"changed after analysis")
    response = route_app["client"].post(
        f"/api/projects/{route_app['project_id']}/v32-generate",
        json={},
    )
    failed = _wait_job(route_app["client"], response.json["id"])
    assert failed["status"] == "error"
    assert "不一致" in failed["message"]
    assert route_app["projects"].load(route_app["project_id"])["active"] == 0


def test_v32_cancel_keeps_candidate_history_unchanged(route_app):
    engine = route_app["engine"]
    engine.wait_for_cancel = True
    engine.generate_started = threading.Event()
    client = route_app["client"]
    submitted = client.post(
        f"/api/projects/{route_app['project_id']}/v32-generate",
        json={},
    )
    assert submitted.status_code == 200
    assert engine.generate_started.wait(3)
    cancelled = client.post(f"/api/jobs/{submitted.json['id']}/cancel")
    assert cancelled.status_code == 200
    job = _wait_job(client, submitted.json["id"])
    assert job["status"] == "cancelled"
    saved = route_app["projects"].load(route_app["project_id"])
    assert saved["active"] == 0
    assert [item["candidate_id"] for item in saved["variants"]] == ["a" * 32]
    remaining = route_app["projects"].path(route_app["project_id"])
    assert not [
        path for path in remaining.glob("v32-*")
        if not path.name.startswith("v32-separation-")
    ]
