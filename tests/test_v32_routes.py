from __future__ import annotations

import ast
import hashlib
import io
from datetime import timedelta
from pathlib import Path
import threading
import time
from types import SimpleNamespace
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
from manialab.v32_routes import build_v32_operation, register_v32_routes, scan_local_loras


ROOT = Path(__file__).parents[1]
_FAKE_TOROSU_CALLS = []
_FAKE_LN_SHAPE_CALLS = []


def _upstream_timing_event_times(reference_path: Path) -> list[int]:
    with zipfile.ZipFile(ROOT / "assets" / "mapperatorinator-source.zip") as archive:
        source = archive.read("osuT5/osuT5/dataset/osu_parser.py").decode("utf-8")
    tree = ast.parse(source)
    parser_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OsuParser"
    )
    method = next(
        node for node in parser_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_timing"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = {
        "timedelta": timedelta,
        "EventType": SimpleNamespace(
            TIMING_POINT="timing_point",
            MEASURE="measure",
            BEAT="beat",
        ),
    }
    exec(compile(ast.fix_missing_locations(module), "<upstream-parse-timing>", "exec"), namespace)

    parsed = parse_osu(reference_path.read_text(encoding="utf-8-sig"))

    class Beatmap:
        def __init__(self):
            self.timing_points = []
            for line in parsed["timing"]:
                fields = line.split(",")
                if len(fields) < 7:
                    continue
                ms_per_beat = float(fields[1])
                red_line = int(fields[6]) == 1
                self.timing_points.append(SimpleNamespace(
                    offset=timedelta(milliseconds=float(fields[0])),
                    bpm=60000 / ms_per_beat if red_line else None,
                    ms_per_beat=ms_per_beat,
                    meter=int(fields[2]),
                ))
            self._hit_objects = [
                SimpleNamespace(
                    time=timedelta(milliseconds=note.start),
                    **(
                        {"end_time": timedelta(milliseconds=note.end)}
                        if note.hold else {}
                    ),
                )
                for note in parsed["notes"]
            ]

        def hit_objects(self, stacking=False):
            return self._hit_objects

    namespace["Beatmap"] = Beatmap

    class Parser:
        add_timing_points = True

        @staticmethod
        def _add_group(event, time, events, event_times, _beatmap, **_kwargs):
            events.append(event)
            event_times.append(int(time.total_seconds() * 1000 + 1e-5))

    return namespace["parse_timing"](Parser(), Beatmap())[1]


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

    def install_runtime(self, *, progress, cancelled, use_mirrors=False):
        assert not cancelled()
        self.install_use_mirrors = use_mirrors
        progress(100, "V32 推理环境已就绪")
        return {"device": "cuda", "gpu": "RTX 4050"}

    def generate(self, **kwargs):
        call = kwargs.copy()
        reference = parse_osu(
            Path(kwargs["reference_chart"]).read_text(encoding="utf-8-sig")
        )
        call["reference_keys"] = reference["keys"]
        call["reference_note_count"] = len(reference["notes"])
        call["reference_note_times"] = [
            (note.start, note.end) for note in reference["notes"]
        ]
        call["reference_timing_offset"] = float(reference["timing"][0].split(",")[0])
        self.calls.append(call)
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
    monkeypatch.setattr(
        routes,
        "ln_shape_osu_text",
        lambda text, mode: _record_ln_shape(text, mode),
    )
    _FAKE_TOROSU_CALLS.clear()
    _FAKE_LN_SHAPE_CALLS.clear()

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
        bpm = float(payload.get("bpm", 120))
        offset = float(payload.get("offset", 0))
        division = payload.get("division", 4)
        audio_shift_ms = float(payload.get("audio_shift_ms", 0))
        return {
            "bpm": bpm,
            "offset": offset,
            "division": division,
            "audio_shift_ms": audio_shift_ms,
            "meter": 4,
            "stable": (
                state["analysis"]["tempo"]["stable"]
                and bpm == 120
                and offset == 0
                and division == 4
                and audio_shift_ms == 0
            ),
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
        task_service=tasks,
    )
    app.extensions.update(task_service=tasks, v32_engine=engine)
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
    _FAKE_TOROSU_CALLS.append({
        "keys": keys,
        "density": density,
        "keep_jacks": _kwargs.get("keep_jacks"),
        "playstyle": _kwargs.get("playstyle"),
    })
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


def _record_ln_shape(osu_text, mode):
    _FAKE_LN_SHAPE_CALLS.append(mode)
    return osu_text


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
    assert route_app["engine"].calls[0]["target_keycount"] == 10
    assert route_app["engine"].calls[0]["reference_keys"] == 7
    assert route_app["engine"].calls[0]["reference_note_count"] == 1

    saved = route_app["projects"].load(project_id)
    assert saved["active"] == 1
    assert len(saved["variants"]) == 2
    candidate = saved["variants"][1]
    assert candidate["mode"] == "audio"
    assert candidate["settings"]["keys"] == 10
    assert candidate["timing_revision"] == 3
    assert candidate["source_audio_sha256"] == route_app["audio_sha"]
    assert candidate["report"]["v32"]["separation"]["profile"] == "six"
    assert candidate["report"]["v32"]["collision_cleanup"]["removed_objects"] == 0
    assert candidate["report"]["grid"]["off_grid"] == 0
    assert candidate["report"]["v32"]["torosu"]["keep_jacks"] is True

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
        "/api/v32/install",
        json={"use_mirrors": True},
    )
    assert submitted.status_code == 200
    job = _wait_job(route_app["client"], submitted.json["id"])
    assert job["status"] == "done", job
    assert job["global"] is True
    assert job["result"] == {"device": "cuda", "gpu": "RTX 4050"}
    assert engine.install_use_mirrors is True
    status = route_app["client"].get(
        f"/api/v32/install/{submitted.json['id']}"
    )
    assert status.status_code == 200
    assert status.json["global"] is True
    assert route_app["client"].get("/api/v32/status").json["install_available"] is True


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"model_variant": "v31"},
        {"generation_strategy": "not-a-strategy"},
        {"torosu_density": "sparse"},
        {"difficulty": True},
        {"hold_note_ratio": 0.9},
        {"mania_column_temperature": 2.1},
        {"timing_confirmed": 1},
        {"keys": 8, "key_layout": "10k_5_5"},
        {"bpm": True},
        {"bpm": 1001},
        {"offset": 600001},
        {"division": True},
        {"division": 5},
        {"audio_shift_ms": 101},
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


@pytest.mark.parametrize(
    "keys,preset,model_keys,keep_jacks,density",
    [
        (6, "rice_stream", 7, False, "dense"),
        (7, "rice_jack", 7, True, "more-dense"),
        (8, "hybrid_adaptive", 7, True, "dense"),
        (9, "ln_focus", 7, True, "dense"),
        (10, "rice_stream", 7, False, "dense"),
    ],
)
def test_v32_named_presets_emit_requested_6_to_10k_and_real_torosu_options(
    route_app, keys, preset, model_keys, keep_jacks, density,
):
    client = route_app["client"]
    project_id = route_app["project_id"]
    response = client.post(
        f"/api/projects/{project_id}/v32-generate",
        json={"preset": preset, "keys": keys, "timing_confirmed": False},
    )
    assert response.status_code == 200, response.json
    job = _wait_job(client, response.json["id"])
    assert job["status"] == "done", job
    assert route_app["engine"].calls[0]["keycount"] == model_keys
    assert route_app["engine"].calls[0]["target_keycount"] == keys
    assert route_app["engine"].calls[0]["preset"] == preset

    saved = route_app["projects"].load(project_id)
    candidate = saved["variants"][saved["active"]]
    assert candidate["settings"]["keys"] == keys
    assert candidate["report"]["v32"]["torosu"]["target_keys"] == keys
    assert candidate["report"]["v32"]["torosu"]["keep_jacks"] is keep_jacks
    assert candidate["report"]["v32"]["torosu"]["density"] == density
    assert _FAKE_TOROSU_CALLS[-1] == {
        "keys": keys,
        "density": density,
        "keep_jacks": keep_jacks,
        "playstyle": {
            6: "3+3",
            7: "3+1+3",
            8: "3+2+3",
            9: "4+1+4",
            10: "5+5",
        }[keys],
    }


def test_v32_applies_explicit_clock_and_layout_to_candidate_and_export(route_app):
    client = route_app["client"]
    project_id = route_app["project_id"]
    response = client.post(
        f"/api/projects/{project_id}/v32-generate",
        json={
            "keys": 8,
            "key_layout": "8k_4_4",
            "generation_strategy": "native",
            "bpm": 137.5,
            "offset": -40,
            "division": 6,
            "audio_shift_ms": 10,
            "timing_confirmed": True,
        },
    )
    assert response.status_code == 200, response.json
    job = _wait_job(client, response.json["id"])
    assert job["status"] == "done", job
    candidate = route_app["projects"].load(project_id)["variants"][-1]
    assert candidate["settings"]["keys"] == 8
    assert candidate["settings"]["key_layout"] == "8k_4_4"
    assert candidate["bpm"] == 137.5
    assert candidate["offset"] == -40
    assert candidate["division"] == 6
    assert candidate["audio_shift_ms"] == 10
    assert route_app["engine"].calls[0]["reference_keys"] == 8
    assert route_app["engine"].calls[0]["reference_note_count"] == 1
    assert route_app["engine"].calls[0]["reference_note_times"] == [(4082, 4082)]

    exported = client.get(f"/api/projects/{project_id}/export/osu")
    assert exported.status_code == 200, exported.json
    parsed = parse_osu(exported.data.decode("utf-8"))
    assert parsed["keys"] == 8
    assert parsed["timing"][0].split(",")[0] == "-40.000"
    assert "BeatDivisor: 6" in exported.data.decode("utf-8")


def test_v32_timing_context_covers_180_second_audio_with_upstream_parser(
    route_app, monkeypatch,
):
    import manialab.v32_routes as routes

    def decode_180_seconds(_audio, work, *, cancelled):
        assert not cancelled()
        path = Path(work) / "preview.wav"
        path.write_bytes(b"decoded 180-second preview")
        return np.zeros(180 * 22050, dtype=np.float32)

    monkeypatch.setattr(routes, "decode", decode_180_seconds)
    engine = route_app["engine"]
    original_generate = engine.generate
    parsed_timing = {}

    def parse_reference(**kwargs):
        reference_path = Path(kwargs["reference_chart"])
        notes = parse_osu(reference_path.read_text(encoding="utf-8-sig"))["notes"]
        assert notes[-1].end >= 179_000
        parsed_timing["event_times"] = _upstream_timing_event_times(reference_path)
        return original_generate(**kwargs)

    monkeypatch.setattr(engine, "generate", parse_reference)
    response = route_app["client"].post(
        f"/api/projects/{route_app['project_id']}/v32-generate",
        json={"timing_confirmed": True},
    )
    assert response.status_code == 200, response.json
    result = _wait_job(route_app["client"], response.json["id"])
    assert result["status"] == "done", result

    event_times = parsed_timing["event_times"]
    assert min(event_times) == 0
    assert max(event_times) >= 179_000
    assert len(event_times) >= 360
    assert max(right - left for left, right in zip(event_times, event_times[1:])) <= 500


def test_v32_route_cleans_snapped_model_collisions_and_reports_drops(
    route_app, monkeypatch,
):
    import manialab.v32_routes as routes

    model_keys = 7
    chart = _osu(model_keys)
    rows = []
    specs = [
        (0, False, 1000),
        (0, True, 2300),
        (1, True, 5000),
        (1, True, 2200),
        (2, False, 1000),
        (3, False, 1000),
        (4, False, 1000),
        (5, False, 1000),
        (6, False, 1000),
    ]
    for lane, hold, end in specs:
        x = int((lane + 0.5) * 512 / model_keys)
        rows.append(
            f"{x},192,1000,{128 if hold else 1},0,"
            f"{end}:0:0:0:0:" if hold else f"{x},192,1000,1,0,0:0:0:0:"
        )
    chart = chart.split("[HitObjects]", 1)[0] + "[HitObjects]\n" + "\n".join(rows) + "\n"

    engine = route_app["engine"]
    original_generate = engine.generate

    def generate_collision_chart(**kwargs):
        result = original_generate(**kwargs)
        Path(result["map"]).write_text(chart, encoding="utf-8")
        return result

    converted_inputs = {}
    original_convert = routes.convert_osu_text

    def capture_torosu_input(osu_text, *args, **kwargs):
        converted_inputs["notes"] = parse_osu(osu_text)["notes"]
        return original_convert(osu_text, *args, **kwargs)

    monkeypatch.setattr(engine, "generate", generate_collision_chart)
    monkeypatch.setattr(routes, "convert_osu_text", capture_torosu_input)
    response = route_app["client"].post(
        f"/api/projects/{route_app['project_id']}/v32-generate",
        json={"preset": "ln_focus", "keys": 10},
    )
    assert response.status_code == 200, response.json
    result = _wait_job(route_app["client"], response.json["id"])
    assert result["status"] == "done", result

    saved = route_app["projects"].load(route_app["project_id"])
    candidate = saved["variants"][saved["active"]]
    cleanup = candidate["report"]["v32"]["collision_cleanup"]
    assert cleanup["objects_before"] == 9
    assert cleanup["objects_after"] == 7
    assert cleanup["removed_objects"] == 2
    assert cleanup["duplicate_track_objects_removed"] == 2
    assert len(converted_inputs["notes"]) == 7
    lanes_by_start = {}
    for note in converted_inputs["notes"]:
        lanes = lanes_by_start.setdefault(note.start, set())
        assert note.lane not in lanes
        lanes.add(note.lane)


def test_v32_timing_placeholder_does_not_precede_positive_first_red_line(route_app):
    response = route_app["client"].post(
        f"/api/projects/{route_app['project_id']}/v32-generate",
        json={
            "bpm": 150,
            "offset": 3181,
            "timing_confirmed": True,
        },
    )
    assert response.status_code == 200, response.json
    result = _wait_job(route_app["client"], response.json["id"])
    assert result["status"] == "done", result

    call = route_app["engine"].calls[0]
    assert call["reference_timing_offset"] == 3181
    assert call["reference_note_times"] == [(4082, 4082)]
    assert call["reference_note_times"][0][0] >= call["reference_timing_offset"]


def test_v32_timing_reference_parses_with_upstream_slider(route_app, monkeypatch):
    slider = pytest.importorskip("slider")
    engine = route_app["engine"]
    original_generate = engine.generate

    def parse_upstream_reference(**kwargs):
        reference_path = Path(kwargs["reference_chart"])
        reference_text = reference_path.read_text(encoding="utf-8-sig")
        assert reference_text.count("timing-context-placeholder") == 1
        assert len(parse_osu(reference_text)["notes"]) == 1
        beatmap = slider.Beatmap.from_path(str(reference_path))
        assert beatmap is not None
        return original_generate(**kwargs)

    monkeypatch.setattr(engine, "generate", parse_upstream_reference)
    response = route_app["client"].post(
        f"/api/projects/{route_app['project_id']}/v32-generate",
        json={"timing_confirmed": True},
    )
    assert response.status_code == 200, response.json
    result = _wait_job(route_app["client"], response.json["id"])
    assert result["status"] == "done", result
    assert route_app["engine"].calls[0]["reference_note_count"] == 1


def test_v32_uses_mixed_rms_fallback_when_six_stem_profile_is_unavailable(
    route_app, monkeypatch,
):
    separator = route_app["app"].extensions["v32_separator"]
    monkeypatch.setattr(separator, "status", lambda: {"available": False})
    monkeypatch.setattr(
        separator,
        "separate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unavailable six-stem profile must not be invoked")
        ),
    )
    response = route_app["client"].post(
        f"/api/projects/{route_app['project_id']}/v32-generate",
        json={"timing_confirmed": True},
    )
    assert response.status_code == 200, response.json
    job = _wait_job(route_app["client"], response.json["id"])
    assert job["status"] == "done", job
    candidate = route_app["projects"].load(route_app["project_id"])["variants"][-1]
    assert candidate["report"]["v32"]["energy_source"] == "mixed_audio_rms"
    assert candidate["report"]["v32"]["separation"]["profile"] == "mixed_rms_fallback"
    status = route_app["client"].get("/api/v32/status")
    assert status.status_code == 200
    assert status.json["available"] is True
    assert status.json["separation_available"] is False


def test_v32_4_plus_4_uses_requested_lane_layout(route_app):
    response = route_app["client"].post(
        f"/api/projects/{route_app['project_id']}/v32-generate",
        json={
            "keys": 8,
            "key_layout": "8k_4_4",
            "generation_strategy": "native",
            "timing_confirmed": True,
        },
    )
    assert response.status_code == 200, response.json
    result = _wait_job(route_app["client"], response.json["id"])
    assert result["status"] == "done", result
    candidate = route_app["projects"].load(route_app["project_id"])["variants"][-1]
    assert candidate["settings"]["key_layout"] == "8k_4_4"
    assert candidate["settings"]["layout_engine"] == "beam"
    assert candidate["report"]["v32"]["torosu"]["target_layout"]["reassigned"] is True
    assert _FAKE_TOROSU_CALLS[-1]["playstyle"] == "3+2+3"


def test_v32_route_and_registered_factory_share_same_operation_builder(route_app):
    assert route_app["app"].extensions["v32_operation_factory"].func is build_v32_operation


def test_v32_route_validates_and_applies_torosuln_mode(route_app):
    client = route_app["client"]
    project_id = route_app["project_id"]
    bad = client.post(
        f"/api/projects/{project_id}/v32-generate",
        json={"preset": "ln_focus", "keys": 10, "torosu_ln_mode": "unknown"},
    )
    assert bad.status_code == 400
    assert not route_app["tasks"].has_active()

    response = client.post(
        f"/api/projects/{project_id}/v32-generate",
        json={"preset": "ln_focus", "keys": 10, "torosu_ln_mode": "short_ln"},
    )
    assert response.status_code == 200
    job = _wait_job(client, response.json["id"])
    assert job["status"] == "done", job
    assert _FAKE_LN_SHAPE_CALLS == ["short_ln"]
    assert route_app["engine"].calls[0]["torosu_ln_mode"] == "short_ln"
    saved = route_app["projects"].load(project_id)
    candidate = saved["variants"][saved["active"]]
    assert candidate["report"]["v32"]["torosu"]["ln_mode"] == "short_ln"


def test_v32_route_persists_creator_preset_and_version_suffix(route_app):
    client = route_app["client"]
    project_id = route_app["project_id"]
    response = client.post(
        f"/api/projects/{project_id}/v32-generate",
        json={
            "preset": "hybrid_adaptive",
            "keys": 10,
            "metadata": {"creator": "zhyzx35607", "version_suffix": "adaptive-v1"},
        },
    )
    assert response.status_code == 200, response.json
    job = _wait_job(client, response.json["id"])
    assert job["status"] == "done", job
    candidate = route_app["projects"].load(project_id)["variants"][-1]
    assert candidate["preset"] == "hybrid_adaptive"
    assert candidate["metadata"] == {
        "creator": "zhyzx35607",
        "version_suffix": "adaptive-v1",
    }


def test_local_lora_scan_lists_only_valid_direct_child_adapters(tmp_path):
    lora_root = tmp_path / "assets" / "lora"
    valid = lora_root / "tiny"
    valid.mkdir(parents=True)
    (valid / "adapter_config.json").write_text("{}", encoding="utf-8")
    (valid / "adapter_model.safetensors").write_bytes(b"weights")
    (valid / "lora-manifest.json").write_text(
        '{"name":"LN Focus","base_model":{"variant":"v32-mini"}}',
        encoding="utf-8",
    )
    incomplete = lora_root / "incomplete"
    incomplete.mkdir()
    (incomplete / "adapter_config.json").write_text("{}", encoding="utf-8")
    nested = lora_root / "nested"
    nested.mkdir()
    hidden = nested / "child"
    hidden.mkdir()
    (hidden / "adapter_config.json").write_text("{}", encoding="utf-8")
    (hidden / "adapter_model.safetensors").write_bytes(b"weights")

    assert scan_local_loras(tmp_path) == [{
        "id": "tiny",
        "label": "LN Focus",
        "path": str(valid.resolve()),
        "base_model": "v32-mini",
        "location": "bundle",
    }]


def test_frozen_local_lora_overrides_same_named_bundled_adapter(tmp_path, monkeypatch):
    import manialab.v32_routes as routes

    monkeypatch.setattr(routes.sys, "frozen", True, raising=False)
    resources = tmp_path / "_internal"
    bundled = resources / "assets" / "lora" / "focus"
    portable = resources.parent / "assets" / "lora" / "focus"
    for directory, content in ((bundled, b"bundle"), (portable, b"portable")):
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(content)

    rows = scan_local_loras(resources)

    assert len(rows) == 1
    assert rows[0]["path"] == str(portable.resolve())
    assert rows[0]["location"] == "portable"
