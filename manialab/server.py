from __future__ import annotations

from dataclasses import replace
import logging
import math
from pathlib import Path
import secrets
from types import SimpleNamespace
from urllib.parse import urlparse
import zipfile

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from . import __version__
from .arrange import DEFAULTS, parameters
from .meter import meter_for_clock, stable_audio_clock, valid_audio_clock
from .timing import DIVISIONS
from .audio import analyze, analyze_deep
from .separation import Separator
from .provided_stems import ProvidedStemSeparator,prepare_provided_stems,provided_stem_identity
from .piano import PianoDetector
from .guitar import GuitarDetector
from .instrument_detector import InstrumentDetector
from .pattern_prior import prior_status
from .style import describe_chart
from .analysis_routes import register_analysis_routes
from .candidate_routes import register_candidate_routes
from .export_routes import register_export_routes
from .planner import plan
from .playback_routes import register_playback_routes
from .project_routes import register_project_routes
from .arrangement_routes import register_arrangement_routes
from .timing_routes import register_timing_routes
from .upstream_routes import register_upstream_routes
from .v32_engine import V32Engine
from .v32_routes import register_v32_routes
from .storage import MAX_IMPORT_BYTES, Projects, safe_path
from .task_service import ProjectTasks
from .upstream import MapperEngine


logger = logging.getLogger(__name__)


def create_app(resources, data_root):
    resources, data_root = Path(resources).resolve(), Path(data_root).resolve()
    app = Flask(__name__, static_folder=str(resources / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_IMPORT_BYTES
    app.config["TOKEN"] = secrets.token_urlsafe(32)
    projects = Projects(Path(data_root) / "projects")
    task_service = ProjectTasks(projects, logger)
    engine = MapperEngine(data_root, resources)
    v32_engine = V32Engine(data_root, resources)
    separator = Separator(resources)
    piano_detector = PianoDetector(resources)
    guitar_detector = GuitarDetector(resources)
    instrument_detector = InstrumentDetector(resources)
    analysis_services = SimpleNamespace(
        parameters=parameters,
        analyze=analyze,
        analyze_deep=analyze_deep,
        Separator=Separator,
        ProvidedStemSeparator=ProvidedStemSeparator,
        prepare_provided_stems=prepare_provided_stems,
        provided_stem_identity=provided_stem_identity,
    )
    app.extensions.update(projects=projects, jobs=task_service.jobs, executor=task_service.executor,
                          task_service=task_service, analysis_services=analysis_services,
                          v32_engine=v32_engine)
    register_project_routes(app, projects, task_service, parameters=parameters, describe_chart=describe_chart)

    @app.before_request
    def protect():
        if request.host.split(":")[0] not in ("127.0.0.1", "localhost"):
            return jsonify(error="仅允许本机访问"), 403
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            return jsonify(error="请求来源不匹配"), 403
        if request.method != "GET" and request.headers.get("X-ManiaLab-Token") != app.config["TOKEN"]:
            return jsonify(error="页面会话已过期，请刷新"), 403

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(Exception)
    def error(err):
        if isinstance(err, HTTPException):
            return jsonify(error="文件过大（上限 512MB）" if err.code == 413 else err.description), err.code
        if isinstance(err, (ValueError, KeyError, FileNotFoundError, zipfile.BadZipFile)):
            return jsonify(error=str(err)), 400
        logger.exception("Request failed")
        return jsonify(error="操作失败，详情已写入本地日志"), 500

    @app.get("/")
    def index():
        return (resources / "static" / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", app.config["TOKEN"])

    @app.get("/api/status")
    def status():
        separator_status = separator.status()
        v32_status = v32_engine.status()
        v32_status["separation_available"] = bool(
            separator_status.get("profiles", {}).get("six", {}).get("available")
        )
        return jsonify(version=__version__, defaults=DEFAULTS, projects=projects.recent(),
                       capabilities={"audio": True, "remap": True, "api_planner": True,
                                     "trained_10k": False, "v32": True},
                       engine=engine.status(), separator=separator_status, piano=piano_detector.status(),
                       guitar=guitar_detector.status(), instruments=instrument_detector.status(),
                       pattern_prior=prior_status(), v32_engine=v32_status)

    def current_audio(state):
        if state.get("audio"):
            return safe_path(projects.path(state["id"]) / "files", state["audio"])
        raise ValueError("项目没有音频；请导入包含音频的 .osz，或同时选择 .osu 与音频")

    def submit(pid, operation, *, identity=None, cleanup=None):
        return jsonify(task_service.submit(pid, operation, identity=identity, cleanup=cleanup))

    @app.get("/api/jobs/<jid>")
    def get_job(jid):
        job = task_service.get(jid)
        if job is None:
            return jsonify(error="任务不存在"), 404
        return jsonify(job)

    @app.post("/api/jobs/<jid>/cancel")
    def cancel(jid):
        if not task_service.cancel(jid):
            return jsonify(error="任务不存在"), 404
        return jsonify(ok=True)

    def requested_shift(payload, state):
        shift = float(payload.get("audio_shift_ms", state.get("audio_shift_ms", 0)))
        if not math.isfinite(shift) or abs(shift) > 100:
            raise ValueError("制谱提前量必须在 -100–100 ms 内")
        return shift

    def requested_tempo(payload, state):
        analysis = state.get("analysis") or {}
        detected = analysis.get("tempo")
        if not valid_audio_clock(detected):
            detected = {"bpm": 120, "offset": 0}
        shift = requested_shift(payload, state)
        detected_offset = math.floor(detected["offset"] - shift + .5)
        bpm, offset = float(payload.get("bpm", detected["bpm"])), float(payload.get("offset", detected_offset))
        if not math.isfinite(bpm) or not 10 <= bpm <= 1000 or not math.isfinite(offset) or abs(offset) > 600000:
            raise ValueError("BPM 或 Offset 无效")
        division = payload.get("division", detected.get("division", 4))
        if type(division) is not int or division not in DIVISIONS:
            raise ValueError("拍内细分必须为 3、4、6、8、12 或 16")
        unchanged = abs(bpm - detected["bpm"]) < .0001 and abs(offset - detected_offset) < .01 and division == detected.get("division", 4)
        clock = dict(bpm=bpm, offset=round(offset), division=division, audio_shift_ms=shift,
                     stable=unchanged and stable_audio_clock(detected))
        return detected | clock | {"meter": meter_for_clock(analysis, clock)}

    register_analysis_routes(
        app, projects=projects, task_service=task_service, resources=resources,
        current_audio=current_audio, submit=submit, requested_shift=requested_shift,
        separator=separator, piano_detector=piano_detector, guitar_detector=guitar_detector,
        instrument_detector=instrument_detector, services=analysis_services,
    )
    register_timing_routes(app, projects=projects, requested_tempo=requested_tempo)
    register_arrangement_routes(
        app,
        projects=projects,
        task_service=task_service,
        resources=resources,
        submit=submit,
        parameters=parameters,
        requested_shift=requested_shift,
        current_audio=current_audio,
        planner=plan,
    )
    register_candidate_routes(
        app,
        submit=submit,
        parameters=parameters,
        requested_tempo=requested_tempo,
    )

    register_upstream_routes(
        app,
        engine=engine,
        projects=projects,
        submit=submit,
        current_audio=current_audio,
        requested_tempo=requested_tempo,
    )
    register_v32_routes(
        app,
        engine=v32_engine,
        projects=projects,
        submit=submit,
        current_audio=current_audio,
        requested_tempo=requested_tempo,
        resources=resources,
    )
    register_export_routes(app, projects=projects, resources=resources, current_audio=current_audio)
    register_playback_routes(
        app,
        projects=projects,
        current_audio=current_audio,
        requested_tempo=requested_tempo,
    )
    return app
