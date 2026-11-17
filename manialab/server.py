from __future__ import annotations

from dataclasses import replace
import json
import logging
import math
from pathlib import Path
import secrets
import time
from types import SimpleNamespace
from urllib.parse import urlparse
import zipfile

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from . import __version__
from .arrange import DEFAULTS, parameters, layout_metrics
from .timing import DIVISIONS, align_candidates, grid_report
from .audio import analyze, analyze_deep
from .separation import Separator
from .provided_stems import ProvidedStemSeparator,prepare_provided_stems,provided_stem_identity
from .piano import PianoDetector
from .guitar import GuitarDetector
from .instrument_detector import InstrumentDetector
from .pattern_prior import prior_status
from .review import review_chart
from .ln_patterns import analyze_ln_patterns
from .editing import edit_notes
from .style import describe_chart
from .analysis_routes import register_analysis_routes
from .domain import clean_meta, note_from, parse_osu, upstream_events, validate
from .export_routes import register_export_routes
from .planner import plan
from .playback_routes import register_playback_routes
from .project_routes import register_project_routes
from .arrangement_routes import register_arrangement_routes
from .timing_routes import register_timing_routes
from .storage import Projects, safe_path
from .task_service import ProjectTasks
from .upstream import MapperEngine


logger = logging.getLogger(__name__)


def create_app(resources, data_root):
    resources, data_root = Path(resources).resolve(), Path(data_root).resolve()
    app = Flask(__name__, static_folder=str(resources / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    app.config["TOKEN"] = secrets.token_urlsafe(32)
    projects = Projects(Path(data_root) / "projects")
    task_service = ProjectTasks(projects, logger)
    engine = MapperEngine(data_root, resources)
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
                          task_service=task_service, analysis_services=analysis_services)
    register_project_routes(app, projects, task_service)

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
        return jsonify(version=__version__, defaults=DEFAULTS, projects=projects.recent(),
                       capabilities={"audio": True, "remap": True, "api_planner": True, "trained_10k": False},
                       engine=engine.status(), separator=separator.status(), piano=piano_detector.status(),
                       guitar=guitar_detector.status(), instruments=instrument_detector.status(), pattern_prior=prior_status())

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
        detected = state.get("analysis", {}).get("tempo", {"bpm": 120, "offset": 0})
        shift = requested_shift(payload, state)
        detected_offset = math.floor(detected["offset"] - shift + .5)
        bpm, offset = float(payload.get("bpm", detected["bpm"])), float(payload.get("offset", detected_offset))
        if not math.isfinite(bpm) or not 10 <= bpm <= 1000 or not math.isfinite(offset) or abs(offset) > 600000:
            raise ValueError("BPM 或 Offset 无效")
        division = payload.get("division", detected.get("division", 4))
        if type(division) is not int or division not in DIVISIONS:
            raise ValueError("拍内细分必须为 3、4、6、8、12 或 16")
        unchanged = abs(bpm - detected["bpm"]) < .0001 and abs(offset - detected_offset) < .01 and division == detected.get("division", 4)
        coverage_checked = detected.get('evidence_source') != 'drums' or detected.get('timing_coverage_revision',0) >= 1
        return detected | dict(bpm=bpm, offset=round(offset), division=division, audio_shift_ms=shift,
                               stable=bool(unchanged and detected.get("stable") and coverage_checked))

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

    @app.post("/api/projects/<pid>/generate")
    def generate(pid):
        payload = request.json
        prefs = parameters(payload.get("settings", {}))
        mode = payload.get("mode", "audio")
        if mode not in ("audio", "remap"):
            raise ValueError("生成模式无效")
        def operation(state, progress, cancelled):
            from .generation import generate_variant
            return generate_variant(state, payload, prefs, requested_tempo(payload, state), progress, cancelled)
        return submit(pid, operation)

    @app.post("/api/projects/<pid>/edit")
    def edit_candidate(pid):
        payload = request.json
        def operation(state, progress, cancelled):
            import copy
            if state['active'] < 0:
                raise ValueError('请先生成候选谱面')
            old = state['variants'][state['active']]
            if payload.get('base_created') != old.get('created'):
                raise ValueError('候选版本已经改变，请重新选择音符再修改')
            progress(20,'检查修改与相邻 LN 占轨')
            length = state.get('analysis',{}).get('duration')
            notes, edit = edit_notes(old,payload,round(length*1000) if length else None)
            prefs = old['settings']
            points = state.get('input_timing',[]) if old['mode']=='remap' else []
            beat = 60000/old['bpm']
            report = validate(notes,prefs['keys'])
            if old['mode']=='audio':
                report['clock'] = old['report'].get('clock',{})
                report['grid'] = grid_report(notes,old)
                if report['grid']['off_grid']:
                    raise ValueError('修改后的音符未通过落线检查')
            report['layout'] = layout_metrics(notes,prefs['keys'],beat,points)
            report['review'] = review_chart(notes,prefs['keys'],beat,points)
            report['ln_patterns'] = analyze_ln_patterns(notes,prefs['keys'],beat,points)
            variant = copy.deepcopy(old)
            variant.update(notes=[n.data() for n in notes],report=report,created=time.time(),elapsed_seconds=0.,
                           event_count=len(upstream_events(notes)),region=None)
            variant['edits'] = [*old.get('edits',[]),dict(parent_created=old['created'],**edit)]
            variant['selection'] = dict(message='手动修改已保存为新候选。原始采音保留，可从候选列表返回上一版。')
            state['variants'] = (state['variants']+[variant])[-8:]
            state['active'] = len(state['variants'])-1
            state['settings'] = prefs.copy()
            return report
        return submit(pid,operation)

    @app.post("/api/projects/<pid>/reference-style")
    def reference_style(pid):
        with task_service.lock:
            task_service.ensure_project_idle(pid)
            state = projects.load(pid)
            file = request.files.get('file')
            if not file or Path(file.filename or '').suffix.lower() != '.osu':
                raise ValueError('请选择一份 .osu 参考谱面')
            raw = file.stream.read(10*1024*1024+1)
            if len(raw) > 10*1024*1024:
                raise ValueError('参考谱面超过 10MB')
            try:
                profile = describe_chart(parse_osu(raw.decode('utf-8-sig')))
            except UnicodeError:
                raise ValueError('参考谱面必须为 UTF-8 文本') from None
            draft = json.loads(request.form.get('settings', '{}'))
            if not isinstance(draft, dict):
                raise ValueError('当前偏好必须为参数对象')
            prefs = parameters(parameters(state['settings'] | draft) | profile['preferences'])
            state['reference_style'] = profile
            # The current target key count remains the user's explicit choice.
            state['settings'] = prefs
            projects.save(state)
        return jsonify(state)

    @app.post("/api/projects/<pid>/engine-install")
    def engine_install(pid):
        def operation(state, progress, cancelled):
            engine.install(progress, cancelled)
            return engine.status()
        return submit(pid, operation)

    @app.post("/api/projects/<pid>/engine-generate")
    def engine_generate(pid):
        payload = request.get_json() or {}
        prefs = parameters(payload.get("settings", {}))
        seconds = payload.get("seconds", 30)
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 1200:
            raise ValueError("时长应为 0–1200 秒；0 表示全曲")
        def operation(state, progress, cancelled):
            # Supply decoded WAV so upstream pydub needs no global ffmpeg installation.
            from .audio import decode
            work = projects.path(pid)
            if not (work / "preview.wav").exists():
                decode(current_audio(state), work)
            tempo=requested_tempo(payload,state) if payload.get('use_timing') is True else None
            if tempo is not None and not (tempo.get('stable') or payload.get('timing_confirmed') is True):
                raise ValueError('请先检查并确认当前 BPM / Offset，再传入原模型')
            raw = engine.generate(work / "preview.wav", work / ("engine-" + secrets.token_hex(4)), prefs, progress, cancelled, seconds, tempo)
            text = raw.read_text(encoding="utf-8-sig")
            import re
            text = re.sub(r"(?m)^AudioFilename:.*$", lambda _: "AudioFilename: " + state["audio"], text)
            for field, value in {"Title": state["title"], "TitleUnicode": state["title"], "Artist": state["artist"], "ArtistUnicode": state["artist"]}.items():
                text = re.sub(r"(?m)^" + field + r":.*$", lambda _, f=field, v=value: f + ":" + clean_meta(v), text)
            alignment = None
            if tempo:
                from .upstream import fixed_timing_output
                text, model_notes, alignment = fixed_timing_output(text, tempo)
            parsed = parse_osu(text)
            selected = model_notes if tempo else parsed['notes']
            # This is the model comparison path. Reassigning every column here
            # erases the learned LN layout before the user can inspect it.
            # The ordinary remap action remains available after generation.
            assigned = selected
            name = "Mapperatorinator-" + secrets.token_hex(4) + ".osu"
            (work / "files" / name).write_text(text, encoding="utf-8")
            state.update(original=name, input_notes=[n.data() for n in selected], input_keys=parsed["keys"], input_timing=parsed["timing"], mode="remap", settings=prefs)
            state["charts"].append({"path": name, "label": f'Mapperatorinator V32 · {prefs["keys"]}K'})
            report = validate(assigned, prefs["keys"])
            beat=60000/tempo['bpm'] if tempo else 500
            report['layout']=layout_metrics(assigned,prefs['keys'],beat,parsed['timing'])
            report['review']=review_chart(assigned,prefs['keys'],beat,parsed['timing'])
            report['ln_patterns']=analyze_ln_patterns(assigned,prefs['keys'],beat,parsed['timing'])
            warnings = getattr(engine, "last_warnings", [])
            message = "已保留 mania 模型生成的键型。当前应用仍使用原版权重；需要重新定轨时可另行生成候选。"
            if not report['valid']:
                message += f" 原始键型存在 {report['issue_count']} 项结构冲突，请先检查或重新定轨。"
            if tempo:
                report['grid'] = grid_report(assigned, tempo)
                message += (f" 已按确认的 BPM / Offset 及细分量化模型输出：移动 {alignment['moved']} 个起点、"
                    f"{alignment['tails_moved']} 个音尾，{alignment['collapsed_holds']} 个不足一格的 LN 转为短键。"
                    f" {alignment['large_attack_shifts']} 个起音移动超过 30ms，需要回听；落线不代表采音正确。")
            if warnings:
                message += "\n原模型警告（可能包含未配对并被上游丢弃的 LN）：\n" + "\n".join(warnings)
            # Source reference is project-level, so discard variants belonging to an older source.
            state["variants"] = [dict(notes=[n.data() for n in assigned], settings=prefs, mode="remap", report=report,
                selection={"message": message, "upstream_warnings": warnings, 'alignment': alignment,
                           'layout_origin': 'mapperatorinator'},
                bpm=tempo['bpm'] if tempo else 120, offset=tempo['offset'] if tempo else 0,
                division=tempo.get('division',4) if tempo else 4,
                audio_shift_ms=tempo.get('audio_shift_ms',0) if tempo else 0,
                timing_confirmed=bool(tempo),
                provided_timing=tempo,created=time.time(), region=None, event_count=len(upstream_events(assigned)))]
            state["active"] = 0
            return report
        return submit(pid, operation)

    register_export_routes(app, projects=projects, resources=resources, current_audio=current_audio)
    register_playback_routes(
        app,
        projects=projects,
        current_audio=current_audio,
        requested_tempo=requested_tempo,
    )
    return app
