from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import secrets
import threading
import time
import traceback
from urllib.parse import urlparse
import zipfile

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from .arrange import DEFAULTS, assign_lanes, parameters, select_notes, layout_metrics
from .timing import align_candidates, grid_diagnostics
from .audio import analyze
from .domain import clean_meta, export_osu, note_from, parse_osu, upstream_events, validate
from .planner import plan
from .storage import AUDIO, Projects, extract_osz, safe_path
from .upstream import MapperEngine


def create_app(resources, data_root):
    resources = Path(resources)
    app = Flask(__name__, static_folder=str(resources / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    app.config["TOKEN"] = secrets.token_urlsafe(32)
    projects = Projects(Path(data_root) / "projects")
    executor, mutex = ThreadPoolExecutor(max_workers=1), threading.RLock()
    jobs = {}
    engine = MapperEngine(data_root, resources)
    app.extensions.update(projects=projects, jobs=jobs, executor=executor)

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
        traceback.print_exc()
        return jsonify(error="操作失败，详情已写入本地日志"), 500

    @app.get("/")
    def index():
        return (resources / "static" / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", app.config["TOKEN"])

    @app.get("/api/status")
    def status():
        return jsonify(version="0.1.0-alpha.2", defaults=DEFAULTS, projects=projects.recent(),
                       capabilities={"audio": True, "remap": True, "api_planner": True, "trained_10k": False}, engine=engine.status())

    def busy(pid):
        if any(j["project"] == pid and j["status"] in ("queued", "running") for j in jobs.values()):
            raise ValueError("该项目已有任务运行，请等待或取消")

    def current_audio(state):
        if state.get("audio"):
            return safe_path(projects.path(state["id"]) / "files", state["audio"])
        raise ValueError("项目没有音频；请导入包含音频的 .osz，或同时选择 .osu 与音频")

    def set_chart(state, relative):
        base = projects.path(state["id"]) / "files"
        source = safe_path(base, relative)
        parsed = parse_osu(source.read_text(encoding="utf-8-sig"))
        state.update(mode="remap", original=relative, input_notes=[n.data() for n in parsed["notes"]],
                     title=parsed["metadata"].get("Title", source.stem), artist=parsed["metadata"].get("Artist", "Unknown"),
                     input_keys=parsed["keys"], input_timing=parsed["timing"], variants=[], active=-1)
        state["settings"]["keys"] = min(10, max(6, parsed["keys"]))
        audio_name = parsed["general"].get("AudioFilename", "")
        audio = safe_path(base, (Path(relative).parent / audio_name).as_posix()) if audio_name else None
        if audio and audio.is_file():
            state["audio"] = audio.relative_to(base).as_posix()
        elif state.get("audio") and Path(state["audio"]).name.casefold() == Path(audio_name).name.casefold():
            # Keep a jointly uploaded audio file only when its name matches the chart reference.
            pass
        else:
            state["audio"] = None
        state.pop("analysis", None)
        for file in ("activations.npz", "preview.wav"):
            (projects.path(state["id"]) / file).unlink(missing_ok=True)

    @app.post("/api/import")
    def import_files():
        files = request.files.getlist("files")
        if not files:
            raise ValueError("请选择音频、.osu 或 .osz 文件")
        pid, path = projects.create()
        state = dict(id=pid, title="Untitled", artist="Unknown", mode="audio", settings=DEFAULTS.copy(),
                     variants=[], active=-1, charts=[], audio=None, original=None, input_notes=[])
        for file in files:
            name = Path(file.filename.replace("\\", "/")).name
            suffix = Path(name).suffix.lower()
            if suffix == ".osz":
                extract_osz(file.stream, path / "files")
            elif suffix in AUDIO | {".osu"}:
                target = safe_path(path / "files", name)
                if target.exists():
                    raise ValueError("同名上传文件，请分别导入")
                file.save(target)
            else:
                raise ValueError("支持 MP3/WAV/FLAC/OGG/M4A/OPUS、.osu 和 .osz")
        candidates = list((path / "files").rglob("*"))
        audio = next((f for f in candidates if f.is_file() and f.suffix.lower() in AUDIO), None)
        if audio:
            state.update(audio=audio.relative_to(path / "files").as_posix(), title=audio.stem)
        for chart in sorted(f for f in candidates if f.suffix.lower() == ".osu"):
            try:
                parsed = parse_osu(chart.read_text(encoding="utf-8-sig"))
                state["charts"].append({"path": chart.relative_to(path / "files").as_posix(),
                                        "label": f'{parsed["keys"]}K · {parsed["metadata"].get("Version", chart.stem)}'})
            except (ValueError, UnicodeError, IndexError):
                continue
        if state["charts"]:
            set_chart(state, state["charts"][0]["path"])
        elif not state["audio"]:
            raise ValueError("没有可用音频或原生 mania 谱面")
        projects.save(state)
        return jsonify(state)

    @app.get("/api/projects/<pid>")
    def get_project(pid):
        state = projects.load(pid)
        if state.get("original") and "input_timing" not in state:
            source = safe_path(projects.path(pid) / "files", state["original"])
            state["input_timing"] = parse_osu(source.read_text(encoding="utf-8-sig"))["timing"]
        return jsonify(state)

    @app.post("/api/projects/<pid>/chart")
    def choose_chart(pid):
        with mutex:
            busy(pid)
            state = projects.load(pid)
            relative = request.json["path"]
            if relative not in [c["path"] for c in state["charts"]]:
                raise ValueError("未知谱面")
            set_chart(state, relative)
            projects.save(state)
            return jsonify(state)

    @app.post("/api/projects/<pid>/variant")
    def choose_variant(pid):
        with mutex:
            busy(pid)
            state = projects.load(pid)
            index = request.json["index"]
            if type(index) is not int or not 0 <= index < len(state["variants"]):
                raise ValueError("候选版本无效")
            state["active"] = index
            state["settings"] = state["variants"][index]["settings"].copy()
            projects.save(state)
            return jsonify(state)

    def submit(pid, operation):
        with mutex:
            busy(pid)
            if any(j["status"] in ("queued", "running") for j in jobs.values()):
                raise ValueError("已有计算任务运行，请完成后再启动新任务")
            jid = secrets.token_hex(10)
            job = dict(id=jid, project=pid, status="queued", progress=0, message="等待运行", cancel=False)
            jobs[jid] = job
            def progress(p, message):
                with mutex:
                    job.update(progress=p, message=message)
            def run():
                started = time.perf_counter()
                job["status"] = "running"
                try:
                    state = projects.load(pid)
                    result = operation(state, progress, lambda: job["cancel"])
                    if job["cancel"]:
                        raise InterruptedError("已取消，原有谱面已保留")
                    projects.save(state)
                    job.update(status="done", progress=100, message="完成", result=result, elapsed_seconds=round(time.perf_counter() - started, 2))
                except InterruptedError as error:
                    job.update(status="cancelled", message=str(error))
                except Exception as error:
                    traceback.print_exc()
                    job.update(status="error", message=str(error) if isinstance(error, ValueError) else "任务失败，请查看本地日志")
            executor.submit(run)
            return jsonify(job)

    @app.get("/api/jobs/<jid>")
    def get_job(jid):
        with mutex:
            return jsonify(jobs[jid])

    @app.post("/api/jobs/<jid>/cancel")
    def cancel(jid):
        with mutex:
            jobs[jid]["cancel"] = True
        return jsonify(ok=True)

    @app.post("/api/projects/<pid>/analyze")
    def analyze_audio(pid):
        prefs = parameters(request.json.get("settings", {}))
        def operation(state, progress, cancelled):
            state["analysis"] = analyze(current_audio(state), projects.path(pid), resources / "assets" / "basic-pitch.onnx", prefs, progress, cancelled)
            state["settings"] = prefs
            return {"candidates": len(state["analysis"]["candidates"])}
        return submit(pid, operation)

    def requested_tempo(payload, state):
        detected = state.get("analysis", {}).get("tempo", {"bpm": 120, "offset": 0})
        bpm, offset = float(payload.get("bpm", detected["bpm"])), float(payload.get("offset", detected["offset"]))
        if not math.isfinite(bpm) or not 10 <= bpm <= 1000 or not math.isfinite(offset) or abs(offset) > 600000:
            raise ValueError("BPM 或 Offset 无效")
        division = payload.get("division", detected.get("division", 4))
        if type(division) is not int or division not in (3, 4):
            raise ValueError("拍内细分必须为 3 或 4")
        unchanged = abs(bpm - detected["bpm"]) < .0001 and abs(offset - detected["offset"]) < .01 and division == detected.get("division", 4)
        return detected | dict(bpm=bpm, offset=offset, division=division, stable=bool(unchanged and detected.get("stable")))

    @app.post("/api/projects/<pid>/timing-check")
    def timing_check(pid):
        state = projects.load(pid)
        tempo = requested_tempo(request.json, state)
        attacks = [note_from(n) for n in state.get("analysis", {}).get("timing_attacks", [])]
        if not attacks:
            raise ValueError("请用新版重新分析音频，才能检查整曲拍线")
        checks = grid_diagnostics(attacks, tempo["bpm"], tempo["offset"], tempo["division"])
        return jsonify(sections=checks, suspect=sum(s["suspect"] for s in checks))

    @app.post("/api/projects/<pid>/generate")
    def generate(pid):
        payload = request.json
        prefs = parameters(payload.get("settings", {}))
        mode = payload.get("mode", "audio")
        if mode not in ("audio", "remap"):
            raise ValueError("生成模式无效")
        def operation(state, progress, cancelled):
            started = time.perf_counter()
            tempo = requested_tempo(payload, state)
            progress(15, "选择音乐物件 / 保留原始物件")
            if mode == "remap":
                if not state["input_notes"]:
                    raise ValueError("请先导入原生 mania 谱面")
                selected = [note_from(n) for n in state["input_notes"]]
                # Scale the imported column prior when changing keycount.
                selected = [replace(n, lane=min(prefs["keys"] - 1, int((n.lane + .5) * prefs["keys"] / state["input_keys"]))) for n in selected]
                selection = {"message": "严格保留音符数量、起止时间、物件类型、音效和原始变速。"}
            else:
                if not state.get("analysis"):
                    raise ValueError("请先完成音频分析")
                if state["analysis"].get("revision", 1) < 2:
                    raise ValueError("该项目使用旧版采音结果，请先点击「分析音频」更新拍线和 LN 候选；音高缓存会自动复用")
                candidates, alignment = align_candidates([note_from(n) for n in state["analysis"]["candidates"]],
                    tempo, enabled=payload.get("snap", True) is True, confirmed=payload.get("timing_confirmed", False) is True)
                progress(25, "按目标 LN 比例求解持续音与未来占轨约束")
                selected, selection = select_notes(candidates, prefs, state.get("plan", {}).get("segments", []))
                selection["alignment"] = alignment
                selection["message"] += f" 起音吸附 {alignment['moved']} 个。" + alignment["reason"]
                if not selected:
                    raise ValueError("未找到可生成的采音候选，请降低检测阈值或更换音源")
            locked = []
            region = payload.get("region")
            if region is not None:
                if state["active"] < 0:
                    raise ValueError("请先生成完整谱面")
                variant = state["variants"][state["active"]]
                if variant["settings"]["keys"] != prefs["keys"] or variant["mode"] != mode:
                    raise ValueError("局部重写不能改变键数或生成模式")
                if not isinstance(region, list) or len(region) != 2 or not all(type(x) in (int, float) and math.isfinite(x) for x in region) or not 0 <= region[0] < region[1]:
                    raise ValueError("局部时间范围无效")
                a, b = region
                previous = [note_from(n) for n in variant["notes"]]
                locked = [n for n in previous if not a <= n.start < b]
                selected = [n for n in selected if a <= n.start < b]
            progress(45, "分配轨道并检查 LN 占轨")
            output = assign_lanes(selected, prefs, locked, beat_ms=60000 / tempo["bpm"], progress=progress, cancelled=cancelled)
            progress(90, "校验谱面并保存候选")
            report = validate(output, prefs["keys"])
            report["layout"] = layout_metrics(output, prefs["keys"], 60000 / tempo["bpm"])
            timing_sections = grid_diagnostics([note_from(n) for n in state.get("analysis", {}).get("timing_attacks", [])],
                                              tempo["bpm"], tempo["offset"], tempo["division"]) if mode == "audio" else []
            if region is not None:
                selection["message"] = f"局部定轨完成；整谱实际 LN 比例 {report['ln_ratio']:.1%}。区域外物件已锁定。"
            variant = dict(notes=[n.data() for n in output], settings=prefs, mode=mode, report=report, selection=selection,
                           bpm=tempo["bpm"], offset=tempo["offset"], division=tempo["division"],
                           timing_sections=timing_sections,
                           snap=payload.get("snap", True) is True, timing_confirmed=payload.get("timing_confirmed", False) is True,
                           elapsed_seconds=round(time.perf_counter() - started, 2), created=time.time(), region=region,
                           event_count=len(upstream_events(output)))
            state["variants"] = (state["variants"] + [variant])[-8:]
            state["active"] = len(state["variants"]) - 1
            state["settings"] = prefs
            return report
        return submit(pid, operation)

    @app.post("/api/projects/<pid>/plan")
    def api_plan(pid):
        payload = request.json
        prefs = parameters(payload.get("settings", {}))
        def operation(state, progress, cancelled):
            if not state.get("analysis"):
                raise ValueError("请先完成音频分析，API 需要真实的段落统计")
            progress(15, "请求 API 编排建议（发送统计与文字，不发送音频）")
            result = plan(payload.get("endpoint", ""), payload.get("model", ""), payload.get("key", ""),
                          payload.get("instruction", ""), state["analysis"]["segments"], prefs)
            state["plan"] = result
            state["settings"] = parameters(prefs | result["preferences"])
            return result
        return submit(pid, operation)

    @app.post("/api/projects/<pid>/clear-plan")
    def clear_plan(pid):
        with mutex:
            busy(pid)
            state = projects.load(pid)
            state.pop("plan", None)
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
        prefs = parameters(request.json.get("settings", {}))
        seconds = request.json.get("seconds", 30)
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 1200:
            raise ValueError("时长应为 0–1200 秒；0 表示全曲")
        def operation(state, progress, cancelled):
            # Supply decoded WAV so upstream pydub needs no global ffmpeg installation.
            from .audio import decode
            work = projects.path(pid)
            if not (work / "preview.wav").exists():
                decode(current_audio(state), work)
            raw = engine.generate(work / "preview.wav", work / ("engine-" + secrets.token_hex(4)), prefs, progress, cancelled, seconds)
            text = raw.read_text(encoding="utf-8-sig")
            import re
            text = re.sub(r"(?m)^AudioFilename:.*$", lambda _: "AudioFilename: " + state["audio"], text)
            for field, value in {"Title": state["title"], "TitleUnicode": state["title"], "Artist": state["artist"], "ArtistUnicode": state["artist"]}.items():
                text = re.sub(r"(?m)^" + field + r":.*$", lambda _, f=field, v=value: f + ":" + clean_meta(v), text)
            parsed = parse_osu(text)
            assigned = assign_lanes(parsed["notes"], prefs)
            name = "Mapperatorinator-" + secrets.token_hex(4) + ".osu"
            (work / "files" / name).write_text(text, encoding="utf-8")
            state.update(original=name, input_notes=[n.data() for n in parsed["notes"]], input_keys=parsed["keys"], input_timing=parsed["timing"], mode="remap", settings=prefs)
            state["charts"].append({"path": name, "label": f'Mapperatorinator V32 · {prefs["keys"]}K'})
            report = validate(assigned, prefs["keys"])
            warnings = getattr(engine, "last_warnings", [])
            message = "原始 mania 模型输出已严格重新定轨。此模型尚未进行 10K 专训。"
            if warnings:
                message += "\n原模型警告（可能包含未配对并被上游丢弃的 LN）：\n" + "\n".join(warnings)
            # Source reference is project-level, so discard variants belonging to an older source.
            state["variants"] = [dict(notes=[n.data() for n in assigned], settings=prefs, mode="remap", report=report,
                selection={"message": message, "upstream_warnings": warnings},
                bpm=120, offset=0, created=time.time(), region=None, event_count=len(upstream_events(assigned)))]
            state["active"] = 0
            return report
        return submit(pid, operation)

    @app.get("/api/projects/<pid>/audio")
    def audio(pid):
        state = projects.load(pid)
        preview = projects.path(pid) / "preview.wav"
        return send_file(preview if preview.exists() else current_audio(state), conditional=True)

    @app.get("/api/projects/<pid>/export/<kind>")
    def export(pid, kind):
        state = projects.load(pid)
        if state["active"] < 0:
            raise ValueError("请先生成谱面")
        variant = state["variants"][state["active"]]
        notes = [note_from(n) for n in variant["notes"]]
        base = projects.path(pid) / "files"
        original = None
        folder = Path()
        if variant["mode"] == "remap":
            original = safe_path(base, state["original"]).read_text(encoding="utf-8-sig")
            folder = Path(state["original"]).parent
        content = export_osu(notes, variant["settings"]["keys"], resources / "vendor" / "mapperatorinator" / "template.osu",
                             title=state["title"], artist=state["artist"], audio=state.get("audio") or "audio.ogg",
                             bpm=variant["bpm"], offset=variant["offset"], original=original,
                             version=f'ManiaLab {variant["settings"]["seed"]}')
        if kind == "osu":
            return send_file(io.BytesIO(content.encode("utf-8")), download_name="ManiaLab.osu", as_attachment=True, mimetype="application/octet-stream")
        if kind == "json":
            evidence = {"title": state["title"], "variant": variant, "analysis": state.get("analysis"), "plan": state.get("plan")}
            return send_file(io.BytesIO(json.dumps(evidence, ensure_ascii=False, indent=2).encode()), download_name="ManiaLab-analysis.json", as_attachment=True, mimetype="application/json")
        if kind != "osz":
            raise ValueError("不支持的导出格式")
        current_audio(state)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr((folder / "ManiaLab.osu").as_posix(), content)
            for file in base.rglob("*"):
                if file.is_file() and file.suffix.lower() != ".osu":
                    archive.write(file, file.relative_to(base).as_posix())
        buffer.seek(0)
        return send_file(buffer, download_name="ManiaLab.osz", as_attachment=True, mimetype="application/octet-stream")

    return app
