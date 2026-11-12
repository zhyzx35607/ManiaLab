from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import secrets
import shutil
import threading
import time
import traceback
from urllib.parse import urlparse
import zipfile

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from . import __version__
from .arrange import DEFAULTS, assign_lanes, parameters, select_notes, layout_metrics
from .timing import DIVISIONS, align_candidates, chart_clock_notes, grid_diagnostics, grid_report, refit_offset
from .audio import analyze, analyze_deep
from .audition import grid_events, render_audition, waveform_window
from .separation import Separator
from .provided_stems import ProvidedStemSeparator,prepare_provided_stems
from .piano import PianoDetector
from .guitar import GuitarDetector
from .instrument_detector import InstrumentDetector
from .voices import apply_phrase_evidence
from .pattern_prior import prior_status
from .review import review_chart
from .ln_patterns import analyze_ln_patterns
from .editing import edit_notes
from .stems import VOICES,stem_path
from .style import describe_chart
from .domain import clean_meta, export_osu, note_from, parse_osu, upstream_events, validate
from .planner import plan
from .storage import AUDIO, Projects, extract_osz, safe_path
from .upstream import MapperEngine


def create_app(resources, data_root):
    resources, data_root = Path(resources).resolve(), Path(data_root).resolve()
    app = Flask(__name__, static_folder=str(resources / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    app.config["TOKEN"] = secrets.token_urlsafe(32)
    projects = Projects(Path(data_root) / "projects")
    executor, mutex = ThreadPoolExecutor(max_workers=1), threading.RLock()
    jobs = {}
    engine = MapperEngine(data_root, resources)
    separator = Separator(resources)
    piano_detector = PianoDetector(resources)
    guitar_detector = GuitarDetector(resources)
    instrument_detector = InstrumentDetector(resources)
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
        return jsonify(version=__version__, defaults=DEFAULTS, projects=projects.recent(),
                       capabilities={"audio": True, "remap": True, "api_planner": True, "trained_10k": False},
                       engine=engine.status(), separator=separator.status(), piano=piano_detector.status(),
                       guitar=guitar_detector.status(), instruments=instrument_detector.status(), pattern_prior=prior_status())

    def busy(pid):
        if any(j["project"] == pid and j["status"] in ("queued", "running") for j in jobs.values()):
            raise ValueError("该项目已有任务运行，请等待或取消")

    def current_audio(state):
        if state.get("audio"):
            return safe_path(projects.path(state["id"]) / "files", state["audio"])
        raise ValueError("项目没有音频；请导入包含音频的 .osz，或同时选择 .osu 与音频")

    def set_chart(state, relative):
        previous_audio=state.get('audio')
        base = projects.path(state["id"]) / "files"
        source = safe_path(base, relative)
        parsed = parse_osu(source.read_text(encoding="utf-8-sig"))
        from .structure import reference_marks
        state['reference_marks'] = reference_marks(source.read_text(encoding='utf-8-sig'))
        state.pop('expression', None)
        state.pop('plan', None)
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
        if state.get('audio')!=previous_audio:state.pop('provided_stems',None)
        state.pop("analysis", None)
        for file in ("activations.npz", "preview.wav"):
            (projects.path(state["id"]) / file).unlink(missing_ok=True)

    @app.post("/api/import")
    def import_files():
        files = request.files.getlist("files")
        if not files:
            raise ValueError("请选择音频、.osu 或 .osz 文件")
        pid, path = projects.create()
        try:
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
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            raise

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

    def submit(pid, operation, *, identity=None):
        with mutex:
            busy(pid)
            if any(j["status"] in ("queued", "running") for j in jobs.values()):
                raise ValueError("已有计算任务运行，请完成后再启动新任务")
            jid = secrets.token_hex(10)
            job = dict(id=jid, project=pid, status="queued", progress=0, message="等待运行", cancel=False)
            if identity is not None:
                job['identity'] = identity
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
        payload = request.json
        if not isinstance(payload,dict):
            raise ValueError('采音请求必须是对象')
        prefs = parameters(payload.get("settings", {}))
        analysis_engine = payload.get("engine", "basic")
        piano_engine = payload.get('piano_engine','general')
        guitar_engine = payload.get('guitar_engine','general')
        allow_guitar_releases = payload.get('allow_guitar_predicted_releases',False)
        instrument_engine = payload.get('instrument_engine','general')
        allow_releases = payload.get('allow_predicted_releases',False)
        if analysis_engine not in ("basic", "deep", "deep6", "provided"):
            raise ValueError("采音引擎无效")
        if piano_engine not in ('general','specialist'):
            raise ValueError('钢琴采音方式无效')
        if guitar_engine not in ('general','fl'):
            raise ValueError('吉他采音方式无效')
        if type(allow_guitar_releases) is not bool:
            raise ValueError('吉他模型音尾选项必须是布尔值')
        if allow_guitar_releases and guitar_engine!='fl':
            raise ValueError('请先选择吉他专用采音，再启用吉他模型音尾')
        if instrument_engine not in ('general','tsumugi'):
            raise ValueError('贝斯与伴奏采音方式无效')
        if type(allow_releases) is not bool:
            raise ValueError('模型音尾选项必须是布尔值')
        if allow_releases and instrument_engine!='tsumugi':
            raise ValueError('请先选择贝斯与其他伴奏专用采音，再启用模型音尾')
        if instrument_engine=='tsumugi':
            if analysis_engine not in ('deep','deep6','provided'):
                raise ValueError('专用采音需要分离或导入的贝斯、其他伴奏声轨')
            if not instrument_detector.status()['available']:
                raise ValueError('当前安装包未包含贝斯与其他伴奏专用组件')
            if analysis_engine=='provided':
                with mutex:
                    provided=projects.load(pid).get('provided_stems',{})
                    if not any(v in ('bass','other') for v in provided.get('voices',[])):
                        raise ValueError('已导入声轨中没有贝斯或其他伴奏')
        if guitar_engine=='fl':
            if analysis_engine not in ('deep6','provided'):
                raise ValueError('吉他专用采音需要六声部分离或已导入的吉他声轨')
            if not guitar_detector.status()['available']:
                raise ValueError('当前安装包未包含吉他专用采音模型')
            if analysis_engine=='provided':
                with mutex:
                    provided=projects.load(pid).get('provided_stems',{})
                    if 'guitar' not in provided.get('voices',[]):
                        raise ValueError('已导入声轨中没有吉他')
        if piano_engine=='specialist':
            if analysis_engine not in ('deep6','provided'):
                raise ValueError('钢琴专用采音需要六声部分离或已导入的钢琴声轨')
            if not piano_detector.status()['available']:
                raise ValueError('当前安装包未包含钢琴专用采音模型')
        def operation(state, progress, cancelled):
            if analysis_engine in ('deep','deep6','provided'):
                selected_separator=(ProvidedStemSeparator(state.get('provided_stems')) if analysis_engine=='provided'
                    else Separator(resources,'six') if analysis_engine=='deep6' else separator)
                state["analysis"] = analyze_deep(current_audio(state), projects.path(pid), resources / "assets/basic-pitch.onnx",
                    prefs, selected_separator, progress, cancelled,
                    piano_detector=piano_detector if piano_engine=='specialist' else None,
                    guitar_detector=guitar_detector if guitar_engine=='fl' else None,
                    allow_guitar_predicted_releases=allow_guitar_releases,
                    instrument_detector=instrument_detector if instrument_engine=='tsumugi' else None,
                    allow_predicted_releases=allow_releases)
            else:
                state["analysis"] = analyze(current_audio(state), projects.path(pid), resources / "assets" / "basic-pitch.onnx", prefs, progress, cancelled)
            state["settings"] = prefs
            state['analysis_options'] = dict(engine=analysis_engine,piano_engine=piano_engine,
                instrument_engine=instrument_engine,allow_predicted_releases=allow_releases,
                guitar_engine=guitar_engine,allow_guitar_predicted_releases=allow_guitar_releases)
            state["audio_shift_ms"] = requested_shift(payload, state)
            from .expression import propose_expression
            if state['analysis'].get('structure'):
                state['expression'] = propose_expression(state['analysis'], prefs)
            else:
                state.pop('expression', None)
            if state.get('plan',{}).get('source')=='local-audio-recommendation':
                state.pop('plan',None)
            # Keep old variants selectable, but never mask a fresh timing result
            # with the previously selected candidate's BPM / Offset.
            if state["mode"] == "audio":
                state["active"] = -1
            return {"candidates": len(state["analysis"]["candidates"])}
        return submit(pid, operation, identity=dict(engine=analysis_engine,piano_engine=piano_engine,
            instrument_engine=instrument_engine,allow_predicted_releases=allow_releases,
            guitar_engine=guitar_engine,allow_guitar_predicted_releases=allow_guitar_releases))

    @app.post('/api/projects/<pid>/provided-stems')
    def import_provided_stems(pid):
        with mutex:
            busy(pid)
            if any(j['status'] in ('queued','running') for j in jobs.values()):
                raise ValueError('已有计算任务运行，请完成后再导入声轨')
            state=projects.load(pid);current_audio(state)
            if not request.files or set(request.files)-set(VOICES):raise ValueError('请选择并标明声部文件')
            # Preserve old sets and analyses; each upload has an isolated path.
            folder=projects.path(pid)/'provided-uploads'/secrets.token_hex(16)
            sources={}
            for voice in request.files:
                uploads=request.files.getlist(voice)
                if len(uploads)!=1:raise ValueError('每个声部只能选择一个文件')
                upload=uploads[0]
                name=Path(upload.filename.replace('\\','/')).name
                if Path(name).suffix.lower() not in AUDIO:raise ValueError('声部文件必须是音频')
                target=safe_path(folder/voice,name);target.parent.mkdir(parents=True,exist_ok=True)
                upload.save(target);sources[voice]=target
            def operation(state,progress,cancelled):
                report=prepare_provided_stems(current_audio(state),sources,projects.path(pid),progress,cancelled)
                state['provided_stems']=report
                return {'voices':report['voices'],'duration_seconds':report['duration_seconds']}
            return submit(pid,operation)

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

    @app.post("/api/projects/<pid>/timing-refit")
    def timing_refit(pid):
        state = projects.load(pid)
        tempo = requested_tempo(request.json, state)
        attacks = chart_clock_notes([note_from(n) for n in state.get("analysis", {}).get("timing_attacks", [])], tempo["audio_shift_ms"])
        fitted = refit_offset(attacks, tempo["bpm"], tempo["division"])
        checks = grid_diagnostics(attacks, fitted["bpm"], fitted["offset"], fitted["division"])
        return jsonify(**fitted, sections=checks, suspect=sum(s["suspect"] for s in checks))

    @app.post("/api/projects/<pid>/timing-check")
    def timing_check(pid):
        state = projects.load(pid)
        tempo = requested_tempo(request.json, state)
        attacks = chart_clock_notes([note_from(n) for n in state.get("analysis", {}).get("timing_attacks", [])], tempo["audio_shift_ms"])
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
            from .recommendations import adjusted_segments, validate_plan, plan_evidence
            segment_preferences=adjusted_segments(state.get('plan',{}),prefs)
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
                validate_plan(state.get('plan',{}),state['analysis'],prefs['keys'],tempo['bpm'])
                if state["analysis"].get("revision", 1) < 3:
                    raise ValueError("该项目使用旧版采音结果，请先点击「分析音频」更新拍线和 LN 候选；音高缓存会自动复用")
                detected, phrases, weak_omitted = plan_evidence(state['analysis'],state.get('plan',{}))
                expression = None
                if state['analysis'].get('structure'):
                    from .expression import propose_expression, validate_expression, expression_evidence, select_expression
                    expression = state.get('expression')
                    if not expression or expression.get('source') == 'automatic':
                        expression = propose_expression(
                            state['analysis'], prefs, segment_preferences=segment_preferences)
                    validate_expression(expression, state['analysis'])
                    evidence, expression_meta, decisions = expression_evidence(
                        state['analysis'], expression, prefs, segment_preferences=segment_preferences)
                    phrase_report = dict(used=sum(n.phrase for n in evidence), explanation='由段落声部策略选择连续乐句表达')
                    state['expression'] = expression
                else:
                    evidence, phrase_report = apply_phrase_evidence([note_from(n) for n in detected],
                        [note_from(n) for n in phrases], prefs["phrase_ln"], segment_preferences)
                candidates, alignment = align_candidates(chart_clock_notes(evidence, tempo["audio_shift_ms"]),
                    tempo, enabled=payload.get("snap", True) is True, confirmed=payload.get("timing_confirmed", False) is True)
                progress(25, "按段落声部方案选择音组，检查完整持续音的占轨")
                if expression:
                    selected, selection = select_expression(candidates, expression_meta, decisions, prefs,
                        60000/tempo['bpm'], segment_preferences)
                    selection['expression'] = expression
                    selection['structure'] = state['analysis']['structure']
                else:
                    selected, selection = select_notes(candidates, prefs, segment_preferences)
                selection["phrases"] = phrase_report
                selection['recommendation_weak_candidates_omitted']=weak_omitted
                selection["alignment"] = alignment
                selection["message"] += (f" 量化起点 {alignment['moved']} 个、LN 尾端 {alignment['tails_moved']} 个；"
                    f"{alignment['large_attack_shifts']} 个候选起音移动超过 30ms，本谱保留 {selection.get('selected_large_attack_shifts', 0)} 个。" + alignment["reason"])
                if not selected:
                    raise ValueError("未找到可生成的采音候选，请检查声部权重、检测阈值或音源")
            locked = []
            region = payload.get("region")
            if region is not None:
                if state["active"] < 0:
                    raise ValueError("请先生成完整谱面")
                variant = state["variants"][state["active"]]
                if variant["settings"]["keys"] != prefs["keys"] or variant["mode"] != mode:
                    raise ValueError("局部重写不能改变键数或生成模式")
                if mode == "audio" and (variant.get("timing_revision", 0) < 3 or
                    any(variant.get(k, 0) != tempo[k] for k in ("bpm", "offset", "division", "audio_shift_ms"))):
                    raise ValueError("Timing 已改变或候选来自旧版，请先重新生成整谱，避免局部重写混用不同拍线")
                if not isinstance(region, list) or len(region) != 2 or not all(type(x) in (int, float) and math.isfinite(x) for x in region) or not 0 <= region[0] < region[1]:
                    raise ValueError("局部时间范围无效")
                a, b = region
                previous = [note_from(n) for n in variant["notes"]]
                locked = [n for n in previous if not a <= n.start < b]
                selected = [n for n in selected if a <= n.start < b]
            progress(45, "分配轨道并检查 LN 占轨")
            original_timing = state.get("input_timing", []) if mode == "remap" else []
            output = assign_lanes(selected, prefs, locked, beat_ms=60000 / tempo["bpm"], progress=progress, cancelled=cancelled,
                                  segment_preferences=segment_preferences, timing_points=original_timing)
            progress(90, "校验谱面并保存候选")
            report = validate(output, prefs["keys"])
            if mode == "audio":
                report["clock"] = dict(audio_shift_ms=tempo["audio_shift_ms"],
                    explanation="正数表示谱面起止比原始采音提前；未改变音频或原始检测。试听按实际导出时间播放提示音。")
                report["grid"] = grid_report(output, tempo)
                if report["grid"]["off_grid"]:
                    raise ValueError("生成结果存在离线音符，未保存；请重新分析并生成整谱")
            report["layout"] = layout_metrics(output, prefs["keys"], 60000 / tempo["bpm"], original_timing)
            report["review"] = review_chart(output, prefs["keys"], 60000 / tempo["bpm"], original_timing)
            report["ln_patterns"] = analyze_ln_patterns(output, prefs["keys"], 60000 / tempo["bpm"], original_timing)
            timing_sections = grid_diagnostics(chart_clock_notes([note_from(n) for n in state.get("analysis", {}).get("timing_attacks", [])], tempo["audio_shift_ms"]),
                                              tempo["bpm"], tempo["offset"], tempo["division"]) if mode == "audio" else []
            if region is not None:
                selection["message"] = f"局部定轨完成；整谱实际 LN 比例 {report['ln_ratio']:.1%}。区域外物件已锁定。"
            variant = dict(notes=[n.data() for n in output], settings=prefs, mode=mode, report=report, selection=selection,
                           bpm=tempo["bpm"], offset=tempo["offset"], division=tempo["division"],
                           audio_shift_ms=tempo["audio_shift_ms"],
                           timing_revision=3,
                           timing_sections=timing_sections,
                           snap=payload.get("snap", True) is True, timing_confirmed=payload.get("timing_confirmed", False) is True,
                           elapsed_seconds=round(time.perf_counter() - started, 2), created=time.time(), region=region,
                           event_count=len(upstream_events(output)))
            state["variants"] = (state["variants"] + [variant])[-8:]
            state["active"] = len(state["variants"]) - 1
            state["settings"] = prefs
            state["audio_shift_ms"] = tempo["audio_shift_ms"]
            return report
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

    @app.post("/api/projects/<pid>/plan")
    def api_plan(pid):
        payload = request.json
        prefs = parameters(payload.get("settings", {}))
        def operation(state, progress, cancelled):
            if not state.get("analysis"):
                raise ValueError("请先完成音频分析，API 需要真实的段落统计")
            progress(15, "请求 API 编排建议（发送统计与文字，不发送音频）")
            review = state["variants"][state["active"]]["report"].get("review") if state["active"] >= 0 else None
            result = plan(payload.get("endpoint", ""), payload.get("model", ""), payload.get("key", ""),
                          payload.get("instruction", ""), state["analysis"]["segments"], prefs, review=review)
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

    @app.post('/api/projects/<pid>/recommendations')
    def local_recommendations(pid):
        from .recommendations import recommend
        payload=request.get_json() or {}
        with mutex:
            state=projects.load(pid)
            prefs=parameters(state['settings'] | payload.get('settings',{}))
            result=recommend(state.get('analysis'),prefs,resources,payload.get('bpm'))
        return jsonify(result)

    @app.post('/api/projects/<pid>/structure')
    def music_structure(pid):
        payload = request.get_json() or {}
        def operation(state, progress, cancelled):
            from .structure import detect_structure, edit_boundaries
            from .expression import propose_expression
            analysis = state.get('analysis')
            if not analysis or 'duration' not in analysis:
                raise ValueError('请先分析音频')
            structure = analysis.get('structure', {})
            action = payload.get('action', 'detect')
            if action == 'detect':
                from .audio import decode, SR
                progress(10, '读取音频，比较音色、能量与声部活动变化')
                samples = decode(current_audio(state), projects.path(pid))
                structure = detect_structure(analysis, samples, SR)
            elif action == 'bookmarks':
                marks = state.get('reference_marks', {}).get('bookmarks', [])
                if not marks:
                    raise ValueError('当前导入谱面没有书签')
                # Bookmarks use the chart clock; acoustic sections use audio.
                shift = requested_shift(payload, state)
                marks = [round(t+shift) for t in marks if 0 < t+shift < analysis['duration']*1000]
                structure = edit_boundaries(structure, marks, analysis, 'reference-bookmarks')
                structure['notice'] = '段落边界来自导入谱面的书签；它们是可编辑提示，不代表自动识别了主歌或副歌。'
            elif action == 'kiai':
                marks = state.get('reference_marks', {}).get('kiai_transitions', [])
                if not marks:
                    raise ValueError('当前导入谱面没有 Kiai 开关点')
                # Kiai state changes use the chart clock; acoustic sections use audio.
                shift = requested_shift(payload, state)
                marks = [round(t+shift) for t in marks if 0 < t+shift < analysis['duration']*1000]
                if not marks:
                    raise ValueError('导入谱面的 Kiai 开关点都不在当前音频范围内')
                structure = edit_boundaries(structure, marks, analysis, 'reference-kiai')
                structure['notice'] = '段落边界来自导入谱面的 Kiai 开关；它们是可编辑提示，不代表自动识别了主歌或副歌。'
            elif action == 'edit':
                structure = edit_boundaries(structure, payload.get('boundaries'), analysis)
            else:
                raise ValueError('分段操作无效')
            analysis['structure'] = structure
            analysis['segments'] = structure['segments']
            state.pop('plan', None)
            state['expression'] = propose_expression(analysis, parameters(state['settings'] | payload.get('settings', {})))
            return dict(sections=len(structure['segments']))
        return submit(pid, operation)

    @app.post('/api/projects/<pid>/expression')
    def source_expression(pid):
        from .expression import validate_expression, propose_expression
        payload = request.get_json() or {}
        with mutex:
            busy(pid)
            state = projects.load(pid)
            if not state.get('analysis', {}).get('structure'):
                raise ValueError('请先检测音乐段落')
            if payload.get('action') == 'automatic':
                prefs = parameters(state['settings'] | payload.get('settings', {}))
                from .recommendations import adjusted_segments
                value = propose_expression(state['analysis'], prefs,
                    segment_preferences=adjusted_segments(state.get('plan', {}), prefs))
            else:
                value = validate_expression(payload.get('expression'), state['analysis'])
                value = dict(value, source='manual')
            state['expression'] = value
            projects.save(state)
        return jsonify(state)

    @app.post('/api/projects/<pid>/apply-recommendation')
    def apply_recommendation(pid):
        from .recommendations import recommend
        payload=request.get_json() or {}
        with mutex:
            busy(pid)
            state=projects.load(pid)
            prefs=parameters(state['settings'] | payload.get('settings',{}))
            result=recommend(state.get('analysis'),prefs,resources,payload.get('bpm'))
            if payload.get('evidence_signature')!=result['evidence_signature']:
                raise ValueError('音频分析已经改变，请刷新建议后再选择')
            if payload.get('context_signature')!=result['context_signature']:
                raise ValueError('建议的键数、BPM、参考统计或生成规则已经改变，请刷新建议后再选择')
            option=next((o for o in result['options'] if o['id']==payload.get('choice')),None)
            if option is None:
                raise ValueError('请选择已有的编排建议')
            state['settings']=parameters(prefs | option['preferences'])
            state['plan']={k:option[k] for k in ('preferences','segments','explanation')}
            state['plan'].update(source='local-audio-recommendation',choice=option['id'],
                evidence_signature=result['evidence_signature'],reference=result['reference'],
                section_reasons=option['section_reasons'],keys=result['keys'],bpm=result['bpm'])
            if state.get('expression', {}).get('source') == 'automatic':
                from .expression import propose_expression
                state['expression'] = propose_expression(state['analysis'], state['settings'],
                    segment_preferences=option['segments'])
            projects.save(state)
        return jsonify(state)

    @app.post("/api/projects/<pid>/reference-style")
    def reference_style(pid):
        with mutex:
            busy(pid)
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

    @app.get("/api/projects/<pid>/audio")
    def audio(pid):
        state = projects.load(pid)
        preview = projects.path(pid) / "preview.wav"
        return send_file(preview if preview.exists() else current_audio(state), conditional=True)

    @app.get("/api/projects/<pid>/stem/<voice>")
    def stem_audio(pid, voice):
        state=projects.load(pid)
        return send_file(stem_path(projects.path(pid),state.get('analysis'),voice),conditional=True)

    @app.post("/api/projects/<pid>/audition")
    def audition(pid):
        state = projects.load(pid)
        payload = request.json
        voice = payload.get("voice", "mix")
        preview = projects.path(pid)/'preview.wav' if voice=='mix' else stem_path(projects.path(pid),state.get('analysis'),voice)
        if not preview.is_file():
            raise ValueError("请先分析音频，再生成同步试听片段")
        start = float(payload.get("start_ms", 0))
        length = float(payload.get("duration_ms", 12000))
        if not math.isfinite(start) or not math.isfinite(length) or start < 0 or not 1000 <= length <= 30000:
            raise ValueError("试听范围必须为 1–30 秒，起点不能为负")
        mode = payload.get("mode", "beats")
        if mode == "beats":
            tempo = requested_tempo(payload, state)
            events = grid_events(tempo["bpm"], tempo["offset"], start, length)
        elif mode in ("chart", "evidence"):
            if mode == "chart":
                if state["active"] < 0:
                    raise ValueError("请先生成候选谱面")
                source = state["variants"][state["active"]]["notes"]
            else:
                source = state.get("analysis", {}).get("candidates", [])
            if voice != "mix":
                source = [n for n in source if n.get("voice", "mix") == voice]
            events = [(n["start"], "head") for n in source]
            if payload.get("releases", True) is True:
                events.extend((n["end"], "tail") for n in source if n["end"] > n["start"])
        else:
            raise ValueError("试听模式无效")
        buffer, report = render_audition(preview, start, length, events,
            music_volume=float(payload.get("music_volume", .7)), click_volume=float(payload.get("click_volume", .3)))
        response = send_file(buffer, mimetype="audio/wav", download_name="ManiaLab-timing-audition.wav")
        response.headers["X-ManiaLab-Audition-Events"] = str(len(report["events"]))
        return response

    @app.get("/api/projects/<pid>/waveform")
    def waveform(pid):
        state=projects.load(pid)
        voice = request.args.get("voice", "mix")
        preview = projects.path(pid)/'preview.wav' if voice=='mix' else stem_path(projects.path(pid),state.get('analysis'),voice)
        if not preview.is_file():
            raise ValueError("请先分析音频，再查看波形细节")
        return jsonify(waveform_window(preview, float(request.args.get("start_ms", 0)),
                                       float(request.args.get("duration_ms", 2000))))

    @app.get("/api/projects/<pid>/export/<kind>")
    def export(pid, kind):
        state = projects.load(pid)
        if kind == "json":
            if not state.get("analysis") and state["active"] < 0:
                raise ValueError("请先分析音频或生成候选，再下载分析 JSON")
            evidence = {"title": state["title"], "variant": state["variants"][state["active"]] if state["active"] >= 0 else None,
                        "analysis": state.get("analysis"), "plan": state.get("plan"), "reference_style": state.get("reference_style"),
                        "expression": state.get('expression'), "reference_marks": state.get('reference_marks')}
            return send_file(io.BytesIO(json.dumps(evidence, ensure_ascii=False, indent=2).encode()), download_name="ManiaLab-analysis.json", as_attachment=True, mimetype="application/json")
        if state["active"] < 0:
            raise ValueError("请先生成谱面")
        variant = state["variants"][state["active"]]
        notes = [note_from(n) for n in variant["notes"]]
        if variant["mode"] == "audio" and variant.get("timing_revision", 0) < 3:
            raise ValueError("该候选来自旧版，未通过完整落线检查；请重新分析音频并生成整谱后导出")
        base = projects.path(pid) / "files"
        original = None
        folder = Path()
        if variant["mode"] == "remap":
            original = safe_path(base, state["original"]).read_text(encoding="utf-8-sig")
            folder = Path(state["original"]).parent
        content = export_osu(notes, variant["settings"]["keys"], resources / "vendor" / "mapperatorinator" / "template.osu",
                             title=state["title"], artist=state["artist"], audio=state.get("audio") or "audio.ogg",
                             bpm=variant["bpm"], offset=variant["offset"], original=original,
                             grid_division=variant["division"] if variant["mode"] == "audio" else None,
                             version=f'ManiaLab {variant["settings"]["seed"]}',
                             bookmarks=[max(0, round(t-variant.get('audio_shift_ms', 0))) for t in
                                 variant.get('selection', {}).get('structure', {}).get('boundaries', [])[1:-1]])
        if variant.get('provided_timing'):
            from .upstream import verify_fixed_timing
            verify_fixed_timing(content, variant['provided_timing'])
        if kind == "osu":
            return send_file(io.BytesIO(content.encode("utf-8")), download_name="ManiaLab.osu", as_attachment=True, mimetype="application/octet-stream")
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
