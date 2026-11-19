import math

from flask import jsonify, request, send_file

from .audition import grid_events, imported_grid_events, render_audition, waveform_window
from .chart_workspace import audio_work
from .domain import parse_osu
from .stems import candidate_stem_path, separation_identity
from .storage import safe_path


def _current_source_reference(state):
    separation = (state.get('analysis') or {}).get('separation')
    if not separation:
        raise ValueError('当前采音结果没有可试听的独立声部')
    try:
        identity = separation_identity(separation)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('当前采音没有有效的声轨来源身份，请重新分析') from exc
    return dict(identity=identity, separation=separation)


def _playback_source_reference(state, context):
    if context not in ('chart', 'evidence'):
        raise ValueError('声部试听上下文无效')
    if context == 'chart' and state.get('active', -1) >= 0:
        variant = state['variants'][state['active']]
        if variant.get('mode') == 'audio':
            reference = variant.get('source_separation')
            if reference is None:
                raise ValueError('此旧候选未保存声轨来源，不能准确回看声部；请重新分析并生成候选')
            return reference
    return _current_source_reference(state)


def _context_stem_path(work, state, voice, context, expected_identity):
    reference = _playback_source_reference(state, context)
    if expected_identity != reference.get('identity'):
        raise ValueError('试听来源已切换，请刷新项目后重试')
    return candidate_stem_path(work, reference, voice)


def _validate_audio_reference(state, expected_audio):
    if expected_audio is not None and expected_audio != (state.get('audio') or ''):
        raise ValueError('试听音频已切换，请刷新项目后重试')


def register_playback_routes(app, *, projects, current_audio, requested_tempo):
    @app.get("/api/projects/<pid>/audio")
    def audio(pid):
        state = projects.load(pid)
        _validate_audio_reference(state, request.args.get('audio'))
        source = current_audio(state)
        preview = audio_work(projects.path(pid), state) / "preview.wav"
        return send_file(preview if preview.is_file() else source, conditional=True)

    @app.get("/api/projects/<pid>/stem/<voice>")
    def stem_audio(pid, voice):
        state = projects.load(pid)
        context = request.args.get('context', 'evidence')
        path = _context_stem_path(
            audio_work(projects.path(pid), state), state, voice, context,
            request.args.get('source_identity'))
        return send_file(path, conditional=True)

    @app.post("/api/projects/<pid>/audition")
    def audition(pid):
        state = projects.load(pid)
        payload = request.json
        voice = payload.get("voice", "mix")
        work = audio_work(projects.path(pid), state)
        _validate_audio_reference(state, payload.get('audio'))
        mode = payload.get("mode", "beats")
        context = ('chart' if mode == 'chart' else 'evidence' if mode == 'evidence'
                   else payload.get('context', 'evidence'))
        preview = work / "preview.wav" if voice == "mix" else _context_stem_path(
            work, state, voice, context, payload.get('source_identity'))
        start = float(payload.get("start_ms", 0))
        length = float(payload.get("duration_ms", 12000))
        if not math.isfinite(start) or not math.isfinite(length) or start < 0 or not 1000 <= length <= 30000:
            raise ValueError("试听范围必须为 1–30 秒，起点不能为负")
        remap_mix = False
        meter = None
        if mode == "beats":
            clock = payload.get("beat_clock")
            if clock is None:
                active = state["active"]
                saved_mode = state["variants"][active]["mode"] if active >= 0 else state["mode"]
                clock = "original" if saved_mode == "remap" else "fixed"
            if clock == "original":
                remap_mix = voice == "mix"
                points = state.get("input_timing")
                if points is None and state.get("original"):
                    chart = safe_path(projects.path(pid) / "files", state["original"])
                    points = parse_osu(chart.read_text(encoding="utf-8-sig"))["timing"]
                events = imported_grid_events(points or [], start, length)
            elif clock == "fixed":
                saved = state["variants"][state["active"]] if state["active"] >= 0 else {}
                saved_audio = saved.get("mode") == "audio"
                fields = ("bpm", "offset", "audio_shift_ms")
                if saved_audio and not any(key in payload for key in fields):
                    payload = {key: saved[key] for key in fields if key in saved} | payload
                tempo = requested_tempo(payload, state)
                meter = tempo["meter"]
                if saved_audio and all(tempo[key] == saved.get(key, 0) for key in fields):
                    meter = saved.get("meter", 4)
                events = grid_events(tempo["bpm"], tempo["offset"], start, length, meter=meter)
            else:
                raise ValueError("试听拍线来源无效")
        elif mode in ("chart", "evidence"):
            if mode == "chart":
                if state["active"] < 0:
                    raise ValueError("请先生成候选谱面")
                variant = state["variants"][state["active"]]
                source = variant["notes"]
                remap_mix = voice == "mix" and variant["mode"] == "remap"
            else:
                source = state.get("analysis", {}).get("candidates", [])
            if voice != "mix":
                source = [note for note in source if note.get("voice", "mix") == voice]
            events = [(note["start"], "head") for note in source]
            if payload.get("releases", True) is True:
                events.extend((note["end"], "tail") for note in source if note["end"] > note["start"])
        else:
            raise ValueError("试听模式无效")
        original_audio = not preview.is_file()
        if original_audio:
            if not (remap_mix and state.get("original")):
                raise ValueError("请先分析音频，再生成同步试听片段")
            preview = current_audio(state)
        buffer, report = render_audition(
            preview,
            start,
            length,
            events,
            music_volume=float(payload.get("music_volume", .7)),
            click_volume=float(payload.get("click_volume", .3)),
            original_audio=original_audio,
        )
        response = send_file(buffer, mimetype="audio/wav", download_name="ManiaLab-timing-audition.wav")
        response.headers["X-ManiaLab-Audition-Events"] = str(len(report["events"]))
        if meter is not None:
            response.headers["X-ManiaLab-Audition-Meter"] = str(meter)
        return response

    @app.get("/api/projects/<pid>/waveform")
    def waveform(pid):
        state = projects.load(pid)
        voice = request.args.get("voice", "mix")
        context = request.args.get('context', 'evidence')
        _validate_audio_reference(state, request.args.get('audio'))
        work = audio_work(projects.path(pid), state)
        preview = work / "preview.wav" if voice == "mix" else _context_stem_path(
            work, state, voice, context, request.args.get('source_identity'))
        if not preview.is_file():
            raise ValueError("请先分析音频，再查看波形细节")
        return jsonify(
            waveform_window(
                preview,
                float(request.args.get("start_ms", 0)),
                float(request.args.get("duration_ms", 2000)),
            )
        )
