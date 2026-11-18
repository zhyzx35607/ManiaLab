import math

from flask import jsonify, request, send_file

from .audition import grid_events, imported_grid_events, render_audition, waveform_window
from .chart_workspace import audio_work
from .domain import parse_osu
from .stems import stem_path
from .storage import safe_path


def register_playback_routes(app, *, projects, current_audio, requested_tempo):
    @app.get("/api/projects/<pid>/audio")
    def audio(pid):
        state = projects.load(pid)
        source = current_audio(state)
        preview = audio_work(projects.path(pid), state) / "preview.wav"
        return send_file(preview if preview.is_file() else source, conditional=True)

    @app.get("/api/projects/<pid>/stem/<voice>")
    def stem_audio(pid, voice):
        state = projects.load(pid)
        return send_file(stem_path(audio_work(projects.path(pid), state), state.get("analysis"), voice), conditional=True)

    @app.post("/api/projects/<pid>/audition")
    def audition(pid):
        state = projects.load(pid)
        payload = request.json
        voice = payload.get("voice", "mix")
        work = audio_work(projects.path(pid), state)
        preview = work / "preview.wav" if voice == "mix" else stem_path(work, state.get("analysis"), voice)
        start = float(payload.get("start_ms", 0))
        length = float(payload.get("duration_ms", 12000))
        if not math.isfinite(start) or not math.isfinite(length) or start < 0 or not 1000 <= length <= 30000:
            raise ValueError("试听范围必须为 1–30 秒，起点不能为负")
        mode = payload.get("mode", "beats")
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
        work = audio_work(projects.path(pid), state)
        preview = work / "preview.wav" if voice == "mix" else stem_path(work, state.get("analysis"), voice)
        if not preview.is_file():
            raise ValueError("请先分析音频，再查看波形细节")
        return jsonify(
            waveform_window(
                preview,
                float(request.args.get("start_ms", 0)),
                float(request.args.get("duration_ms", 2000)),
            )
        )
