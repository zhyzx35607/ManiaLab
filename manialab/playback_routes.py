import math

from flask import jsonify, request, send_file

from .audition import grid_events, render_audition, waveform_window
from .stems import stem_path


def register_playback_routes(app, *, projects, current_audio, requested_tempo):
    @app.get("/api/projects/<pid>/audio")
    def audio(pid):
        state = projects.load(pid)
        preview = projects.path(pid) / "preview.wav"
        return send_file(preview if preview.exists() else current_audio(state), conditional=True)

    @app.get("/api/projects/<pid>/stem/<voice>")
    def stem_audio(pid, voice):
        state = projects.load(pid)
        return send_file(stem_path(projects.path(pid), state.get("analysis"), voice), conditional=True)

    @app.post("/api/projects/<pid>/audition")
    def audition(pid):
        state = projects.load(pid)
        payload = request.json
        voice = payload.get("voice", "mix")
        preview = projects.path(pid) / "preview.wav" if voice == "mix" else stem_path(
            projects.path(pid), state.get("analysis"), voice
        )
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
                source = [note for note in source if note.get("voice", "mix") == voice]
            events = [(note["start"], "head") for note in source]
            if payload.get("releases", True) is True:
                events.extend((note["end"], "tail") for note in source if note["end"] > note["start"])
        else:
            raise ValueError("试听模式无效")
        buffer, report = render_audition(
            preview,
            start,
            length,
            events,
            music_volume=float(payload.get("music_volume", .7)),
            click_volume=float(payload.get("click_volume", .3)),
        )
        response = send_file(buffer, mimetype="audio/wav", download_name="ManiaLab-timing-audition.wav")
        response.headers["X-ManiaLab-Audition-Events"] = str(len(report["events"]))
        return response

    @app.get("/api/projects/<pid>/waveform")
    def waveform(pid):
        state = projects.load(pid)
        voice = request.args.get("voice", "mix")
        preview = projects.path(pid) / "preview.wav" if voice == "mix" else stem_path(
            projects.path(pid), state.get("analysis"), voice
        )
        if not preview.is_file():
            raise ValueError("请先分析音频，再查看波形细节")
        return jsonify(
            waveform_window(
                preview,
                float(request.args.get("start_ms", 0)),
                float(request.args.get("duration_ms", 2000)),
            )
        )
