from flask import jsonify, request

from .domain import note_from
from .timing import analysis_timing_diagnostics, chart_clock_notes, refit_offset


def register_timing_routes(app, projects, requested_tempo):
    @app.post("/api/projects/<pid>/timing-refit")
    def timing_refit(pid):
        state = projects.load(pid)
        tempo = requested_tempo(request.json, state)
        attacks = chart_clock_notes([note_from(note) for note in state.get("analysis", {}).get("timing_attacks", [])],
                                    tempo["audio_shift_ms"])
        fitted = refit_offset(attacks, tempo["bpm"], tempo["division"])
        checks = analysis_timing_diagnostics(state.get("analysis", {}), tempo | fitted)
        return jsonify(fitted | checks)

    @app.post("/api/projects/<pid>/timing-check")
    def timing_check(pid):
        state = projects.load(pid)
        tempo = requested_tempo(request.json, state)
        if not state.get("analysis", {}).get("timing_attacks"):
            raise ValueError("请用新版重新分析音频，才能检查整曲拍线")
        return jsonify(analysis_timing_diagnostics(state["analysis"], tempo))
