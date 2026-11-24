import copy
import time

from flask import request

from .arrange import layout_metrics
from .candidate_identity import new_candidate_id
from .domain import upstream_events, validate
from .editing import edit_notes
from .generation import generate_variant
from .ln_patterns import analyze_ln_patterns
from .review import review_chart
from .selection_report import generation_context, resolve_generation_origin
from .timing import grid_report


def register_candidate_routes(app, *, submit, parameters, requested_tempo):
    @app.post("/api/projects/<pid>/generate")
    def generate(pid):
        payload = request.json
        prefs = parameters(payload.get("settings", {}))
        mode = payload.get("mode", "audio")
        if mode not in ("audio", "remap"):
            raise ValueError("生成模式无效")

        def operation(state, progress, cancelled):
            return generate_variant(state, payload, prefs, requested_tempo(payload, state), progress, cancelled)

        return submit(pid, operation)

    @app.post("/api/projects/<pid>/edit")
    def edit_candidate(pid):
        payload = request.json

        def operation(state, progress, cancelled):
            if state["active"] < 0:
                raise ValueError("请先生成候选谱面")
            old = state["variants"][state["active"]]
            base_candidate_id = payload.get("base_candidate_id")
            if (not isinstance(base_candidate_id, str)
                    or base_candidate_id != old.get("candidate_id")):
                raise ValueError("候选版本已经改变或页面版本过旧，请刷新后再修改")
            progress(20, "检查修改与相邻 LN 占轨")
            length = state.get("analysis", {}).get("duration")
            notes, edit = edit_notes(old, payload, round(length * 1000) if length else None)
            prefs = old["settings"]
            points = state.get("input_timing", []) if old["mode"] == "remap" else []
            beat = 60000 / old["bpm"]
            report = validate(notes, prefs["keys"])
            if old["mode"] == "audio":
                report["clock"] = old["report"].get("clock", {})
                report["grid"] = grid_report(notes, old)
                if report["grid"]["off_grid"]:
                    raise ValueError("修改后的音符未通过落线检查")
            report["layout"] = layout_metrics(notes, prefs["keys"], beat, points,
                                              key_layout=prefs["key_layout"])
            report["review"] = review_chart(notes, prefs["keys"], beat, points,
                                            key_layout=prefs["key_layout"])
            report["ln_patterns"] = analyze_ln_patterns(
                notes, prefs["keys"], beat, points, key_layout=prefs["key_layout"],
            )
            variant = copy.deepcopy(old | {"selection": generation_context(old.get("selection", {}))})
            variant["generation_origin"] = resolve_generation_origin(old, state["variants"])
            variant.update(
                candidate_id=new_candidate_id(),
                notes=[note.data() for note in notes],
                report=report,
                created=time.time(),
                elapsed_seconds=0.0,
                event_count=len(upstream_events(notes)),
                region=None,
            )
            variant["edits"] = [*old.get("edits", []), dict(
                parent_created=old["created"], parent_candidate_id=old["candidate_id"], **edit)]
            variant["selection"]["message"] = "手动修改已保存为新候选。原始采音保留，可从候选列表返回上一版。"
            state["variants"] = (state["variants"] + [variant])[-8:]
            state["active"] = len(state["variants"]) - 1
            state["settings"] = prefs.copy()
            return report

        return submit(pid, operation)
