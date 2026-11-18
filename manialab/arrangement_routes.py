from flask import jsonify, request
from .chart_workspace import audio_work
from .recommendation_plan import adjusted_segments


def register_arrangement_routes(app, *, projects, task_service, resources, submit,
                                parameters, requested_shift, current_audio, planner):
    app.extensions["planner"] = planner

    @app.post("/api/projects/<pid>/plan")
    def api_plan(pid):
        payload = request.json
        prefs = parameters(payload.get("settings", {}))

        def operation(state, progress, cancelled):
            if not state.get("analysis"):
                raise ValueError("请先完成音频分析，API 需要真实的段落统计")
            progress(15, "请求 API 编排建议（发送统计与文字，不发送音频）")
            review = state["variants"][state["active"]]["report"].get("review") if state["active"] >= 0 else None
            result = app.extensions["planner"](
                payload.get("endpoint", ""), payload.get("model", ""), payload.get("key", ""),
                payload.get("instruction", ""), state["analysis"]["segments"], prefs, review=review,
            )
            state["plan"] = result
            state["settings"] = parameters(prefs | result["preferences"])
            return result

        return submit(pid, operation)

    @app.post("/api/projects/<pid>/clear-plan")
    def clear_plan(pid):
        with task_service.lock:
            task_service.ensure_project_idle(pid)
            state = projects.load(pid)
            state.pop("plan", None)
            projects.save(state)
        return jsonify(state)

    @app.post("/api/projects/<pid>/recommendations")
    def local_recommendations(pid):
        from .recommendations import recommend

        payload = request.get_json() or {}
        with task_service.lock:
            state = projects.load(pid)
            prefs = parameters(state["settings"] | payload.get("settings", {}))
            result = recommend(state.get("analysis"), prefs, resources, payload.get("bpm"))
        return jsonify(result)

    @app.post("/api/projects/<pid>/structure")
    def music_structure(pid):
        payload = request.get_json() or {}

        def operation(state, progress, cancelled):
            from .structure import detect_structure, edit_boundaries
            from .expression import propose_expression

            analysis = state.get("analysis")
            if not analysis or "duration" not in analysis:
                raise ValueError("请先分析音频")
            structure = analysis.get("structure", {})
            action = payload.get("action", "detect")
            if action == "detect":
                from .audio_config import SR
                from .audio_io import decode

                progress(10, "读取音频，比较音色、能量与声部活动变化")
                samples = decode(current_audio(state), audio_work(projects.path(pid), state))
                structure = detect_structure(analysis, samples, SR)
            elif action == "bookmarks":
                marks = state.get("reference_marks", {}).get("bookmarks", [])
                if not marks:
                    raise ValueError("当前导入谱面没有书签")
                # Bookmarks use the chart clock; acoustic sections use audio.
                shift = requested_shift(payload, state)
                marks = [round(t + shift) for t in marks if 0 < t + shift < analysis["duration"] * 1000]
                structure = edit_boundaries(structure, marks, analysis, "reference-bookmarks")
                structure["notice"] = "段落边界来自导入谱面的书签；它们是可编辑提示，不代表自动识别了主歌或副歌。"
            elif action == "kiai":
                marks = state.get("reference_marks", {}).get("kiai_transitions", [])
                if not marks:
                    raise ValueError("当前导入谱面没有 Kiai 开关点")
                # Kiai state changes use the chart clock; acoustic sections use audio.
                shift = requested_shift(payload, state)
                marks = [round(t + shift) for t in marks if 0 < t + shift < analysis["duration"] * 1000]
                if not marks:
                    raise ValueError("导入谱面的 Kiai 开关点都不在当前音频范围内")
                structure = edit_boundaries(structure, marks, analysis, "reference-kiai")
                structure["notice"] = "段落边界来自导入谱面的 Kiai 开关；它们是可编辑提示，不代表自动识别了主歌或副歌。"
            elif action == "edit":
                structure = edit_boundaries(structure, payload.get("boundaries"), analysis)
            else:
                raise ValueError("分段操作无效")
            analysis["structure"] = structure
            analysis["segments"] = structure["segments"]
            state.pop("plan", None)
            state["expression"] = propose_expression(analysis, parameters(state["settings"] | payload.get("settings", {})))
            return dict(sections=len(structure["segments"]))

        return submit(pid, operation)

    @app.post("/api/projects/<pid>/expression")
    def source_expression(pid):
        from .expression import validate_expression, propose_expression

        payload = request.get_json() or {}
        with task_service.lock:
            task_service.ensure_project_idle(pid)
            state = projects.load(pid)
            if not state.get("analysis", {}).get("structure"):
                raise ValueError("请先检测音乐段落")
            if payload.get("action") == "automatic":
                prefs = parameters(state["settings"] | payload.get("settings", {}))
                value = propose_expression(
                    state["analysis"], prefs,
                    segment_preferences=adjusted_segments(state.get("plan", {}), prefs),
                )
            else:
                value = validate_expression(payload.get("expression"), state["analysis"])
                value = dict(value, source="manual")
            state["expression"] = value
            projects.save(state)
        return jsonify(state)

    @app.post("/api/projects/<pid>/apply-recommendation")
    def apply_recommendation(pid):
        from .recommendations import recommend

        payload = request.get_json() or {}
        with task_service.lock:
            task_service.ensure_project_idle(pid)
            state = projects.load(pid)
            prefs = parameters(state["settings"] | payload.get("settings", {}))
            result = recommend(state.get("analysis"), prefs, resources, payload.get("bpm"))
            if payload.get("evidence_signature") != result["evidence_signature"]:
                raise ValueError("音频分析已经改变，请刷新建议后再选择")
            if payload.get("context_signature") != result["context_signature"]:
                raise ValueError("建议的键数、BPM、参考统计或生成规则已经改变，请刷新建议后再选择")
            option = next((item for item in result["options"] if item["id"] == payload.get("choice")), None)
            if option is None:
                raise ValueError("请选择已有的编排建议")
            state["settings"] = parameters(prefs | option["preferences"])
            state["plan"] = {key: option[key] for key in ("preferences", "segments", "explanation")}
            state["plan"].update(
                source="local-audio-recommendation", choice=option["id"],
                evidence_signature=result["evidence_signature"], reference=result["reference"],
                section_reasons=option["section_reasons"], keys=result["keys"], bpm=result["bpm"],
            )
            if state.get("expression", {}).get("source") == "automatic":
                from .expression import propose_expression

                state["expression"] = propose_expression(
                    state["analysis"], state["settings"], segment_preferences=option["segments"],
                )
            projects.save(state)
        return jsonify(state)
