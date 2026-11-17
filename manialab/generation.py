"""Generate a candidate from saved project evidence, independent of HTTP/storage.

The caller owns serialization and commits state only after successful completion.
"""
from dataclasses import replace
from copy import deepcopy
import math
import time

from .arrange import assign_lanes, select_notes, layout_metrics
from .domain import note_from, validate, upstream_events
from .timing import align_candidates, chart_clock_notes, grid_report, analysis_timing_diagnostics
from .voices import apply_phrase_evidence
from .review import review_chart
from .ln_patterns import analyze_ln_patterns
from .preferences import ChartPreferences
from .evidence import capture_source_evidence
from .selection_report import regional_selection_report


def generate_variant(state, payload, prefs, tempo, progress, cancelled):
    mode = payload.get("mode", "audio")
    started = time.perf_counter()
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
        segment_preferences = ChartPreferences(segment_preferences, tempo["audio_shift_ms"])
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
    previous_evidence = None
    locked_edits = []
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
        previous_evidence = variant.get('source_evidence')
        locked_ids = {n.id for n in locked}
        locked_edits = [deepcopy(edit) for edit in variant.get('edits', [])
            if (edit.get('after') or {}).get('id') in locked_ids]
        selected = [n for n in selected if a <= n.start < b]
        if locked_ids & {n.id for n in selected}:
            raise ValueError('局部采音与保留物件的标识冲突；请重新生成整谱，避免混用不同分析的音符')
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
    timing_diagnostics = analysis_timing_diagnostics(state.get('analysis', {}), tempo) if mode == 'audio' else None
    if region is not None:
        selection = regional_selection_report(selection, output, locked)
    variant = dict(notes=[n.data() for n in output], settings=prefs, mode=mode, report=report, selection=selection,
                   bpm=tempo["bpm"], offset=tempo["offset"], division=tempo["division"],
                   audio_shift_ms=tempo["audio_shift_ms"],
                   timing_revision=3,
                   timing_sections=timing_diagnostics['sections'] if timing_diagnostics else [],
                   timing_diagnostics=timing_diagnostics,
                   snap=payload.get("snap", True) is True, timing_confirmed=payload.get("timing_confirmed", False) is True,
                   elapsed_seconds=round(time.perf_counter() - started, 2), created=time.time(), region=region,
                   event_count=len(upstream_events(output)))
    if mode == 'audio':
        variant['source_evidence'] = capture_source_evidence(state['analysis'], evidence, output,
            locked=locked, previous=previous_evidence)
    if locked_edits:
        variant['edits'] = locked_edits
    state["variants"] = (state["variants"] + [variant])[-8:]
    state["active"] = len(state["variants"]) - 1
    state["settings"] = prefs
    state["audio_shift_ms"] = tempo["audio_shift_ms"]
    return report
