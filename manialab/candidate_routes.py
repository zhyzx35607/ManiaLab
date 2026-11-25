import copy
import hashlib
from pathlib import Path
import re
import time

from flask import jsonify, request

from .arrange import layout_metrics
from .candidate_identity import new_candidate_id
from .domain import upstream_events, validate
from .editing import edit_notes
from .generation import generate_variant
from .ln_patterns import analyze_ln_patterns
from .review import review_chart
from .selection_report import generation_context, resolve_generation_origin
from .timing import grid_report
from .arrange_config import DEFAULT_KEY_LAYOUTS
from .storage import safe_path


BATCH_GENERATION_PRESETS = {
    "onset_only": dict(density=8, ln_ratio=0, phrase_ln=0, min_ln=240,
                       max_held=5, chord_size=4, focus="balanced"),
    "balanced": dict(density=7, ln_ratio=.55, phrase_ln=.65, min_ln=240,
                     max_held=5, chord_size=4, focus="balanced"),
    "sustain": dict(density=6, ln_ratio=.85, phrase_ln=.85, min_ln=300,
                    max_held=4, chord_size=3, focus="pitched"),
    "rhythm": dict(density=8, ln_ratio=.25, phrase_ln=.25, min_ln=240,
                   max_held=3, chord_size=4, focus="transient"),
}


def _analysis_audio_sha256(analysis):
    identity = analysis.get("audio_sha256")
    if not isinstance(identity, str):
        separation = analysis.get("separation")
        identity = separation.get("audio_sha256") if isinstance(separation, dict) else None
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", identity):
        return None
    return identity.lower()


def _audio_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_candidate_routes(app, *, submit, parameters, requested_tempo,
                              projects=None, task_service=None, current_audio=None):
    projects = projects or app.extensions.get("projects")
    task_service = task_service or app.extensions.get("task_service")
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

    if projects is not None and task_service is not None:
        @app.post("/api/batch/generate")
        def generate_batch():
            payload = request.get_json()
            if not isinstance(payload, dict):
                raise ValueError("批量生成请求必须为对象")
            project_ids = payload.get("projects")
            if (not isinstance(project_ids, list)
                    or not 1 <= len(project_ids) <= 10
                    or any(not isinstance(project_id, str) or not project_id
                           for project_id in project_ids)):
                raise ValueError("批量生成需要选择 1–10 首已完成分析的歌曲")
            if len(set(project_ids)) != len(project_ids):
                raise ValueError("批量生成不能重复选择同一首歌曲")
            preset = payload.get("preset")
            if not isinstance(preset, str):
                raise ValueError("批量制谱预设无效")
            preset_settings = BATCH_GENERATION_PRESETS.get(preset)
            if preset_settings is None:
                raise ValueError("批量制谱预设无效")
            keys = payload.get("keys")
            if type(keys) is not int or keys not in DEFAULT_KEY_LAYOUTS:
                raise ValueError("批量键数必须为 6–10K")
            key_layout = payload.get("key_layout", DEFAULT_KEY_LAYOUTS[keys])
            density = payload.get("density", preset_settings["density"])
            timing_confirmed = payload.get("timing_confirmed", False)
            snap = payload.get("snap", True)
            if type(timing_confirmed) is not bool or type(snap) is not bool:
                raise ValueError("Timing 确认和量化选项必须为布尔值")
            shared_settings = parameters({
                **preset_settings,
                "keys": keys,
                "key_layout": key_layout,
                "density": density,
            })
            generation_payload = {
                "mode": "audio",
                "settings": shared_settings,
                "snap": snap,
                "timing_confirmed": timing_confirmed,
            }
            entries = []
            for project_id in project_ids:
                state = projects.load(project_id)
                if state.get("id") != project_id or state.get("mode") != "audio":
                    raise ValueError("批量制谱只能选择音频项目")
                analysis = state.get("analysis")
                if not isinstance(analysis, dict):
                    raise ValueError(f"{state.get('title') or project_id} 尚未完成音频分析")
                if analysis.get("revision", 1) < 3:
                    raise ValueError(
                        f"{state.get('title') or project_id} 的采音结果过旧，请重新分析"
                    )
                audio_name = state.get("audio")
                if not audio_name:
                    raise ValueError(
                        f"{state.get('title') or project_id} 没有可用于导出的音频"
                    )
                if current_audio is not None:
                    audio_path = current_audio(state)
                else:
                    audio_path = safe_path(projects.path(project_id) / "files", audio_name)
                if not audio_path.is_file():
                    raise ValueError(
                        f"{state.get('title') or project_id} 的音频文件不存在"
                    )
                source_sha256 = _analysis_audio_sha256(analysis)
                if source_sha256 is None:
                    raise ValueError(
                        f"{state.get('title') or project_id} 缺少有效音频身份，请重新分析"
                    )
                if _audio_file_sha256(audio_path) != source_sha256:
                    raise ValueError(
                        f"{state.get('title') or project_id} 的当前音频与分析音频不一致，请重新分析"
                    )
                audio_stat = audio_path.stat()
                tempo = requested_tempo({}, state)
                song_settings = dict(shared_settings)
                old_settings = state.get("settings")
                if isinstance(old_settings, dict):
                    seed = old_settings.get("seed")
                    if isinstance(seed, int) and not isinstance(seed, bool) \
                            and 0 <= seed <= 2147483647:
                        song_settings["seed"] = seed

                def operation(song, progress, cancelled, *, options=generation_payload,
                              prefs=song_settings, clock=tempo, title=state.get("title") or project_id,
                              expected_revision=analysis.get("revision", 1),
                              expected_audio_name=audio_name, expected_audio_sha=source_sha256,
                              expected_audio_stat=(audio_stat.st_size, audio_stat.st_mtime_ns)):
                    fresh_analysis = song.get("analysis")
                    fresh_source_sha = (_analysis_audio_sha256(fresh_analysis)
                                        if isinstance(fresh_analysis, dict) else None)
                    fresh_audio_name = song.get("audio")
                    if (song.get("mode") != "audio"
                            or not isinstance(fresh_analysis, dict)
                            or fresh_analysis.get("revision", 1) != expected_revision
                            or fresh_source_sha != expected_audio_sha
                            or fresh_audio_name != expected_audio_name):
                        raise ValueError(
                            f"{title} 的分析或音频在批量任务启动前已改变，请重新分析"
                        )
                    fresh_path = (current_audio(song) if current_audio is not None else
                                  safe_path(projects.path(song["id"]) / "files", fresh_audio_name))
                    if not fresh_path.is_file():
                        raise ValueError(f"{title} 的音频文件不存在")
                    fresh_stat = fresh_path.stat()
                    if (fresh_stat.st_size, fresh_stat.st_mtime_ns) != expected_audio_stat:
                        raise ValueError(
                            f"{title} 的音频文件在批量任务启动前已改变，请重新分析"
                        )
                    if _audio_file_sha256(fresh_path) != expected_audio_sha:
                        raise ValueError(
                            f"{title} 的音频内容在批量任务启动前已改变，请重新分析"
                        )
                    fresh_clock = requested_tempo({}, song)
                    if any(fresh_clock[key] != clock[key] for key in (
                            "bpm", "offset", "division", "audio_shift_ms", "stable")):
                        raise ValueError(
                            f"{title} 的拍线在批量任务启动前已改变，请重新分析并检查拍线"
                        )
                    if not clock["stable"] and not options["timing_confirmed"]:
                        raise ValueError(
                            f"{title} 的自动 Timing 不可靠；先单独检查并试听，或明确确认本批次所有歌曲的当前拍线"
                        )
                    report = generate_variant(
                        song, options, prefs, clock, progress, cancelled,
                    )
                    active = song.get("active", -1)
                    variant = (song.get("variants", [])[active]
                               if isinstance(active, int)
                               and 0 <= active < len(song.get("variants", []))
                               else None)
                    if not isinstance(variant, dict) or not variant.get("candidate_id"):
                        raise ValueError("批量生成没有提交有效候选")
                    return {"candidate_id": variant["candidate_id"], "report": report}

                entries.append(dict(project=project_id, title=state.get("title"),
                                    operation=operation))
            identity = {
                "projects": project_ids,
                "preset": preset,
                "settings": {key: shared_settings[key]
                             for key in ("keys", "key_layout", "density", "ln_ratio",
                                         "phrase_ln", "focus")},
            }
            return jsonify(task_service.submit_batch(
                entries, max_parallel=1, identity=identity, min_items=1,
            ))

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
