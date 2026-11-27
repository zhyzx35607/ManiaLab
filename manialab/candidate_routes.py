import copy
import hashlib
import math
from pathlib import Path
import re
import secrets
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
from .arrange_config import DEFAULT_KEY_LAYOUTS, TOROSU_PLAYSTYLES
from .storage import safe_path
from .task_service import MAX_BATCH_SIZE


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

    @app.get("/api/presets")
    def list_presets():
        from .presets import PRESET_IDS, get_preset

        return jsonify(items=[get_preset(preset_id) for preset_id in PRESET_IDS])

    @app.get("/api/batch/jobs")
    def list_batch_jobs():
        if task_service is None:
            return jsonify(items=[])
        try:
            limit = int(request.args.get("limit", "20"))
        except (TypeError, ValueError) as error:
            raise ValueError("批量历史 limit 必须为整数") from error
        return jsonify(items=task_service.recent_batches(limit=limit))

    @app.get("/api/batch/jobs/<job_id>")
    def get_batch_job(job_id):
        if task_service is None:
            return jsonify(error="当前服务不支持批量任务"), 404
        job = task_service.get(job_id)
        if not job or not job.get("batch"):
            return jsonify(error="批量任务不存在"), 404
        return jsonify(job)

    def completed_lineage(previous, spec):
        rows = {}
        sources = list(spec.get("lineage_items", []))
        sources.extend(previous.get("attempt_items", previous.get("items", [])))
        for item in sources:
            if (not isinstance(item, dict) or item.get("status") != "done"
                    or not isinstance(item.get("project"), str)):
                continue
            row = {
                "project": item["project"],
                "title": item.get("title") or item["project"],
                "status": "done",
                "progress": 100,
                "message": item.get("message", "已完成"),
            }
            candidate_id = item.get("candidate_id")
            if not isinstance(candidate_id, str):
                result = item.get("result")
                candidate_id = result.get("candidate_id") if isinstance(result, dict) else None
            if isinstance(candidate_id, str):
                row["candidate_id"] = candidate_id
                row["result"] = {"candidate_id": candidate_id}
            rows[row["project"]] = row
        return list(rows.values())

    @app.post("/api/batch/jobs/<job_id>/resume")
    def resume_batch_job(job_id):
        if task_service is None:
            raise ValueError("当前服务不支持批量任务")
        previous = task_service.get(job_id)
        spec = task_service.batch_resume_spec(job_id)
        if not previous or not previous.get("batch") or not isinstance(spec, dict):
            raise ValueError("该批次缺少可恢复请求")
        if previous.get("status") in ("queued", "running"):
            raise ValueError("该批次仍在运行，不能重复续跑")
        attempt_items = previous.get("attempt_items", previous.get("items", []))
        candidates = [
            item for item in attempt_items
            if item.get("status") != "done"
        ]
        lineage_items = completed_lineage(previous, spec)
        resume_chain = list(spec.get("resume_chain", []))
        if job_id not in resume_chain:
            resume_chain.append(job_id)
        resume_payload = request.get_json(silent=True)
        selected_projects = resume_payload.get("projects") if isinstance(resume_payload, dict) else None
        if selected_projects is not None and (
                not isinstance(selected_projects, list)
                or any(not isinstance(project_id, str) for project_id in selected_projects)):
            raise ValueError("续跑项目列表无效")
        selected = set(selected_projects) if selected_projects is not None else None
        if selected is not None:
            unknown = selected - {item.get("project") for item in candidates}
            if unknown:
                raise ValueError("续跑项目不属于该批次未完成歌曲")
            candidates = [item for item in candidates if item.get("project") in selected]
        if not candidates:
            raise ValueError("该批次没有待续跑歌曲")
        kind = spec.get("kind")
        if kind == "analysis":
            runner = app.extensions.get("batch_analysis_submit")
            if runner is None:
                raise ValueError("批量分析恢复服务不可用")
            projects_to_run = [item["project"] for item in candidates]
            return jsonify(runner(
                projects_to_run,
                spec["options"],
                spec["max_parallel"],
                persistent_spec={
                    **spec,
                    "projects": projects_to_run,
                    "resumed_from": job_id,
                    "resume_chain": resume_chain,
                    "lineage_items": lineage_items,
                },
                min_items=1,
            ))
        if kind in ("generation", "v32-generation"):
            payload = dict(spec["payload"])
            payload["projects"] = [item["project"] for item in candidates]
            payload["resumed_from"] = job_id
            return jsonify(generate_batch_payload(
                payload,
                lineage_items=lineage_items,
                resume_chain=resume_chain,
            ))
        raise ValueError("批次类型不支持恢复")

    def generate_batch_payload(payload, *, lineage_items=(), resume_chain=()):
        if not isinstance(payload, dict):
            raise ValueError("批量生成请求必须为对象")
        project_ids = payload.get("projects")
        if (not isinstance(project_ids, list)
                or not 1 <= len(project_ids) <= MAX_BATCH_SIZE
                or any(not isinstance(project_id, str) or not project_id
                       for project_id in project_ids)):
            raise ValueError(f"批量生成需要选择 1–{MAX_BATCH_SIZE} 首已完成分析的歌曲")
        if len(set(project_ids)) != len(project_ids):
            raise ValueError("批量生成不能重复选择同一首歌曲")
        preset = payload.get("preset")
        if not isinstance(preset, str):
            raise ValueError("批量制谱预设无效")
        from .presets import get_preset, legacy_generation_settings

        preset_definition = get_preset(preset)
        if preset_definition["mode"] != "audio":
            raise ValueError("音频批量只能使用四种音频风格预设；已有谱改造请在单曲工作流中执行")
        keys = payload.get("keys")
        if type(keys) is not int or keys not in DEFAULT_KEY_LAYOUTS:
            raise ValueError("批量键数必须为 6–10K")
        key_layout = payload.get("key_layout", DEFAULT_KEY_LAYOUTS[keys])
        difficulty = payload.get("difficulty", 5.5)
        scribe = payload.get("scribe", "zhyzx35607")
        suffix = payload.get("version_suffix", "")
        timing_confirmed = payload.get("timing_confirmed", False)
        snap = payload.get("snap", True)
        if (not isinstance(scribe, str) or not scribe.strip()
                or len(scribe.strip()) > 64 or any(ord(char) < 32 for char in scribe)):
            raise ValueError("统一谱师名必须为 1–64 个可见字符")
        if (not isinstance(suffix, str) or len(suffix) > 48
                or any(ord(char) < 32 for char in suffix)):
            raise ValueError("预设版本后缀不能超过 48 个可见字符")
        if type(timing_confirmed) is not bool or type(snap) is not bool:
            raise ValueError("Timing 确认和量化选项必须为布尔值")
        if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)) \
                or not math.isfinite(difficulty) or not 0 <= difficulty <= 10:
            raise ValueError("目标难度必须在 0–10 星之间")
        if payload.get("torosu_density") is not None:
            selected_density = payload["torosu_density"]
            if selected_density not in ("none", "dense", "more-dense", "full"):
                raise ValueError("Torosu 密度无效")
        else:
            selected_density = preset_definition["torosu"]["density"]
        selected_ln_mode = payload.get(
            "torosu_ln_mode", preset_definition["torosu"]["ln_mode"],
        )
        if selected_ln_mode not in ("keep", "fln", "short_ln"):
            raise ValueError("Torosu LN 模式无效")
        settings = legacy_generation_settings(
            preset,
            keys=keys,
            density=payload.get("density"),
        )
        settings["key_layout"] = key_layout
        settings["torosu_density"] = selected_density
        settings["torosu_ln_mode"] = selected_ln_mode
        if key_layout not in TOROSU_PLAYSTYLES and settings.get("layout_engine") == "torosu":
            settings["layout_engine"] = "beam"
        shared_settings = parameters(settings)
        if preset_definition["id"] == "torosu_remap":
            shared_settings["layout_engine"] = "torosu"
        mode = payload.get("engine", "traditional")
        if mode not in ("traditional", "v32", "v32-mini"):
            raise ValueError("批量生成引擎无效")
        if preset_definition["id"] == "torosu_remap" and mode != "traditional":
            raise ValueError("已有谱 Torosu 改造必须使用传统 Torosu 引擎")
        generation_payload = {
            "mode": "audio",
            "settings": shared_settings,
            "snap": snap,
            "timing_confirmed": timing_confirmed,
            "metadata": {"creator": scribe.strip(), "version_suffix": suffix.strip()},
        }
        timing_payload = {
            field: payload[field]
            for field in ("bpm", "offset", "division", "audio_shift_ms")
            if field in payload
        }
        persistent_spec = {
            "kind": "v32-generation" if mode in ("v32", "v32-mini") else "generation",
            "payload": {
                key: value for key, value in payload.items()
                if key != "resumed_from"
            },
            "batch_key": payload.get("batch_key"),
            "lineage_items": list(lineage_items),
            "resume_chain": list(resume_chain),
            "settings": {
                "keys": keys,
                "key_layout": key_layout,
                "engine": mode,
                "preset": preset_definition["id"],
                "difficulty": difficulty,
                "hold_note_ratio": payload.get(
                    "hold_note_ratio", preset_definition["v32"]["hold_note_ratio"],
                ),
                "torosu_density": selected_density,
                "torosu_keep_jacks": preset_definition["torosu"]["keep_jacks"],
                "torosu_ln_mode": selected_ln_mode,
                "lora_path": payload.get("lora_path"),
                "timing": timing_payload,
                "shared": shared_settings,
            },
        }
        batch_key = persistent_spec["batch_key"]
        if not isinstance(batch_key, str) or not re.fullmatch(r"[0-9a-f]{24}", batch_key):
            batch_key = secrets.token_hex(12)
            persistent_spec["batch_key"] = batch_key
            persistent_spec["payload"]["batch_key"] = batch_key
        factory = app.extensions.get("v32_operation_factory")
        v32_engine = mode in ("v32", "v32-mini")
        if v32_engine and factory is None:
            raise ValueError("当前服务未注册 V32 批量生成器")
        entries = []
        identities = {}
        for project_id in project_ids:
            state = projects.load(project_id)
            if state.get("id") != project_id or state.get("mode") != "audio":
                raise ValueError("批量制谱只能选择音频项目")
            analysis = state.get("analysis")
            if not isinstance(analysis, dict) or analysis.get("revision", 1) < 3:
                raise ValueError(f"{state.get('title') or project_id} 尚未完成最新音频分析")
            audio_name = state.get("audio")
            if not audio_name:
                raise ValueError(f"{state.get('title') or project_id} 没有可用于导出的音频")
            audio_path = current_audio(state) if current_audio is not None else safe_path(
                projects.path(project_id) / "files", audio_name,
            )
            source_sha = _analysis_audio_sha256(analysis)
            if not audio_path.is_file() or source_sha is None or _audio_file_sha256(audio_path) != source_sha:
                raise ValueError(f"{state.get('title') or project_id} 的当前音频与分析音频不一致")
            audio_stat = audio_path.stat()
            clock = requested_tempo(timing_payload, state)
            item_payload = {
                **generation_payload,
                "bpm": clock["bpm"],
                "offset": clock["offset"],
                "division": clock["division"],
                "audio_shift_ms": clock["audio_shift_ms"],
                "timing_confirmed": timing_confirmed,
            }
            if v32_engine:
                options = {
                    "preset": preset_definition["id"],
                    "keys": keys,
                    "key_layout": key_layout,
                    "model_variant": mode if mode != "traditional" else "v32-mini",
                    "difficulty": difficulty,
                    "hold_note_ratio": payload.get(
                        "hold_note_ratio", preset_definition["v32"]["hold_note_ratio"],
                    ),
                    "torosu_density": selected_density,
                    "torosu_keep_jacks": preset_definition["torosu"]["keep_jacks"],
                    "torosu_ln_mode": selected_ln_mode,
                    "lora_path": payload.get("lora_path"),
                    "timing_confirmed": timing_confirmed,
                    "metadata": item_payload["metadata"],
                    **timing_payload,
                }
                v32_operation = factory(project_id, options)

                def operation(song, progress, cancelled, *, inner=v32_operation,
                              meta=item_payload["metadata"], operation_batch_key=batch_key):
                    for saved in reversed(song.get("variants", [])):
                        if saved.get("batch_identity") == operation_batch_key:
                            return {"candidate_id": saved.get("candidate_id"), "recovered": True}
                    result = inner(song, progress, cancelled)
                    active = song.get("active", -1)
                    variants = song.get("variants", [])
                    if isinstance(active, int) and 0 <= active < len(variants):
                        variants[active]["metadata"] = meta
                        variants[active]["batch_identity"] = operation_batch_key
                    return result

                operation_cleanup = getattr(v32_operation, "cleanup", None)
                item_payload["v32"] = options
            else:
                operation_cleanup = None
                item_payload["settings"] = shared_settings

                def operation(
                    song,
                    progress,
                    cancelled,
                    *,
                    expected_revision=analysis.get("revision", 1),
                    expected_audio_name=audio_name,
                    expected_audio_sha=source_sha,
                    expected_audio_stat=(audio_stat.st_size, audio_stat.st_mtime_ns),
                    expected_clock=clock,
                    requested_clock=timing_payload,
                    song_payload=item_payload,
                    song_settings=shared_settings,
                    title=state.get("title") or project_id,
                    operation_batch_key=batch_key,
                ):
                    for saved in reversed(song.get("variants", [])):
                        if saved.get("batch_identity") == operation_batch_key:
                            return {"candidate_id": saved.get("candidate_id"), "recovered": True}
                    current_analysis = song.get("analysis")
                    current_sha = (_analysis_audio_sha256(current_analysis)
                                   if isinstance(current_analysis, dict) else None)
                    if (song.get("mode") != "audio"
                            or not isinstance(current_analysis, dict)
                            or current_analysis.get("revision", 1) != expected_revision
                            or current_sha != expected_audio_sha
                            or song.get("audio") != expected_audio_name):
                        raise ValueError(f"{title} 的分析或音频已变化，不能续跑")
                    audio_now = current_audio(song) if current_audio else safe_path(
                        projects.path(song["id"]) / "files", song.get("audio", ""),
                    )
                    stat = audio_now.stat()
                    if (stat.st_size, stat.st_mtime_ns) != expected_audio_stat \
                            or _audio_file_sha256(audio_now) != expected_audio_sha:
                        raise ValueError(f"{title} 的音频内容已变化，不能续跑")
                    current_clock = requested_tempo(requested_clock, song)
                    if any(current_clock[key] != expected_clock[key] for key in (
                            "bpm", "offset", "division", "audio_shift_ms", "stable",
                    )):
                        raise ValueError(f"{title} 的拍线已变化，请重新检查")
                    if not current_clock["stable"] and not song_payload["timing_confirmed"]:
                        raise ValueError(
                            f"{title} 的自动 Timing 不可靠；先单独检查并试听，"
                            "或明确确认本批次所有歌曲的当前拍线"
                        )
                    report = generate_variant(
                        song, song_payload, song_settings, current_clock, progress, cancelled,
                    )
                    active = song.get("active", -1)
                    variant = song.get("variants", [])[active]
                    if not variant.get("candidate_id"):
                        raise ValueError("批量生成没有提交有效候选")
                    variant["metadata"] = song_payload["metadata"]
                    variant["preset"] = preset_definition["id"]
                    variant["batch_identity"] = operation_batch_key
                    return {"candidate_id": variant["candidate_id"], "report": report}
            entries.append({
                "project": project_id,
                "title": state.get("title"),
                "operation": operation,
                "completed_check": lambda song, operation_batch_key=batch_key: next(
                    (
                        {"candidate_id": saved.get("candidate_id"), "recovered": True}
                        for saved in reversed(song.get("variants", []))
                        if saved.get("batch_identity") == operation_batch_key
                    ),
                    None,
                ),
                **({"cleanup": operation_cleanup} if operation_cleanup else {}),
            })
            identities[project_id] = {
                "analysis_revision": analysis.get("revision", 1),
                "audio_sha256": source_sha,
                "audio_name": audio_name,
                "clock": clock,
            }
        identity = {
            "projects": project_ids,
            "preset": preset_definition["id"],
            "engine": mode,
            "keys": keys,
            "key_layout": key_layout,
            "settings": persistent_spec["settings"],
            "scribe": scribe.strip(),
            "version_suffix": suffix.strip(),
        }
        if v32_engine:
            engine_status = app.extensions["v32_engine"].status() \
                if app.extensions.get("v32_engine") is not None else None
            if engine_status is None:
                raise ValueError("V32 引擎状态不可用")
            if not engine_status.get("available"):
                raise ValueError("Mapperatorinator V32 推理环境未就绪")
        return task_service.submit_batch(
            entries,
            max_parallel=1,
            identity=identity,
            min_items=1,
            persistent_spec={
                **persistent_spec,
                "item_identity": identities,
                "resumed_from": payload.get("resumed_from"),
            },
        )
    @app.post("/api/projects/<pid>/generate")
    def generate(pid):
        payload = request.json
        raw_settings = payload.get("settings", {})
        preset_id = payload.get("preset")
        mode = payload.get("mode", "audio")
        if preset_id:
            from .presets import get_preset, legacy_generation_settings

            preset = get_preset(preset_id)
            mode = payload.get("mode", preset["mode"])
            if preset["mode"] == "remap":
                mode = "remap"
            keys = raw_settings.get("keys", 10) if isinstance(raw_settings, dict) else 10
            raw_settings = {
                **legacy_generation_settings(
                    preset_id,
                    keys=keys,
                    density=raw_settings.get("density") if isinstance(raw_settings, dict) else None,
                ),
                **(raw_settings if isinstance(raw_settings, dict) else {}),
            }
        prefs = parameters(raw_settings)
        if mode not in ("audio", "remap"):
            raise ValueError("生成模式无效")
        payload = {**payload, "mode": mode}

        def operation(state, progress, cancelled):
            report = generate_variant(state, payload, prefs, requested_tempo(payload, state), progress, cancelled)
            active = state.get("active", -1)
            variants = state.get("variants", [])
            if isinstance(active, int) and 0 <= active < len(variants):
                variants[active]["metadata"] = payload.get("metadata", {})
                variants[active]["preset"] = preset_id
            return report

        return submit(pid, operation)

    if projects is not None and task_service is not None:
        @app.post("/api/batch/generate")
        def generate_batch():
            return jsonify(generate_batch_payload(request.get_json(silent=True)))

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
