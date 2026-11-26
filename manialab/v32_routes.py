"""HTTP route for cancellable Mapperatorinator V32 candidates."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np
from flask import jsonify, request

from .arrange import layout_metrics, parameters
from .audio_io import decode
from .audio_cache import sha256_file
from .chart_workspace import audio_work
from .candidate_identity import new_candidate_id
from .domain import (
    export_osu,
    parse_osu,
    upstream_events,
    validate,
)
from .ln_patterns import analyze_ln_patterns
from .review import review_chart
from .stems import PROFILES
from .timing import grid_report
from .v32_charts import (
    _format_sections,
    _set_field,
    repair_native_10k_collisions,
    sections,
    sanitize_zero_length_model_holds,
    snap_chart_to_grid,
    trim_air_ln_tails,
)
from .separation import Separator
from .torosu_bridge import convert_osu_text


_ALLOWED_PAYLOAD = {
    "model_variant",
    "generation_strategy",
    "torosu_density",
    "difficulty",
    "hold_note_ratio",
    "mania_column_temperature",
    "lora_path",
    "timing_confirmed",
}
_TOROSU_LAYOUT = "10k_5_5"


def _analysis_audio_sha256(analysis: dict) -> str | None:
    identity = analysis.get("audio_sha256")
    if not isinstance(identity, str):
        separation = analysis.get("separation")
        identity = separation.get("audio_sha256") if isinstance(separation, dict) else None
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", identity):
        return None
    return identity.lower()


def _request_options(payload: dict, engine) -> tuple[dict, bool]:
    unknown = set(payload) - _ALLOWED_PAYLOAD
    if unknown:
        raise ValueError("V32 请求含未知字段：" + ", ".join(sorted(unknown)))

    model_variant = payload.get("model_variant", "v32")
    generation_strategy = payload.get("generation_strategy", "7k_to_torosu_10k")
    torosu_density = payload.get("torosu_density", "dense")
    if not isinstance(model_variant, str):
        raise ValueError("模型版本无效")
    if not isinstance(generation_strategy, str):
        raise ValueError("生成策略无效")
    if not isinstance(torosu_density, str):
        raise ValueError("Torosu 密度无效")
    difficulty = payload.get("difficulty", 5.5)
    hold_note_ratio = payload.get("hold_note_ratio", 0.55)
    mania_column_temperature = payload.get("mania_column_temperature", 1.0)
    lora_path = payload.get("lora_path")
    if lora_path is not None and not isinstance(lora_path, str):
        raise ValueError("LoRA 路径必须为文本")
    if lora_path is not None:
        lora_path = lora_path.strip() or None
    timing_confirmed = payload.get("timing_confirmed", False)
    if type(timing_confirmed) is not bool:
        raise ValueError("Timing 确认选项必须为布尔值")

    options = engine.validate_options(
        model_variant=model_variant,
        generation_strategy=generation_strategy,
        torosu_density=torosu_density,
        difficulty=difficulty,
        hold_note_ratio=hold_note_ratio,
        mania_column_temperature=mania_column_temperature,
        lora_path=lora_path,
        device="auto",
    )
    return options, timing_confirmed


def _separation_energy(separator, audio_path, work_path, source_sha256, progress, cancelled):
    if not separator.status().get("available"):
        raise ValueError("V32 的分轨 LN 校验需要可用的六声部分离组件")
    Path(work_path).mkdir(parents=True, exist_ok=True)
    stem_directory, report = separator.separate(
        audio_path,
        work_path,
        progress=progress,
        cancelled=cancelled,
        include_beats=False,
    )
    if report.get("audio_sha256") != source_sha256:
        raise ValueError("六声部分离结果与当前输入音频身份不一致")
    voices = PROFILES["six"]["voices"]
    arrays = []
    for voice in voices:
        path = Path(stem_directory) / f"{voice}.npy"
        if not path.is_file():
            raise ValueError(f"六声部分离结果缺少 {voice} 声部")
        samples = np.load(path, allow_pickle=False)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError(f"{voice} 声部能量数据无效")
        arrays.append(samples.astype(np.float32, copy=False))
    length = min(map(len, arrays), default=0)
    if length <= 0:
        raise ValueError("六声部分离没有有效采样")
    energy = np.zeros(length, dtype=np.float32)
    for samples in arrays:
        energy += np.square(samples[:length], dtype=np.float32)
    np.sqrt(energy, out=energy)
    return energy, report


def register_v32_routes(
    app,
    *,
    engine,
    projects,
    submit,
    current_audio,
    requested_tempo,
    resources,
):
    resources = Path(resources)

    @app.post("/api/projects/<pid>/v32-install")
    def install_v32(pid):
        status = engine.status()
        if not status.get("install_available"):
            raise ValueError("当前发行包不含 V32 自动安装组件；请配置固定 Mapperatorinator 源码与运行环境")

        def operation(_state, progress, cancelled):
            result = engine.install_runtime(progress=progress, cancelled=cancelled)
            progress(100, "V32 推理环境已就绪")
            return result

        return submit(pid, operation)

    @app.post("/api/projects/<pid>/v32-generate")
    def generate_v32(pid):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("V32 请求必须为 JSON 对象")
        options, timing_confirmed = _request_options(payload, engine)
        if not engine.status().get("available"):
            raise ValueError("Mapperatorinator V32 推理环境未就绪")

        run = {}
        separator = Separator(resources, profile="six")

        def cleanup(_committed):
            run_path = run.get("path")
            if run_path is not None and run_path.exists():
                shutil.rmtree(run_path)

        def operation(state, progress, cancelled):
            if state.get("mode") != "audio":
                raise ValueError("V32 生成仅支持音频项目")
            analysis = state.get("analysis")
            if not isinstance(analysis, dict):
                raise ValueError("请先完成音频分析")
            if type(analysis.get("revision")) is not int or analysis["revision"] < 3:
                raise ValueError("该项目采音结果过旧，请重新分析后生成 V32 候选")
            audio_path = current_audio(state)
            if not audio_path.is_file():
                raise ValueError("项目音频文件不存在")
            source_sha256 = sha256_file(audio_path, cancelled=cancelled)
            if source_sha256 != _analysis_audio_sha256(analysis):
                raise ValueError("当前音频与分析音频不一致，请重新分析")

            tempo = requested_tempo(payload, state)
            if not (tempo.get("stable") or timing_confirmed):
                raise ValueError("当前自动 Timing 不可靠；请检查拍线并明确勾选 Timing 确认")
            if cancelled():
                raise InterruptedError("已取消，原有候选已保留")

            project_path = projects.path(pid)
            work = audio_work(project_path, state)
            work.mkdir(parents=True, exist_ok=True)
            run_path = Path(tempfile.mkdtemp(prefix="v32-", dir=work))
            run["path"] = run_path
            preview = run_path / "preview.wav"
            decode(audio_path, run_path, cancelled=cancelled)
            if cancelled():
                raise InterruptedError("已取消，原有候选已保留")

            strategy = options["generation_strategy"]
            model_keys = 7 if strategy == "7k_to_torosu_10k" else 10
            prefs = parameters({
                **(state.get("settings") if isinstance(state.get("settings"), dict) else {}),
                "keys": 10,
                "key_layout": _TOROSU_LAYOUT,
                "layout_engine": "torosu",
            })
            template = resources / "vendor" / "mapperatorinator" / "template.osu"
            reference = export_osu(
                [],
                model_keys,
                template,
                title=state.get("title", "Untitled"),
                artist=state.get("artist", "Unknown"),
                audio=preview.name,
                bpm=tempo["bpm"],
                offset=tempo["offset"],
                meter=tempo.get("meter", 4),
                version=f"V32 timing context {model_keys}K",
            )
            reference_parts = sections(reference)
            reference_parts["Editor"] = _set_field(
                reference_parts.get("Editor", []),
                "BeatDivisor",
                str(tempo["division"]),
            )
            reference = _format_sections(reference_parts)
            reference_path = run_path / "timing-context.osu"
            reference_path.write_text(reference, encoding="utf-8")
            inference_path = run_path / "inference"
            progress(3, "准备六声部 RMS 能量校验")
            energy, separation_report = _separation_energy(
                separator,
                audio_path,
                work / ("v32-separation-" + source_sha256[:16]),
                source_sha256,
                progress,
                cancelled,
            )
            if cancelled():
                raise InterruptedError("已取消，原有候选已保留")

            inference = engine.generate(
                model_variant=options["model_variant"],
                audio_path=preview,
                reference_chart=reference_path,
                output_dir=inference_path,
                keycount=model_keys,
                difficulty=options["difficulty"],
                hold_note_ratio=options["hold_note_ratio"],
                mania_column_temperature=options["mania_column_temperature"],
                lora_path=options["lora_path"],
                device="auto",
                progress=progress,
                cancelled=cancelled,
            )
            if cancelled():
                raise InterruptedError("已取消，原有候选已保留")
            raw_chart = Path(inference["map"]).read_text(encoding="utf-8-sig")
            raw_chart, hold_repair = sanitize_zero_length_model_holds(raw_chart)
            parsed_model = parse_osu(raw_chart)
            if parsed_model["keys"] != model_keys:
                raise ValueError(
                    f"V32 输出应为 {model_keys}K，实际为 {parsed_model['keys']}K"
                )
            snapped, grid_snap = snap_chart_to_grid(
                raw_chart,
                bpm=tempo["bpm"],
                offset_ms=tempo["offset"],
                division=tempo["division"],
            )
            progress(93, "按六声部 RMS 检查空气 LN 尾部")
            trimmed, trim_report = trim_air_ln_tails(
                snapped,
                energy,
                22050,
                grid_bpm=tempo["bpm"],
                grid_offset_ms=tempo["offset"],
                grid_division=tempo["division"],
            )

            torosu_report = {"reassigned": False, "density": "none"}
            if strategy == "7k_to_torosu_10k":
                converted = convert_osu_text(
                    trimmed,
                    10,
                    playstyle="5+5",
                    density=options["torosu_density"],
                    keep_jacks=True,
                    strict=True,
                    cancelled=cancelled,
                )
                torosu_report["density"] = options["torosu_density"]
            else:
                repaired, repair_report = repair_native_10k_collisions(trimmed)
                converted = convert_osu_text(
                    repaired,
                    10,
                    playstyle="5+5",
                    density="none",
                    keep_jacks=True,
                    strict=True,
                    cancelled=cancelled,
                )
                torosu_report.update(repair_report)
            if cancelled():
                raise InterruptedError("已取消，原有候选已保留")

            converted_notes = parse_osu(converted)["notes"]
            output_text = export_osu(
                converted_notes,
                10,
                None,
                original=reference,
                version=f"V32 {options['model_variant']} · {strategy}",
            )
            notes = parse_osu(output_text)["notes"]
            report = validate(notes, 10)
            if not report["valid"]:
                raise ValueError("V32/Torosu 输出未通过结构校验：" + report["issues"][0])
            beat_ms = 60000 / tempo["bpm"]
            report["grid"] = grid_report(notes, tempo)
            if report["grid"]["off_grid"]:
                raise ValueError("V32/Torosu 输出存在未落拍物件，未保存候选")
            report["layout"] = layout_metrics(
                notes, 10, beat_ms, key_layout=prefs["key_layout"],
            )
            report["review"] = review_chart(
                notes, 10, beat_ms, key_layout=prefs["key_layout"],
            )
            report["ln_patterns"] = analyze_ln_patterns(
                notes, 10, beat_ms, key_layout=prefs["key_layout"],
            )
            report["v32"] = {
                "model_variant": inference["model_variant"],
                "model_revision": inference["model_revision"],
                "generation_strategy": strategy,
                "device": inference["device"],
                "fast_decoder_loop_active": inference["fast_decoder_loop_active"],
                "lora_loaded": inference["lora_loaded"],
                "lora_path": inference["lora_path"],
                "elapsed_seconds": inference.get("elapsed_seconds"),
                "peak_rss_bytes": inference.get("peak_rss_bytes"),
                "hold_repair": hold_repair,
                "grid_snap": grid_snap,
                "energy_trim": trim_report,
                "separation": {
                    "audio_sha256": separation_report.get("audio_sha256"),
                    "model": separation_report.get("model"),
                    "model_sha256": separation_report.get("model_sha256"),
                    "profile": separation_report.get("profile"),
                },
                "torosu": torosu_report,
            }
            candidate = {
                "candidate_id": new_candidate_id(),
                "notes": [note.data() for note in notes],
                "settings": prefs,
                "mode": "audio",
                "report": report,
                "selection": {
                    "message": (
                        f"由 Mapperatorinator {options['model_variant']} 生成并经 Torosu 10K "
                        "校验；分轨 RMS 只用于裁切持续近静音的 LN 尾端。"
                    ),
                    "layout_origin": "mapperatorinator_v32",
                    "generation_strategy": strategy,
                    "model_revision": inference["model_revision"],
                    "torosu": torosu_report,
                },
                "bpm": tempo["bpm"],
                "offset": tempo["offset"],
                "division": tempo["division"],
                "meter": tempo.get("meter", 4),
                "audio_shift_ms": tempo["audio_shift_ms"],
                "timing_revision": 3,
                "timing_confirmed": timing_confirmed,
                "created": time.time(),
                "elapsed_seconds": inference.get("elapsed_seconds", 0.0),
                "region": None,
                "generation_origin": {"kind": "full_song_generation"},
                "event_count": len(upstream_events(notes)),
                "source_audio_sha256": source_sha256,
            }

            # Do not publish a candidate if its backing audio changed mid-run.
            if sha256_file(audio_path, cancelled=cancelled) != source_sha256:
                raise ValueError("生成期间项目音频发生变化，候选未保存")
            if cancelled():
                raise InterruptedError("已取消，原有候选已保留")
            state["variants"] = (state.get("variants", []) + [candidate])[-8:]
            state["active"] = len(state["variants"]) - 1
            state["settings"] = prefs.copy()
            state["audio_shift_ms"] = tempo["audio_shift_ms"]
            return {
                "candidate_id": candidate["candidate_id"],
                "report": report,
            }

        return submit(pid, operation, cleanup=cleanup)
