"""HTTP route for cancellable Mapperatorinator V32 candidates."""
from __future__ import annotations

from collections import Counter
from functools import partial
from pathlib import Path
import re
import shutil
import tempfile
import time
import json
import sys

import numpy as np
from flask import jsonify, request

from .arrange import assign_lanes, layout_metrics, parameters
from .arrange_config import (
    DEFAULT_KEY_LAYOUTS,
    KEY_LAYOUT_PROFILES,
    TOROSU_PLAYSTYLES,
)
from .audio_io import decode
from .audio_cache import sha256_file
from .chart_workspace import audio_work
from .candidate_identity import new_candidate_id
from .domain import (
    Note,
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
    sections,
    sanitize_zero_length_model_holds,
    snap_chart_to_grid,
    trim_air_ln_tails,
)
from .separation import Separator
from .torosu_bridge import analyze_osu_text, convert_osu_text, ln_shape_osu_text


DIVISIONS = (3, 4, 6, 8, 12, 16)
TIMING_FIELDS = ("bpm", "offset", "division", "audio_shift_ms")
_ALLOWED_PAYLOAD = {
    "preset",
    "model_variant",
    "generation_strategy",
    "keys",
    "key_layout",
    *TIMING_FIELDS,
    "torosu_density",
    "torosu_keep_jacks",
    "torosu_ln_mode",
    "difficulty",
    "hold_note_ratio",
    "mania_column_temperature",
    "lora_path",
    "timing_confirmed",
    "metadata",
}


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

    from .presets import v32_preset_options

    preset_id = payload.get("preset")
    if preset_id is not None:
        base = v32_preset_options(
            preset_id,
            keys=payload.get("keys", 10),
            model_variant=payload.get("model_variant"),
            difficulty=payload.get("difficulty"),
            hold_note_ratio=payload.get("hold_note_ratio"),
            mania_column_temperature=payload.get("mania_column_temperature"),
            torosu_density=payload.get("torosu_density"),
            torosu_keep_jacks=payload.get("torosu_keep_jacks"),
            torosu_ln_mode=payload.get("torosu_ln_mode"),
        )
        preset_id = base["preset"]
    else:
        base = {
            "model_variant": "v32-mini",
            "generation_strategy": "7k_to_torosu_10k",
            "difficulty": 5.5,
            "hold_note_ratio": 0.55,
            "mania_column_temperature": 1.0,
            "keys": 10,
            "torosu_density": "dense",
            "torosu_keep_jacks": True,
            "torosu_ln_mode": "keep",
            "preset": None,
        }
    model_variant = payload.get("model_variant", base["model_variant"])
    generation_strategy = payload.get(
        "generation_strategy", base["generation_strategy"],
    )
    torosu_density = payload.get("torosu_density", base["torosu_density"])
    keys = payload.get("keys", base["keys"])
    torosu_keep_jacks = payload.get(
        "torosu_keep_jacks", base["torosu_keep_jacks"],
    )
    torosu_ln_mode = payload.get(
        "torosu_ln_mode", base["torosu_ln_mode"],
    )
    if not isinstance(model_variant, str):
        raise ValueError("模型版本无效")
    if not isinstance(generation_strategy, str):
        raise ValueError("生成策略无效")
    if not isinstance(torosu_density, str):
        raise ValueError("Torosu 密度无效")
    difficulty = payload.get("difficulty", base["difficulty"])
    hold_note_ratio = payload.get("hold_note_ratio", base["hold_note_ratio"])
    mania_column_temperature = payload.get(
        "mania_column_temperature", base["mania_column_temperature"],
    )
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
        keys=keys,
        torosu_keep_jacks=torosu_keep_jacks,
        torosu_ln_mode=torosu_ln_mode,
        preset=preset_id,
    )
    key_layout = payload.get("key_layout", DEFAULT_KEY_LAYOUTS[options["keys"]])
    if (not isinstance(key_layout, str)
            or key_layout not in KEY_LAYOUT_PROFILES[options["keys"]]):
        raise ValueError(f"{options['keys']}K 键型无效")
    for field, low, high in (
        ("bpm", 10, 1000),
        ("offset", -600000, 600000),
        ("audio_shift_ms", -100, 100),
    ):
        if field not in payload:
            continue
        value = payload[field]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or not low <= value <= high):
            raise ValueError(f"{field} 超出有效范围")
    if "division" in payload and (
            type(payload["division"]) is not int or payload["division"] not in DIVISIONS):
        raise ValueError("拍内细分必须为 3、4、6、8、12 或 16")
    options["key_layout"] = key_layout
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


def _mixed_audio_rms(samples: np.ndarray, *, sample_rate: int = 22050) -> np.ndarray:
    """Build a bounded acoustic envelope for lightweight portable profiles."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("混音 RMS 输入无效")
    window = max(1, round(sample_rate * 0.04))
    padded_length = ((audio.size + window - 1) // window) * window
    padded = np.pad(audio, (0, padded_length - audio.size))
    frames = padded.reshape(-1, window)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
    return np.repeat(rms, window)[:audio.size].astype(np.float32, copy=False)


def scan_local_loras(resources: str | Path) -> list[dict]:
    """List valid bundled or user-added LoRAs without following directory links."""
    def is_link(path: Path) -> bool:
        junction_check = getattr(path, "is_junction", None)
        return path.is_symlink() or bool(junction_check and junction_check())

    resource_root = Path(resources).resolve()
    roots = []
    if getattr(sys, "frozen", False):
        # The sibling directory is writable and lets users add adapters after
        # unpacking. It wins over a packaged adapter with the same folder name.
        roots.append(resource_root.parent / "assets" / "lora")
    roots.append(resource_root / "assets" / "lora")
    rows = []
    seen_ids = set()
    for root in roots:
        if not root.is_dir() or is_link(root):
            continue
        root = root.resolve()
        for directory in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if (not directory.is_dir()
                    or is_link(directory)
                    or not directory.resolve().is_relative_to(root)):
                continue
            adapter_id = directory.name.casefold()
            if adapter_id in seen_ids:
                continue
            config = directory / "adapter_config.json"
            weights = directory / "adapter_model.safetensors"
            if (is_link(config) or is_link(weights)
                    or not config.is_file() or not weights.is_file()):
                continue
            try:
                if weights.stat().st_size <= 0:
                    continue
                config_data = json.loads(config.read_text(encoding="utf-8"))
                if not isinstance(config_data, dict):
                    continue
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            manifest_path = directory / "lora-manifest.json"
            manifest = {}
            if manifest_path.is_file() and not is_link(manifest_path):
                try:
                    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(raw_manifest, dict):
                        manifest = raw_manifest
                except (OSError, UnicodeError, json.JSONDecodeError):
                    pass
            model = manifest.get("base_model")
            variant = model.get("variant") if isinstance(model, dict) else None
            rows.append({
                "id": directory.name,
                "label": str(manifest.get("name") or directory.name),
                "path": str(directory.resolve()),
                "base_model": variant if isinstance(variant, str) else None,
                "location": "portable" if root == roots[0].resolve() and getattr(sys, "frozen", False) else "bundle",
            })
            seen_ids.add(adapter_id)
    return rows


def _repair_native_collisions(
    osu_text: str,
    keys: int,
    *,
    key_layout: str | None = None,
    force_reassign: bool = False,
    cancelled=lambda: False,
) -> tuple[str, dict]:
    if cancelled():
        raise InterruptedError("已取消，原有候选已保留")
    parsed = parse_osu(osu_text)
    before = validate(parsed["notes"], keys)
    if before["valid"] and not force_reassign:
        return osu_text, {
            "reassigned": False,
            "objects_before": len(parsed["notes"]),
            "objects_after": len(parsed["notes"]),
            "issues_before": [],
        }
    source_objects = Counter(
        tuple(field.strip() for field in note.raw.split(",")[1:])
        for note in parsed["notes"]
    )
    notes = assign_lanes(
        parsed["notes"],
        {
            "keys": keys,
            "key_layout": key_layout or DEFAULT_KEY_LAYOUTS[keys],
            "layout_engine": "beam",
        },
        timing_points=parsed["timing"],
        cancelled=cancelled,
    )
    if cancelled():
        raise InterruptedError("已取消，原有候选已保留")
    output = export_osu(
        notes,
        keys,
        None,
        original=osu_text,
        version=parsed["metadata"].get("Version", "V32 native"),
    )
    result = parse_osu(output)
    after = validate(result["notes"], keys)
    output_objects = Counter(
        tuple(field.strip() for field in note.raw.split(",")[1:])
        for note in result["notes"]
    )
    if not after["valid"] or output_objects != source_objects:
        raise ValueError(
            "native 冲突重排未保留全部物件字段："
            + (after["issues"][0] if after["issues"] else "物件数量/字段发生变化")
        )
    return output, {
        "reassigned": True,
        "objects_before": len(parsed["notes"]),
        "objects_after": len(result["notes"]),
        "issues_before": before["issues"],
        "issues_after": after["issues"],
    }


def build_v32_operation(
    pid,
    payload,
    *,
    projects,
    current_audio,
    engine,
    resources,
    separator,
    requested_tempo,
):
    """Create the single-project V32 operation used by both single and batch routes."""
    if not isinstance(payload, dict):
        raise ValueError("V32 请求必须为 JSON 对象")
    options, timing_confirmed = _request_options(payload, engine)
    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, dict) or set(raw_metadata) - {"creator", "version_suffix"}:
        raise ValueError("V32 谱面元数据必须只包含 creator 与 version_suffix")
    creator = raw_metadata.get("creator", "ManiaLab")
    version_suffix = raw_metadata.get("version_suffix", "")
    if (not isinstance(creator, str) or not creator.strip()
            or len(creator.strip()) > 64 or any(ord(char) < 32 for char in creator)):
        raise ValueError("统一谱师名必须为 1–64 个可见字符")
    if (not isinstance(version_suffix, str) or len(version_suffix) > 48
            or any(ord(char) < 32 for char in version_suffix)):
        raise ValueError("预设版本后缀不能超过 48 个可见字符")
    metadata = {
        "creator": creator.strip(),
        "version_suffix": version_suffix.strip(),
    }
    if not engine.status().get("available"):
        raise ValueError("Mapperatorinator V32 推理环境未就绪")
    resources = Path(resources)
    run = {}

    def cleanup(_committed=False):
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

        target_keys = options["keys"]
        model_keys = options["model_keycount"]
        strategy = options["generation_strategy"]
        key_layout = options["key_layout"]
        torosu_key_layout = (
            key_layout if key_layout in TOROSU_PLAYSTYLES
            else DEFAULT_KEY_LAYOUTS[target_keys]
        )
        prefs = parameters({
            **(state.get("settings") if isinstance(state.get("settings"), dict) else {}),
            "keys": target_keys,
            "key_layout": key_layout,
            "layout_engine": (
                "torosu" if key_layout in TOROSU_PLAYSTYLES else "beam"
            ),
            "torosu_keep_jacks": options["torosu_keep_jacks"],
            "torosu_density": options["torosu_density"],
        })
        project_path = projects.path(pid)
        work = audio_work(project_path, state)
        work.mkdir(parents=True, exist_ok=True)
        run_path = Path(tempfile.mkdtemp(prefix="v32-", dir=work))
        run["path"] = run_path
        preview = run_path / "preview.wav"
        decoded_samples = decode(audio_path, run_path, cancelled=cancelled)
        if cancelled():
            raise InterruptedError("已取消，原有候选已保留")

        template = resources / "vendor" / "mapperatorinator" / "template.osu"
        # Mapperatorinator's slider parser requires a HitObjects section and at
        # least one object even when only TIMING is supplied as context.
        placeholder_time = max(0, round(tempo["offset"]))
        reference = export_osu(
            [Note(
                placeholder_time,
                placeholder_time,
                0,
                id="timing-context-placeholder",
            )],
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
        if separator.status().get("available"):
            progress(3, "准备六声部 RMS 能量校验")
            energy, separation_report = _separation_energy(
                separator,
                audio_path,
                work / ("v32-separation-" + source_sha256[:16]),
                source_sha256,
                progress,
                cancelled,
            )
            energy_source = "six_stem_rms"
        else:
            progress(3, "便携轻量模式：从原混音计算 RMS 能量包络")
            energy = _mixed_audio_rms(decoded_samples)
            separation_report = {
                "audio_sha256": source_sha256,
                "model": None,
                "model_sha256": None,
                "profile": "mixed_rms_fallback",
            }
            energy_source = "mixed_audio_rms"
        if cancelled():
            raise InterruptedError("已取消，原有候选已保留")

        inference = engine.generate(
            model_variant=options["model_variant"],
            audio_path=preview,
            reference_chart=reference_path,
            output_dir=inference_path,
            keycount=model_keys,
            target_keycount=target_keys,
            generation_strategy=strategy,
            torosu_density=options["torosu_density"],
            torosu_keep_jacks=options["torosu_keep_jacks"],
            torosu_ln_mode=options["torosu_ln_mode"],
            preset=options["preset"],
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
        if options["torosu_ln_mode"] != "keep":
            progress(94, f"应用 Torosu {options['torosu_ln_mode']} LN 形状")
            trimmed = ln_shape_osu_text(trimmed, options["torosu_ln_mode"])
        if cancelled():
            raise InterruptedError("已取消，原有候选已保留")

        # Preserve the old native 10K route's no-density default unless a named
        # preset or the generalized 7K->target route was explicitly requested.
        torosu_density = options["torosu_density"]
        if (options["preset"] is None
                and strategy in ("native_10k", "native")):
            torosu_density = "none"
        repaired_chart, repair_report = _repair_native_collisions(
            trimmed, target_keys, cancelled=cancelled,
        )
        converted = convert_osu_text(
            repaired_chart,
            target_keys,
            playstyle=TOROSU_PLAYSTYLES[torosu_key_layout],
            density=torosu_density,
            keep_jacks=options["torosu_keep_jacks"],
            strict=True,
            cancelled=cancelled,
        )
        layout_repair_report = None
        if key_layout != torosu_key_layout:
            converted, layout_repair_report = _repair_native_collisions(
                converted,
                target_keys,
                key_layout=key_layout,
                force_reassign=True,
                cancelled=cancelled,
            )
        if cancelled():
            raise InterruptedError("已取消，原有候选已保留")

        converted_notes = parse_osu(converted)["notes"]
        output_text = export_osu(
            converted_notes,
            target_keys,
            None,
            original=reference,
            version=(
                f"V32 {options['model_variant']} · {strategy}"
                + (f" · {options['preset']}" if options["preset"] else "")
            ),
        )
        notes = parse_osu(output_text)["notes"]
        report = validate(notes, target_keys)
        if not report["valid"]:
            raise ValueError("V32/Torosu 输出未通过结构校验：" + report["issues"][0])
        beat_ms = 60000 / tempo["bpm"]
        report["grid"] = grid_report(notes, tempo)
        if report["grid"]["off_grid"]:
            raise ValueError("V32/Torosu 输出存在未落拍物件，未保存候选")
        report["layout"] = layout_metrics(
            notes, target_keys, beat_ms, key_layout=key_layout,
        )
        report["review"] = review_chart(
            notes, target_keys, beat_ms, key_layout=key_layout,
        )
        report["ln_patterns"] = analyze_ln_patterns(
            notes, target_keys, beat_ms, key_layout=key_layout,
        )
        torosu_report = {
            **repair_report,
            **({"target_layout": layout_repair_report} if layout_repair_report else {}),
            "density": torosu_density,
            "keep_jacks": options["torosu_keep_jacks"],
            "ln_mode": options["torosu_ln_mode"],
            "target_keys": target_keys,
            "playstyle": TOROSU_PLAYSTYLES[torosu_key_layout],
            "key_layout": key_layout,
        }
        torosu_report.update(analyze_osu_text(
            output_text,
            keys=target_keys,
            playstyle=TOROSU_PLAYSTYLES[torosu_key_layout],
        ))
        torosu_report["playstyle"] = TOROSU_PLAYSTYLES[torosu_key_layout]
        torosu_report["key_layout"] = key_layout
        report["v32"] = {
            "model_variant": inference["model_variant"],
            "model_revision": inference["model_revision"],
            "generation_strategy": strategy,
            "preset": options["preset"],
            "model_keycount": model_keys,
            "target_keys": target_keys,
            "device": inference["device"],
            "fast_decoder_loop_active": inference["fast_decoder_loop_active"],
            "lora_loaded": inference["lora_loaded"],
            "lora_path": inference["lora_path"],
            "elapsed_seconds": inference.get("elapsed_seconds"),
            "peak_rss_bytes": inference.get("peak_rss_bytes"),
            "hold_repair": hold_repair,
            "grid_snap": grid_snap,
            "energy_trim": trim_report,
            "energy_source": energy_source,
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
                    f"由 Mapperatorinator {options['model_variant']} 生成并经 Torosu "
                    f"{target_keys}K 校验；分轨 RMS 只用于裁切持续近静音的 LN 尾端。"
                ),
                "layout_origin": "mapperatorinator_v32",
                "generation_strategy": strategy,
                "preset": options["preset"],
                "model_revision": inference["model_revision"],
                "torosu": torosu_report,
            },
            "preset": options["preset"],
            "metadata": metadata,
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
        if sha256_file(audio_path, cancelled=cancelled) != source_sha256:
            raise ValueError("生成期间项目音频发生变化，候选未保存")
        if cancelled():
            raise InterruptedError("已取消，原有候选已保留")
        state["variants"] = (state.get("variants", []) + [candidate])[-8:]
        state["active"] = len(state["variants"]) - 1
        state["settings"] = prefs.copy()
        state["audio_shift_ms"] = tempo["audio_shift_ms"]
        return {"candidate_id": candidate["candidate_id"], "report": report}

    operation.options = options
    operation.cleanup = cleanup
    return operation


def register_v32_routes(
    app,
    *,
    engine,
    projects,
    submit,
    current_audio,
    requested_tempo,
    resources,
    separator=None,
    task_service=None,
):
    resources = Path(resources)
    separator = separator or Separator(resources, profile="six")
    task_service = task_service or app.extensions.get("task_service")
    app.extensions["v32_separator"] = separator
    app.extensions["v32_operation_factory"] = partial(
        build_v32_operation,
        projects=projects,
        current_audio=current_audio,
        engine=engine,
        resources=resources,
        separator=separator,
        requested_tempo=requested_tempo,
    )

    @app.get("/api/loras")
    def local_loras():
        return jsonify(items=scan_local_loras(resources))

    @app.get("/api/v32/status")
    def v32_status():
        status = engine.status()
        status["separation_available"] = bool(separator.status().get("available"))
        return jsonify(status)

    def submit_v32_install():
        status = engine.status()
        if not status.get("install_available"):
            raise ValueError("当前发行包不含 V32 自动安装组件；请配置固定 Mapperatorinator 源码与运行环境")
        if task_service is None:
            raise ValueError("当前服务不支持全局后台任务")

        def operation(_state, progress, cancelled):
            result = engine.install_runtime(progress=progress, cancelled=cancelled)
            progress(100, "V32 推理环境已就绪")
            return result

        return jsonify(task_service.submit_global(
            operation,
            identity={"kind": "v32-runtime-install"},
        ))

    @app.post("/api/v32/install")
    def install_v32_global():
        return submit_v32_install()

    @app.get("/api/v32/install/<job_id>")
    def get_v32_install(job_id):
        if task_service is None:
            return jsonify(error="当前服务不支持全局后台任务"), 404
        job = task_service.get(job_id)
        if (not job or not job.get("global")
                or job.get("identity") != {"kind": "v32-runtime-install"}):
            return jsonify(error="V32 安装任务不存在"), 404
        return jsonify(job)

    @app.post("/api/projects/<pid>/v32-install")
    def install_v32_for_project(pid):
        return submit_v32_install()

    @app.post("/api/projects/<pid>/v32-generate")
    def generate_v32(pid):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("V32 请求必须为 JSON 对象")
        operation = app.extensions["v32_operation_factory"](pid, payload)
        return submit(pid, operation, cleanup=operation.cleanup)
