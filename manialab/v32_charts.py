"""V32 chart cleanup, timing alignment, and validation."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math

import numpy as np

from .domain import export_osu, parse_osu, sections, validate


def sanitize_zero_length_model_holds(osu_text: str) -> tuple[str, dict]:
    """Keep an invalid V32 hold's attack as a tap instead of dropping it."""
    content = sections(osu_text)
    lines = content.get("HitObjects")
    if lines is None:
        raise ValueError("Mapperatorinator 输出缺少 HitObjects")
    downgraded = []
    repaired = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split(",")
        if len(fields) < 6:
            repaired.append(line)
            continue
        try:
            kind = int(fields[3])
            start = int(fields[2])
            end = int(fields[5].split(":", 1)[0])
        except ValueError:
            repaired.append(line)
            continue
        if not kind & 128 or end > start:
            repaired.append(line)
            continue
        fields[3] = str((kind & ~128) | 1)
        tail = fields[5].split(":")
        sample = ":".join(tail[1:]) if len(tail) > 1 else "0:0:0:0:"
        fields[5] = sample or "0:0:0:0:"
        repaired.append(",".join(fields))
        downgraded.append({"line": line_number, "start_ms": start, "invalid_end_ms": end})
    content["HitObjects"] = repaired
    result = _format_sections(content)
    parse_osu(result)
    return result, {
        "downgraded": len(downgraded),
        "preserved_attacks": len(downgraded),
        "objects_before": len(lines),
        "objects_after": len(repaired),
        "details": downgraded,
    }


def trim_air_ln_tails(
    osu_text: str,
    audio: np.ndarray,
    sample_rate: int,
    *,
    minimum_silence_ms: int = 150,
    relative_threshold: float = 0.01,
    window_ms: int = 25,
    hop_ms: int = 10,
    grid_bpm: float | None = None,
    grid_offset_ms: float = 0,
    grid_division: int = 16,
) -> tuple[str, dict]:
    """Trim only LN tails followed by a sustained >150ms near-silent region."""
    if minimum_silence_ms < 150:
        raise ValueError("空气长条门槛不得低于 150ms")
    wave = np.asarray(audio, dtype=np.float64)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if wave.ndim != 1 or sample_rate <= 0 or not np.isfinite(wave).all():
        raise ValueError("LN 能量包络输入无效")
    if not 0 < relative_threshold < 1:
        raise ValueError("静音相对阈值无效")
    parsed = parse_osu(osu_text)
    window = max(1, round(sample_rate * window_ms / 1000))
    hop = max(1, round(sample_rate * hop_ms / 1000))
    starts = np.arange(0, max(1, len(wave)), hop, dtype=np.int64)
    ends = np.minimum(len(wave), starts + window)
    prefix = np.concatenate(([0.0], np.cumsum(np.square(wave), dtype=np.float64)))
    envelope = np.sqrt(
        np.maximum(0, (prefix[ends] - prefix[starts]) / np.maximum(1, ends - starts))
    )
    threshold = max(1e-5, float(np.percentile(envelope, 95)) * relative_threshold)
    period = 60000 / grid_bpm / grid_division if grid_bpm and grid_bpm > 0 else None

    replacements: dict[str, str] = {}
    trimmed = []
    for note in parsed["notes"]:
        if not note.hold or note.end <= note.start:
            continue
        left = max(0, int(note.start * sample_rate / 1000))
        right = min(len(wave), int(note.end * sample_rate / 1000))
        if right <= left:
            continue
        mask = (starts < right) & (ends > left) & (envelope >= threshold)
        active = np.flatnonzero(mask)
        if not len(active):
            continue
        active_end = round(ends[active[-1]] * 1000 / sample_rate)
        silence = note.end - active_end
        if silence < minimum_silence_ms:
            continue
        new_end = active_end
        if period:
            new_end = round((new_end - grid_offset_ms) / period) * period + grid_offset_ms
            new_end = int(round(new_end))
        if new_end <= note.start + 20 or new_end >= note.end:
            continue
        fields = note.raw.split(",")
        tail = fields[5].split(":")
        tail[0] = str(new_end)
        fields[5] = ":".join(tail)
        replacements.setdefault(note.raw, ",".join(fields))
        trimmed.append({
            "start_ms": note.start,
            "old_end_ms": note.end,
            "new_end_ms": new_end,
            "quiet_tail_ms": silence,
        })

    if not replacements:
        return osu_text, {
            "examined_lns": sum(note.hold for note in parsed["notes"]),
            "trimmed_lns": 0,
            "threshold_rms": threshold,
            "trims": [],
        }
    counts = Counter()
    output_lines = []
    in_objects = False
    for line in osu_text.splitlines():
        if line.strip() == "[HitObjects]":
            in_objects = True
            output_lines.append(line)
            continue
        if line.startswith("[") and line.endswith("]") and line.strip() != "[HitObjects]":
            in_objects = False
        if in_objects and line in replacements:
            used = counts[line]
            counts[line] += 1
            if used == 0:
                line = replacements[line]
        output_lines.append(line)
    result = "\n".join(output_lines) + ("\n" if osu_text.endswith("\n") else "")
    verify = parse_osu(result)
    if len(verify["notes"]) != len(parsed["notes"]):
        raise ValueError("LN 能量裁切意外改变了物件数")
    return result, {
        "examined_lns": sum(note.hold for note in parsed["notes"]),
        "trimmed_lns": len(trimmed),
        "threshold_rms": threshold,
        "trims": trimmed,
    }


def validate_map_grid(
    osu_text: str,
    *,
    bpm: float,
    offset_ms: float,
    division: int = 48,
) -> dict:
    """Require all generated starts and LN releases to land on the clip clock."""
    from .timing import grid_report

    parsed = parse_osu(osu_text)
    report = grid_report(
        parsed["notes"],
        {"bpm": float(bpm), "offset": float(offset_ms), "division": division},
    )
    if report["off_grid"]:
        raise ValueError(
            f"{parsed['keys']}K 谱存在 {report['off_grid']} 个离线起点/LN 尾端，"
            f"最大偏差 {report['max_error_ms']}ms"
        )
    return report


def snap_chart_to_grid(
    osu_text: str,
    *,
    bpm: float,
    offset_ms: float,
    division: int = 48,
) -> tuple[str, dict]:
    """Snap generated starts and LN releases to the confirmed timing clock."""
    parsed = parse_osu(osu_text)
    tick = 60000 / float(bpm) / division
    if not math.isfinite(tick) or tick <= 0:
        raise ValueError("拍格周期无效")
    notes = []
    moved = []
    for note in parsed["notes"]:
        start = round(offset_ms + round((note.start - offset_ms) / tick) * tick)
        end = (
            round(offset_ms + round((note.end - offset_ms) / tick) * tick)
            if note.hold else start
        )
        if note.hold and end <= start:
            end = start + max(1, round(tick))
        if start != note.start or end != note.end:
            moved.append({
                "old_start_ms": note.start,
                "new_start_ms": start,
                "old_end_ms": note.end,
                "new_end_ms": end,
            })
        fields = note.raw.split(",")
        fields[2] = str(start)
        if note.hold:
            tail = fields[5].split(":")
            tail[0] = str(end)
            fields[5] = ":".join(tail)
        notes.append(replace(note, start=start, end=end, raw=",".join(fields)))
    content = sections(osu_text)
    content["HitObjects"] = [
        note.raw for note in sorted(notes, key=lambda item: (item.start, item.lane))
    ]
    result = _format_sections(content)
    parse_osu(result)
    return result, {
        "division": division,
        "tick_ms": tick,
        "adjusted_objects": len(moved),
        "max_endpoint_shift_ms": max(
            (
                max(abs(row["old_start_ms"] - row["new_start_ms"]),
                    abs(row["old_end_ms"] - row["new_end_ms"]))
                for row in moved
            ),
            default=0,
        ),
        "adjustments": moved,
    }


def repair_native_10k_collisions(osu_text: str) -> tuple[str, dict]:
    """Reassign only lanes when native 10K has same-track collisions."""
    from .arrange import assign_lanes

    parsed = parse_osu(osu_text)
    before = validate(parsed["notes"], 10)
    if before["valid"]:
        return osu_text, {
            "reassigned": False,
            "objects_before": len(parsed["notes"]),
            "objects_after": len(parsed["notes"]),
            "issues_before": [],
        }
    original_payload = Counter(
        tuple(field.strip() for field in note.raw.split(",")[1:])
        for note in parsed["notes"]
    )
    arranged = assign_lanes(
        parsed["notes"],
        {"keys": 10, "layout_engine": "beam"},
        timing_points=parsed["timing"],
    )
    result = export_osu(
        arranged,
        10,
        None,
        original=osu_text,
        version=parsed["metadata"].get("Version", "V32 native 10K"),
    )
    verified = parse_osu(result)
    after = validate(verified["notes"], 10)
    actual_payload = Counter(
        tuple(field.strip() for field in note.raw.split(",")[1:])
        for note in verified["notes"]
    )
    if not after["valid"] or actual_payload != original_payload:
        raise ValueError(
            "native 10K 冲突重排未保留全部物件字段："
            + (after["issues"][0] if after["issues"] else "物件数量/字段发生变化")
        )
    return result, {
        "reassigned": True,
        "objects_before": len(parsed["notes"]),
        "objects_after": len(verified["notes"]),
        "issues_before": before["issues"],
        "issues_after": after["issues"],
    }


def shift_reference_timing(
    source_text: str,
    *,
    segment_start_ms: int,
    segment_duration_ms: int | None = None,
    bpm: float | None = None,
    offset_ms: float | None = None,
    audio_filename: str | None = None,
) -> tuple[str, float, float]:
    """Build a clip-relative reference map and return its base BPM/offset."""
    content = sections(source_text)
    timing = content.get("TimingPoints", [])
    if bpm is not None:
        if not math.isfinite(bpm) or not 10 <= bpm <= 1000:
            raise ValueError("BPM 必须在 10–1000 之间")
        source_offset = float(offset_ms if offset_ms is not None else 0)
        relative_offset = source_offset - segment_start_ms
        timing = [f"{relative_offset:.3f},{60000 / bpm:.8f},4,2,0,70,1,0"]
    else:
        redlines = []
        for line in timing:
            fields = line.split(",")
            if len(fields) < 2:
                continue
            try:
                beat = float(fields[1])
                position = float(fields[0])
            except ValueError:
                continue
            if beat > 0:
                redlines.append((position, beat))
        if not redlines:
            raise ValueError("参考谱面没有有效的红线 BPM")
        position, beat = redlines[0]
        bpm = 60000 / beat
        source_offset = position if offset_ms is None else float(offset_ms)
        relative_offset = source_offset - segment_start_ms
        timing = [
            ",".join([f"{float(line.split(',')[0]) - segment_start_ms:.3f}",
                      *line.split(",")[1:]])
            if line.strip() and "," in line else line
            for line in timing
        ]
    content["TimingPoints"] = timing
    editor = content.get("Editor", [])
    for line in editor:
        if line.split(":", 1)[0].strip().casefold() != "bookmarks":
            continue
        try:
            values = [
                round(float(value.strip()) - segment_start_ms)
                for value in line.split(":", 1)[1].split(",")
                if value.strip()
            ]
        except ValueError:
            continue
        values = [
            value for value in values
            if value >= 0 and (segment_duration_ms is None or value < segment_duration_ms)
        ]
        content["Editor"] = _set_field(editor, "Bookmarks", ",".join(map(str, values)))
        break
    general = content.get("General", [])
    preview = next(
        (line.split(":", 1)[1].strip() for line in general
         if line.split(":", 1)[0].strip().casefold() == "previewtime"),
        None,
    )
    if preview is not None:
        try:
            preview_ms = round(float(preview) - segment_start_ms)
        except ValueError:
            preview_ms = 0
        if preview_ms < 0 or (
            segment_duration_ms is not None and preview_ms >= segment_duration_ms
        ):
            preview_ms = 0
        content["General"] = _set_field(general, "PreviewTime", str(preview_ms))
    if audio_filename is not None:
        general = content.setdefault("General", [])
        content["General"] = _set_field(general, "AudioFilename", audio_filename)
    return _format_sections(content), float(bpm), float(relative_offset)


def _format_sections(content: dict[str, list[str]]) -> str:
    return "\n".join(
        ["\n".join(content.get("", ["osu file format v14"]))]
        + [f"[{name}]\n" + "\n".join(body) for name, body in content.items() if name]
    ) + "\n"


def _set_field(lines: list[str], key: str, value: str) -> list[str]:
    result = []
    found = False
    for line in lines:
        if line.split(":", 1)[0].strip().casefold() == key.casefold():
            if not found:
                result.append(f"{key}:{value}")
                found = True
        else:
            result.append(line)
    if not found:
        result.append(f"{key}:{value}")
    return result
