"""Summarize LN tail changes by frozen structural segment and voice."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LARGE_SHIFT_MS = 30
DEFAULT_PLAN = Path("test-data/frozen-fullsong-evidence-20261117/plan.json")
DEFAULT_REPLAY = Path("test-data/frozen-fullsong-evidence-20261117/report.json")
DEFAULT_OUTPUT = Path("test-data/frozen-ln-segment-report-20261118")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _rooted(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _finite_ms(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite millisecond value")
    return float(value)


def _case_index(rows: object, label: str) -> dict[str, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must be a nonempty list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} rows must be objects")
        case_id = row.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
            raise ValueError(f"{label} contains an invalid case ID")
        if case_id in result:
            raise ValueError(f"{label} contains duplicate case ID: {case_id}")
        result[case_id] = row
    return result


def _sections(summary: dict, case_id: str) -> list[dict]:
    rows = summary.get("sections")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Frozen replay has no structural sections: {case_id}")
    result = []
    seen_indices = set()
    previous_end = None
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid structural section in {case_id}")
        index = row.get("index")
        if type(index) is not int or index < 0 or index in seen_indices:
            raise ValueError(f"Invalid or duplicate section index in {case_id}")
        start = _finite_ms(row.get("start_ms"), f"{case_id}.section.start_ms")
        end = _finite_ms(row.get("end_ms"), f"{case_id}.section.end_ms")
        if start is None or end is None or end <= start:
            raise ValueError(f"Invalid structural section interval in {case_id}")
        if previous_end is not None and start < previous_end:
            raise ValueError(f"Overlapping or unordered structural sections in {case_id}")
        seen_indices.add(index)
        previous_end = end
        result.append({"index": index, "start_ms": start, "end_ms": end})
    return result


def _release_reason(row: object) -> str:
    if not isinstance(row, dict):
        return "unknown"
    value = row.get("release_reason")
    if value is None or value == "":
        return "unknown"
    if not isinstance(value, str):
        raise ValueError("release_reason must be a string when present")
    return value


def _direction(delta: float) -> str:
    if delta > 0:
        return "later"
    if delta < 0:
        return "earlier"
    return "unchanged"


def _movement_summary(values: list[float], missing: int) -> dict:
    directions = Counter(_direction(value) for value in values)
    return {
        "measured_count": len(values),
        "missing_count": missing,
        "later_count": directions["later"],
        "earlier_count": directions["earlier"],
        "unchanged_count": directions["unchanged"],
        "abs_gt_30ms_count": sum(abs(value) > LARGE_SHIFT_MS for value in values),
        "max_abs_ms": max((abs(value) for value in values), default=0.0),
    }


def _new_group() -> dict:
    return {
        "ln_count": 0,
        "release_reasons": {
            "detected": Counter(),
            "expressed": Counter(),
            "final": Counter(),
        },
        "source_tail_deltas_ms": [],
        "source_tail_missing_count": 0,
        "quantization_deltas_ms": [],
        "quantization_missing_count": 0,
        "missing_source_snapshot_count": 0,
        "missing_detected_end_count": 0,
        "missing_expressed_end_count": 0,
        "large_movement_candidates": [],
    }


def _snapshot_for(snapshots: object, note_id: str) -> dict | None:
    if not isinstance(snapshots, dict):
        return None
    snapshot = snapshots.get(note_id)
    return snapshot if isinstance(snapshot, dict) else None


def _audio_start(note: dict, snapshot: dict | None, label: str) -> tuple[float | None, str | None]:
    if note.get("audio_start") is not None:
        return _finite_ms(note["audio_start"], f"{label}.audio_start"), None
    detected = snapshot.get("detected") if snapshot else None
    if isinstance(detected, dict) and detected.get("start") is not None:
        return _finite_ms(detected["start"], f"{label}.source_evidence.detected.start"), None
    return None, "missing_audio_start"


def _section_for(audio_start_ms: float, sections: list[dict]) -> dict | None:
    for section in sections:
        if section["start_ms"] <= audio_start_ms < section["end_ms"]:
            return section
    return None


def _add_candidate(
    group: dict,
    note: dict,
    audio_start_ms: float | None,
    source_delta: float | None,
    quantization_delta: float | None,
    detected_reason: str,
    expressed_reason: str,
    final_reason: str,
) -> bool:
    movements = {}
    if source_delta is not None and abs(source_delta) > LARGE_SHIFT_MS:
        movements["source_tail"] = {
            "direction": _direction(source_delta),
            "delta_ms": source_delta,
        }
    if quantization_delta is not None and abs(quantization_delta) > LARGE_SHIFT_MS:
        movements["quantization"] = {
            "direction": _direction(quantization_delta),
            "delta_ms": quantization_delta,
        }
    if not movements:
        return False
    group["large_movement_candidates"].append({
        "id": note["id"],
        "pitch": note.get("pitch"),
        "audio_start_ms": audio_start_ms,
        "movements": movements,
        "release_reasons": {
            "detected": detected_reason,
            "expressed": expressed_reason,
            "final": final_reason,
        },
    })
    return True


def _summarize_case(plan_case: dict, replay_case: dict, variant: dict) -> dict:
    case_id = plan_case["id"]
    sections = _sections(replay_case, case_id)
    notes = variant.get("notes")
    if not isinstance(notes, list):
        raise ValueError(f"Frozen variant has no notes: {case_id}")
    snapshots = variant.get("source_evidence")
    note_ids = set()
    voices = set()
    parsed = []
    total_ln = 0

    for position, note in enumerate(notes):
        if not isinstance(note, dict):
            raise ValueError(f"Invalid note in {case_id}")
        note_id = note.get("id")
        if not isinstance(note_id, str) or not note_id or note_id in note_ids:
            raise ValueError(f"Missing or duplicate note ID in {case_id}")
        note_ids.add(note_id)
        voice = note.get("voice", "unknown")
        if not isinstance(voice, str) or not voice:
            raise ValueError(f"Invalid voice for {case_id}:{note_id}")
        pitch = note.get("pitch")
        if pitch is not None and type(pitch) is not int:
            raise ValueError(f"Invalid pitch for {case_id}:{note_id}")
        voices.add(voice)
        start = _finite_ms(note.get("start"), f"{case_id}.{note_id}.start")
        end = _finite_ms(note.get("end"), f"{case_id}.{note_id}.end")
        if start is None or end is None or end < start:
            raise ValueError(f"Invalid note interval for {case_id}:{note_id}")
        is_ln = end > start
        if is_ln:
            total_ln += 1
        snapshot = _snapshot_for(snapshots, note_id)
        audio_start_ms, audio_start_source = _audio_start(note, snapshot, f"{case_id}.{note_id}")
        section = _section_for(audio_start_ms, sections) if audio_start_ms is not None else None
        detected = snapshot.get("detected") if snapshot and isinstance(snapshot.get("detected"), dict) else None
        expressed = snapshot.get("expressed") if snapshot and isinstance(snapshot.get("expressed"), dict) else None
        detected_reason = _release_reason(detected)
        expressed_reason = _release_reason(expressed)
        final_reason = _release_reason(note)
        detected_end = _finite_ms(
            detected.get("end") if detected else None,
            f"{case_id}.{note_id}.source_evidence.detected.end",
        )
        expressed_end = _finite_ms(
            expressed.get("end") if expressed else None,
            f"{case_id}.{note_id}.source_evidence.expressed.end",
        )
        evidence_end = _finite_ms(note.get("evidence_end"), f"{case_id}.{note_id}.evidence_end")
        source_delta = (
            expressed_end - detected_end
            if detected_end is not None and expressed_end is not None
            else None
        )
        quantization_delta = end - evidence_end if evidence_end is not None else None
        parsed.append({
            "note": note,
            "is_ln": is_ln,
            "voice": voice,
            "snapshot": snapshot,
            "section": section,
            "audio_start_ms": audio_start_ms,
            "audio_start_source": audio_start_source,
            "detected": detected,
            "expressed": expressed,
            "detected_reason": detected_reason,
            "expressed_reason": expressed_reason,
            "final_reason": final_reason,
            "detected_end": detected_end,
            "expressed_end": expressed_end,
            "source_delta": source_delta,
            "quantization_delta": quantization_delta,
        })

    voices = sorted(voices or {"unknown"})
    group_map = {
        (section["index"], voice): _new_group()
        for section in sections
        for voice in voices
    }
    unassigned_ln = []

    for row in parsed:
        if not row["is_ln"]:
            continue
        note = row["note"]
        section = row["section"]
        if section is None:
            reason = row["audio_start_source"] or "outside_structural_segments"
            unassigned_ln.append({
                "id": note["id"],
                "voice": row["voice"],
                "pitch": note.get("pitch"),
                "audio_start_ms": row["audio_start_ms"],
                "reason": reason,
            })
            continue

        group = group_map[(section["index"], row["voice"])]
        group["ln_count"] += 1
        for name, reason in (
            ("detected", row["detected_reason"]),
            ("expressed", row["expressed_reason"]),
            ("final", row["final_reason"]),
        ):
            group["release_reasons"][name][reason] += 1
        if row["snapshot"] is None:
            group["missing_source_snapshot_count"] += 1
        if row["detected_end"] is None:
            group["missing_detected_end_count"] += 1
        if row["expressed_end"] is None:
            group["missing_expressed_end_count"] += 1
        if row["source_delta"] is None:
            group["source_tail_missing_count"] += 1
        else:
            group["source_tail_deltas_ms"].append(row["source_delta"])
        if row["quantization_delta"] is None:
            group["quantization_missing_count"] += 1
        else:
            group["quantization_deltas_ms"].append(row["quantization_delta"])
        _add_candidate(
            group,
            note,
            row["audio_start_ms"],
            row["source_delta"],
            row["quantization_delta"],
            row["detected_reason"],
            row["expressed_reason"],
            row["final_reason"],
        )

    large_unassigned = []
    for row in parsed:
        if not row["is_ln"] or row["section"] is not None:
            continue
        candidate = {"large_movement_candidates": []}
        _add_candidate(
            candidate,
            row["note"],
            row["audio_start_ms"],
            row["source_delta"],
            row["quantization_delta"],
            row["detected_reason"],
            row["expressed_reason"],
            row["final_reason"],
        )
        if candidate["large_movement_candidates"]:
            large_unassigned.extend(candidate["large_movement_candidates"])

    section_rows = []
    for section in sections:
        voice_rows = []
        for voice in voices:
            group = group_map[(section["index"], voice)]
            voice_rows.append({
                "voice": voice,
                "ln_count": group["ln_count"],
                "release_reasons": {
                    name: dict(sorted(counter.items()))
                    for name, counter in group["release_reasons"].items()
                },
                "source_tail_movement_audio_clock": _movement_summary(
                    group["source_tail_deltas_ms"], group["source_tail_missing_count"]
                ),
                "quantization_movement_chart_clock": _movement_summary(
                    group["quantization_deltas_ms"], group["quantization_missing_count"]
                ),
                "missing_source_snapshot_count": group["missing_source_snapshot_count"],
                "missing_detected_end_count": group["missing_detected_end_count"],
                "missing_expressed_end_count": group["missing_expressed_end_count"],
                "large_movement_candidates": sorted(
                    group["large_movement_candidates"], key=lambda row: row["id"]
                ),
            })
        section_rows.append({
            **section,
            "voices": voice_rows,
        })

    assigned_ln = sum(group["ln_count"] for group in group_map.values())
    return {
        "id": case_id,
        "keys": plan_case.get("settings", {}).get("keys"),
        "ln_count": total_ln,
        "assigned_ln_count": assigned_ln,
        "unassigned_ln_count": len(unassigned_ln),
        "sections": section_rows,
        "unassigned_ln": sorted(unassigned_ln, key=lambda row: row["id"]),
        "unassigned_large_movement_candidates": sorted(large_unassigned, key=lambda row: row["id"]),
    }


def build_report(
    plan: dict,
    replay: dict,
    variants: dict[str, dict],
    plan_sha256: str,
    replay_sha256: str,
) -> dict:
    if plan.get("schema") != "manialab-frozen-generation-v1":
        raise ValueError("Expected a frozen-generation-v1 plan")
    if replay.get("schema") != "manialab-frozen-generation-results-v1":
        raise ValueError("Expected a frozen-generation-results-v1 replay")
    if replay.get("plan_sha256") != plan_sha256:
        raise ValueError("Replay report does not match the frozen plan hash")
    plan_cases = _case_index(plan.get("cases"), "Plan cases")
    replay_cases = _case_index(replay.get("cases"), "Replay cases")
    if set(plan_cases) != set(replay_cases):
        raise ValueError("Plan and replay case IDs do not match")
    if set(variants) != set(plan_cases):
        raise ValueError("Loaded variant case IDs do not match the frozen plan")
    if replay.get("passed") is not True:
        raise ValueError("Frozen replay report did not pass")

    variant_hashes = {}
    for case_id, plan_case in plan_cases.items():
        replay_case = replay_cases[case_id]
        if replay_case.get("passed") is not True:
            raise ValueError(f"Frozen replay case did not pass: {case_id}")
        settings = plan_case.get("settings")
        if not isinstance(settings, dict) or type(settings.get("keys")) is not int:
            raise ValueError(f"Frozen plan has no valid key count: {case_id}")
        if replay_case.get("keys") != settings["keys"]:
            raise ValueError(f"Frozen replay key count does not match plan: {case_id}")
        _sections(replay_case, case_id)
        variant_hash = replay_case.get("variant_sha256")
        if not isinstance(variant_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", variant_hash):
            raise ValueError(f"Invalid variant hash in replay case: {case_id}")
        variant_hashes[case_id] = variant_hash

    cases = []
    for case_id, plan_case in plan_cases.items():
        replay_case = replay_cases[case_id]
        variant = variants[case_id]
        if not isinstance(variant, dict):
            raise ValueError(f"Missing loaded frozen variant: {case_id}")
        cases.append(_summarize_case(plan_case, replay_case, variant))

    cases.sort(key=lambda row: row["id"])
    return {
        "schema": "manialab-frozen-generation-ln-segments-v1",
        "identity": {
            "plan_sha256": plan_sha256,
            "replay_sha256": replay_sha256,
            "tool_sha256": sha256_file(Path(__file__).resolve()),
            "case_ids": [row["id"] for row in cases],
            "variant_sha256": {
                case_id: variant_hashes[case_id] for case_id in sorted(variant_hashes)
            },
            "large_shift_threshold_ms": LARGE_SHIFT_MS,
            "source_tail_clock": "raw audio milliseconds: expressed.end - detected.end",
            "quantization_clock": "chart milliseconds: note.end - note.evidence_end",
        },
        "summary": {
            "case_count": len(cases),
            "ln_count": sum(row["ln_count"] for row in cases),
            "assigned_ln_count": sum(row["assigned_ln_count"] for row in cases),
            "unassigned_ln_count": sum(row["unassigned_ln_count"] for row in cases),
        },
        "cases": cases,
        "limitations": {
            "audio_read": False,
            "reference_chart_read": False,
            "model_inference": False,
            "physical_key_release_verified": False,
            "musical_value_scored": False,
            "purpose": "Locate listening candidates only; model-predicted offsets are not physical key releases.",
        },
    }


def run(plan_path: Path, replay_path: Path, output: Path) -> dict:
    plan_path, replay_path, output = map(_rooted, (plan_path, replay_path, output))
    test_data = (ROOT / "test-data").resolve()
    try:
        output.relative_to(test_data)
    except ValueError as error:
        raise ValueError("Output must remain inside ignored test-data") from error
    if output.exists():
        raise FileExistsError(f"Preserve previous runs: {output}")

    plan_sha = sha256_file(plan_path)
    replay_sha = sha256_file(replay_path)
    plan = read_json(plan_path)
    replay = read_json(replay_path)
    replay_root = replay_path.parent.resolve()
    replay_cases = _case_index(replay.get("cases"), "Replay cases")
    loaded = {}
    source_hashes = {}
    for case_id, replay_case in replay_cases.items():
        variant_path = (replay_root / case_id / "variant.json").resolve()
        try:
            variant_path.relative_to(replay_root)
        except ValueError as error:
            raise ValueError(f"Frozen variant path escapes replay root: {case_id}") from error
        expected = replay_case.get("variant_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid variant hash in replay case: {case_id}")
        actual = sha256_file(variant_path)
        if actual != expected:
            raise ValueError(f"Frozen variant hash mismatch: {case_id}")
        source_hashes[case_id] = actual
        loaded[case_id] = read_json(variant_path)

    report = build_report(plan, replay, loaded, plan_sha, replay_sha)
    if sha256_file(plan_path) != plan_sha or sha256_file(replay_path) != replay_sha:
        raise ValueError("Frozen plan or replay report changed during summarization")
    if any(
        sha256_file(replay_root / case_id / "variant.json") != expected
        for case_id, expected in source_hashes.items()
    ):
        raise ValueError("Frozen variant changed during summarization")

    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run(args.plan, args.replay, args.output)
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
