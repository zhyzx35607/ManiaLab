"""Aggregate frozen generation replay reports by input, key count, and profile."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _count(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _profile(case: dict) -> str:
    plan = case.get("plan") or {}
    if not isinstance(plan, dict):
        raise ValueError(f"Invalid plan for case {case.get('id')}")
    source, choice = plan.get("source"), plan.get("choice")
    if source == "local-audio-recommendation":
        if not isinstance(choice, str) or not choice:
            raise ValueError(f"Recommendation case has no choice: {case.get('id')}")
        return f"recommendation:{choice}"
    if any(
        isinstance(segment, dict)
        and isinstance(segment.get("preferences"), dict)
        and "max_held" in segment["preferences"]
        for segment in plan.get("segments", [])
    ):
        return "held-boundary-pressure"
    if not plan:
        return "default"
    return f"custom:{source or 'unspecified'}"


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _index(rows: object, label: str) -> dict[str, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must be a nonempty list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} rows must be objects")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"{label} contains a missing or duplicate ID")
        result[identity] = row
    return result


def _voice_metrics(row: dict, case_id: str) -> dict:
    voices = row.get("voices")
    if not isinstance(voices, dict):
        raise ValueError(f"Missing per-voice summary for {case_id}")
    result = {}
    for voice, values in sorted(voices.items()):
        if not isinstance(voice, str) or not voice or not isinstance(values, dict):
            raise ValueError(f"Invalid voice summary for {case_id}")
        counts = {
            key: _count(values.get(source), f"{case_id}.{voice}.{source}")
            for key, source in (
                ("candidate_count", "candidates"),
                ("selected_count", "selected"),
                ("retained_raw_ids", "retained_raw_ids"),
                ("ln_count", "ln"),
                ("union_ln_ms", "union_ln_ms"),
                ("unknown_release_ln", "unknown_release_ln"),
                ("large_attack_shifts", "large_attack_shifts"),
                ("max_attack_shift_ms", "max_attack_shift_ms"),
                ("large_release_shifts", "large_release_shifts"),
                ("max_release_shift_ms", "max_release_shift_ms"),
            )
        }
        release_sources = values.get("release_sources")
        omitted_reasons = values.get("omitted_reasons")
        if not isinstance(release_sources, dict) or not isinstance(omitted_reasons, dict):
            raise ValueError(f"Missing release or omission categories for {case_id}.{voice}")
        release_sources = {
            name: _count(count, f"{case_id}.{voice}.release_sources.{name}")
            for name, count in sorted(release_sources.items())
        }
        omitted_reasons = {
            name: _count(count, f"{case_id}.{voice}.omitted_reasons.{name}")
            for name, count in sorted(omitted_reasons.items())
        }
        if sum(release_sources.values()) != counts["ln_count"]:
            raise ValueError(f"Release sources do not account for all LN in {case_id}.{voice}")
        if release_sources.get("unknown", 0) != counts["unknown_release_ln"]:
            raise ValueError(f"Unknown release count disagrees in {case_id}.{voice}")
        known_release_ln = counts["ln_count"] - counts["unknown_release_ln"]
        result[voice] = {
            **counts,
            "known_release_ln": known_release_ln,
            "candidate_selection_rate": (
                round(counts["selected_count"] / counts["candidate_count"], 6)
                if counts["candidate_count"]
                else None
            ),
            "unknown_release_fraction": (
                round(counts["unknown_release_ln"] / counts["ln_count"], 6)
                if counts["ln_count"]
                else None
            ),
            "release_sources": release_sources,
            "omitted_reasons": omitted_reasons,
        }
    if sum(row["selected_count"] for row in result.values()) != _count(
        row.get("notes"), f"{case_id}.notes"
    ):
        raise ValueError(f"Per-voice selected counts do not match total notes in {case_id}")
    if sum(row["ln_count"] for row in result.values()) != _count(
        row.get("ln"), f"{case_id}.ln"
    ):
        raise ValueError(f"Per-voice LN counts do not match total LN in {case_id}")
    return result


def _key_mapping(
    notes: object,
    keys: int,
    case_id: str,
    voices: dict,
) -> tuple[dict, dict[str, dict]]:
    if not isinstance(notes, list):
        raise ValueError(f"Variant notes must be a list for {case_id}")
    lane_counts = [0] * keys
    lane_counts_by_voice: dict[str, list[int]] = defaultdict(lambda: [0] * keys)
    indexed = {}
    selected_by_voice = Counter()
    for note in notes:
        if not isinstance(note, dict):
            raise ValueError(f"Variant notes must be objects for {case_id}")
        identity, lane, start, end, voice = (
            note.get("id"),
            note.get("lane"),
            note.get("start"),
            note.get("end"),
            note.get("voice"),
        )
        if not isinstance(identity, str) or not identity or identity in indexed:
            raise ValueError(f"Variant note IDs must be unique for {case_id}")
        if type(lane) is not int or lane not in range(keys):
            raise ValueError(f"Variant note lane is outside {keys}K for {case_id}")
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
        ):
            raise ValueError(f"Variant note time is invalid for {case_id}")
        if not isinstance(voice, str) or not voice:
            raise ValueError(f"Variant note voice is invalid for {case_id}")
        indexed[identity] = note
        lane_counts[lane] += 1
        lane_counts_by_voice[voice][lane] += 1
        selected_by_voice[voice] += 1

    expected_by_voice = {
        voice: values["selected_count"] for voice, values in voices.items()
    }
    if dict(selected_by_voice) != {
        voice: count for voice, count in expected_by_voice.items() if count
    }:
        raise ValueError(f"Variant per-voice counts disagree for {case_id}")
    return (
        {
            "lane_counts": lane_counts,
            "lane_counts_by_voice": dict(sorted(lane_counts_by_voice.items())),
        },
        indexed,
    )


def _key_mapping_comparisons(cases: list[dict], note_maps: dict[str, dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in cases:
        grouped[(case["analysis_sha256"], case["profile"])].append(case)

    comparisons: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for (analysis_sha, profile), rows in grouped.items():
        for left_index, left in enumerate(sorted(rows, key=lambda row: row["keys"])):
            for right in sorted(rows, key=lambda row: row["keys"])[left_index + 1 :]:
                left_notes = note_maps[left["id"]]
                right_notes = note_maps[right["id"]]
                left_ids, right_ids = set(left_notes), set(right_notes)
                common = left_ids & right_ids
                lane_changed_by_voice = Counter()
                timing_changed_by_voice = Counter()
                pitch_changed_by_voice = Counter()
                voice_changed = 0
                for identity in common:
                    before, after = left_notes[identity], right_notes[identity]
                    voice = before["voice"]
                    if before["lane"] != after["lane"]:
                        lane_changed_by_voice[voice] += 1
                    if (before["start"], before["end"]) != (after["start"], after["end"]):
                        timing_changed_by_voice[voice] += 1
                    if before.get("pitch") != after.get("pitch"):
                        pitch_changed_by_voice[voice] += 1
                    if voice != after["voice"]:
                        voice_changed += 1
                added_by_voice = Counter(
                    right_notes[identity]["voice"] for identity in right_ids - left_ids
                )
                removed_by_voice = Counter(
                    left_notes[identity]["voice"] for identity in left_ids - right_ids
                )
                comparisons[(profile, left["keys"], right["keys"])].append(
                    {
                        "analysis_sha256": analysis_sha,
                        "from_case": left["id"],
                        "to_case": right["id"],
                        "from_notes": len(left_ids),
                        "to_notes": len(right_ids),
                        "common_note_ids": len(common),
                        "added_note_ids": len(right_ids - left_ids),
                        "removed_note_ids": len(left_ids - right_ids),
                        "lane_changed_note_ids": sum(lane_changed_by_voice.values()),
                        "lane_changed_by_voice": dict(sorted(lane_changed_by_voice.items())),
                        "timing_changed_note_ids": sum(timing_changed_by_voice.values()),
                        "timing_changed_by_voice": dict(sorted(timing_changed_by_voice.items())),
                        "pitch_changed_note_ids": sum(pitch_changed_by_voice.values()),
                        "voice_changed_note_ids": voice_changed,
                        "added_by_voice": dict(sorted(added_by_voice.items())),
                        "removed_by_voice": dict(sorted(removed_by_voice.items())),
                    }
                )

    result = []
    for (profile, from_keys, to_keys), rows in sorted(comparisons.items()):
        common = sum(row["common_note_ids"] for row in rows)
        lane_changed = sum(row["lane_changed_note_ids"] for row in rows)
        lane_changed_by_voice = Counter()
        timing_changed_by_voice = Counter()
        for row in rows:
            lane_changed_by_voice.update(row["lane_changed_by_voice"])
            timing_changed_by_voice.update(row["timing_changed_by_voice"])
        result.append(
            {
                "profile": profile,
                "from_keys": from_keys,
                "to_keys": to_keys,
                "analysis_count": len(rows),
                "from_notes": sum(row["from_notes"] for row in rows),
                "to_notes": sum(row["to_notes"] for row in rows),
                "common_note_ids": common,
                "added_note_ids": sum(row["added_note_ids"] for row in rows),
                "removed_note_ids": sum(row["removed_note_ids"] for row in rows),
                "lane_changed_note_ids": lane_changed,
                "lane_changed_fraction": round(lane_changed / common, 6) if common else None,
                "lane_changed_by_voice": dict(sorted(lane_changed_by_voice.items())),
                "timing_changed_note_ids": sum(
                    row["timing_changed_note_ids"] for row in rows
                ),
                "timing_changed_by_voice": dict(sorted(timing_changed_by_voice.items())),
                "pitch_changed_note_ids": sum(
                    row["pitch_changed_note_ids"] for row in rows
                ),
                "voice_changed_note_ids": sum(
                    row["voice_changed_note_ids"] for row in rows
                ),
                "analyses": sorted(rows, key=lambda row: row["analysis_sha256"]),
            }
        )
    return result


def _median(values: list[float | int]) -> float | None:
    return round(float(statistics.median(values)), 6) if values else None


def build_matrix_report(
    plan: dict,
    replay: dict,
    *,
    plan_sha256: str,
    replay_sha256: str,
    tool_sha256: str,
    variant_notes_by_case: dict[str, list[dict]] | None = None,
) -> dict:
    if plan.get("schema") != "manialab-frozen-generation-v1":
        raise ValueError("Expected a frozen generation plan")
    if replay.get("schema") != "manialab-frozen-generation-results-v1":
        raise ValueError("Expected a frozen generation replay report")
    if replay.get("plan_sha256") != plan_sha256:
        raise ValueError("Replay report and plan hashes differ")
    if replay.get("passed") is not True:
        raise ValueError("Replay report is not fully passing")
    planned = _index(plan.get("cases"), "plan cases")
    reported = _index(replay.get("cases"), "replay cases")
    if planned.keys() != reported.keys():
        missing = sorted(planned.keys() - reported.keys())
        extra = sorted(reported.keys() - planned.keys())
        raise ValueError(f"Plan/replay case IDs differ; missing={missing[:3]}, extra={extra[:3]}")

    if variant_notes_by_case is not None and variant_notes_by_case.keys() != planned.keys():
        raise ValueError("Plan/variant case IDs differ")

    cases = []
    note_maps = {}
    profile_units = set()
    timing_by_analysis: dict[str, dict] = {}
    for case_id in sorted(planned):
        spec, row = planned[case_id], reported[case_id]
        settings = spec.get("settings")
        if not isinstance(settings, dict):
            raise ValueError(f"Missing settings for {case_id}")
        keys = _count(settings.get("keys"), f"{case_id}.keys")
        if keys not in range(6, 11) or row.get("keys") != keys:
            raise ValueError(f"Invalid or mismatched key count for {case_id}")
        analysis_sha = spec.get("analysis_sha256")
        if not isinstance(analysis_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", analysis_sha):
            raise ValueError(f"Invalid analysis hash for {case_id}")
        analysis_path = spec.get("analysis")
        if not isinstance(analysis_path, str) or not analysis_path:
            raise ValueError(f"Missing analysis path for {case_id}")
        if row.get("passed") is not True:
            raise ValueError(f"Replay case did not pass: {case_id}")

        profile = _profile(spec)
        unit = (analysis_sha, keys, profile)
        if unit in profile_units:
            raise ValueError(
                f"Multiple cases share one analysis/key/profile unit: {case_id}; "
                "give distinct profiles to sensitivity cases"
            )
        profile_units.add(unit)

        gate = row.get("automatic_timing_gate")
        if not isinstance(gate, dict) or type(gate.get("accepted")) is not bool:
            raise ValueError(f"Missing automatic Timing result for {case_id}")
        tempo = spec.get("tempo")
        if not isinstance(tempo, dict):
            raise ValueError(f"Missing frozen clock for {case_id}")
        clock = {
            key: tempo.get(key)
            for key in ("bpm", "offset", "division", "audio_shift_ms", "stable")
        }
        analysis_timing = timing_by_analysis.get(analysis_sha)
        timing = {
            "clock": clock,
            "automatic_timing_gate_accepted": gate["accepted"],
        }
        if analysis_timing and analysis_timing["result"] != timing:
            raise ValueError(f"Timing result changes across configs for the same analysis: {case_id}")
        if not analysis_timing:
            timing_by_analysis[analysis_sha] = {
                "analysis": analysis_path,
                "result": timing,
                "case_ids": [],
                "keys": set(),
            }
        timing_by_analysis[analysis_sha]["case_ids"].append(case_id)
        timing_by_analysis[analysis_sha]["keys"].add(keys)

        voice_rows = _voice_metrics(row, case_id)
        case = {
            "id": case_id,
            "analysis": analysis_path,
            "analysis_sha256": analysis_sha,
            "keys": keys,
            "profile": profile,
            "configuration_sha256": _canonical_sha(
                {
                    "settings": settings,
                    "tempo": tempo,
                    "plan": spec.get("plan"),
                }
            ),
            "notes": _count(row.get("notes"), f"{case_id}.notes"),
            "ln": _count(row.get("ln"), f"{case_id}.ln"),
            "source_snapshot_notes": _count(
                row.get("source_snapshot_notes"), f"{case_id}.source_snapshot_notes"
            ),
            "timing": timing,
            "timing_suspect_windows": _count(
                row.get("timing_suspect_windows"), f"{case_id}.timing_suspect_windows"
            ),
            "voices": voice_rows,
        }
        if variant_notes_by_case is not None:
            mapping, note_maps[case_id] = _key_mapping(
                variant_notes_by_case[case_id],
                keys,
                case_id,
                voice_rows,
            )
            if len(note_maps[case_id]) != case["notes"]:
                raise ValueError(f"Variant note count disagrees for {case_id}")
            case["key_mapping"] = mapping
        cases.append(case)

    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for case in cases:
        grouped[(case["keys"], case["profile"])].append(case)

    by_key_profile = []
    for (keys, profile), rows in sorted(grouped.items()):
        voice_names = sorted({name for case in rows for name in case["voices"]})
        voice_summary = {}
        for voice in voice_names:
            samples = [
                case["voices"].get(
                    voice,
                    {
                        "candidate_count": 0,
                        "selected_count": 0,
                        "retained_raw_ids": 0,
                        "ln_count": 0,
                        "union_ln_ms": 0,
                        "unknown_release_ln": 0,
                        "known_release_ln": 0,
                        "large_attack_shifts": 0,
                        "max_attack_shift_ms": 0,
                        "large_release_shifts": 0,
                        "max_release_shift_ms": 0,
                        "candidate_selection_rate": None,
                        "unknown_release_fraction": None,
                        "release_sources": {},
                        "omitted_reasons": {},
                    },
                )
                for case in rows
            ]
            rates = [
                sample["candidate_selection_rate"]
                for sample in samples
                if sample["candidate_selection_rate"] is not None
            ]
            release_sources = Counter()
            omitted_reasons = Counter()
            for sample in samples:
                release_sources.update(sample["release_sources"])
                omitted_reasons.update(sample["omitted_reasons"])
            candidate_total = sum(sample["candidate_count"] for sample in samples)
            selected_total = sum(sample["selected_count"] for sample in samples)
            ln_total = sum(sample["ln_count"] for sample in samples)
            unknown_ln = sum(sample["unknown_release_ln"] for sample in samples)
            voice_summary[voice] = {
                "analyses_with_candidates": sum(sample["candidate_count"] > 0 for sample in samples),
                "analyses_without_candidates": sum(sample["candidate_count"] == 0 for sample in samples),
                "candidate_count": candidate_total,
                "selected_count": selected_total,
                "pooled_selection_rate": (
                    round(selected_total / candidate_total, 6) if candidate_total else None
                ),
                "median_per_analysis_selection_rate": _median(rates),
                "ln_count": ln_total,
                "median_ln_per_analysis": _median([sample["ln_count"] for sample in samples]),
                "unknown_release_ln": unknown_ln,
                "known_release_ln": ln_total - unknown_ln,
                "unknown_release_fraction": round(unknown_ln / ln_total, 6) if ln_total else None,
                "union_ln_ms": sum(sample["union_ln_ms"] for sample in samples),
                "large_attack_shifts": sum(sample["large_attack_shifts"] for sample in samples),
                "large_release_shifts": sum(sample["large_release_shifts"] for sample in samples),
                "release_sources": dict(sorted(release_sources.items())),
                "omitted_reasons": dict(sorted(omitted_reasons.items())),
            }
        by_key_profile.append(
            {
                "keys": keys,
                "profile": profile,
                "case_count": len(rows),
                "analysis_count": len({case["analysis_sha256"] for case in rows}),
                "automatic_timing_accepted_cases": sum(
                    case["timing"]["automatic_timing_gate_accepted"] for case in rows
                ),
                "automatic_timing_rejected_cases": sum(
                    not case["timing"]["automatic_timing_gate_accepted"] for case in rows
                ),
                "notes": sum(case["notes"] for case in rows),
                "ln": sum(case["ln"] for case in rows),
                "source_snapshot_notes": sum(case["source_snapshot_notes"] for case in rows),
                "voices": voice_summary,
            }
        )

    analyses = []
    for analysis_sha, values in sorted(timing_by_analysis.items()):
        analyses.append(
            {
                "analysis": values["analysis"],
                "analysis_sha256": analysis_sha,
                "case_ids": sorted(values["case_ids"]),
                "keys": sorted(values["keys"]),
                **values["result"],
            }
        )
    timing_accepted = sum(
        row["automatic_timing_gate_accepted"] for row in analyses
    )
    result = {
        "schema": "manialab-frozen-generation-matrix-v1",
        "plan_sha256": plan_sha256,
        "replay_report_sha256": replay_sha256,
        "tool_sha256": tool_sha256,
        "replay_source_files": replay.get("source_files", {}),
        "summary": {
            "case_count": len(cases),
            "analysis_count": len(analyses),
            "key_counts": sorted({case["keys"] for case in cases}),
            "automatic_timing_accepted_analyses": timing_accepted,
            "automatic_timing_rejected_analyses": len(analyses) - timing_accepted,
        },
        "analyses": analyses,
        "by_key_profile": by_key_profile,
        "cases": cases,
        "limitations": {
            "audio_read": False,
            "neural_inference": False,
            "reference_chart_read": False,
            "independent_generalization": False,
            "musical_value_scored": False,
            "unknown_release_is_not_classified_as_good_or_bad": True,
            "profile_totals_are_descriptive_not_causal": True,
            "analysis_is_the_deduplication_unit_not_a_claimed_song_group": True,
        },
    }
    if variant_notes_by_case is not None:
        result["key_mapping_comparisons"] = _key_mapping_comparisons(cases, note_maps)
        result["limitations"]["lane_changes_are_not_player_validated"] = True
    return result


def run(plan_path: Path, replay_path: Path, output: Path) -> dict:
    def rooted(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    plan_path, replay_path, output = map(rooted, (plan_path, replay_path, output))
    output_root = (ROOT / "test-data").resolve()
    try:
        output.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("Output must remain inside ignored test-data") from exc
    if output.exists():
        raise FileExistsError(f"Preserve previous runs: {output}")
    plan_sha, replay_sha = _sha256_file(plan_path), _sha256_file(replay_path)
    replay = _read_json(replay_path)
    variant_notes_by_case = {}
    for row in _index(replay.get("cases"), "replay cases").values():
        case_id = row["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
            raise ValueError(f"Unsafe replay case ID: {case_id}")
        variant_path = replay_path.parent / case_id / "variant.json"
        expected_sha = row.get("variant_sha256")
        if not isinstance(expected_sha, str) or _sha256_file(variant_path) != expected_sha:
            raise ValueError(f"Replay variant hash mismatch: {case_id}")
        variant = _read_json(variant_path)
        notes = variant.get("notes")
        if not isinstance(notes, list):
            raise ValueError(f"Replay variant has no notes: {case_id}")
        variant_notes_by_case[case_id] = notes
    report = build_matrix_report(
        _read_json(plan_path),
        replay,
        plan_sha256=plan_sha,
        replay_sha256=replay_sha,
        tool_sha256=_sha256_file(Path(__file__)),
        variant_notes_by_case=variant_notes_by_case,
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.plan, args.replay, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": report["summary"],
                "profiles": [
                    {
                        "keys": row["keys"],
                        "profile": row["profile"],
                        "analysis_count": row["analysis_count"],
                        "ln": row["ln"],
                    }
                    for row in report["by_key_profile"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
