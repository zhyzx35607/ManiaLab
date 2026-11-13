"""Audit frozen onset-selection decisions for held-note review markers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manialab.domain import UNOBSERVED_RELEASES


INPUTS = {
    "analysis": "test-data/sakura-tsumugi-expression-onset-v3/analysis.json",
    "selection": "test-data/expression-precision-v1/default/10k/selection.json",
    "execution_plan": "test-data/expression-precision-v1/execution-plan.json",
}
EXPECTED_SHA256 = {
    "analysis": "26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a",
    "selection": "24f23cc1d59e7452e5ede424f10054d285b0d2ecdd26b33095e651716f0b0b44",
    "execution_plan": "e6d47956ec469a98c81997f9e09d36011b8c2da6a8b75041aca4c4ad42b1b31e",
}
SOURCE_PATHS = (
    "scripts/summarize_expression_tradeoffs.py",
    "manialab/domain.py",
    "manialab/expression.py",
    "docs/expression-tradeoff-audit-protocol.md",
)
OMITTED_REASONS = {
    "duplicate_pitch",
    "group_capacity_or_density",
    "represented_by_chord",
    "voice_not_selected",
    "weak_detection",
}


def _index(rows: object, label: str) -> dict[str, dict]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} row must be an object")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"{label} contains a missing or duplicate ID")
        result[identity] = row
    return result


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap_ms(start: int, end: int, intervals: list[tuple[int, int]]) -> int:
    return sum(max(0, min(end, right) - max(start, left)) for left, right in intervals)


def _uncovered_intervals(
    start: int,
    end: int,
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    cursor = start
    result = []
    for left, right in intervals:
        if right <= cursor:
            continue
        if left >= end:
            break
        if left > cursor:
            result.append((cursor, min(left, end)))
        cursor = max(cursor, right)
        if cursor >= end:
            break
    if cursor < end:
        result.append((cursor, end))
    return result


def _hold_eligible(decision: dict, candidate: dict, min_ln_ms: int) -> bool:
    articulation = decision.get("articulation", {})
    if not isinstance(articulation, dict):
        raise ValueError(f"Invalid articulation evidence for candidate {decision.get('id')}")
    start = decision.get("start")
    end = decision.get("expression_end")
    return (
        candidate.get("source") == "pitch"
        and decision.get("mode") not in {"onsets", "omit"}
        and candidate.get("release_reason") not in UNOBSERVED_RELEASES
        and articulation.get("tap") is not True
        and type(start) is int
        and type(end) is int
        and end - start >= min_ln_ms
    )


def build_expression_tradeoff_audit(
    analysis: dict,
    selection: dict,
    min_ln_ms: int,
) -> dict:
    if type(min_ln_ms) is not int or min_ln_ms <= 0:
        raise ValueError("min_ln_ms must be a positive integer")
    candidates = _index(analysis.get("candidates"), "analysis candidates")
    decisions = _index(selection.get("decisions"), "selection decisions")
    if candidates.keys() != decisions.keys():
        missing = sorted(candidates.keys() - decisions.keys())
        unknown = sorted(decisions.keys() - candidates.keys())
        raise ValueError(
            f"Analysis and selection IDs differ; missing={missing[:3]}, unknown={unknown[:3]}"
        )

    sections = (analysis.get("structure") or {}).get("segments")
    if not isinstance(sections, list):
        raise ValueError("Analysis structure segments are missing")

    selected_count = 0
    omitted_count = 0
    omitted_reasons = Counter()
    actual_selected_ln = []
    selected_ln_onsets = set()
    section_voice_counts = defaultdict(Counter)
    omitted_hold_candidates = []

    for identity, decision in decisions.items():
        candidate = candidates[identity]
        start = candidate.get("start")
        voice = candidate.get("voice")
        section = decision.get("section")
        if (
            type(start) is not int
            or decision.get("start") != start
            or not isinstance(voice, str)
            or not voice
            or decision.get("voice") != voice
        ):
            raise ValueError(f"Candidate and selection onset/voice differ for {identity}")
        if type(section) is not int or not 0 <= section < len(sections):
            raise ValueError(f"Invalid section for candidate {identity}")
        if decision.get("decision") not in {"selected", "omitted"}:
            raise ValueError(f"Invalid selection state for candidate {identity}")
        if not isinstance(decision.get("reason"), str) or not decision["reason"]:
            raise ValueError(f"Missing selection reason for candidate {identity}")

        key = (section, voice)
        section_voice_counts[key]["candidate_count"] += 1
        if decision["decision"] == "selected":
            selected_count += 1
            section_voice_counts[key]["selected_candidate_count"] += 1
            chart_start = decision.get("chart_start")
            chart_end = decision.get("chart_end")
            if type(chart_start) is not int or type(chart_end) is not int:
                raise ValueError(f"Selected candidate has no chart interval: {identity}")
            if chart_end > chart_start:
                if not _hold_eligible(decision, candidate, min_ln_ms):
                    raise ValueError(f"Selected LN does not satisfy hold evidence: {identity}")
                actual_selected_ln.append(decision)
                selected_ln_onsets.add((section, voice, start))
                section_voice_counts[key]["selected_ln_candidate_count"] += 1
        else:
            omitted_count += 1
            reason = decision["reason"]
            if reason not in OMITTED_REASONS:
                raise ValueError(f"Unknown omission reason for candidate {identity}: {reason}")
            omitted_reasons[reason] += 1
            section_voice_counts[key]["omitted_candidate_count"] += 1
            if _hold_eligible(decision, candidate, min_ln_ms):
                omitted_hold_candidates.append((decision, candidate))

    if selection.get("selected") != selected_count or selection.get("omitted") != omitted_count:
        raise ValueError("Selection totals do not match its decision rows")
    if selection.get("omission_reasons") != dict(sorted(omitted_reasons.items())):
        raise ValueError("Selection omission reasons do not match its decision rows")

    selected_intervals_by_voice = defaultdict(list)
    for decision in actual_selected_ln:
        start = decision["start"]
        end = decision.get("expression_end")
        if type(end) is not int or end <= start:
            raise ValueError(f"Selected LN has no analysis-clock interval: {decision['id']}")
        selected_intervals_by_voice[decision["voice"]].append((start, end))
    selected_intervals_by_voice = {
        voice: _merge_intervals(intervals)
        for voice, intervals in selected_intervals_by_voice.items()
    }

    groups = defaultdict(list)
    for decision, candidate in omitted_hold_candidates:
        key = (decision["section"], decision["voice"], decision["start"])
        groups[key].append((decision, candidate))

    section_rows = []
    by_section_voice = []
    for key, grouped_candidates in sorted(groups.items()):
        section, voice, start = key
        candidates_sorted = sorted(grouped_candidates, key=lambda row: row[0]["id"])
        end = max(decision["expression_end"] for decision, _ in candidates_sorted)
        intervals = selected_intervals_by_voice.get(voice, [])
        uncovered = _uncovered_intervals(start, end, intervals)
        section_info = sections[section]
        reason_counts = Counter(decision["reason"] for decision, _ in candidates_sorted)
        row = dict(
            section_index=section,
            section_label=section_info.get("label", ""),
            section_start_ms=section_info.get("start_ms"),
            section_end_ms=section_info.get("end_ms"),
            voice=voice,
            start_analysis_ms=start,
            max_expression_end_analysis_ms=end,
            candidate_count=len(candidates_sorted),
            candidate_ids=[decision["id"] for decision, _ in candidates_sorted],
            omission_reasons=dict(sorted(reason_counts.items())),
            has_selected_ln_at_same_voice_onset=key in selected_ln_onsets,
            overlap_with_current_same_voice_ln_ms=_overlap_ms(start, end, intervals),
            individually_uncovered_ms=sum(right - left for left, right in uncovered),
            individually_uncovered_intervals_analysis_ms=[
                [left, right] for left, right in uncovered
            ],
        )
        section_rows.append(row)
        counts = section_voice_counts[(section, voice)]
        counts["omitted_hold_candidate_count"] += len(candidates_sorted)
        counts["omitted_hold_onset_group_count"] += 1
        if key not in selected_ln_onsets:
            counts["omitted_hold_onset_groups_without_selected_ln"] += 1
        counts["omitted_hold_onset_group_reason_rows"] += len(candidates_sorted)

    section_pairs = sorted({
        (section, voice)
        for section, voice in section_voice_counts
    })
    for section, voice in section_pairs:
        counts = section_voice_counts[(section, voice)]
        relevant_groups = [
            row for row in section_rows
            if row["section_index"] == section and row["voice"] == voice
        ]
        reason_counts = Counter()
        for row in relevant_groups:
            reason_counts.update(row["omission_reasons"])
        section_info = sections[section]
        by_section_voice.append(dict(
            section_index=section,
            section_label=section_info.get("label", ""),
            section_start_ms=section_info.get("start_ms"),
            section_end_ms=section_info.get("end_ms"),
            voice=voice,
            candidate_count=counts["candidate_count"],
            selected_candidate_count=counts["selected_candidate_count"],
            omitted_candidate_count=counts["omitted_candidate_count"],
            selected_ln_candidate_count=counts["selected_ln_candidate_count"],
            omitted_hold_candidate_count=counts["omitted_hold_candidate_count"],
            omitted_hold_onset_group_count=counts["omitted_hold_onset_group_count"],
            omitted_hold_onset_groups_without_selected_ln=
                counts["omitted_hold_onset_groups_without_selected_ln"],
            omitted_hold_candidate_reasons=dict(sorted(reason_counts.items())),
        ))

    selected_ln_group_count = len(selected_ln_onsets)
    no_same_onset_ln_count = sum(
        not row["has_selected_ln_at_same_voice_onset"] for row in section_rows
    )
    review_queue = sorted(
        section_rows,
        key=lambda row: (
            -row["individually_uncovered_ms"],
            row["section_index"],
            row["voice"],
            row["start_analysis_ms"],
            row["candidate_ids"],
        ),
    )[:30]
    return dict(
        schema="manialab-expression-tradeoff-audit-v1",
        scope=dict(
            dataset="Sakura Tsumugi expression onset v3",
            selection="expression precision default 10K",
            min_ln_ms=min_ln_ms,
            interval_clock="raw analysis candidate start/expression_end; not chart timing",
            section_identity="frozen selection section index",
            source_identity="selection voice field; model class is not treated as a source",
        ),
        counts=dict(
            analysis_candidate_count=len(candidates),
            selected_candidate_count=selected_count,
            omitted_candidate_count=omitted_count,
            selected_ln_candidate_count=len(actual_selected_ln),
            selected_ln_onset_group_count=selected_ln_group_count,
            omitted_hold_eligible_candidate_count=len(omitted_hold_candidates),
            omitted_hold_eligible_onset_group_count=len(section_rows),
            omitted_hold_groups_without_same_onset_selected_ln=no_same_onset_ln_count,
        ),
        omitted_hold_candidate_reasons=dict(
            sorted(Counter(
                decision["reason"] for decision, _ in omitted_hold_candidates
            ).items())
        ),
        by_section_voice=by_section_voice,
        omitted_hold_onset_groups=section_rows,
        structural_review_queue=dict(
            ordering="descending individually_uncovered_ms; not a musical-value ranking",
            candidates=review_queue,
        ),
        interpretation_limits=dict(
            neural_inference=False,
            selection_rerun=False,
            audio_read=False,
            reference_notes_read=False,
            selected_ln_count_is_chart_export=True,
            omitted_hold_is_pre_arrangement_eligibility=True,
            omitted_intervals_are_not_additive_gains=True,
            capacity_or_density_constraints_not_resolved_again=True,
            musical_value_scored=False,
            timing_acceptance=False,
            musical_acceptance=False,
            production_default_changed=False,
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output: Path) -> dict:
    output = output if output.is_absolute() else ROOT / output
    output = output.resolve()
    try:
        output.relative_to((ROOT / "test-data").resolve())
    except ValueError as exc:
        raise ValueError("Output must remain inside ignored test-data") from exc
    if output.exists():
        raise FileExistsError(f"Preserve previous runs: {output}")

    paths = {name: ROOT / relative for name, relative in INPUTS.items()}
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED_SHA256:
        raise ValueError(f"Frozen input hashes changed: {hashes}")
    loaded = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    settings = loaded["execution_plan"].get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Frozen execution plan has no settings")
    report = build_expression_tradeoff_audit(
        loaded["analysis"], loaded["selection"], settings.get("min_ln")
    )
    report["input_sha256"] = {
        paths[name].relative_to(ROOT).as_posix(): digest
        for name, digest in hashes.items()
    }
    report["source_sha256"] = {
        name: _sha256_file(ROOT / name) for name in SOURCE_PATHS
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.output)
    print(json.dumps(dict(
        output=str(args.output),
        counts=report["counts"],
        omitted_hold_candidate_reasons=report["omitted_hold_candidate_reasons"],
        review_queue=report["structural_review_queue"]["candidates"][:5],
        limits=report["interpretation_limits"],
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
