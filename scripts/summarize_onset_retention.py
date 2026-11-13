"""Build a frozen-input onset retention ledger without rerunning selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "analysis": "test-data/sakura-tsumugi-expression-onset-v3/analysis.json",
    "chart_candidates": "test-data/chord-evidence-ties-v1/10k/candidates.json",
    "selection": "test-data/expression-precision-v1/default/10k/selection.json",
    "legacy_scopes": "test-data/expression-rhythm-v1/default/full/10k/scopes.json",
}
EXPECTED_SHA256 = {
    "analysis": "26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a",
    "chart_candidates": "c20742509baea4d94c63493d6f5454fa3ee5c4b72636917f4ea2e89fab071895",
    "selection": "24f23cc1d59e7452e5ede424f10054d285b0d2ecdd26b33095e651716f0b0b44",
    "legacy_scopes": "185634495f21d9ad98cd33362ac4474c2a146966e81e67fe371b3343338ad46d",
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


def _reason_counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(
        row["selection_reason"] for row in rows if row["decision"] == "omitted"
    ).items()))


def _summarize_onset(start: int, rows: list[dict]) -> dict:
    by_voice: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_voice[(row["section"], row["voice"])].append(row)

    voice_rows = []
    for (section, voice), candidates in sorted(by_voice.items()):
        selected = sum(row["decision"] == "selected" for row in candidates)
        omitted = len(candidates) - selected
        groups = sorted({
            row["selection_group_index"] for row in candidates
            if row["selection_group_index"] is not None
        })
        class_labels = sorted({
            row["model_class_label"] for row in candidates if row["model_class_label"]
        })
        voice_rows.append(dict(
            section=section,
            voice=voice,
            candidate_count=len(candidates),
            chart_candidate_count=sum(row["chart_candidate"] for row in candidates),
            selected_candidate_count=selected,
            omitted_candidate_count=omitted,
            onset_state=("omitted" if not selected else
                         "selected" if not omitted else "partially_selected"),
            omitted_reasons=_reason_counts(candidates),
            selection_group_indices=groups,
            model_class_labels=class_labels,
            candidate_ids=sorted(row["candidate_id"] for row in candidates),
            candidates=sorted(candidates, key=lambda row: row["candidate_id"]),
        ))

    unique_voices = {row["voice"] for row in voice_rows}
    return dict(
        start_ms=start,
        voice_count=len(unique_voices),
        section_voice_onset_count=len(voice_rows),
        same_time_across_voices=len(unique_voices) > 1,
        candidate_count=len(rows),
        voice_onsets=voice_rows,
    )


def build_onset_retention_ledger(
    analysis: dict,
    chart_candidates: list[dict],
    selection: dict,
    legacy_scopes: list[dict],
) -> dict:
    analysis_rows = _index(analysis.get("candidates"), "analysis candidates")
    decisions = _index(selection.get("decisions"), "selection decisions")
    if analysis_rows.keys() != decisions.keys():
        missing = sorted(analysis_rows.keys() - decisions.keys())
        unknown = sorted(decisions.keys() - analysis_rows.keys())
        raise ValueError(
            f"Analysis and selection IDs differ; missing={missing[:3]}, unknown={unknown[:3]}"
        )

    chart_rows = _index(chart_candidates, "chart candidates")
    if not chart_rows.keys() <= analysis_rows.keys():
        raise ValueError("Chart candidate IDs must be a subset of the analysis candidates")
    selected_ids = {
        identity for identity, decision in decisions.items()
        if decision.get("decision") == "selected"
    }
    if not selected_ids <= chart_rows.keys():
        raise ValueError("Selected candidate IDs must be present in the chart candidates")

    scope_ids = []
    if not isinstance(legacy_scopes, list):
        raise ValueError("Legacy scopes must be a list")
    for scope in legacy_scopes:
        if not isinstance(scope, dict) or not isinstance(scope.get("rows"), list):
            raise ValueError("Legacy scope rows are malformed")
        for row in scope["rows"]:
            identities = row.get("candidate_ids") if isinstance(row, dict) else None
            if not isinstance(identities, list) or any(not isinstance(i, str) for i in identities):
                raise ValueError("Legacy scope candidate IDs are malformed")
            scope_ids.extend(identities)
    if len(scope_ids) != len(set(scope_ids)) or set(scope_ids) != chart_rows.keys():
        raise ValueError("Legacy scope IDs must cover chart candidates exactly once")

    rows_by_start: dict[int, list[dict]] = defaultdict(list)
    rows_by_section_voice_start: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    for identity, candidate in analysis_rows.items():
        decision = decisions[identity]
        start = candidate.get("start")
        voice = candidate.get("voice")
        if type(start) is not int or decision.get("start") != start:
            raise ValueError(f"Raw onset mismatch for candidate {identity}")
        if not isinstance(voice, str) or not voice or decision.get("voice") != voice:
            raise ValueError(f"Voice mismatch for candidate {identity}")
        section = decision.get("section")
        if type(section) is not int or section < 0:
            raise ValueError(f"Invalid section for candidate {identity}")
        if decision.get("decision") not in {"selected", "omitted"}:
            raise ValueError(f"Invalid selection state for candidate {identity}")
        if not isinstance(decision.get("reason"), str) or not decision["reason"]:
            raise ValueError(f"Missing selection reason for candidate {identity}")
        group = decision.get("group")
        if group is not None and (type(group) is not int or group < 0):
            raise ValueError(f"Invalid selection group index for candidate {identity}")
        represented_by = decision.get("represented_by", [])
        if not isinstance(represented_by, list) or any(
            not isinstance(value, str) for value in represented_by
        ):
            raise ValueError(f"Invalid represented_by field for candidate {identity}")

        chart = chart_rows.get(identity)
        row = dict(
            candidate_id=identity,
            start_ms=start,
            voice=voice,
            section=section,
            decision=decision["decision"],
            selection_reason=decision["reason"],
            selection_group_index=group,
            represented_by_candidate_ids=represented_by,
            represented_at_ms=decision.get("represented_at"),
            chart_candidate=chart is not None,
            chart_start_ms=chart.get("start") if chart is not None else None,
            model_class_label=decision.get("instrument_id", ""),
            source_label=candidate.get("source"),
            strength_value=candidate.get("strength"),
            evidence_start_ms=candidate.get("evidence_start"),
        )
        rows_by_start[start].append(row)
        rows_by_section_voice_start[(section, voice, start)].append(row)

    selected_count = sum(row["decision"] == "selected" for rows in rows_by_start.values()
                         for row in rows)
    omitted_count = sum(row["decision"] == "omitted" for rows in rows_by_start.values()
                        for row in rows)
    reported_reasons = selection.get("omission_reasons")
    actual_reasons = dict(sorted(Counter(
        decision["reason"] for decision in decisions.values()
        if decision["decision"] == "omitted"
    ).items()))
    if reported_reasons != actual_reasons:
        raise ValueError("Selection omission_reasons do not match its decision rows")
    if selection.get("selected") != selected_count or selection.get("omitted") != omitted_count:
        raise ValueError("Selection totals do not match its decision rows")

    onsets = [
        _summarize_onset(start, rows)
        for start, rows in sorted(rows_by_start.items())
    ]
    by_voice: dict[str, list[dict]] = defaultdict(list)
    for onset in onsets:
        for row in onset["voice_onsets"]:
            by_voice[row["voice"]].append(row)
    voice_summary = {}
    for voice, rows in sorted(by_voice.items()):
        reason_totals = Counter()
        for row in rows:
            reason_totals.update(row["omitted_reasons"])
        voice_summary[voice] = dict(
            candidate_count=sum(row["candidate_count"] for row in rows),
            section_voice_onset_count=len(rows),
            chart_candidate_count=sum(row["chart_candidate_count"] for row in rows),
            selected_candidate_count=sum(row["selected_candidate_count"] for row in rows),
            omitted_candidate_count=sum(row["omitted_candidate_count"] for row in rows),
            selected_onset_count=sum(row["onset_state"] == "selected" for row in rows),
            partially_selected_onset_count=sum(
                row["onset_state"] == "partially_selected" for row in rows
            ),
            omitted_onset_count=sum(row["onset_state"] == "omitted" for row in rows),
            omitted_reasons=dict(sorted(reason_totals.items())),
        )

    by_section_voice = []
    section_groups: dict[tuple[int, str], list[list[dict]]] = defaultdict(list)
    for (section, voice, _), rows in sorted(rows_by_section_voice_start.items()):
        section_groups[(section, voice)].append(rows)
    for (section, voice), groups in sorted(section_groups.items()):
        flattened = [row for group in groups for row in group]
        reason_totals = Counter(
            row["selection_reason"] for row in flattened if row["decision"] == "omitted"
        )
        by_section_voice.append(dict(
            section=section,
            voice=voice,
            raw_onset_count=len(groups),
            candidate_count=len(flattened),
            chart_candidate_count=sum(row["chart_candidate"] for row in flattened),
            selected_candidate_count=sum(row["decision"] == "selected" for row in flattened),
            omitted_candidate_count=sum(row["decision"] == "omitted" for row in flattened),
            omitted_reasons=dict(sorted(reason_totals.items())),
            candidate_ids=sorted(row["candidate_id"] for row in flattened),
        ))

    outside_chart = [
        row for rows in rows_by_start.values() for row in rows if not row["chart_candidate"]
    ]
    outside_by_voice_reason: dict[str, Counter] = defaultdict(Counter)
    for row in outside_chart:
        outside_by_voice_reason[row["voice"]][row["selection_reason"]] += 1
    multi_voice = [row for row in onsets if row["same_time_across_voices"]]
    return dict(
        schema="manialab-onset-retention-ledger-v1",
        aggregation=dict(
            onset_clock="original analysis candidate start; chart_start is retained separately",
            voice_identity="selection decision voice field",
            instrument_identity="instrument_id is retained as a model-class label, not counted as a separate source",
            selection_group="selection decision group field, preserved as selection_group_index",
            strength_semantics="copied raw value; not interpreted as confidence or musical value",
        ),
        counts=dict(
            analysis_candidate_count=len(analysis_rows),
            chart_candidate_count=len(chart_rows),
            selected_candidate_count=selected_count,
            omitted_candidate_count=omitted_count,
            unique_raw_onset_times=len(onsets),
            unique_section_voice_onsets=sum(
                len(row["voice_onsets"]) for row in onsets
            ),
            same_time_multi_voice_onset_count=len(multi_voice),
            same_time_multi_voice_candidate_count=sum(row["candidate_count"] for row in multi_voice),
            outside_chart_candidate_count=len(outside_chart),
        ),
        omission_reasons=actual_reasons,
        outside_chart_candidates_by_voice_reason={
            voice: dict(sorted(reasons.items()))
            for voice, reasons in sorted(outside_by_voice_reason.items())
        },
        by_voice=voice_summary,
        by_section_voice=by_section_voice,
        onsets=onsets,
        checks=dict(
            analysis_selection_ids_equal=True,
            chart_candidate_ids_are_analysis_subset=True,
            selected_ids_are_chart_candidate_subset=True,
            legacy_scopes_cover_chart_candidates_once=True,
            selector_totals_match_decisions=True,
            selector_omission_reasons_match_decisions=True,
        ),
        limitations=dict(
            neural_inference=False,
            audio_read=False,
            reference_notes_read=False,
            timing_acceptance=False,
            musical_acceptance=False,
            music_value_scored=False,
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
        raise ValueError("Ledger output must remain inside ignored test-data") from exc
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
    report = build_onset_retention_ledger(
        loaded["analysis"], loaded["chart_candidates"], loaded["selection"],
        loaded["legacy_scopes"],
    )
    report["input_sha256"] = {
        paths[name].relative_to(ROOT).as_posix(): digest
        for name, digest in hashes.items()
    }
    sources = (
        "scripts/summarize_onset_retention.py",
        "scripts/check_expression_rhythm.py",
        "manialab/expression.py",
        "docs/onset-retention-ledger-protocol.md",
    )
    report["source_sha256"] = {name: _sha256_file(ROOT / name) for name in sources}
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
    print(json.dumps(dict(output=str(args.output), counts=report["counts"],
                          omission_reasons=report["omission_reasons"],
                          outside_chart_candidates_by_voice_reason=
                          report["outside_chart_candidates_by_voice_reason"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
