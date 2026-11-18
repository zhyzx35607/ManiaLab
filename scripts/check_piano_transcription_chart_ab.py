"""Compare cached Kong and Transkun piano attacks through chart generation."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manialab.arrange import assign_lanes, layout_metrics, parameters
from manialab.domain import export_osu, note_from, parse_osu, validate
from manialab.expression import (
    expression_evidence,
    select_expression,
    signature as expression_signature,
    validate_expression,
)
from manialab.timing import align_candidates, chart_clock_notes, grid_report


DEFAULT_INPUT = Path("test-data/sakura-benchmark/piano-full-pipeline-v1")
DEFAULT_TRANSKUN = Path("test-data/piano-transkun-music-v1/sakura-nova-piano-result.json")
DEFAULT_PRESERVED = Path("test-data/piano-onset-preservation-v1/music-sakura-nova-piano-result.json")
COMMON_PIANO_STRENGTH = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _onset_only_note(note: dict, *, identity: str, strength: float) -> dict:
    start = note.get("start")
    pitch = note.get("pitch")
    if type(start) is not int or start < 0:
        raise ValueError("Piano onset must be a nonnegative integer millisecond")
    if type(pitch) is not int or not 0 <= pitch <= 127:
        raise ValueError("Piano pitch must be a MIDI note number")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Common piano strength must be between 0 and 1")
    return {
        "start": start,
        "end": start,
        "lane": -1,
        "pitch": pitch,
        "strength": strength,
        "source": "pitch",
        "id": identity,
        "raw": "",
        "voice": "piano",
        "phrase": False,
        "support": [],
        "support_coverage": 0.0,
        "evidence_start": None,
        "evidence_end": None,
        "release_reason": "unresolved_or_postprocessed",
    }


def normalize_kong_attacks(candidates: list[dict], *, strength: float = COMMON_PIANO_STRENGTH) -> list[dict]:
    """Keep Kong pitch/onset candidates while neutralizing confidence and releases."""
    result = []
    for index, candidate in enumerate(candidates):
        if candidate.get("voice") != "piano":
            continue
        note = _onset_only_note(
            candidate,
            identity=candidate["id"],
            strength=strength,
        )
        result.append(note)
    if not result:
        raise ValueError("No Kong piano candidates found")
    return result


def transkun_attack_candidates(events: list[dict], *, prefix: str, strength: float = COMMON_PIANO_STRENGTH) -> list[dict]:
    """Convert Transkun events using onset/pitch only; ignore velocity and offset."""
    if not isinstance(events, list) or not events:
        raise ValueError("Transkun piano events must be a nonempty list")
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("Transkun piano events must be objects")
    indexed = list(enumerate(events))
    indexed.sort(key=lambda row: (row[1].get("onset", -1), row[1].get("pitch", -1), row[0]))
    result = []
    for _, event in indexed:
        if not isinstance(event, dict) or event.get("has_onset") is not True:
            raise ValueError("Every selected Transkun event must carry an onset")
        onset = event.get("onset")
        if isinstance(onset, bool) or not isinstance(onset, (int, float)) or not math.isfinite(onset) or onset < 0:
            raise ValueError("Transkun onset must be a finite nonnegative number of seconds")
        pitch = event.get("pitch")
        if type(pitch) is not int or not 0 <= pitch <= 127:
            raise ValueError("Transkun pitch must be a MIDI note number")
        start = math.floor(onset * 1000 + 0.5)
        result.append(
            _onset_only_note(
                {"start": start, "pitch": pitch},
                identity=f"piano:{prefix}:{len(result):05d}",
                strength=strength,
            )
        )
    return result


def replace_piano_candidates(analysis: dict, piano_candidates: list[dict]) -> dict:
    result = copy.deepcopy(analysis)
    result["candidates"] = sorted(
        [candidate for candidate in result["candidates"] if candidate.get("voice") != "piano"]
        + piano_candidates,
        key=lambda row: (row["start"], row["pitch"], row["voice"], row["id"]),
    )
    return result


def _counter(values) -> dict:
    return dict(sorted(Counter(values).items()))


def _event_key(note) -> tuple:
    return (note.voice, note.pitch, note.start, note.end)


def compare_to_reference_chart(reference: list[dict], output: list[dict]) -> dict:
    """Measure event-set changes only; no authored chart is used as truth."""
    reference_counts, output_counts = Counter(map(_event_key, reference)), Counter(map(_event_key, output))
    added = output_counts - reference_counts
    removed = reference_counts - output_counts
    by_voice_added = Counter(key[0] for key, count in added.items() for _ in range(count))
    by_voice_removed = Counter(key[0] for key, count in removed.items() for _ in range(count))
    return {
        "identical_events": sum((reference_counts & output_counts).values()),
        "added_events": sum(added.values()),
        "removed_events": sum(removed.values()),
        "added_by_voice": dict(sorted(by_voice_added.items())),
        "removed_by_voice": dict(sorted(by_voice_removed.items())),
    }


def run_case(
    *,
    name: str,
    analysis: dict,
    frozen_plan: dict,
    settings: dict,
    clock: dict,
    output: Path,
    audio_reference: str,
) -> dict:
    begin = time.monotonic()
    plan = copy.deepcopy(frozen_plan)
    plan["evidence_signature"] = expression_signature(analysis)
    validate_expression(plan, analysis)
    evidence, metadata, decisions = expression_evidence(analysis, plan, settings)
    aligned, alignment = align_candidates(
        chart_clock_notes(evidence, clock["audio_shift_ms"]),
        clock,
        confirmed=True,
    )
    selected, selection = select_expression(aligned, metadata, decisions, settings, 60000 / clock["bpm"])
    placed = assign_lanes(selected, settings, beat_ms=60000 / clock["bpm"])
    structure = validate(placed, settings["keys"])
    grid = grid_report(placed, clock)
    if not structure["valid"] or grid["off_grid"] or len(placed) != len(selected):
        raise ValueError(f"Generated chart failed structural checks: {name}")
    if any(note.voice == "piano" and note.hold for note in placed):
        raise ValueError(f"Onset-only piano candidates produced an LN: {name}")
    content = export_osu(
        placed,
        settings["keys"],
        ROOT / "vendor/mapperatorinator/template.osu",
        title="Sakura piano onset source comparison",
        audio=audio_reference,
        bpm=clock["bpm"],
        offset=clock["offset"],
        version=f"{name} research",
        grid_division=clock["division"],
    )
    parsed = parse_osu(content)
    if parsed["keys"] != settings["keys"] or not validate(parsed["notes"], settings["keys"])["valid"]:
        raise ValueError(f"Exported chart failed read-back: {name}")

    chart_path = output / f"{name}.osu"
    notes_path = output / f"{name}-notes.json"
    selection_path = output / f"{name}-selection.json"
    chart_path.write_text(content, encoding="utf-8")
    notes_path.write_text(
        json.dumps([note.data() for note in placed], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "name": name,
        "keys": settings["keys"],
        "elapsed_seconds": round(time.monotonic() - begin, 3),
        "candidate_counts": _counter(row["voice"] for row in analysis["candidates"]),
        "selected_counts": _counter(note.voice for note in placed),
        "selected_attack_groups": _counter(
            voice for voice, _ in {(note.voice, note.start) for note in placed}
        ),
        "selected_holds": _counter(note.voice for note in placed if note.hold),
        "alignment": alignment,
        "structure": structure,
        "grid": grid,
        "layout": layout_metrics(placed, settings["keys"], 60000 / clock["bpm"]),
        "selection_optimal": selection["attack_optimal"],
        "omission_reasons": selection["omission_reasons"],
        "chart": chart_path.name,
        "notes": notes_path.name,
        "selection": selection_path.name,
        "audio_reference": audio_reference,
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    cli.add_argument("--transkun", type=Path, default=DEFAULT_TRANSKUN)
    cli.add_argument("--preserved", type=Path, default=DEFAULT_PRESERVED)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError("Preserve completed runs by choosing a new output directory")

    read = read_json
    baseline_report = read(args.input / "report.json")
    if baseline_report.get("reference_chart_read") is not False:
        raise ValueError("The cached source analysis must not have read the reference chart")
    analysis = read(args.input / "specialist-analysis.json")
    frozen_plan = read(args.input / "specialist-expression.json")
    transkun = read(args.transkun)
    preserved = read(args.preserved)
    if transkun.get("kind") != "nova" or transkun.get("trace", {}).get("release_truth_available") is not False:
        raise ValueError("Expected the cached Nova Transkun run with unverified release times")
    if preserved.get("scope") != "music" or preserved.get("name") != transkun.get("name"):
        raise ValueError("Preserved-onset report does not match the Transkun music case")
    if preserved.get("nova", {}).get("acoustic_offset_truth_available") is not False:
        raise ValueError("Preserved Transkun offsets must remain unverified")

    sections = (analysis.get("structure") or {}).get("segments", [])
    if not sections or any(
        section.get("voices", {}).get("piano", {}).get("mode") != "onsets"
        for section in frozen_plan.get("sections", [])
    ):
        raise ValueError("Frozen expression plan must use onset-only piano mode in every section")
    if len(sections) != len(frozen_plan.get("sections", [])):
        raise ValueError("Frozen plan and source analysis have different section counts")
    for section, policy in zip(sections, frozen_plan["sections"]):
        if any(section.get(key) != policy.get(key) for key in ("start_ms", "end_ms")):
            raise ValueError("Frozen plan boundaries differ from the source analysis")

    original_piano = [row for row in analysis["candidates"] if row.get("voice") == "piano"]
    if len(original_piano) != 1557:
        raise ValueError("Unexpected cached Kong piano candidate count")
    transkun_events = transkun["trace"]["piano_events"]
    official_piano = transkun_attack_candidates(transkun_events, prefix="transkun")
    preserved_events = [
        row["note"]
        for row in preserved["evidence"]["candidates"]
        if row.get("note", {}).get("has_onset") is True
    ]
    preserved_piano = transkun_attack_candidates(preserved_events, prefix="transkun-preserved")
    if len(official_piano) != 1275 or len(preserved_piano) != 1277:
        raise ValueError("Unexpected Transkun candidate counts")
    if len(preserved_piano) - len(official_piano) != 2:
        raise ValueError("Expected two preserved onset proposals")

    output = args.output.resolve()
    output.mkdir(parents=True)
    audio_reference = Path(os.path.relpath((args.input / "audio.mp3").resolve(), output)).as_posix()
    clock = baseline_report["identity"]["generation_clock"]
    settings_base = baseline_report["identity"]["settings"]
    source_sha = {
        "baseline_report": sha256(args.input / "report.json"),
        "kong_analysis": sha256(args.input / "specialist-analysis.json"),
        "frozen_expression": sha256(args.input / "specialist-expression.json"),
        "transkun_result": sha256(args.transkun),
        "preserved_onset_result": sha256(args.preserved),
        "audio": sha256(args.input / "audio.mp3"),
        "runner": sha256(Path(__file__)),
    }
    common_non_piano_sha = hashlib.sha256(
        json.dumps(
            [row for row in analysis["candidates"] if row.get("voice") != "piano"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    variants = {
        "kong": normalize_kong_attacks(original_piano),
        "transkun": official_piano,
        "transkun-preserved": preserved_piano,
    }
    rows = []
    for keys in range(6, 11):
        settings = parameters(settings_base | {"keys": keys})
        for variant, piano_candidates in variants.items():
            current = replace_piano_candidates(analysis, piano_candidates)
            row = run_case(
                name=f"{variant}-{keys}k",
                analysis=current,
                frozen_plan=frozen_plan,
                settings=settings,
                clock=clock,
                output=output,
                audio_reference=audio_reference,
            )
            row["piano_input_candidates"] = len(piano_candidates)
            row["common_non_piano_candidates_sha256"] = common_non_piano_sha
            rows.append(row)
            (output / "progress.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps({k: row[k] for k in (
                "name", "selected_counts", "selected_holds", "layout", "elapsed_seconds"
            )}, ensure_ascii=False), flush=True)

    baseline_by_key = {row["keys"]: row for row in rows if row["name"].startswith("kong-")}
    for row in rows:
        if row["name"].startswith("kong-"):
            continue
        key = baseline_by_key[row["keys"]]["name"]
        reference_notes = json.loads((output / f"{key}-notes.json").read_text(encoding="utf-8"))
        candidate_notes = json.loads((output / f"{row['name']}-notes.json").read_text(encoding="utf-8"))
        row["delta_vs_kong"] = compare_to_reference_chart(
            [note_from(note) for note in reference_notes],
            [note_from(note) for note in candidate_notes],
        )

    report = {
        "identity": {
            "source_sha256": source_sha,
            "common_non_piano_candidates_sha256": common_non_piano_sha,
            "generation_clock": clock,
            "keys": [6, 7, 8, 9, 10],
            "settings_base": settings_base,
            "common_piano_strength": COMMON_PIANO_STRENGTH,
            "common_expression_policy_sha256": hashlib.sha256(
                json.dumps(
                    frozen_plan["sections"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "transkun_official_candidates": len(official_piano),
            "transkun_preserved_candidates": len(preserved_piano),
            "preserved_extra_onset_proposals": len(preserved_piano) - len(official_piano),
            "transkun_velocity_used_as_confidence": False,
            "transkun_offsets_used_for_ln": False,
            "piano_ln_enabled": False,
            "reference_chart_read": False,
            "audio_copied": False,
            "application_enabled": False,
            "direct_listening_performed": False,
            "gameplay_acceptance": False,
            "musical_acceptance": False,
        },
        "cases": rows,
        "notice": (
            "Same cached non-piano evidence, expression policy, manually confirmed clock, "
            "and generation settings. All piano events are reduced to onset-only candidates "
            "with common synthetic strength 0.5; model velocity/confidence and note offsets "
            "are ignored. Layout metrics and authored-chart differences are diagnostics, not "
            "evidence of acoustic truth, musical quality, LN quality, or playability."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
