"""Re-estimate Timing from the SHA-pinned frozen generation analyses.

This reuses the stored selected attacks and beat-model timestamps. It does not
read audio, rerun source selection, or invoke any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.domain import note_from
from manialab.timing import estimate_timing

DEFAULT_PLAN = Path("test-data/frozen-generation-plan-20261117/plan.json")
FROZEN_PLAN_SHA256 = "0696d759598b57ceaaefe3eb52ca3e78f0d443cee60c0816c4cda2d733fad593"
SOURCE_FILES = (
    "manialab/domain.py",
    "manialab/timing.py",
    "manialab/timing_fitting.py",
)
CANDIDATE_FIELDS = (
    "bpm",
    "offset",
    "division",
    "coverage",
    "median_error_ms",
    "score",
    "exact_bpm",
    "rounding_drift_ms",
    "beat_support",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_snapshot() -> dict[str, str]:
    snapshot = {path: sha(ROOT / path) for path in SOURCE_FILES}
    snapshot["scripts/reestimate_frozen_timing.py"] = sha(Path(__file__).resolve())
    return snapshot


def _safe_relative_analysis(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Frozen analysis paths must stay inside the repository")
    resolved = (ROOT / relative).resolve()
    allowed = (ROOT / "test-data" / "audio-choice-pairs").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError("Frozen analyses must come from audio-choice-pairs") from error
    return resolved


def _group_cases(plan: dict) -> list[dict]:
    if plan.get("schema") != "manialab-frozen-generation-v1" or not plan.get("cases"):
        raise ValueError("Expected a nonempty frozen generation plan")
    groups = {}
    case_ids = set()
    for case in plan["cases"]:
        case_id = case.get("id")
        if (not isinstance(case_id, str) or not case_id
                or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in case_id)
                or case_id in case_ids):
            raise ValueError("Case IDs must be unique simple names")
        case_ids.add(case_id)
        analysis_path = _safe_relative_analysis(case["analysis"])
        expected_hash = case.get("analysis_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("Frozen analysis SHA-256 is missing or invalid: " + case_id)
        key = analysis_path.as_posix()
        group = groups.setdefault(key, dict(path=analysis_path, sha256=expected_hash, cases=[]))
        if group["sha256"] != expected_hash:
            raise ValueError("Cases disagree on the frozen analysis identity: " + case_id)
        if group["cases"] and case.get("tempo") != group["cases"][0].get("tempo"):
            raise ValueError("Cases sharing an analysis must use the same frozen clock")
        group["cases"].append(case)
    return list(groups.values())


def _candidate_summary(candidate: dict) -> dict:
    return {key: candidate[key] for key in CANDIDATE_FIELDS if key in candidate}


def _chart_offset(audio_offset: float, audio_shift_ms: float) -> int:
    # Match chart_clock_notes and the UI's Math.round behavior at half values.
    return math.floor(audio_offset - audio_shift_ms + 0.5)


def _tempo_summary(tempo: dict) -> dict:
    return {key: tempo.get(key) for key in (
        "bpm", "offset", "division", "stable", "coverage", "median_error_ms",
        "confidence", "evidence_source", "exact_bpm", "rounding_drift_ms",
    ) if key in tempo}


def _reestimate(group: dict) -> dict:
    path = group["path"]
    if sha(path) != group["sha256"]:
        raise ValueError("Frozen analysis changed: " + str(path.relative_to(ROOT)))
    analysis = read(path)
    attacks_data = analysis.get("timing_attacks")
    if not isinstance(attacks_data, list):
        raise ValueError("Frozen analysis is missing timing_attacks: " + str(path.relative_to(ROOT)))
    duration = analysis.get("duration")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Frozen analysis has an invalid duration: " + str(path.relative_to(ROOT)))
    saved_tempo = analysis.get("tempo")
    if not isinstance(saved_tempo, dict):
        raise ValueError("Frozen analysis is missing its selected tempo: " + str(path.relative_to(ROOT)))
    case_tempo = group["cases"][0].get("tempo")
    if not isinstance(case_tempo, dict):
        raise ValueError("Frozen plan is missing its generation clock")
    for case in group["cases"][1:]:
        if case.get("tempo") != case_tempo:
            raise ValueError("Cases sharing an analysis must use the same frozen clock")

    separation = analysis.get("separation") or {}
    beat_evidence = separation.get("beat_evidence") or {}
    beats_ms = beat_evidence.get("beats_ms")
    if beats_ms is not None and not isinstance(beats_ms, list):
        raise ValueError("Frozen beat evidence must be a list of timestamps")
    attacks = [note_from(note) for note in attacks_data]
    result = estimate_timing(attacks, float(duration), beats_ms)
    audio_shift = float(case_tempo.get("audio_shift_ms", 0))
    estimated_chart_offset = _chart_offset(float(result["offset"]), audio_shift)
    saved_analysis_offset = float(saved_tempo["offset"])
    plan_offset = float(case_tempo["offset"])

    return dict(
        analysis=path.relative_to(ROOT).as_posix(),
        analysis_sha256=group["sha256"],
        case_ids=[case["id"] for case in group["cases"]],
        attack_count=len(attacks),
        duration_seconds=float(duration),
        frozen=dict(
            analysis_tempo=_tempo_summary(saved_tempo),
            plan_clock={key: case_tempo.get(key) for key in (
                "bpm", "offset", "division", "audio_shift_ms", "stable",
            )},
            beat_evidence=dict(
                used=bool(beats_ms),
                timestamp_count=len(beats_ms or []),
                model_sha256=beat_evidence.get("model_sha256"),
            ),
        ),
        reestimated=dict(
            bpm=result["bpm"],
            offset=result["offset"],
            chart_offset=estimated_chart_offset,
            division=result.get("division", 4),
            stable=result["stable"],
            coverage=result.get("coverage"),
            median_error_ms=result.get("median_error_ms"),
            confidence=result.get("confidence"),
            tempo_ambiguity=result.get("tempo_ambiguity"),
            sparse_windows=result.get("sparse_windows", []),
            candidates=[_candidate_summary(candidate) for candidate in result.get("candidates", [])],
            precise_candidates=[
                _candidate_summary(candidate) for candidate in result.get("precise_candidates", [])
            ],
        ),
        delta=dict(
            bpm_from_frozen_analysis=round(float(result["bpm"]) - float(saved_tempo["bpm"]), 6),
            audio_offset_ms_from_frozen_analysis=round(float(result["offset"]) - saved_analysis_offset, 3),
            chart_offset_ms_from_plan=estimated_chart_offset - plan_offset,
            division_matches_analysis=result.get("division", 4) == saved_tempo.get("division", 4),
            stable_matches_analysis=result["stable"] == saved_tempo.get("stable"),
        ),
    )


def _validate_output_path(output: Path) -> Path:
    resolved = output if output.is_absolute() else ROOT / output
    resolved = resolved.resolve()
    expected_parent = (ROOT / "test-data").resolve()
    if resolved.parent != expected_parent or not resolved.name.startswith("frozen-timing-reestimate-"):
        raise ValueError("Output must be a new frozen-timing-reestimate-* directory directly under test-data")
    if resolved.exists():
        raise FileExistsError("Refusing to overwrite existing output: " + str(resolved))
    return resolved


def run(plan_path: Path, output: Path) -> dict:
    plan_path = plan_path if plan_path.is_absolute() else ROOT / plan_path
    plan_path = plan_path.resolve()
    try:
        plan_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Frozen plan must be inside the repository") from error
    output = _validate_output_path(output)
    plan_hash = sha(plan_path)
    if plan_hash != FROZEN_PLAN_SHA256:
        raise ValueError("Frozen plan SHA-256 does not match the regression protocol")
    plan = read(plan_path)
    groups = _group_cases(plan)
    sources_before = source_snapshot()
    input_hashes = {str(group["path"]): group["sha256"] for group in groups}

    rows = []
    for group in groups:
        row = _reestimate(group)
        rows.append(row)
        print(json.dumps({
            "id": row["case_ids"][0],
            "bpm": row["reestimated"]["bpm"],
            "offset": row["reestimated"]["offset"],
            "chart_offset": row["reestimated"]["chart_offset"],
            "division": row["reestimated"]["division"],
            "stable": row["reestimated"]["stable"],
            "delta": row["delta"],
        }, ensure_ascii=True), flush=True)

    if (source_snapshot() != sources_before or sha(plan_path) != plan_hash
            or any(sha(Path(path)) != digest for path, digest in input_hashes.items())):
        raise ValueError("Source, plan, or frozen analysis changed during re-estimation")

    report = dict(
        schema="manialab-frozen-timing-reestimate-v1",
        plan_sha256=plan_hash,
        source_files=sources_before,
        unique_analyses=len(rows),
        plan_cases=len(plan["cases"]),
        cases=rows,
        runtime=dict(
            python=platform.python_version(),
            numpy=numpy.__version__,
            scipy=scipy.__version__,
            threads={key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            )},
        ),
        limitations=dict(
            audio_read=False,
            reference_chart_read=False,
            beat_model_inference=False,
            frozen_beat_evidence_used=True,
            selected_timing_attacks_reused=True,
            drum_mix_source_selection_replayed=False,
            musical_acceptance=False,
            independent_generalization=False,
            note_level_contents_omitted=True,
        ),
    )
    output.mkdir(parents=False, exist_ok=False)
    write_new(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.plan, args.output)
    print(json.dumps({
        "unique_analyses": report["unique_analyses"],
        "plan_cases": report["plan_cases"],
        "report": str(args.output / "report.json"),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
