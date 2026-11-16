"""Explain stored Sakura tempo aliases using one frozen timing snapshot.

This does not estimate BPM or offset. It reuses the frozen source candidates,
then measures their fixed grids against the selected timing attacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manialab.domain import note_from
from manialab.timing import grid_diagnostics

INPUT_PATH = Path("test-data/sakura-tsumugi-expression-onset-v3/analysis.json")
EXPECTED_INPUT_SHA256 = "26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a"
TEMPO_FAMILY = (74.0, 148.0, 296.0)
WINDOW_MS = 12_000


def _wrapped_delta_ms(left: float, right: float, period_ms: float) -> float:
    return (left - right + period_ms / 2) % period_ms - period_ms / 2


def _grid_coverage(times: np.ndarray, weights: np.ndarray, offset: float, unit_ms: float) -> dict:
    tolerance = min(25.0, unit_ms * 0.2)
    distance = np.abs((times - offset + unit_ms / 2) % unit_ms - unit_ms / 2)
    return {
        "weighted_coverage": round(float(np.average(distance < tolerance, weights=weights)), 4),
        "tolerance_ms": round(tolerance, 3),
    }


def _window_summary(sections: list[dict]) -> dict:
    if not sections:
        return {
            "window_count": 0,
            "suspect_window_count": 0,
            "minimum_coverage": None,
            "maximum_absolute_phase_error_ms": None,
            "suspect_windows": [],
        }
    suspect = [section for section in sections if section.get("suspect")]
    return {
        "window_count": len(sections),
        "suspect_window_count": len(suspect),
        "minimum_coverage": min(section["coverage"] for section in sections),
        "maximum_absolute_phase_error_ms": round(
            max(abs(section["phase_error_ms"]) for section in sections), 2
        ),
        "suspect_windows": [
            {
                "start_ms": section["start_ms"],
                "end_ms": section["end_ms"],
                "attacks": section["attacks"],
                "coverage": section["coverage"],
                "phase_error_ms": section["phase_error_ms"],
            }
            for section in suspect
        ],
    }


def _candidate_by_bpm(source: dict, bpm: float) -> dict | None:
    matches = [candidate for candidate in source["candidates"] if abs(candidate["bpm"] - bpm) < 0.01]
    if len(matches) > 1:
        raise ValueError(f"Duplicate {bpm:g} BPM candidates")
    return matches[0] if matches else None


def build_report(analysis: dict, input_sha256: str) -> dict:
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError("Frozen analysis SHA-256 does not match the protocol")
    duration = analysis.get("duration")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Analysis duration must be a positive finite number")

    attacks_data = analysis.get("timing_attacks")
    if not isinstance(attacks_data, list) or len(attacks_data) < 8:
        raise ValueError("Frozen timing_attacks are missing or too sparse")
    attacks = [note_from(value) for value in attacks_data]
    if any(attacks[index].start > attacks[index + 1].start for index in range(len(attacks) - 1)):
        raise ValueError("Frozen timing_attacks must be sorted by start time")
    times = np.asarray([attack.start for attack in attacks], dtype=float)
    weights = np.asarray([max(0.05, attack.strength) for attack in attacks], dtype=float)

    tempo = analysis.get("tempo")
    if not isinstance(tempo, dict):
        raise ValueError("Frozen tempo data is missing")
    selected_source = tempo.get("evidence_source")
    comparisons = tempo.get("comparisons")
    if not isinstance(selected_source, str) or not isinstance(comparisons, dict):
        raise ValueError("Frozen tempo comparisons or selected evidence source are missing")
    if selected_source not in comparisons:
        raise ValueError("Selected evidence source has no frozen comparison")
    selected_comparison = comparisons[selected_source]

    selected_candidates = {}
    for bpm in TEMPO_FAMILY:
        candidate = _candidate_by_bpm(selected_comparison, bpm)
        if candidate is None:
            raise ValueError(f"Frozen selected-source comparison is missing {bpm:g} BPM")
        if candidate.get("division", 0) <= 0:
            raise ValueError(f"Invalid subdivision at {bpm:g} BPM")
        selected_candidates[bpm] = candidate

    grid_hypotheses = []
    for bpm, candidate in selected_candidates.items():
        period = 60_000 / candidate["bpm"]
        tick = period / candidate["division"]
        beat = _grid_coverage(times, weights, candidate["offset"], period)
        half_beat = _grid_coverage(times, weights, candidate["offset"], period / 2)
        subdivision = _grid_coverage(times, weights, candidate["offset"], tick)
        sections = grid_diagnostics(
            attacks,
            candidate["bpm"],
            candidate["offset"],
            candidate["division"],
        )
        if abs(subdivision["weighted_coverage"] - candidate["coverage"]) > 0.002:
            raise ValueError(f"Recomputed {bpm:g} BPM subdivision coverage differs from snapshot")
        grid_hypotheses.append(
            {
                "bpm": candidate["bpm"],
                "beat_period_ms": round(period, 3),
                "offset_ms": candidate["offset"],
                "division": candidate["division"],
                "score": candidate["score"],
                "subdivision_coverage": subdivision["weighted_coverage"],
                "beat_coverage": beat["weighted_coverage"],
                "half_beat_coverage": half_beat["weighted_coverage"],
                "median_error_ms": candidate["median_error_ms"],
                "windows": _window_summary(sections),
            }
        )

    hypotheses_by_bpm = {row["bpm"]: row for row in grid_hypotheses}
    alias_relationships = []
    for slower_bpm, faster_bpm in ((74.0, 148.0), (148.0, 296.0)):
        slower = selected_candidates[slower_bpm]
        faster = selected_candidates[faster_bpm]
        faster_period = 60_000 / faster_bpm
        phase_delta = _wrapped_delta_ms(
            faster["offset"], slower["offset"], faster_period
        )
        slower_grid = hypotheses_by_bpm[slower_bpm]
        faster_grid = hypotheses_by_bpm[faster_bpm]
        alias_relationships.append(
            {
                "slower_bpm": slower_bpm,
                "faster_bpm": faster_bpm,
                "period_ratio": round((60_000 / slower_bpm) / faster_period, 4),
                "offset_phase_delta_mod_faster_beat_ms": round(phase_delta, 2),
                "slower_half_beat_coverage": slower_grid["half_beat_coverage"],
                "faster_beat_coverage": faster_grid["beat_coverage"],
                "coverage_difference": round(
                    abs(
                        slower_grid["half_beat_coverage"]
                        - faster_grid["beat_coverage"]
                    ),
                    4,
                ),
            }
        )

    source_summaries = {}
    for source_name, source in comparisons.items():
        if not isinstance(source, dict) or not isinstance(source.get("candidates"), list):
            raise ValueError(f"Invalid frozen comparison for source {source_name}")
        source_candidate_rows = []
        for bpm in TEMPO_FAMILY:
            candidate = _candidate_by_bpm(source, bpm)
            if candidate is None:
                continue
            source_candidate_rows.append(
                {
                    "bpm": candidate["bpm"],
                    "offset_ms": candidate["offset"],
                    "division": candidate["division"],
                    "score": candidate["score"],
                    "coverage": candidate["coverage"],
                    "median_error_ms": candidate["median_error_ms"],
                }
            )
        source_summaries[source_name] = {
            "selected": {
                key: source.get(key)
                for key in ("bpm", "offset", "division", "score", "coverage", "median_error_ms", "stable")
            },
            "stored_windows": _window_summary(source.get("sections", [])),
            "tempo_family_candidates": source_candidate_rows,
        }

    cross_source_phase = []
    for bpm in TEMPO_FAMILY:
        drum_candidate = _candidate_by_bpm(comparisons.get("drums", {}), bpm)
        mix_candidate = _candidate_by_bpm(comparisons.get("mix", {}), bpm)
        if drum_candidate is None or mix_candidate is None:
            continue
        period = 60_000 / bpm
        cross_source_phase.append(
            {
                "bpm": bpm,
                "drums_offset_ms": drum_candidate["offset"],
                "mix_offset_ms": mix_candidate["offset"],
                "wrapped_phase_difference_ms": round(
                    _wrapped_delta_ms(mix_candidate["offset"], drum_candidate["offset"], period),
                    2,
                ),
            }
        )

    return {
        "schema_version": 1,
        "protocol": (
            "Read only the SHA-pinned analysis JSON. Reuse its stored tempo candidates "
            "and compare their fixed grids to the selected timing_attacks; do not estimate "
            "BPM/Offset or read audio, charts, or reference notes."
        ),
        "input": {
            "path": INPUT_PATH.as_posix(),
            "sha256": input_sha256,
            "duration_ms": round(duration * 1000),
            "timing_attack_count": len(attacks),
            "timing_attack_source": selected_source,
        },
        "selected_timing": {
            key: tempo.get(key)
            for key in ("bpm", "offset", "division", "score", "coverage", "median_error_ms", "stable")
        },
        "source_comparisons": source_summaries,
        "selected_attack_grid_hypotheses": grid_hypotheses,
        "half_double_relationships": alias_relationships,
        "cross_source_phase": cross_source_phase,
        "interpretation_limits": {
            "physical_offset_verified": False,
            "metrical_beat_identified": False,
            "music_quality_evaluated": False,
            "stable_gate_changed": False,
            "note": (
                "Grid coverage measures proximity of detected attacks to candidate beat or "
                "subdivision grids. It cannot identify strong beats, prove physical audio "
                "timing, or establish which half/double-time interpretation a listener hears."
            ),
        },
    }


def load_frozen_analysis(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_INPUT_SHA256:
        raise ValueError(f"Unexpected frozen analysis SHA-256: {digest}")
    return json.loads(raw.decode("utf-8")), digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / INPUT_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "test-data/sakura-timing-ambiguity-v1-20261116",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {args.output_dir}")
    analysis, input_sha256 = load_frozen_analysis(args.input)
    report = build_report(analysis, input_sha256)
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["timing_module_sha256"] = hashlib.sha256((ROOT / "manialab/timing.py").read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "selected": report["selected_timing"],
        "suspect_windows_by_grid": {
            str(row["bpm"]): row["windows"]["suspect_window_count"]
            for row in report["selected_attack_grid_hypotheses"]
        },
        "cross_source_phase": report["cross_source_phase"],
        "output": output.as_posix(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
