"""Compare the sequence prior on complete held-out mania charts.

This is a fixed-event lane-layout diagnostic. It does not measure audio
transcription, LN-duration quality, or whether a layout feels good to play.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from manialab.arrange import assign_lanes, layout_metrics, parameters
from manialab.layout import SEQUENCE_SCORE_CANDIDATE_CAP
from manialab.domain import export_osu, parse_osu, validate
from manialab.ln_patterns import analyze_ln_patterns
from manialab.sequence_prior import load_sequence_prior

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = Path("test-data/sequence-prior-v1/report.json")
DEFAULT_MODEL = Path("test-data/sequence-prior-v1/model.json")
DEFAULT_MAPS = Path("test-data/corpus-expanded/snapshot-verified/maps")
DEFAULT_OUTPUT = Path("test-data/sequence-prior-remap-v4/report.json")
METRIC_NAMES = (
    "cost_per_note",
    "short_jacks_per_100",
    "release_repeats_per_100",
    "lane_change_rate",
    "ln_reference_lane_change_rate",
    "ln_lane_entropy",
    "ln_inverse_profile_error",
    "ln_reverse_shield_profile_error",
    "ln_release_only_profile_error",
    "ln_split_release_profile_error",
    "ln_release_press_profile_error",
    "same_hand_hybrid_profile_error",
    "ln_occupied_key_fraction",
)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_maps(report, keys, split):
    """Select exactly one frozen group split and reject ambiguous identities."""
    if split not in ("validation", "test"):
        raise ValueError("Full-chart evaluation only accepts validation or test")
    maps = report.get("prep", {}).get("sampling_maps")
    if not isinstance(maps, list):
        raise ValueError("Training report has no sampling map manifest")

    selected = [
        row for row in maps
        if row.get("keys") == keys and row.get("split") == split
    ]
    if not selected:
        raise ValueError(f"No {split} maps found for {keys}K")

    seen = set()
    for row in selected:
        beatmap_id = row.get("id")
        if (isinstance(beatmap_id, bool) or not isinstance(beatmap_id, int)
                or beatmap_id < 0 or beatmap_id in seen):
            raise ValueError(f"Invalid or duplicate map identity: {beatmap_id!r}")
        if (not isinstance(row.get("sha256"), str)
                or len(row["sha256"]) != 64
                or not isinstance(row.get("group"), str)
                or not row["group"]):
            raise ValueError(f"Map {beatmap_id} has incomplete frozen identity")
        seen.add(beatmap_id)
    return sorted(selected, key=lambda row: (row["group"], row["id"]))


def _non_lane_signature(notes):
    return Counter(tuple(note.raw.split(",")[1:]) for note in notes)


def _object_lane_signature(notes):
    return Counter(
        (tuple(note.raw.split(",")[1:]), note.lane)
        for note in notes
    )


def _ln_lane_entropy(notes, keys):
    counts = np.bincount(
        [note.lane for note in notes if note.hold], minlength=keys
    ).astype(np.float64)
    total = counts.sum()
    if not total:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum()
                 / math.log(keys))


def _validate_roundtrip(notes, keys, original_text, original_chart, template):
    exported = export_osu(
        notes, keys, template, original=original_text
    )
    reread = parse_osu(exported)
    if reread["timing"] != original_chart["timing"]:
        raise AssertionError("TimingPoints changed during .osu round-trip")
    if _non_lane_signature(reread["notes"]) != _non_lane_signature(
            original_chart["notes"]):
        raise AssertionError("Non-lane hit-object fields changed on export")
    if Counter((note.start, note.end) for note in reread["notes"]) != Counter(
            (note.start, note.end) for note in original_chart["notes"]):
        raise AssertionError("Hit-object times or LN endpoints changed on export")
    if _object_lane_signature(reread["notes"]) != _object_lane_signature(notes):
        raise AssertionError("Exported lane coordinates differ from the layout")
    validation = validate(reread["notes"], keys)
    if not validation["valid"]:
        raise AssertionError(validation["issues"][:3])
    return validation


def _chart_metrics(notes, original_notes, keys, timing_points, baseline_notes,
                   seconds, roundtrip):
    ergonomic = layout_metrics(notes, keys, timing_points=timing_points)
    count = max(1, len(notes))
    reference_lanes = {note.id: note.lane for note in original_notes}
    baseline_lanes = {note.id: note.lane for note in baseline_notes}
    moved_from_reference = sum(
        note.lane != reference_lanes[note.id] for note in notes
    )
    ln_reference_lane_changes = sum(
        note.hold and note.lane != reference_lanes[note.id] for note in notes
    )
    moved_from_baseline = sum(
        note.lane != baseline_lanes[note.id] for note in notes
    )
    ln_count = sum(note.hold for note in notes)
    tap_count = len(notes) - ln_count
    ln_patterns = analyze_ln_patterns(
        notes, keys, timing_points=timing_points
    )
    reference_patterns = analyze_ln_patterns(
        original_notes, keys, timing_points=timing_points
    )
    pattern_counts = ln_patterns["counts"]
    reference_pattern_counts = reference_patterns["counts"]
    per_100_ln = 100 / max(1, ln_count)
    per_100_tap = 100 / max(1, tap_count)

    def normalized_pattern(counts, name, scale):
        return counts.get(name, 0) * scale

    def profile_error(name, scale):
        return abs(
            normalized_pattern(pattern_counts, name, scale)
            - normalized_pattern(reference_pattern_counts, name, scale)
        )

    return {
        "objects": len(notes),
        "ln": ln_count,
        "ln_ratio": round(ln_count / count, 6),
        "max_held": round(float(roundtrip["max_held"]), 6),
        "lane_counts": list(roundtrip["lane_counts"]),
        "ln_lane_counts": [
            sum(note.hold and note.lane == lane for note in notes)
            for lane in range(keys)
        ],
        "seconds": round(float(seconds), 3),
        "reference_lane_changes": moved_from_reference,
        "reference_ln_lane_changes": ln_reference_lane_changes,
        "baseline_lane_changes": moved_from_baseline,
        "ln_pattern_counts": dict(pattern_counts),
        "reference_ln_pattern_counts": dict(reference_pattern_counts),
        "metrics": {
            "cost_per_note": round(ergonomic["cost"] / count, 6),
            "short_jacks_per_100": round(100 * ergonomic["short_jacks"] / count, 6),
            "release_repeats_per_100": round(
                100 * ergonomic["release_repeats"] / count, 6
            ),
            "lane_change_rate": round(moved_from_reference / count, 6),
            "ln_reference_lane_change_rate": (
                round(ln_reference_lane_changes / ln_count, 6)
                if ln_count else None
            ),
            "ln_lane_entropy": (
                round(_ln_lane_entropy(notes, keys), 6)
                if ln_count else None
            ),
            "ln_inverse_profile_error": round(
                profile_error("inverse_heads", per_100_ln), 6
            ),
            "ln_reverse_shield_profile_error": round(
                profile_error("reverse_shield_heads", per_100_ln), 6
            ),
            "ln_release_only_profile_error": round(
                profile_error("release_only_rows", per_100_ln), 6
            ),
            "ln_split_release_profile_error": round(
                profile_error("split_release_hands", per_100_ln), 6
            ),
            "ln_release_press_profile_error": round(
                profile_error("release_press_hands", per_100_ln), 6
            ),
            "same_hand_hybrid_profile_error": round(
                profile_error("same_hand_hybrid_taps", per_100_tap), 6
            ),
            "ln_occupied_key_fraction": (
                ln_patterns["occupied_key_fraction"] if ln_count else None
            ),
        },
    }


def summarize(rows, *, bootstrap_samples=5000, seed=20261112):
    """Report paired song-group macro metrics separately for each key count."""
    rng = np.random.default_rng(seed)
    grouped = defaultdict(lambda: {"baseline": [], "sequence_prior": []})
    for row in rows:
        if row.get("passed") is False:
            continue
        group = (row["keys"], row["group"])
        for variant in ("baseline", "sequence_prior"):
            grouped[group][variant].append(row[variant]["metrics"])
    if not grouped:
        return {"by_keys": {}}

    result = {}
    for keys in sorted({key for key, _ in grouped}):
        key_groups = {
            group: variants for (key, group), variants in grouped.items()
            if key == keys
        }
        metrics = {}
        for name in METRIC_NAMES:
            group_values = {
                group: {
                    variant: float(np.mean([
                        chart[name] for chart in values[variant]
                        if chart[name] is not None
                    ]))
                    for variant in ("baseline", "sequence_prior")
                    if any(chart[name] is not None for chart in values[variant])
                }
                for group, values in key_groups.items()
            }
            paired = np.asarray([
                values["sequence_prior"] - values["baseline"]
                for values in group_values.values()
                if "baseline" in values and "sequence_prior" in values
            ], dtype=np.float64)
            if not len(paired):
                continue
            samples = rng.integers(
                0, len(paired), size=(bootstrap_samples, len(paired))
            )
            low, high = np.quantile(
                paired[samples].mean(axis=1), [0.025, 0.975]
            )
            baseline = np.mean([
                values["baseline"] for values in group_values.values()
                if "baseline" in values
            ])
            sequence = np.mean([
                values["sequence_prior"] for values in group_values.values()
                if "sequence_prior" in values
            ])
            metrics[name] = {
                "groups": len(paired),
                "baseline_macro": round(float(baseline), 6),
                "sequence_prior_macro": round(float(sequence), 6),
                "delta_prior_minus_baseline": round(float(paired.mean()), 6),
                "delta_ci_95": [round(float(low), 6), round(float(high), 6)],
            }
        result[str(keys)] = {
            "groups": len(key_groups),
            "metrics": metrics,
        }
    return {"by_keys": result}


def _validate_strength(strength):
    if (isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(strength)
            or not 0 <= strength <= 1):
        raise ValueError("strength must be finite and between 0 and 1")


def _validate_frozen_validation(artifact, report, model_sha256,
                                report_sha256, keys, strength, seed):
    if artifact.get("revision") != "sequence-prior-remap-v4":
        raise ValueError("Validation artifact has an unexpected revision")
    if artifact.get("split") != "validation":
        raise ValueError("Frozen strength must come from the validation split")
    if artifact.get("model_sha256") != model_sha256:
        raise ValueError("Frozen validation uses a different model")
    if artifact.get("report_sha256") != report_sha256:
        raise ValueError("Frozen validation uses a different training report")
    if artifact.get("training_manifest_sha256") != report.get(
            "training_manifest_sha256"):
        raise ValueError("Frozen validation uses a different training manifest")
    if artifact.get("keys") != sorted(keys):
        raise ValueError("Frozen validation key counts do not match test keys")
    if artifact.get("seed") != seed:
        raise ValueError("Frozen validation seed does not match test seed")
    if artifact.get("score_candidate_cap") != SEQUENCE_SCORE_CANDIDATE_CAP:
        raise ValueError("Frozen validation uses a different scorer candidate cap")
    try:
        artifact_strength = float(artifact["strength"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Frozen validation has no valid strength") from exc
    if not math.isclose(artifact_strength, strength, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Test strength differs from the validation artifact")
    if artifact.get("selection_limit") is not None:
        raise ValueError("A limited validation run cannot freeze test strength")
    validation_ids = []
    for key_count in keys:
        validation_ids.extend(
            row["id"] for row in select_maps(report, key_count, "validation")
        )
    expected_ids = sorted(validation_ids)
    actual_ids = sorted(artifact.get("selected_map_ids", []))
    rows = artifact.get("rows")
    if (actual_ids != expected_ids or not isinstance(rows, list)
            or len(rows) != len(expected_ids)
            or {row.get("beatmap_id") for row in rows if row.get("passed")}
            != set(expected_ids)):
        raise ValueError("Validation artifact is incomplete or contains failures")


def _write_atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(*, report_path=DEFAULT_REPORT, model_path=DEFAULT_MODEL,
        map_root=DEFAULT_MAPS, output=DEFAULT_OUTPUT, keys=(10,),
        split="validation", strength=0.35, seed=42, limit=None,
        validation_artifact_path=None, resume=False):
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    _validate_strength(strength)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if split == "test" and validation_artifact_path is None:
        raise ValueError(
            "Test split requires --validation-artifact from a complete "
            "validation run"
        )
    keys = tuple(sorted(keys))
    if (not keys or len(set(keys)) != len(keys)
            or any(isinstance(value, bool) or value not in range(6, 11)
                   for value in keys)):
        raise ValueError("keys must be unique values in the 6–10 range")
    report = _read_json(report_path)
    if report.get("revision") != "sequence-window-v1":
        raise ValueError("Unexpected sequence-prior report revision")
    prior = load_sequence_prior(model_path)
    if report.get("training_manifest_sha256") != prior.training_manifest_sha256:
        raise ValueError("Model/report training manifest hash mismatch")
    model_sha256 = prior.sha256
    report_sha256 = _sha256(report_path)
    frozen_validation_sha256 = None
    if split == "test":
        if validation_artifact_path is None:
            raise ValueError(
                "Test split requires --validation-artifact from a complete "
                "validation run"
            )
        frozen_validation_sha256 = _sha256(validation_artifact_path)
        _validate_frozen_validation(
            _read_json(validation_artifact_path),
            report,
            model_sha256,
            report_sha256,
            keys,
            float(strength),
            seed,
        )
    map_root = Path(map_root)
    output = Path(output).resolve()
    local_data = (ROOT / "test-data").resolve()
    if not output.is_relative_to(local_data):
        raise ValueError("Keep full-chart research outputs under test-data")
    if output.exists() and not resume:
        raise FileExistsError(f"Research output already exists: {output}")
    if resume and not output.is_file():
        raise FileNotFoundError(f"No report to resume: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    selected = []
    for key_count in keys:
        if key_count not in range(6, 11):
            raise ValueError("keys must be in the 6–10 range")
        selected.extend(select_maps(report, key_count, split))
    selected.sort(key=lambda row: (row["keys"], row["group"], row["id"]))
    selected_ids = [row["id"] for row in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Selected map IDs must be unique across key counts")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No maps selected for full-chart evaluation")

    template = ROOT / "vendor/mapperatorinator/template.osu"
    selected_map_ids = [entry["id"] for entry in selected]
    selected_groups_by_key = {
        str(key_count): sorted({
            row["group"] for row in selected if row["keys"] == key_count
        })
        for key_count in keys
    }
    identity = {
        "split": split,
        "keys": list(keys),
        "strength": float(strength),
        "seed": seed,
        "score_candidate_cap": SEQUENCE_SCORE_CANDIDATE_CAP,
        "model_sha256": model_sha256,
        "training_manifest_sha256": prior.training_manifest_sha256,
        "report_sha256": report_sha256,
        "validation_artifact_sha256": frozen_validation_sha256,
        "selection_limit": limit,
        "selected_map_ids": selected_map_ids,
    }
    if resume:
        result = _read_json(output)
        if result.get("run_identity") != identity:
            raise ValueError("Existing report does not match this run configuration")
    else:
        result = {
            "revision": "sequence-prior-remap-v4",
            "notice": (
                "Fixed-event lane-layout diagnostic only; structural metrics and "
                "reference similarity do not establish musical or play quality."
            ),
            "split": split,
            "keys": list(keys),
            "strength": float(strength),
            "strength_frozen_on_validation": split == "test",
            "seed": int(seed),
            "score_candidate_cap": SEQUENCE_SCORE_CANDIDATE_CAP,
            "model_sha256": model_sha256,
            "training_manifest_sha256": prior.training_manifest_sha256,
            "report_sha256": report_sha256,
            "validation_artifact_sha256": frozen_validation_sha256,
            "selection_limit": limit,
            "selected_maps": len(selected),
            "selected_map_ids": selected_map_ids,
            "selected_groups_by_key": selected_groups_by_key,
            "run_identity": identity,
            "status": "running",
            "rows": [],
        }
        _write_atomic(output, result)

    existing_rows = {
        row["beatmap_id"]: row for row in result.get("rows", [])
        if isinstance(row.get("beatmap_id"), int)
    }
    rows = list(existing_rows.values())

    for index, entry in enumerate(selected, start=1):
        beatmap_id = entry["id"]
        key_count = entry["keys"]
        previous = existing_rows.get(beatmap_id)
        if previous is not None and previous.get("passed"):
            continue
        item = {
            "beatmap_id": beatmap_id,
            "keys": key_count,
            "group": entry["group"],
            "sha256": entry["sha256"],
        }
        stage = "source-read"
        try:
            path = map_root / f"{beatmap_id}.osu"
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"Frozen source hash mismatch for map {beatmap_id}")
            original_text = raw.decode("utf-8-sig")
            chart = parse_osu(original_text)
            if chart["keys"] != key_count:
                raise ValueError(f"Key-count mismatch for map {beatmap_id}")
            settings = parameters({
                "keys": key_count,
                "layout_engine": "beam",
                "seed": seed,
            })

            stage = "baseline-layout"
            started = time.perf_counter()
            baseline = assign_lanes(
                list(chart["notes"]), settings, timing_points=chart["timing"]
            )
            baseline_seconds = time.perf_counter() - started
            baseline_roundtrip = _validate_roundtrip(
                baseline, key_count, original_text, chart, template
            )
            item["baseline"] = _chart_metrics(
                baseline, chart["notes"], key_count, chart["timing"], baseline,
                baseline_seconds, baseline_roundtrip,
            )

            stage = "sequence-prior-layout"
            started = time.perf_counter()
            mapped = assign_lanes(
                list(chart["notes"]), settings, timing_points=chart["timing"],
                sequence_prior=prior, sequence_strength=strength,
            )
            mapped_seconds = time.perf_counter() - started
            mapped_roundtrip = _validate_roundtrip(
                mapped, key_count, original_text, chart, template
            )
            item["sequence_prior"] = _chart_metrics(
                mapped, chart["notes"], key_count, chart["timing"], baseline,
                mapped_seconds, mapped_roundtrip,
            )
            item["passed"] = True
        except Exception as error:
            item.update(
                passed=False,
                failure_stage=stage,
                error=f"{type(error).__name__}: {error}",
            )
        existing_rows[beatmap_id] = item
        rows = [existing_rows[map_id] for map_id in selected_map_ids
                if map_id in existing_rows]
        result["rows"] = rows
        result["summary"] = summarize(rows)
        result["status"] = (
            "running" if index < len(selected) else
            ("complete" if all(row.get("passed") for row in rows) else "failed")
        )
        _write_atomic(output, result)
        print(json.dumps({
            "index": index,
            "total": len(selected),
            "beatmap_id": beatmap_id,
            "group": entry["group"],
            "passed": item["passed"],
            "failure_stage": item.get("failure_stage"),
            "baseline_seconds": item.get("baseline", {}).get("seconds"),
            "sequence_seconds": item.get("sequence_prior", {}).get("seconds"),
        }), flush=True)

    if any(not row.get("passed") for row in rows):
        raise RuntimeError(
            f"{sum(not row.get('passed') for row in rows)} charts failed; "
            f"details saved in {output}"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keys", type=int, action="append", default=None)
    parser.add_argument("--split", choices=("validation", "test"),
                        default="validation")
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validation-artifact", type=Path,
        help="Complete validation report used to freeze the strength for test.",
    )
    args = parser.parse_args()
    run(
        report_path=args.report,
        model_path=args.model,
        map_root=args.maps,
        output=args.output,
        keys=tuple(args.keys or (10,)),
        split=args.split,
        strength=args.strength,
        seed=args.seed,
        limit=args.limit,
        validation_artifact_path=args.validation_artifact,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
