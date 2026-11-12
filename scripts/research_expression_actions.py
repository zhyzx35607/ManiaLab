"""Probe whether local osu!mania expression actions generalize across songs.

Eight-beat windows are sampling context, not verse/chorus or phrase labels.
Features use attack timing and chord shape; targets describe charted LN and
release choices. This chart-only study does not infer instruments or audio
release truth and cannot enable the generator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, validate
from manialab.sequence_prior import sequence_beat_positions
from manialab.timing import BeatClock
from scripts.train_layout_network import records_and_groups
from scripts.train_pattern_prior import split_for

WINDOW_BEATS = 8.0
ONSET_BINS = 32
KEY_COUNTS = tuple(range(6, 11))
TARGET_NAMES = (
    "ln_fraction",
    "ln_median_beats",
    "ln_long_fraction_2beat",
    "ln_chord_fraction",
    "staggered_release_fraction",
    "hybrid_tap_fraction",
    "inverse_ln_fraction",
    "release_only_fraction",
)
FEATURE_NAMES = (
    *(f"keys_{keys}" for keys in KEY_COUNTS),
    "chart_progress",
    "attack_rows_per_beat",
    "note_heads_per_beat",
    "chord_note_fraction",
    "max_chord_share",
    "onbeat_row_fraction",
    "previous_attack_rows_per_beat",
    "attack_density_change",
    "similarity_previous_window",
    "similarity_two_windows_back",
    *(f"onset_bin_{index}" for index in range(ONSET_BINS)),
)
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
def _cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _active_holds_before_attacks(notes):
    starts, ends = defaultdict(int), defaultdict(int)
    attacks = {note.start for note in notes}
    for note in notes:
        if note.hold:
            starts[note.start] += 1
            ends[note.end] += 1
    active, result = 0, {}
    for time in sorted(attacks | set(starts) | set(ends)):
        active -= ends[time]
        result[time] = max(0, active)
        active += starts[time]
    return result


def _chart_window_targets(
        notes, rows, beat_by_time, clock, active_before,
        all_attack_starts, previous_by_identity):
    holds = [note for note in notes if note.hold]
    taps = [note for note in notes if not note.hold]
    lengths = np.asarray(
        [beat_by_time[note.end] - beat_by_time[note.start] for note in holds],
        dtype=np.float64,
    )
    ln_chords = [
        [note for note in row if note.hold]
        for row in rows.values()
        if sum(note.hold for note in row) >= 2
    ]
    ln_chord_notes = sum(len(group) for group in ln_chords)
    staggered = sum(len({note.end for note in group}) > 1 for group in ln_chords)

    hybrid_taps = sum(active_before.get(note.start, 0) > 0 for note in taps)
    inverse_heads = 0
    for note in notes:
        previous = previous_by_identity[id(note)]
        if note.hold and previous is not None and previous.hold:
            gap = note.start - previous.end
            if 0 < gap <= min(150.0, clock(note.start) * 0.3):
                inverse_heads += 1

    release_only = sum(
        note.end not in all_attack_starts for note in holds
    )
    targets = {
        "ln_fraction": len(holds) / max(1, len(notes)),
        "ln_median_beats": float(np.median(lengths)) if len(lengths) else np.nan,
        "ln_long_fraction_2beat": float(np.mean(lengths >= 2.0)) if len(lengths) else np.nan,
        "ln_chord_fraction": ln_chord_notes / len(holds) if holds else np.nan,
        "staggered_release_fraction": (
            staggered / len(ln_chords) if ln_chords else np.nan
        ),
        "hybrid_tap_fraction": hybrid_taps / len(taps) if taps else np.nan,
        "inverse_ln_fraction": inverse_heads / len(holds) if holds else np.nan,
        "release_only_fraction": release_only / len(holds) if holds else np.nan,
    }
    return np.asarray([targets[name] for name in TARGET_NAMES], dtype=np.float64)


def extract_chart_windows(parsed, window_beats=WINDOW_BEATS):
    """Return attack-context features and authored expression targets.

    Window boundaries are relative to the first chart attack. Empty windows
    remain in the history used for density changes, but are not target samples.
    """
    if (not np.isfinite(window_beats) or window_beats <= 0
            or window_beats > 64):
        raise ValueError("window_beats must be finite and in (0, 64]")
    keys, notes = parsed["keys"], parsed["notes"]
    if keys not in KEY_COUNTS:
        raise ValueError("expression action research supports 6K through 10K")
    report = validate(notes, keys)
    if not report["valid"]:
        raise ValueError("Invalid chart: " + str(report["issues"][:1]))

    notes = sorted(notes, key=lambda note: (note.start, note.lane, note.end))
    times = sorted({value for note in notes for value in (note.start, note.end)})
    positions = sequence_beat_positions(times, parsed.get("timing", ()), fallback=500.0)
    beat_by_time = dict(zip(times, positions.tolist()))
    origin = beat_by_time[min(note.start for note in notes)]
    by_start = defaultdict(list)
    by_window = defaultdict(list)
    grouped_rows_by_window = defaultdict(dict)
    for note in notes:
        by_start[note.start].append(note)
        index = int(np.floor((beat_by_time[note.start] - origin) / window_beats + 1e-9))
        by_window[index].append(note)
        grouped_rows_by_window[index][note.start] = by_start[note.start]

    if not by_window:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0, len(TARGET_NAMES)))
    last_window = max(by_window)
    active_before = _active_holds_before_attacks(notes)
    previous_by_identity = {}
    lane_history = {}
    for note in notes:
        previous_by_identity[id(note)] = lane_history.get(note.lane)
        lane_history[note.lane] = note
    clock = BeatClock(parsed.get("timing", ()), fallback=500.0)
    grids, row_densities = [], []
    rows_by_window = []
    for index in range(last_window + 1):
        local_rows = grouped_rows_by_window[index]
        local_notes = by_window.get(index, [])
        grid = np.zeros(ONSET_BINS, dtype=np.float64)
        row_positions = []
        for start, row in local_rows.items():
            relative = beat_by_time[start] - origin - index * window_beats
            bin_index = min(
                ONSET_BINS - 1,
                max(0, int(np.floor(relative / window_beats * ONSET_BINS))),
            )
            grid[bin_index] += 1.0
            row_positions.append(beat_by_time[start])
        grids.append(grid)
        row_densities.append(len(local_rows) / window_beats)
        rows_by_window.append((local_rows, local_notes, row_positions))

    feature_rows, target_rows = [], []
    key_vector = np.zeros(len(KEY_COUNTS), dtype=np.float64)
    key_vector[KEY_COUNTS.index(keys)] = 1.0
    for index, (local_rows, local_notes, row_positions) in enumerate(rows_by_window):
        if not local_notes:
            continue
        row_count = len(local_rows)
        chord_notes = sum(len(row) for row in local_rows.values() if len(row) > 1)
        onbeat_rows = sum(
            min(value % 1.0, 1.0 - value % 1.0) <= 0.10
            for value in row_positions
        )
        previous_density = row_densities[index - 1] if index else 0.0
        progress = index / max(1, last_window)
        normalized_grid = grids[index] / max(1.0, grids[index].sum())
        feature = np.concatenate((
            key_vector,
            np.asarray([
                progress,
                row_count / window_beats,
                len(local_notes) / window_beats,
                chord_notes / max(1, len(local_notes)),
                max((len(row) for row in local_rows.values()), default=0) / keys,
                onbeat_rows / max(1, row_count),
                previous_density,
                row_count / window_beats - previous_density,
                _cosine(grids[index], grids[index - 1]) if index else 0.0,
                _cosine(grids[index], grids[index - 2]) if index >= 2 else 0.0,
            ]),
            normalized_grid,
        ))
        feature_rows.append(feature)
        target_rows.append(_chart_window_targets(
            local_notes, local_rows, beat_by_time, clock, active_before,
            set(by_start), previous_by_identity,
        ))
    if not feature_rows:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0, len(TARGET_NAMES)))
    x = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(target_rows, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite expression context feature")
    if not np.isfinite(y[:, 0]).all():
        raise ValueError("Non-finite LN fraction target")
    return x, y


def _group_weights(groups):
    counts = Counter(groups)
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    if len(weights):
        weights *= len(weights) / weights.sum()
    return weights


def _target_transform(values, target_name):
    values = np.asarray(values, dtype=np.float64)
    return np.log1p(np.maximum(0.0, values)) if target_name == "ln_median_beats" else values


def _target_inverse(values, target_name):
    return np.expm1(values) if target_name == "ln_median_beats" else values


def fit_ridge(features, targets, groups, target_name, alpha=1.0):
    """Fit one masked, song-group-weighted ridge target."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be finite and positive")
    target_index = TARGET_NAMES.index(target_name)
    valid = np.isfinite(targets[:, target_index])
    x, y = features[valid], targets[valid, target_index]
    local_groups = [group for group, keep in zip(groups, valid) if keep]
    if len(x) < 2 or len(set(local_groups)) < 2:
        raise ValueError(f"Not enough independent training groups for {target_name}")
    weights = _group_weights(local_groups)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-9] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), standardized))
    transformed = _target_transform(y, target_name)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * alpha
    regularizer[0, 0] = 0.0
    lhs = design.T @ (weights[:, None] * design) + regularizer
    rhs = design.T @ (weights * transformed)
    coefficients = np.linalg.solve(lhs, rhs)
    return dict(mean=mean, scale=scale, coefficients=coefficients)


def predict_ridge(model, features, target_name):
    standardized = (features - model["mean"]) / model["scale"]
    design = np.column_stack((np.ones(len(features)), standardized))
    prediction = np.maximum(
        0.0,
        _target_inverse(design @ model["coefficients"], target_name),
    )
    return prediction if target_name == "ln_median_beats" else np.minimum(1.0, prediction)


def _density_position_bins(
        features, keys, density_edges_by_key, fallback_edges, progress_edges):
    density = features[:, _FEATURE_INDEX["attack_rows_per_beat"]]
    progress = features[:, _FEATURE_INDEX["chart_progress"]]
    keys = np.asarray(keys)
    density_bins = np.zeros(len(features), dtype=int)
    for key in KEY_COUNTS:
        selected = keys == key
        edges = density_edges_by_key.get(key, fallback_edges)
        density_bins[selected] = np.digitize(density[selected], edges)
    return (
        density_bins,
        np.digitize(progress, progress_edges).astype(int),
    )


def _weighted_quantiles(values, weights, quantiles):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (values.ndim != 1 or weights.shape != values.shape or len(values) == 0
            or np.any(weights <= 0) or not np.isfinite(values).all()
            or not np.isfinite(weights).all()):
        raise ValueError("weighted quantiles require finite values and positive weights")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    indices = np.searchsorted(
        cumulative,
        np.asarray(quantiles, dtype=np.float64) * cumulative[-1],
        side="left",
    ).clip(0, len(order) - 1)
    return values[order[indices]]


def predict_stratified_baseline(
        train_features, train_targets, train_keys, train_groups,
        evaluation_features, evaluation_keys, target_name, min_cell_groups=4):
    """Predict by key × onset-density tertile × chart-position tertile."""
    target_index = TARGET_NAMES.index(target_name)
    valid = np.isfinite(train_targets[:, target_index])
    y = train_targets[:, target_index]
    valid_groups = [
        group for group, keep in zip(train_groups, valid) if keep
    ]
    valid_weights = _group_weights(valid_groups)
    train_density = train_features[valid, _FEATURE_INDEX["attack_rows_per_beat"]]
    global_density_edges = (
        _weighted_quantiles(train_density, valid_weights, [1 / 3, 2 / 3])
        if len(train_density) else np.zeros(2)
    )
    valid_keys = np.asarray(train_keys)
    density_edges_by_key = {}
    for key in KEY_COUNTS:
        key_valid = valid & (valid_keys == key)
        key_groups = {
            group for group, keep in zip(train_groups, key_valid) if keep
        }
        if len(key_groups) >= min_cell_groups:
            key_weights = _group_weights([
                group for group, keep in zip(train_groups, key_valid) if keep
            ])
            density_edges_by_key[key] = _weighted_quantiles(
                train_features[key_valid, _FEATURE_INDEX["attack_rows_per_beat"]],
                key_weights,
                [1 / 3, 2 / 3],
            )
    progress_edges = np.asarray([1 / 3, 2 / 3], dtype=np.float64)
    train_density_bin, train_progress_bin = _density_position_bins(
        train_features, train_keys, density_edges_by_key,
        global_density_edges, progress_edges,
    )
    eval_density_bin, eval_progress_bin = _density_position_bins(
        evaluation_features, evaluation_keys, density_edges_by_key,
        global_density_edges, progress_edges,
    )
    transformed_y = np.full(len(train_groups), np.nan, dtype=np.float64)
    transformed_y[valid] = _target_transform(y[valid], target_name)
    key_means, key_group_counts = {}, {}
    cell_means = {}
    valid_keys = np.asarray(train_keys)
    for key in KEY_COUNTS:
        selected = valid & (valid_keys == key)
        if np.any(selected):
            selected_group_rows = [
                group for group, keep in zip(train_groups, selected) if keep
            ]
            selected_groups = set(selected_group_rows)
            key_means[key] = float(np.average(
                transformed_y[selected],
                weights=_group_weights(selected_group_rows),
            ))
            key_group_counts[key] = len(selected_groups)
        for density_bin in range(3):
            for progress_bin in range(3):
                cell = (
                    selected
                    & (train_density_bin == density_bin)
                    & (train_progress_bin == progress_bin)
                )
                if not np.any(cell):
                    continue
                selected_groups = {
                    group for group, keep in zip(train_groups, cell) if keep
                }
                cell_key = (key, density_bin, progress_bin)
                if len(selected_groups) >= min_cell_groups:
                    cell_group_rows = [
                        group for group, keep in zip(train_groups, cell) if keep
                    ]
                    cell_means[cell_key] = float(np.average(
                        transformed_y[cell],
                        weights=_group_weights(cell_group_rows),
                    ))
    global_mean = (
        float(np.average(transformed_y[valid], weights=valid_weights))
        if np.any(valid) else 0.0
    )
    predictions = np.empty(len(evaluation_features), dtype=np.float64)
    for index, key in enumerate(evaluation_keys):
        cell_key = (int(key), int(eval_density_bin[index]), int(eval_progress_bin[index]))
        if cell_key in cell_means:
            mean = cell_means[cell_key]
        else:
            mean = key_means.get(int(key), global_mean)
        predictions[index] = _target_inverse(mean, target_name)
    predictions = np.maximum(0.0, predictions)
    if target_name != "ln_median_beats":
        predictions = np.minimum(1.0, predictions)
    return predictions, dict(
        density_edges={
            str(key): [float(value) for value in edges]
            for key, edges in density_edges_by_key.items()
        },
        fallback_density_edges=[float(value) for value in global_density_edges],
        minimum_cell_groups=min_cell_groups,
        populated_cells=len(cell_means),
        key_group_counts={str(key): count for key, count in key_group_counts.items()},
    )


def score_group_macro(targets, predictions, groups, target_name):
    target_index = TARGET_NAMES.index(target_name)
    valid = np.isfinite(targets[:, target_index])
    by_group = defaultdict(list)
    for truth, predicted, group, keep in zip(
            targets[:, target_index], predictions, groups, valid):
        if keep:
            by_group[group].append(abs(float(truth) - float(predicted)))
    errors = [float(np.mean(values)) for values in by_group.values()]
    return dict(
        samples=int(sum(map(len, by_group.values()))),
        groups=len(errors),
        group_macro_mae=round(float(np.mean(errors)), 6) if errors else None,
    )


def _dataset_arrays(rows):
    if not rows:
        return (
            np.empty((0, len(FEATURE_NAMES))),
            np.empty((0, len(TARGET_NAMES))),
            np.empty((0,), dtype=int),
            [],
            [],
        )
    return (
        np.vstack([row["features"] for row in rows]),
        np.vstack([row["targets"] for row in rows]),
        np.asarray([row["keys"] for row in rows], dtype=int),
        [row["group"] for row in rows],
        [row["beatmap_id"] for row in rows],
    )


def _evaluate(train, validation, alpha=1.0):
    train_x, train_y, train_keys, train_groups, _ = _dataset_arrays(train)
    val_x, val_y, val_keys, val_groups, _ = _dataset_arrays(validation)
    if not len(train_x) or not len(val_x):
        raise ValueError("Both train and validation expression windows are required")
    results = {}
    ridge_models = {}
    for target_name in TARGET_NAMES:
        valid_train = np.isfinite(train_y[:, TARGET_NAMES.index(target_name)])
        if len(set(np.asarray(train_groups)[valid_train])) < 2:
            continue
        model = fit_ridge(train_x, train_y, train_groups, target_name, alpha=alpha)
        ridge_models[target_name] = model
        ridge_prediction = predict_ridge(model, val_x, target_name)
        baseline_prediction, baseline_config = predict_stratified_baseline(
            train_x, train_y, train_keys, train_groups,
            val_x, val_keys, target_name,
        )
        results[target_name] = dict(
            ridge=score_group_macro(val_y, ridge_prediction, val_groups, target_name),
            stratified_baseline=score_group_macro(
                val_y, baseline_prediction, val_groups, target_name
            ),
            baseline_config=baseline_config,
        )
    per_key = {}
    for key in KEY_COUNTS:
        selected = np.flatnonzero(val_keys == key)
        if not len(selected):
            continue
        groups = [val_groups[index] for index in selected]
        metrics = {}
        for target_name in TARGET_NAMES:
            if target_name not in ridge_models:
                continue
            baseline_prediction, _ = predict_stratified_baseline(
                train_x, train_y, train_keys, train_groups,
                val_x[selected], val_keys[selected], target_name,
            )
            ridge_prediction = predict_ridge(
                ridge_models[target_name], val_x[selected], target_name
            )
            metrics[target_name] = dict(
                ridge=score_group_macro(val_y[selected], ridge_prediction, groups, target_name),
                stratified_baseline=score_group_macro(
                    val_y[selected], baseline_prediction, groups, target_name
                ),
            )
        per_key[str(key)] = metrics
    return dict(overall=results, by_key=per_key)


def build_report(root, output, *, window_beats=WINDOW_BEATS):
    selected, excluded = records_and_groups(root, exclude_splits=("test",))
    records_by_split = defaultdict(list)
    for record in selected:
        split = split_for(record["group"])
        records_by_split[split].append(record)

    rows_by_split = defaultdict(list)
    source_manifest = []
    chart_errors = []
    for split in ("train", "validation"):
        records = sorted(
            records_by_split.get(split, ()),
            key=lambda row: (row["group"], int(row["beatmap_id"])),
        )
        for index, record in enumerate(records):
            chart_path = root / "maps" / f'{record["beatmap_id"]}.osu'
            try:
                parsed = parse_osu(chart_path.read_text(encoding="utf-8-sig"))
                if parsed["keys"] != int(record["keys"]):
                    raise ValueError("manifest and chart keycounts differ")
                features, targets = extract_chart_windows(parsed, window_beats)
                if len(features):
                    for feature, target in zip(features, targets):
                        rows_by_split[split].append(dict(
                            features=feature,
                            targets=target,
                            keys=parsed["keys"],
                            group=record["group"],
                            beatmap_id=record["beatmap_id"],
                        ))
                source_manifest.append(dict(
                    beatmap_id=record["beatmap_id"],
                    sha256=record["sha256"],
                    song_group=record["group"],
                    split=split,
                    keys=parsed["keys"],
                    windows=len(features),
                ))
            except Exception as error:
                chart_errors.append(dict(
                    beatmap_id=record["beatmap_id"],
                    split=split,
                    error=str(error),
                ))
            if index % 100 == 0:
                print(json.dumps(dict(
                    split=split, visited=index, total=len(records),
                    windows=len(rows_by_split[split]),
                )), flush=True)
    if chart_errors:
        raise ValueError(
            f"{len(chart_errors)} included charts failed; first: {chart_errors[0]}"
        )
    if not rows_by_split["train"] or not rows_by_split["validation"]:
        raise ValueError("No train or validation action windows were extracted")

    train_groups = {row["group"] for row in rows_by_split["train"]}
    validation_groups = {row["group"] for row in rows_by_split["validation"]}
    if train_groups & validation_groups:
        raise ValueError("Song groups overlap between train and validation")
    result = _evaluate(rows_by_split["train"], rows_by_split["validation"])
    manifest_sha = hashlib.sha256(json.dumps(
        source_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    report = dict(
        revision=1,
        purpose="Chart-only expression action predictability; research only, no generator integration.",
        window=dict(beats=window_beats, note="Context unit only; not a semantic phrase label."),
        features=list(FEATURE_NAMES),
        targets=list(TARGET_NAMES),
        split=dict(
            policy="Existing song-group split; test split excluded before chart reads.",
            train=dict(
                charts=sum(row["split"] == "train" for row in source_manifest),
                groups=len(train_groups),
                windows=len(rows_by_split["train"]),
            ),
            validation=dict(
                charts=sum(row["split"] == "validation" for row in source_manifest),
                groups=len(validation_groups),
                windows=len(rows_by_split["validation"]),
            ),
            excluded=excluded,
            source_manifest_sha256=manifest_sha,
        ),
        source_maps=source_manifest,
        evaluation=result,
        limitations=[
            "No audio, stems, or instrument identities are used.",
            "A charted LN endpoint is an author choice, not evidence of an audible release.",
            "An uncharted sound is not treated as a negative audio event.",
            "Validation errors describe chart-action predictability, not musical quality.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("test-data/corpus-expanded"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("test-data/expression-action-predictability-v1/report.json"),
    )
    parser.add_argument("--window-beats", type=float, default=WINDOW_BEATS)
    args = parser.parse_args()
    report = build_report(
        args.corpus,
        args.output,
        window_beats=args.window_beats,
    )
    print(json.dumps(dict(
        output=str(args.output),
        train=report["split"]["train"],
        validation=report["split"]["validation"],
        keys=sorted(report["evaluation"]["by_key"]),
    )))


if __name__ == "__main__":
    main()
