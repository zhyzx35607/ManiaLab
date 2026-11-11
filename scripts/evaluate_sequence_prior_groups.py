"""Evaluate the sequence-window prior with song-group macro accuracy.

This is a local ranking diagnostic under fixed note events and LN endpoints.
It is not a measure of whole-chart generation or play quality.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from manialab.pattern_prior import score_network
from manialab.pattern_sequence import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    mirror_window_features,
)
from scripts.train_sequence_prior import (
    CACHE_REVISION,
    _feature_schema_sha256,
    _load_chart_cache,
)

DEFAULT_REPORT = Path("test-data/sequence-prior-v1/report.json")
DEFAULT_MODEL = Path("test-data/sequence-prior-v1/model.json")
DEFAULT_CACHE = Path("test-data/corpus-expanded/sequence-window-cache-v3")
METRICS = ("overall", "ln", "tap")
BOOTSTRAP_CONFIDENCE = 0.95


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON artifact {path}: {exc}") from exc


def _validate_artifacts(report, model):
    expected_features = list(FEATURE_NAMES)
    if report.get("revision") != "sequence-window-v1":
        raise ValueError("Unexpected sequence-prior report revision")
    if model.get("revision") != "sequence-window-v1":
        raise ValueError("Unexpected sequence-prior model revision")
    if report.get("feature_names") != expected_features:
        raise ValueError("Report feature schema does not match this evaluator")
    if model.get("features") != expected_features:
        raise ValueError("Model feature schema does not match this evaluator")
    training_hash = report.get("training_manifest_sha256")
    if not training_hash or model.get("training_manifest_sha256") != training_hash:
        raise ValueError("Model/report training manifest hash mismatch")

    prep = report.get("prep")
    if not isinstance(prep, dict) or not isinstance(prep.get("configs"), dict):
        raise ValueError("Report is missing cache configuration")
    config = prep["configs"]
    if config.get("cache_revision") != CACHE_REVISION:
        raise ValueError("Report does not reference the expected v3 cache")
    if config.get("feature_count") != FEATURE_COUNT:
        raise ValueError("Report feature count does not match this evaluator")
    if config.get("feature_schema_sha256") != _feature_schema_sha256():
        raise ValueError("Report feature schema hash does not match this evaluator")

    raw_layers = model.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("Model has no network layers")
    layers = []
    previous_width = FEATURE_COUNT
    for index, layer in enumerate(raw_layers):
        if not isinstance(layer, dict) or "weight" not in layer or "bias" not in layer:
            raise ValueError(f"Invalid model layer {index}")
        weight = np.asarray(layer["weight"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        if (weight.ndim != 2 or weight.shape[1] != previous_width
                or bias.shape != (weight.shape[0],) or weight.shape[0] == 0):
            raise ValueError(f"Invalid model layer dimensions at layer {index}")
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError(f"Nonfinite model values at layer {index}")
        layers.append({"weight": weight, "bias": bias})
        previous_width = weight.shape[0]
    if previous_width != 1:
        raise ValueError("Model network must produce one ranking score")
    return config, layers


def pair_accuracy_counts(positive_scores, negative_scores, ln_context):
    """Return fractional correct-pair counts; exact ties count as one half."""
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    context = np.asarray(ln_context, dtype=bool)
    if (positive.ndim != 1 or negative.ndim != 2 or len(positive) != len(negative)
            or context.shape != positive.shape or negative.shape[1] == 0):
        raise ValueError("Scores and LN context have inconsistent shapes")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Ranking scores must be finite")

    margins = positive[:, None] - negative
    result = {}
    for name, mask in (
        ("overall", np.ones(len(context), dtype=bool)),
        ("ln", context),
        ("tap", ~context),
    ):
        selected = margins[mask]
        result[name] = [
            float(np.count_nonzero(selected > 0)
                  + 0.5 * np.count_nonzero(selected == 0)),
            int(selected.size),
        ]
    return result


def _merge_counts(target, source):
    for metric in METRICS:
        target[metric][0] += source[metric][0]
        target[metric][1] += source[metric][1]


def _rounded(value):
    return round(float(value), 6)


def summarize_group_counts(group_counts, *, bootstrap_samples=5000, seed=20261112):
    """Macro-average group accuracies and bootstrap independent groups."""
    if (isinstance(bootstrap_samples, bool)
            or not isinstance(bootstrap_samples, (int, np.integer))
            or bootstrap_samples < 1):
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    result = {}
    for metric in METRICS:
        values = [
            counts[metric][0] / counts[metric][1]
            for counts in group_counts.values()
            if counts[metric][1] > 0
        ]
        pair_count = sum(counts[metric][1] for counts in group_counts.values())
        if not values:
            result[metric] = dict(
                groups=0, pairs=0, macro_accuracy=None, ci_95=None)
            continue
        values = np.asarray(values, dtype=np.float64)
        indices = rng.integers(
            0, len(values), size=(int(bootstrap_samples), len(values)))
        estimates = values[indices].mean(axis=1)
        low, high = np.quantile(estimates, [0.025, 0.975])
        result[metric] = dict(
            groups=len(values),
            pairs=int(pair_count),
            macro_accuracy=_rounded(values.mean()),
            ci_95=[_rounded(low), _rounded(high)],
        )
    return result


def _score_mirror_averaged(features, layers, keys):
    features = np.asarray(features, dtype=np.float32)
    mirrored = mirror_window_features(features, keys)
    direct = score_network(features, layers)
    reflected = score_network(mirrored, layers)
    return (direct + reflected) / 2


def _validated_map_arrays(row, cache_root, config):
    beatmap_id = row.get("id")
    sha256 = row.get("sha256")
    if (isinstance(beatmap_id, bool) or not isinstance(beatmap_id, int)
            or not isinstance(sha256, str) or len(sha256) != 64):
        raise ValueError("Invalid beatmap identity in sampling map")
    if not isinstance(row.get("group"), str) or not row["group"]:
        raise ValueError(f"Beatmap {beatmap_id} has no song group")
    cache_path = Path(cache_root) / (
        f"{sha256}-{beatmap_id}-sequence-window-v3.npz")
    positive, negative, context, _ = _load_chart_cache(cache_path, config)
    if (positive.ndim != 2 or positive.shape[1] != FEATURE_COUNT
            or negative.ndim != 3 or negative.shape[0] != len(positive)
            or negative.shape[2] != FEATURE_COUNT
            or context.shape != (len(positive),)):
        raise ValueError(f"Invalid cached sample shapes for beatmap {beatmap_id}")
    if (not np.isfinite(positive).all() or not np.isfinite(negative).all()):
        raise ValueError(f"Nonfinite cached features for beatmap {beatmap_id}")

    before = row.get("training_windows_before_group_budget")
    retained = row.get("training_windows")
    if (isinstance(before, bool) or not isinstance(before, int)
            or before != len(positive) or isinstance(retained, bool)
            or not isinstance(retained, int) or not 0 <= retained <= before):
        raise ValueError(f"Sampling report/cache mismatch for beatmap {beatmap_id}")
    if retained < len(positive):
        indices = np.random.default_rng(beatmap_id).choice(
            len(positive), retained, replace=False)
        positive, negative, context = (
            positive[indices], negative[indices], context[indices])
    return positive, negative, context


def evaluate_key(report, cache_root, config, layers, keys, *,
                 bootstrap_samples=5000, seed=20261112, batch_size=512):
    sampling_maps = report.get("prep", {}).get("sampling_maps")
    if not isinstance(sampling_maps, list):
        raise ValueError("Report is missing per-map sampling records")
    selected_maps = [
        row for row in sampling_maps
        if row.get("keys") == keys and row.get("split") == "test"
    ]
    if not selected_maps:
        raise ValueError(f"No held-out test maps found for {keys}K")

    group_counts = defaultdict(
        lambda: {metric: [0.0, 0] for metric in METRICS})
    group_charts = defaultdict(int)
    group_windows = defaultdict(int)
    seen_ids = set()
    for row in selected_maps:
        beatmap_id = row.get("id")
        if beatmap_id in seen_ids:
            raise ValueError(f"Duplicate test map {beatmap_id} for {keys}K")
        seen_ids.add(beatmap_id)
        group = row.get("group")
        positive, negative, context = _validated_map_arrays(
            row, cache_root, config)
        if not len(positive):
            continue
        if negative.shape[1] == 0:
            raise ValueError(f"Beatmap {beatmap_id} has no negative examples")
        group_charts[group] += 1
        group_windows[group] += len(positive)
        for begin in range(0, len(positive), batch_size):
            end = min(len(positive), begin + batch_size)
            pos_score = _score_mirror_averaged(
                positive[begin:end], layers, keys)
            neg = negative[begin:end].reshape(-1, FEATURE_COUNT)
            neg_score = _score_mirror_averaged(neg, layers, keys).reshape(
                end - begin, negative.shape[1])
            counts = pair_accuracy_counts(
                pos_score, neg_score, context[begin:end])
            _merge_counts(group_counts[group], counts)

    if not group_counts:
        raise ValueError(f"No retained test windows found for {keys}K")
    macro = summarize_group_counts(
        group_counts, bootstrap_samples=bootstrap_samples, seed=seed)
    by_group = []
    for group in sorted(group_counts):
        counts = group_counts[group]
        metrics = {}
        for metric in METRICS:
            correct, pairs = counts[metric]
            metrics[metric] = dict(
                pairs=int(pairs),
                accuracy=_rounded(correct / pairs) if pairs else None,
            )
        by_group.append(dict(
            group=group,
            charts=group_charts[group],
            windows=group_windows[group],
            metrics=metrics,
        ))
    return dict(
        keys=keys,
        source_test_maps=len(selected_maps),
        evaluated_charts=sum(group_charts.values()),
        evaluated_groups=len(group_counts),
        evaluated_windows=sum(group_windows.values()),
        macro=macro,
        by_group=by_group,
    )


def evaluate(report_path=DEFAULT_REPORT, model_path=DEFAULT_MODEL,
             cache_root=DEFAULT_CACHE, *, keys=(10,),
             bootstrap_samples=5000, seed=20261112, batch_size=512):
    report = _read_json(report_path)
    model = _read_json(model_path)
    config, layers = _validate_artifacts(report, model)
    if (isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    if (isinstance(bootstrap_samples, bool)
            or not isinstance(bootstrap_samples, (int, np.integer))
            or bootstrap_samples < 1):
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    requested_keys = tuple(dict.fromkeys(keys))
    if not requested_keys or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value not in range(6, 11)
            for value in requested_keys):
        raise ValueError("keys must contain one or more integers from 6 through 10")
    bootstrap_samples, seed, batch_size = (
        int(bootstrap_samples), int(seed), int(batch_size))

    results = {}
    for key in requested_keys:
        results[str(key)] = evaluate_key(
            report, cache_root, config, layers, key,
            bootstrap_samples=bootstrap_samples, seed=seed,
            batch_size=batch_size)
    return dict(
        revision="sequence-window-group-evaluation-v1",
        scope=("Test-set local window layout ranking under fixed authored note "
               "events and LN endpoints; not whole-chart generation quality."),
        split="test",
        baseline=dict(
            pair_accuracy=0.5,
            meaning="Chance level for binary pair ranking; not a runtime-prior comparison.",
        ),
        bootstrap=dict(
            unit="independent song group",
            samples=bootstrap_samples,
            seed=seed,
            confidence=BOOTSTRAP_CONFIDENCE,
            method="percentile",
        ),
        artifact=dict(
            report=str(report_path),
            model=str(model_path),
            cache_revision=CACHE_REVISION,
        ),
        keys=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--keys", nargs="+", type=int, choices=range(6, 11),
                        default=[10], help="Key counts to evaluate (default: 10).")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path,
                        help="Optional JSON path; existing files are not overwritten.")
    args = parser.parse_args()

    result = evaluate(
        args.report, args.model, args.cache,
        keys=args.keys,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
