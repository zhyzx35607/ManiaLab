"""Compare two sequence priors on the same chart-window candidate split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_sequence_prior_groups import (
    BOOTSTRAP_CONFIDENCE,
    METRICS,
    _merge_counts,
    _read_json,
    _score_mirror_averaged,
    _validate_model_artifact,
    _validate_report_config,
    _validated_map_arrays,
    pair_accuracy_counts,
    summarize_group_counts,
)

DEFAULT_REPORT_A = Path("test-data/sequence-prior-v1/report.json")
DEFAULT_MODEL_A = Path("test-data/sequence-prior-v1/model.json")
DEFAULT_REPORT_B = Path("test-data/sequence-prior-ln-transition-v1/report.json")
DEFAULT_MODEL_B = Path("test-data/sequence-prior-ln-transition-v1/model.json")
DEFAULT_SAMPLE_REPORT = DEFAULT_REPORT_A
DEFAULT_CACHE = Path("test-data/corpus-expanded/sequence-window-cache-v3")


def summarize_paired_group_deltas(counts_a, counts_b, *,
                                  bootstrap_samples=5000, seed=20261112):
    """Bootstrap model B minus A over the same independent song groups."""
    if (isinstance(bootstrap_samples, bool)
            or not isinstance(bootstrap_samples, (int, np.integer))
            or bootstrap_samples < 1):
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))
    result = {}
    for metric in METRICS:
        deltas = []
        pair_count = 0
        for group in sorted(set(counts_a) & set(counts_b)):
            correct_a, pairs_a = counts_a[group][metric]
            correct_b, pairs_b = counts_b[group][metric]
            if not pairs_a or not pairs_b:
                continue
            if pairs_a != pairs_b:
                raise ValueError(f"Unpaired candidate counts for group {group}")
            deltas.append(correct_b / pairs_b - correct_a / pairs_a)
            pair_count += pairs_a

        if not deltas:
            result[metric] = dict(groups=0, pairs=0, mean_delta=None, ci_95=None)
            continue
        values = np.asarray(deltas, dtype=np.float64)
        indices = rng.integers(
            0, len(values), size=(int(bootstrap_samples), len(values)))
        estimates = values[indices].mean(axis=1)
        low, high = np.quantile(estimates, [0.025, 0.975])
        result[metric] = dict(
            groups=len(values),
            pairs=int(pair_count),
            mean_delta=round(float(values.mean()), 6),
            ci_95=[round(float(low), 6), round(float(high), 6)],
        )
    return result


def _new_group_counts():
    return defaultdict(lambda: {metric: [0.0, 0] for metric in METRICS})


def _evaluate_key(sample_report, cache_root, sample_config, layers_a, layers_b,
                  keys, *, split, bootstrap_samples, seed, batch_size):
    sampling_maps = sample_report.get("prep", {}).get("sampling_maps")
    if not isinstance(sampling_maps, list):
        raise ValueError("Sample report is missing per-map sampling records")
    selected_maps = [
        row for row in sampling_maps
        if row.get("keys") == keys and row.get("split") == split
    ]
    if not selected_maps:
        raise ValueError(f"No {split} maps found for {keys}K")

    counts_a, counts_b = _new_group_counts(), _new_group_counts()
    group_charts = defaultdict(int)
    group_windows = defaultdict(int)
    seen_ids = set()
    for row in selected_maps:
        beatmap_id = row.get("id")
        if beatmap_id in seen_ids:
            raise ValueError(f"Duplicate {split} map {beatmap_id} for {keys}K")
        seen_ids.add(beatmap_id)
        group = row.get("group")
        positive, negative, context = _validated_map_arrays(
            row, cache_root, sample_config)
        if not len(positive):
            continue
        if negative.shape[1] == 0:
            raise ValueError(f"Beatmap {beatmap_id} has no negative examples")

        group_charts[group] += 1
        group_windows[group] += len(positive)
        for begin in range(0, len(positive), batch_size):
            end = min(len(positive), begin + batch_size)
            pos = positive[begin:end]
            neg = negative[begin:end].reshape(-1, positive.shape[1])
            pos_a = _score_mirror_averaged(pos, layers_a, keys)
            neg_a = _score_mirror_averaged(neg, layers_a, keys).reshape(
                end - begin, negative.shape[1])
            pos_b = _score_mirror_averaged(pos, layers_b, keys)
            neg_b = _score_mirror_averaged(neg, layers_b, keys).reshape(
                end - begin, negative.shape[1])
            _merge_counts(counts_a[group], pair_accuracy_counts(
                pos_a, neg_a, context[begin:end]))
            _merge_counts(counts_b[group], pair_accuracy_counts(
                pos_b, neg_b, context[begin:end]))

    if not counts_a:
        raise ValueError(f"No retained {split} windows found for {keys}K")
    macro_a = summarize_group_counts(
        counts_a, bootstrap_samples=bootstrap_samples, seed=seed)
    macro_b = summarize_group_counts(
        counts_b, bootstrap_samples=bootstrap_samples, seed=seed)
    by_group = []
    for group in sorted(counts_a):
        metrics = {}
        for metric in METRICS:
            correct_a, pairs = counts_a[group][metric]
            correct_b, pairs_b = counts_b[group][metric]
            if pairs != pairs_b:
                raise ValueError(f"Unpaired candidate counts for group {group}")
            accuracy_a = correct_a / pairs if pairs else None
            accuracy_b = correct_b / pairs if pairs else None
            metrics[metric] = dict(
                pairs=int(pairs),
                model_a_accuracy=round(float(accuracy_a), 6)
                if accuracy_a is not None else None,
                model_b_accuracy=round(float(accuracy_b), 6)
                if accuracy_b is not None else None,
                delta=round(float(accuracy_b - accuracy_a), 6)
                if accuracy_a is not None else None,
            )
        by_group.append(dict(
            group=group,
            charts=group_charts[group],
            windows=group_windows[group],
            metrics=metrics,
        ))

    return dict(
        keys=keys,
        source_maps=len(selected_maps),
        evaluated_charts=sum(group_charts.values()),
        evaluated_groups=len(counts_a),
        evaluated_windows=sum(group_windows.values()),
        model_a=dict(macro=macro_a, by_group=by_group),
        model_b=dict(macro=macro_b, by_group=by_group),
        paired_delta_b_minus_a=summarize_paired_group_deltas(
            counts_a, counts_b,
            bootstrap_samples=bootstrap_samples, seed=seed),
    )


def compare(report_a_path=DEFAULT_REPORT_A, model_a_path=DEFAULT_MODEL_A,
            report_b_path=DEFAULT_REPORT_B, model_b_path=DEFAULT_MODEL_B,
            sample_report_path=DEFAULT_SAMPLE_REPORT, cache_root=DEFAULT_CACHE,
            *, keys=(10,), split="test", bootstrap_samples=5000, seed=20261112,
            batch_size=512, label_a="model-a", label_b="model-b"):
    report_a = _read_json(report_a_path)
    model_a = _read_json(model_a_path)
    report_b = _read_json(report_b_path)
    model_b = _read_json(model_b_path)
    config_a = _validate_report_config(report_a)
    config_b = _validate_report_config(report_b)
    layers_a = _validate_model_artifact(report_a, model_a)
    layers_b = _validate_model_artifact(report_b, model_b)
    sample_report = _read_json(sample_report_path)
    sample_config = _validate_report_config(sample_report)

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
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    bootstrap_samples, seed, batch_size = (
        int(bootstrap_samples), int(seed), int(batch_size))

    results = {}
    for key in requested_keys:
        results[str(key)] = _evaluate_key(
            sample_report, cache_root, sample_config, layers_a, layers_b, key,
            split=split, bootstrap_samples=bootstrap_samples, seed=seed,
            batch_size=batch_size)
    return dict(
        revision="sequence-window-cross-model-evaluation-v1",
        scope=("Paired held-out local window ranking on identical candidates "
               "with fixed authored note events and LN endpoints; not chart quality."),
        split=split,
        bootstrap=dict(
            unit="independent song group",
            samples=bootstrap_samples,
            seed=seed,
            confidence=BOOTSTRAP_CONFIDENCE,
            method="paired percentile",
        ),
        models=dict(
            a=dict(label=label_a, report=str(report_a_path), model=str(model_a_path),
                   cache_revision=config_a["cache_revision"]),
            b=dict(label=label_b, report=str(report_b_path), model=str(model_b_path),
                   cache_revision=config_b["cache_revision"]),
        ),
        samples=dict(
            report=str(sample_report_path),
            cache=str(cache_root),
            cache_revision=sample_config["cache_revision"],
        ),
        keys=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-a", type=Path, default=DEFAULT_REPORT_A)
    parser.add_argument("--model-a", type=Path, default=DEFAULT_MODEL_A)
    parser.add_argument("--report-b", type=Path, default=DEFAULT_REPORT_B)
    parser.add_argument("--model-b", type=Path, default=DEFAULT_MODEL_B)
    parser.add_argument("--sample-report", type=Path, default=DEFAULT_SAMPLE_REPORT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--label-a", default="model-a")
    parser.add_argument("--label-b", default="model-b")
    parser.add_argument("--keys", nargs="+", type=int, choices=range(6, 11),
                        default=[10])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare(
        args.report_a, args.model_a, args.report_b, args.model_b,
        args.sample_report, args.cache,
        keys=args.keys,
        split=args.split,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite comparison: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
