import json

import numpy as np
import pytest

from manialab.pattern_sequence import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_COUNTS,
    FEATURE_SCHEMA_NAMES,
    LN_INTERACTION_SCHEMA,
)
from scripts.evaluate_sequence_prior_cross_model import (
    compare,
    summarize_paired_group_deltas,
)
from scripts.train_sequence_prior import (
    CACHE_REVISION,
    LN_TRANSITION_CACHE_REVISION,
    _feature_schema_sha256,
)


def _report(manifest, cache_revision, sampling_maps=None,
            feature_schema="sequence-window-v1"):
    feature_names = FEATURE_SCHEMA_NAMES[feature_schema]
    config = {
        "cache_revision": cache_revision,
        "feature_count": len(feature_names),
        "feature_schema_sha256": _feature_schema_sha256(feature_names),
    }
    if feature_schema != "sequence-window-v1":
        config["feature_schema"] = feature_schema
    return {
        "revision": feature_schema,
        "feature_names": list(feature_names),
        "training_manifest_sha256": manifest,
        "prep": {"configs": config, "sampling_maps": sampling_maps or []},
    }, config


def _model(manifest, feature_schema="sequence-window-v1"):
    feature_names = FEATURE_SCHEMA_NAMES[feature_schema]
    return {
        "revision": feature_schema,
        "features": list(feature_names),
        "training_manifest_sha256": manifest,
        "layers": [{
            "weight": np.zeros(
                (1, FEATURE_SCHEMA_COUNTS[feature_schema]),
                dtype=np.float32).tolist(),
            "bias": [0.0],
        }],
    }


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _training_map(group, keys):
    return {
        "id": 999,
        "sha256": "c" * 64,
        "keys": keys,
        "split": "train",
        "group": group,
        "training_windows_before_group_budget": 2,
        "training_windows": 2,
    }


@pytest.mark.parametrize("split", ("validation", "test"))
def test_cross_model_eval_uses_one_v4_sample_report_for_both_models(
        tmp_path, split):
    chart_id = 123
    chart_hash = "a" * 64
    sampling_maps = [{
        "id": chart_id,
        "sha256": chart_hash,
        "keys": 6,
        "split": split,
        "group": "audio:one",
        "training_windows_before_group_budget": 2,
        "training_windows": 2,
    }]
    report_a, _ = _report(
        "manifest-a", CACHE_REVISION, [_training_map("audio:train-a", 6)])
    report_b, _ = _report(
        "manifest-b", LN_TRANSITION_CACHE_REVISION,
        [_training_map("audio:train-b", 6)])
    sample_report, sample_config = _report(
        "manifest-samples", LN_TRANSITION_CACHE_REVISION, sampling_maps)
    report_a_path = _write_json(tmp_path / "report-a.json", report_a)
    report_b_path = _write_json(tmp_path / "report-b.json", report_b)
    sample_report_path = _write_json(tmp_path / "samples.json", sample_report)
    model_a_path = _write_json(tmp_path / "model-a.json", _model("manifest-a"))
    model_b_path = _write_json(tmp_path / "model-b.json", _model("manifest-b"))

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_path = cache_root / (
        f"{chart_hash}-{chart_id}-sequence-window-v4-ln-transition.npz")
    np.savez_compressed(
        cache_path,
        positive=np.zeros((2, FEATURE_COUNT), dtype=np.float16),
        negative=np.zeros((2, 1, FEATURE_COUNT), dtype=np.float16),
        context=np.array([True, False]),
        diagnostics=np.asarray("{}"),
        config=np.asarray(json.dumps(sample_config)),
    )

    result = compare(
        report_a_path,
        model_a_path,
        report_b_path,
        model_b_path,
        sample_report_path,
        cache_root,
        keys=(6,),
        split=split,
        bootstrap_samples=100,
        seed=7,
    )

    key_result = result["keys"]["6"]
    assert result["split"] == split
    assert result["group_isolation"]["overlapping_groups"] == 0
    assert result["samples"]["cache_revision"] == LN_TRANSITION_CACHE_REVISION
    assert key_result["evaluated_groups"] == 1
    assert key_result["model_a"]["macro"]["overall"]["macro_accuracy"] == 0.5
    assert key_result["model_b"]["macro"]["overall"]["macro_accuracy"] == 0.5
    assert key_result["paired_delta_b_minus_a"]["overall"]["mean_delta"] == 0.0


def test_cross_model_eval_compares_v1_and_ln_schema_on_v2_candidates(tmp_path):
    chart_id = 321
    chart_hash = "b" * 64
    sampling_maps = [{
        "id": chart_id,
        "sha256": chart_hash,
        "keys": 10,
        "split": "validation",
        "group": "audio:ln",
        "training_windows_before_group_budget": 2,
        "training_windows": 2,
    }]
    report_a, _ = _report(
        "manifest-a", CACHE_REVISION, [_training_map("audio:train-a", 10)])
    report_b, _ = _report(
        "manifest-b",
        "sequence-window-cache-v5-ln-interactions",
        [_training_map("audio:train-b", 10)],
        feature_schema=LN_INTERACTION_SCHEMA,
    )
    sample_report, sample_config = _report(
        "manifest-samples",
        "sequence-window-cache-v5-ln-interactions",
        sampling_maps,
        feature_schema=LN_INTERACTION_SCHEMA,
    )
    report_a_path = _write_json(tmp_path / "report-a.json", report_a)
    report_b_path = _write_json(tmp_path / "report-b.json", report_b)
    sample_report_path = _write_json(tmp_path / "samples.json", sample_report)
    model_a_path = _write_json(tmp_path / "model-a.json", _model("manifest-a"))
    model_b_path = _write_json(
        tmp_path / "model-b.json",
        _model("manifest-b", LN_INTERACTION_SCHEMA),
    )

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    width = FEATURE_SCHEMA_COUNTS[LN_INTERACTION_SCHEMA]
    cache_path = cache_root / (
        f"{chart_hash}-{chart_id}-sequence-window-v5-ln-interactions.npz")
    np.savez_compressed(
        cache_path,
        positive=np.zeros((2, width), dtype=np.float16),
        negative=np.zeros((2, 1, width), dtype=np.float16),
        context=np.array([True, False]),
        diagnostics=np.asarray("{}"),
        config=np.asarray(json.dumps(sample_config)),
    )

    result = compare(
        report_a_path,
        model_a_path,
        report_b_path,
        model_b_path,
        sample_report_path,
        cache_root,
        keys=(10,),
        split="validation",
        bootstrap_samples=100,
        seed=7,
    )

    key_result = result["keys"]["10"]
    assert key_result["evaluated_groups"] == 1
    assert key_result["model_a"]["macro"]["overall"]["macro_accuracy"] == 0.5
    assert key_result["model_b"]["macro"]["overall"]["macro_accuracy"] == 0.5
    assert key_result["paired_delta_b_minus_a"]["overall"]["mean_delta"] == 0.0


def test_cross_model_eval_rejects_evaluation_group_seen_in_training(tmp_path):
    overlapping_group = "audio:leaked"
    report_a, _ = _report(
        "manifest-a", CACHE_REVISION,
        [_training_map(overlapping_group, 10)])
    report_b, _ = _report(
        "manifest-b", LN_TRANSITION_CACHE_REVISION,
        [_training_map("audio:train-b", 10)])
    sample_report, _ = _report(
        "manifest-samples",
        LN_TRANSITION_CACHE_REVISION,
        [{
            "id": 321,
            "sha256": "d" * 64,
            "keys": 10,
            "split": "validation",
            "group": overlapping_group,
            "training_windows_before_group_budget": 2,
            "training_windows": 2,
        }],
    )

    with pytest.raises(ValueError, match="training groups overlap"):
        compare(
            _write_json(tmp_path / "report-a.json", report_a),
            _write_json(tmp_path / "model-a.json", _model("manifest-a")),
            _write_json(tmp_path / "report-b.json", report_b),
            _write_json(tmp_path / "model-b.json", _model("manifest-b")),
            _write_json(tmp_path / "samples.json", sample_report),
            tmp_path / "cache",
            keys=(10,),
            split="validation",
            bootstrap_samples=100,
            seed=7,
        )


def test_paired_group_delta_rejects_mismatched_candidate_counts():
    counts_a = {
        "group": {
            "overall": [1.0, 2],
            "ln": [1.0, 1],
            "tap": [0.0, 1],
        },
    }
    counts_b = {
        "group": {
            "overall": [1.0, 3],
            "ln": [1.0, 1],
            "tap": [0.0, 2],
        },
    }

    try:
        summarize_paired_group_deltas(counts_a, counts_b, bootstrap_samples=100)
    except ValueError as exc:
        assert "Unpaired candidate counts" in str(exc)
    else:
        raise AssertionError("Expected mismatched candidate counts to be rejected")
