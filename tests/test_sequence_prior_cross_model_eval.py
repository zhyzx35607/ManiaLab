import json

import numpy as np

from manialab.pattern_sequence import FEATURE_COUNT, FEATURE_NAMES
from scripts.evaluate_sequence_prior_cross_model import (
    compare,
    summarize_paired_group_deltas,
)
from scripts.train_sequence_prior import (
    CACHE_REVISION,
    LN_TRANSITION_CACHE_REVISION,
    _feature_schema_sha256,
)


def _report(manifest, cache_revision, sampling_maps=None):
    config = {
        "cache_revision": cache_revision,
        "feature_count": FEATURE_COUNT,
        "feature_schema_sha256": _feature_schema_sha256(),
    }
    return {
        "revision": "sequence-window-v1",
        "feature_names": list(FEATURE_NAMES),
        "training_manifest_sha256": manifest,
        "prep": {"configs": config, "sampling_maps": sampling_maps or []},
    }, config


def _model(manifest):
    return {
        "revision": "sequence-window-v1",
        "features": list(FEATURE_NAMES),
        "training_manifest_sha256": manifest,
        "layers": [{
            "weight": np.zeros((1, FEATURE_COUNT), dtype=np.float32).tolist(),
            "bias": [0.0],
        }],
    }


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_cross_model_eval_uses_one_v4_sample_report_for_both_models(tmp_path):
    chart_id = 123
    chart_hash = "a" * 64
    sampling_maps = [{
        "id": chart_id,
        "sha256": chart_hash,
        "keys": 6,
        "split": "test",
        "group": "audio:one",
        "training_windows_before_group_budget": 2,
        "training_windows": 2,
    }]
    report_a, _ = _report("manifest-a", CACHE_REVISION)
    report_b, _ = _report("manifest-b", LN_TRANSITION_CACHE_REVISION)
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
        bootstrap_samples=100,
        seed=7,
    )

    key_result = result["keys"]["6"]
    assert result["samples"]["cache_revision"] == LN_TRANSITION_CACHE_REVISION
    assert key_result["evaluated_groups"] == 1
    assert key_result["model_a"]["macro"]["overall"]["macro_accuracy"] == 0.5
    assert key_result["model_b"]["macro"]["overall"]["macro_accuracy"] == 0.5
    assert key_result["paired_delta_b_minus_a"]["overall"]["mean_delta"] == 0.0


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
