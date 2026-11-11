import hashlib
import json
import sys

import numpy as np
import pytest

import scripts.evaluate_sequence_prior_remap as remap
from scripts.evaluate_sequence_prior_remap import (
    _limit_maps_per_group,
    _load_prior_artifacts,
    _selection_metadata,
    select_maps,
    summarize,
)
from manialab.domain import Note, parse_osu
from manialab.pattern_sequence import FEATURE_SCHEMA_COUNTS, FEATURE_SCHEMA_NAMES
from scripts.train_sequence_prior import _feature_schema_sha256


def _write_prior_pair(tmp_path, schema):
    feature_names = list(FEATURE_SCHEMA_NAMES[schema])
    count = FEATURE_SCHEMA_COUNTS[schema]
    manifest_hash = "a" * 64
    artifact = {
        "revision": schema,
        "features": feature_names,
        "training_keys": [10],
        "training_manifest_sha256": manifest_hash,
        "layers": [{
            "weight": np.zeros((1, count), dtype=np.float32).tolist(),
            "bias": [0.0],
        }],
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    cache_revision = (
        "sequence-window-cache-v5-ln-interactions"
        if schema != "sequence-window-v1"
        else "sequence-window-cache-v3"
    )
    config = {
        "cache_revision": cache_revision,
        "feature_count": count,
        "feature_schema_sha256": _feature_schema_sha256(feature_names),
    }
    if schema != "sequence-window-v1":
        config["feature_schema"] = schema
    report = {
        "revision": schema,
        "feature_names": feature_names,
        "training_manifest_sha256": manifest_hash,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "prep": {"configs": config, "sampling_maps": []},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, model_path, report


def _row(beatmap_id, split, group, keys=10):
    return {
        "id": beatmap_id,
        "sha256": f"{beatmap_id:064x}",
        "keys": keys,
        "split": split,
        "group": group,
    }


def test_select_maps_uses_only_requested_song_group_split():
    report = {
        "prep": {
            "sampling_maps": [
                _row(1, "train", "train-group"),
                _row(2, "validation", "validation-group"),
                _row(3, "test", "test-group"),
                _row(4, "test", "other-key-group", keys=9),
            ]
        }
    }

    selected = select_maps(report, 10, "test")

    assert [row["id"] for row in selected] == [3]


def test_select_maps_rejects_duplicate_map_id_within_split():
    report = {
        "prep": {
            "sampling_maps": [
                _row(3, "test", "group-a"),
                _row(3, "test", "group-b"),
            ]
        }
    }

    with pytest.raises(ValueError, match="duplicate"):
        select_maps(report, 10, "test")


def test_limit_maps_per_group_is_seeded_deterministic_and_preserves_coverage():
    rows = [
        _row(beatmap_id, "validation", group)
        for group, beatmap_ids in (
            ("group-a", (1, 2, 3)),
            ("group-b", (4, 5, 6)),
            ("group-c", (7, 8, 9)),
        )
        for beatmap_id in beatmap_ids
    ]

    first = _limit_maps_per_group(rows, 2, 42)
    repeated = _limit_maps_per_group(rows, 2, 42)
    different_seed = _limit_maps_per_group(rows, 2, 43)

    assert [row["id"] for row in first] == [row["id"] for row in repeated]
    assert [row["id"] for row in first] != [
        row["id"] for row in different_seed
    ]
    assert {row["group"] for row in first} == {
        "group-a", "group-b", "group-c"
    }
    assert all(
        sum(row["group"] == group for row in first) <= 2
        for group in ("group-a", "group-b", "group-c")
    )
    assert _limit_maps_per_group(rows, None, 42) == rows
    assert _selection_metadata(None) == {}
    assert _selection_metadata(2) == {"selection_limit_per_group": 2}


def test_limit_maps_per_group_keeps_key_counts_as_separate_groups():
    rows = [
        _row(1, "validation", "shared-audio", keys=6),
        _row(2, "validation", "shared-audio", keys=6),
        _row(3, "validation", "shared-audio", keys=10),
        _row(4, "validation", "shared-audio", keys=10),
    ]

    selected = _limit_maps_per_group(rows, 1, 42)

    assert len(selected) == 2
    assert {row["keys"] for row in selected} == {6, 10}
    assert all(row["group"] == "shared-audio" for row in selected)


@pytest.mark.parametrize("limit", (0, -1, True, 1.5))
def test_limit_maps_per_group_requires_a_positive_integer(limit):
    with pytest.raises(ValueError, match="positive integer"):
        _limit_maps_per_group([_row(1, "validation", "group-a")], limit, 42)


def test_run_records_and_evaluates_seeded_per_group_selection(
        tmp_path, monkeypatch):
    report_path, model_path, report = _write_prior_pair(
        tmp_path, "sequence-window-v1")
    rows = []
    map_root = tmp_path / "maps"
    map_root.mkdir()
    for group_index, group in enumerate(("group-a", "group-b", "group-c")):
        for map_index in range(3):
            beatmap_id = group_index * 10 + map_index + 1
            raw = f"synthetic-map-{beatmap_id}".encode()
            (map_root / f"{beatmap_id}.osu").write_bytes(raw)
            rows.append({
                "id": beatmap_id,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "keys": 10,
                "split": "validation",
                "group": group,
            })
    report["prep"]["sampling_maps"] = rows
    report_path.write_text(json.dumps(report), encoding="utf-8")

    calls = {"baseline": [], "sequence_prior": []}
    monkeypatch.setattr(remap, "ROOT", tmp_path)
    monkeypatch.setattr(
        remap, "parse_osu",
        lambda _: {"keys": 10, "notes": [], "timing": []})
    monkeypatch.setattr(remap, "parameters", lambda settings: settings)
    monkeypatch.setattr(
        remap, "assign_lanes",
        lambda notes, *_args, **kwargs: (
            calls["sequence_prior" if kwargs.get("sequence_prior") is not None
                  else "baseline"].append(list(notes)) or []
        ))
    monkeypatch.setattr(remap, "_validate_roundtrip", lambda *_args: {})
    metric_values = {name: 0.0 for name in remap.METRIC_NAMES}
    monkeypatch.setattr(
        remap, "_chart_metrics",
        lambda *_args: {"metrics": metric_values})

    result = remap.run(
        report_path=report_path,
        model_path=model_path,
        map_root=map_root,
        output=tmp_path / "test-data" / "limited-report.json",
        keys=(10,),
        split="validation",
        strength=0.15,
        seed=123,
        limit_per_group=1,
    )

    assert result["selection_limit_per_group"] == 1
    assert result["run_identity"]["selection_limit_per_group"] == 1
    assert result["selected_maps"] == 3
    assert len(result["rows"]) == 3
    assert len(calls["baseline"]) == len(calls["sequence_prior"]) == 3
    assert {row["group"] for row in result["rows"]} == {
        "group-a", "group-b", "group-c"
    }
    assert all(
        sum(row["group"] == group for row in result["rows"]) == 1
        for group in ("group-a", "group-b", "group-c")
    )


@pytest.mark.parametrize(
    "schema",
    ("sequence-window-v1", "sequence-window-ln-interactions-v1"),
)
def test_prior_artifacts_accept_supported_schema_and_bind_model_report(
        tmp_path, schema):
    report_path, model_path, report = _write_prior_pair(tmp_path, schema)

    loaded_report, prior, model_sha256 = _load_prior_artifacts(
        report_path, model_path)

    assert loaded_report == report
    assert prior.feature_schema == schema
    assert model_sha256 == report["model_sha256"]


def test_prior_artifacts_reject_report_model_hash_mismatch(tmp_path):
    report_path, model_path, report = _write_prior_pair(
        tmp_path, "sequence-window-ln-interactions-v1")
    report["model_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="model SHA256"):
        _load_prior_artifacts(report_path, model_path)


def test_prior_artifacts_reject_model_report_schema_mismatch(tmp_path):
    report_path, model_path, report = _write_prior_pair(
        tmp_path, "sequence-window-ln-interactions-v1")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["revision"] = "sequence-window-v1"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    report.pop("model_sha256")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="model revision"):
        _load_prior_artifacts(report_path, model_path)


def _metric_block(cost, jacks, releases, lane_change, entropy):
    values = {name: 0.0 for name in remap.METRIC_NAMES}
    values.update(
        cost_per_note=cost,
        short_jacks_per_100=jacks,
        release_repeats_per_100=releases,
        lane_change_rate=lane_change,
        ln_reference_lane_change_rate=lane_change,
        ln_lane_entropy=entropy,
    )
    return {"metrics": values}


def test_summary_bootstraps_paired_song_group_differences():
    rows = [
        {
            "keys": 10,
            "group": "a",
            "passed": True,
            "baseline": _metric_block(2.0, 4.0, 1.0, 0.2, 0.5),
            "sequence_prior": _metric_block(1.0, 2.0, 0.0, 0.1, 0.6),
        },
        {
            "keys": 10,
            "group": "b",
            "passed": True,
            "baseline": _metric_block(3.0, 5.0, 2.0, 0.3, 0.4),
            "sequence_prior": _metric_block(4.0, 6.0, 3.0, 0.4, 0.3),
        },
    ]

    result = summarize(rows, bootstrap_samples=500, seed=19)

    metrics = result["by_keys"]["10"]["metrics"]
    assert result["by_keys"]["10"]["groups"] == 2
    assert metrics["cost_per_note"]["baseline_macro"] == 2.5
    assert metrics["cost_per_note"]["sequence_prior_macro"] == 2.5
    assert metrics["cost_per_note"]["delta_prior_minus_baseline"] == 0
    assert metrics["cost_per_note"]["delta_ci_95"] == [-1.0, 1.0]


def test_summary_keeps_key_counts_separate():
    rows = []
    for keys in (6, 10):
        rows.append({
            "keys": keys,
            "group": "same-song",
            "passed": True,
            "baseline": _metric_block(1, 2, 3, 0.1, 0.5),
            "sequence_prior": _metric_block(1, 2, 3, 0.1, 0.5),
        })

    result = summarize(rows, bootstrap_samples=50, seed=9)

    assert set(result["by_keys"]) == {"6", "10"}
    assert result["by_keys"]["6"]["groups"] == 1
    assert result["by_keys"]["10"]["groups"] == 1


def test_validation_freeze_binds_strength_model_report_keys_and_seed():
    report = {
        "revision": "sequence-window-v1",
        "training_manifest_sha256": "a" * 64,
        "prep": {
            "sampling_maps": [
                _row(1, "validation", "validation-group"),
                _row(2, "test", "test-group"),
            ]
        },
    }
    artifact = {
        "revision": "sequence-prior-remap-v4",
        "split": "validation",
        "model_sha256": "b" * 64,
        "report_sha256": "c" * 64,
        "training_manifest_sha256": "a" * 64,
        "keys": [10],
        "seed": 42,
        "score_candidate_cap": remap.SEQUENCE_SCORE_CANDIDATE_CAP,
        "strength": 0.35,
        "selection_limit": None,
        "selected_map_ids": [1],
        "rows": [{"beatmap_id": 1, "passed": True}],
    }

    remap._validate_frozen_validation(
        artifact, report, "b" * 64, "c" * 64, (10,), 0.35, 42
    )
    with pytest.raises(ValueError, match="differs from the validation artifact"):
        remap._validate_frozen_validation(
            artifact, report, "b" * 64, "c" * 64, (10,), 0.2, 42
        )
    with pytest.raises(ValueError, match="limited"):
        remap._validate_frozen_validation(
            {**artifact, "selection_limit": 1},
            report,
            "b" * 64,
            "c" * 64,
            (10,),
            0.35,
            42,
        )
    with pytest.raises(ValueError, match="feature schema"):
        remap._validate_frozen_validation(
            {**artifact, "feature_schema": "sequence-window-ln-interactions-v1"},
            report,
            "b" * 64,
            "c" * 64,
            (10,),
            0.35,
            42,
        )
    with pytest.raises(ValueError, match="limited validation"):
        remap._validate_frozen_validation(
            {**artifact, "selection_limit_per_group": 1},
            report,
            "b" * 64,
            "c" * 64,
            (10,),
            0.35,
            42,
        )


def test_test_split_requires_a_validation_artifact():
    with pytest.raises(ValueError, match="validation-artifact"):
        remap.run(split="test")


def test_test_split_rejects_per_group_limit():
    with pytest.raises(ValueError, match="not allowed for the test split"):
        remap.run(split="test", limit_per_group=1)


def test_cli_forwards_limit_per_group(monkeypatch):
    captured = {}
    monkeypatch.setattr(sys, "argv", [
        "evaluate_sequence_prior_remap.py",
        "--output", "report.json",
        "--strength", "0.15",
        "--limit-per-group", "2",
    ])
    monkeypatch.setattr(remap, "run", lambda **kwargs: captured.update(kwargs))

    remap.main()

    assert captured["limit_per_group"] == 2


def test_roundtrip_rejects_lane_coordinate_mismatch(monkeypatch):
    original_text = """osu file format v14
[General]
Mode:3
[Difficulty]
CircleSize:10
[TimingPoints]
0,500,4,2,0,70,1,0
[HitObjects]
25,192,1000,1,0,0:0:0:0:
"""
    chart = parse_osu(original_text)
    target = [Note(1000, 1000, lane=9, id="i0", raw=chart["notes"][0].raw)]
    monkeypatch.setattr(remap, "export_osu", lambda *args, **kwargs: original_text)

    with pytest.raises(AssertionError, match="lane coordinates"):
        remap._validate_roundtrip(
            target, 10, original_text, chart, None
        )


def test_object_lane_signature_keeps_lane_tied_to_hitobject_identity():
    first = Note(
        1000, 1000, lane=2, id="first",
        raw="100,192,1000,1,0,0:0:0:0:",
    )
    second = Note(
        1000, 1000, lane=7, id="second",
        raw="400,192,1000,1,8,0:0:0:0:",
    )
    swapped = [
        Note(**(first.data() | {"lane": 7})),
        Note(**(second.data() | {"lane": 2})),
    ]

    assert remap._non_lane_signature([first, second]) == (
        remap._non_lane_signature(swapped)
    )
    assert remap._object_lane_signature([first, second]) != (
        remap._object_lane_signature(swapped)
    )


def test_ln_profile_metrics_are_zero_for_reference_layout():
    notes = [
        Note(1000, 2000, lane=2, id="ln-a"),
        Note(1250, 1250, lane=4, id="tap"),
        Note(2000, 2500, lane=2, id="inverse"),
    ]
    result = remap._chart_metrics(
        notes,
        notes,
        6,
        (),
        notes,
        1.0,
        {"max_held": 1, "lane_counts": [0, 0, 2, 0, 1, 0]},
    )

    assert result["metrics"]["ln_reference_lane_change_rate"] == 0
    assert result["metrics"]["ln_inverse_profile_error"] == 0
    assert result["metrics"]["ln_release_only_profile_error"] == 0
    assert result["metrics"]["ln_occupied_key_fraction"] > 0
