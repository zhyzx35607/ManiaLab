import pytest

import scripts.evaluate_sequence_prior_remap as remap
from scripts.evaluate_sequence_prior_remap import (
    select_maps,
    summarize,
)
from manialab.domain import Note, parse_osu


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


def test_test_split_requires_a_validation_artifact():
    with pytest.raises(ValueError, match="validation-artifact"):
        remap.run(split="test")


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
