import hashlib
import json

import numpy as np
import pytest

from manialab.domain import Note
from scripts.research_expression_actions import (
    build_report,
)
from scripts.research_expression_actions import (
    FEATURE_NAMES,
    TARGET_NAMES,
    extract_chart_windows,
    fit_ridge,
    predict_ridge,
    predict_stratified_baseline,
    score_group_macro,
)
from scripts.train_pattern_prior import split_for


def _smoke_chart_text(beatmap_id):
    keys = 6
    objects = [
        (0, 2000, 0),
        (500, 500, 1),
        (1000, 2500, 2),
        (1000, 2000, 3),
    ]
    hit_objects = []
    for start, end, lane in objects:
        x = int((lane + 0.5) * 512 / keys)
        if end > start:
            hit_objects.append(f"{x},192,{start},128,0,{end}:0:0:0:0:")
        else:
            hit_objects.append(f"{x},192,{start},1,0,0:0:0:0:")
    return "\n".join((
        "osu file format v14",
        "[General]",
        "Mode:3",
        "[Metadata]",
        "Title:Expression smoke",
        "Artist:Test",
        "Creator:Test",
        f"BeatmapID:{beatmap_id}",
        "BeatmapSetID:-1",
        "Version:10K",
        "[Difficulty]",
        "CircleSize:6",
        "[TimingPoints]",
        "0,500,4,2,0,70,1,0",
        "[HitObjects]",
        *hit_objects,
        "",
    ))


def test_build_report_uses_song_groups_and_never_reads_test_chart(tmp_path):
    corpus = tmp_path / "corpus"
    maps = corpus / "maps"
    maps.mkdir(parents=True)
    names_by_split = {"train": [], "validation": [], "test": []}
    for index in range(10000):
        name = f"song-{index}"
        split = split_for("song:" + name)
        if len(names_by_split[split]) < (2 if split != "test" else 1):
            names_by_split[split].append(name)
        if all(len(names_by_split[key]) == (2 if key != "test" else 1)
               for key in names_by_split):
            break

    manifest = []
    test_ids = set()
    beatmap_id = 100
    for split, names in names_by_split.items():
        for name in names:
            beatmap_id += 1
            record = dict(
                beatmap_id=beatmap_id,
                beatmapset_id=0,
                song_key=name,
                keys=6,
                approved=1,
                eligible=True,
                title="Expression smoke",
                version="10K",
                tags="",
            )
            if split == "test":
                record["sha256"] = hashlib.sha256(name.encode()).hexdigest()
                test_ids.add(beatmap_id)
            else:
                raw = _smoke_chart_text(beatmap_id).encode("utf-8")
                (maps / f"{beatmap_id}.osu").write_bytes(raw)
                record["sha256"] = hashlib.sha256(raw).hexdigest()
            manifest.append(record)
    (corpus / "manifest-smoke.jsonl").write_text(
        "\n".join(json.dumps(record) for record in manifest),
        encoding="utf-8",
    )

    report = build_report(corpus, tmp_path / "report.json")

    assert report["split"]["train"]["groups"] == 2
    assert report["split"]["validation"]["groups"] == 2
    assert report["split"]["train"]["charts"] == 2
    assert report["split"]["validation"]["charts"] == 2
    assert test_ids.isdisjoint(
        {row["beatmap_id"] for row in report["source_maps"]}
    )
    assert (tmp_path / "report.json").exists()


def test_action_windows_use_attacks_as_features_and_authored_holds_as_targets():
    notes = [
        Note(0, 2000, 0),
        Note(500, 500, 1),
        Note(1000, 2500, 2),
        Note(1000, 2000, 3),
        Note(1000, 1050, 4),
        Note(1100, 2200, 4),
    ]
    parsed = dict(
        keys=6,
        notes=notes,
        timing=["0,500,4,2,0,70,1,0"],
    )

    features, targets = extract_chart_windows(parsed)
    target = dict(zip(TARGET_NAMES, targets[0]))

    assert features.shape == (1, len(FEATURE_NAMES))
    assert target["ln_fraction"] == pytest.approx(5 / 6)
    assert target["ln_median_beats"] == pytest.approx(2.2)
    assert target["ln_long_fraction_2beat"] == pytest.approx(4 / 5)
    assert target["ln_chord_fraction"] == pytest.approx(3 / 5)
    assert target["staggered_release_fraction"] == pytest.approx(1.0)
    assert target["hybrid_tap_fraction"] == pytest.approx(1.0)
    assert target["inverse_ln_fraction"] == pytest.approx(1 / 5)


def test_action_features_do_not_leak_ln_endpoints():
    notes = [Note(0, 1500, 0), Note(500, 500, 1), Note(1000, 2000, 2)]
    parsed = dict(keys=6, notes=notes, timing=["0,500,4,2,0,70,1,0"])
    original_x, original_y = extract_chart_windows(parsed)
    taps = [Note(note.start, note.start, note.lane) for note in notes]
    changed_x, changed_y = extract_chart_windows(parsed | {"notes": taps})

    np.testing.assert_array_equal(original_x, changed_x)
    assert original_y[0, TARGET_NAMES.index("ln_fraction")] > 0
    assert changed_y[0, TARGET_NAMES.index("ln_fraction")] == 0


def test_ln_duration_integrates_across_red_line_changes():
    parsed = dict(
        keys=6,
        notes=[Note(1500, 3000, 0)],
        timing=[
            "0,500,4,2,0,70,1,0",
            "2000,250,4,2,0,70,1,0",
        ],
    )
    _, targets = extract_chart_windows(parsed)
    target = dict(zip(TARGET_NAMES, targets[0]))

    assert target["ln_median_beats"] == pytest.approx(5.0)


def test_group_weighted_ridge_and_stratified_baseline_are_finite():
    feature_count = len(FEATURE_NAMES)
    features = np.zeros((6, feature_count), dtype=np.float64)
    features[:, 0] = 1.0
    features[:, 5] = np.arange(6) / 5
    features[:, FEATURE_NAMES.index("attack_rows_per_beat")] = np.arange(6) / 10
    targets = np.full((6, len(TARGET_NAMES)), np.nan, dtype=np.float64)
    targets[:, TARGET_NAMES.index("ln_fraction")] = np.arange(6) / 5
    groups = ["a", "a", "b", "b", "c", "c"]

    model = fit_ridge(features, targets, groups, "ln_fraction")
    predictions = predict_ridge(model, features, "ln_fraction")
    baseline, _ = predict_stratified_baseline(
        features[:4], targets[:4], np.full(4, 6), groups[:4],
        features[4:], np.full(2, 6), "ln_fraction",
    )
    score = score_group_macro(targets[4:], baseline, groups[4:], "ln_fraction")

    assert predictions.shape == (6,)
    assert np.isfinite(predictions).all()
    assert baseline.shape == (2,)
    assert score["samples"] == 2
    assert score["groups"] == 1


def test_stratified_baseline_reweights_song_groups_within_key_fallback():
    features = np.zeros((8, len(FEATURE_NAMES)), dtype=np.float64)
    features[:, FEATURE_NAMES.index("chart_progress")] = 0.2
    features[:, FEATURE_NAMES.index("attack_rows_per_beat")] = 0.1
    targets = np.full((8, len(TARGET_NAMES)), np.nan, dtype=np.float64)
    targets[:, TARGET_NAMES.index("ln_fraction")] = [0, 0, 0, 0, 1, 1, 1, 1]
    keys = np.asarray([6, 6, 6, 7, 6, 7, 7, 7])
    groups = ["a"] * 4 + ["b"] * 4
    evaluation_features = features[:1]

    predictions, _ = predict_stratified_baseline(
        features, targets, keys, groups,
        evaluation_features, np.asarray([6]), "ln_fraction",
        min_cell_groups=3,
    )

    assert predictions[0] == pytest.approx(0.5)


def test_stratified_baseline_reweights_song_groups_within_density_cell():
    features = np.zeros((8, len(FEATURE_NAMES)), dtype=np.float64)
    features[:, FEATURE_NAMES.index("chart_progress")] = 0.2
    features[:, FEATURE_NAMES.index("attack_rows_per_beat")] = [
        0.1, 0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9,
    ]
    targets = np.full((8, len(TARGET_NAMES)), np.nan, dtype=np.float64)
    targets[:, TARGET_NAMES.index("ln_fraction")] = [0, 0, 0, 0, 1, 1, 1, 1]
    keys = np.full(8, 6)
    groups = ["a"] * 4 + ["b"] * 4
    evaluation = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
    evaluation[:, FEATURE_NAMES.index("chart_progress")] = 0.2
    evaluation[:, FEATURE_NAMES.index("attack_rows_per_beat")] = 0.1

    predictions, _ = predict_stratified_baseline(
        features, targets, keys, groups,
        evaluation, np.asarray([6]), "ln_fraction",
        min_cell_groups=2,
    )

    assert predictions[0] == pytest.approx(0.5)


def test_ridge_requires_independent_song_groups():
    features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
    targets = np.zeros((2, len(TARGET_NAMES)), dtype=np.float64)
    with pytest.raises(ValueError, match="independent training groups"):
        fit_ridge(features, targets, ["same", "same"], "ln_fraction")
