import hashlib
import json

import numpy as np
import pytest

from manialab.domain import Note
from manialab.pattern_sequence import (
    FEATURE_COUNT,
    FEATURE_SCHEMA_COUNTS,
    FEATURE_SCHEMA_NAMES,
    LN_INTERACTION_FEATURE_NAMES,
    LN_INTERACTION_SCHEMA,
    LN_PHRASE_CONTEXT_FEATURES,
    LN_PHRASE_CONTEXT_FEATURE_NAMES,
    LN_PHRASE_CONTEXT_SCHEMA,
    window_features_with_ln_phrase_context,
)
from scripts.train_sequence_prior import (
    CACHE_REVISION,
    LN_INTERACTION_CACHE_REVISION,
    LN_PHRASE_CONTEXT_CACHE_REVISION,
    LN_TRANSITION_CACHE_REVISION,
    NEGATIVE_POLICY_LN_TRANSITION,
    MIN_SEQUENCE_BEAT_MS,
    MAX_SEQUENCE_BEAT_MS,
    _feature_schema_sha256,
    _find_ln_transition_negatives,
    _ln_transition_signature,
    _load_chart_cache,
    _load_initial_model_weights,
    _manifest_sha256,
    _validate_initial_model_report,
    _evaluation_splits,
    _sequence_beat_clock,
    _sequence_beat_positions,
    _training_keys,
    chart_samples,
)
from scripts.evaluate_sequence_prior_groups import (
    CACHE_FILE_SUFFIXES,
    _validate_report_config,
)
from scripts.train_pattern_prior import split_for


def test_window_training_keeps_fixed_notes_and_generates_feasible_negatives():
    notes = [
        Note(0, 900, 0, id="hold-a"),
        Note(0, 0, 4, id="tap-a"),
        Note(500, 500, 2, id="tap-b"),
        Note(900, 1300, 1, id="hold-b"),
        Note(1500, 1500, 3, id="tap-c"),
    ]
    positive, negative, context = chart_samples(
        notes, 6, 17, windows_per_chart=5, negatives=1)

    assert positive.ndim == 2 and positive.shape[1] == FEATURE_COUNT
    assert negative.shape == (len(positive), 1, FEATURE_COUNT)
    assert len(context) == len(positive)
    assert np.isfinite(positive).all() and np.isfinite(negative).all()


def test_ln_transition_signature_distinguishes_same_hand_hybrid_from_split_hands():
    rows = [(500, [Note(500, 500, lane=4, id="tap")])]
    initial_ends = np.full(6, -100000, dtype=np.int64)
    initial_last = initial_ends.copy()
    initial_ends[0], initial_last[0] = 900, 0

    same_hand = _ln_transition_signature(
        rows, [(2,)], 6, [500], initial_ends, initial_last)
    other_hand = _ln_transition_signature(
        rows, [(4,)], 6, [500], initial_ends, initial_last)

    assert dict(same_hand)["same_hand_hybrid_taps"] == 1
    assert dict(other_hand)["same_hand_hybrid_taps"] == 0
    assert same_hand != other_hand


def test_ln_transition_signature_distinguishes_inverse_repress_from_other_lane():
    rows = [(1000, [Note(1000, 1500, lane=0, id="next-hold")])]
    initial_ends = np.full(6, -100000, dtype=np.int64)
    initial_last = initial_ends.copy()
    initial_ends[0], initial_last[0] = 900, 0

    inverse = _ln_transition_signature(
        rows, [(0,)], 6, [500], initial_ends, initial_last)
    other_lane = _ln_transition_signature(
        rows, [(2,)], 6, [500], initial_ends, initial_last)

    assert dict(inverse)["inverse_heads"] == 1
    assert dict(other_lane)["inverse_heads"] == 0


def test_ln_transition_signature_keeps_recent_release_before_window_edge():
    rows = [(500, [Note(500, 500, lane=4, id="tap")])]
    initial_ends = np.full(6, -100000, dtype=np.int64)
    initial_last = initial_ends.copy()
    initial_ends[0], initial_last[0] = 450, 0
    initial_lns = ((0, 450, 0),)

    same_hand = _ln_transition_signature(
        rows, [(2,)], 6, [500], initial_ends, initial_last, initial_lns)
    other_hand = _ln_transition_signature(
        rows, [(4,)], 6, [500], initial_ends, initial_last, initial_lns)

    assert dict(same_hand)["near_release_press_hands"] == 1
    assert dict(other_hand)["near_release_press_hands"] == 0


def test_ln_transition_policy_finds_fixed_time_feasible_hard_negatives():
    notes = [
        Note(0, 900, lane=0, id="hold"),
        Note(500, 500, lane=4, id="tap-under-hold"),
        Note(1000, 1000, lane=5, id="tap-after-release"),
    ]

    positive, negative, context, diagnostics = chart_samples(
        notes,
        6,
        31,
        windows_per_chart=3,
        negatives=1,
        negative_policy=NEGATIVE_POLICY_LN_TRANSITION,
        return_diagnostics=True,
    )

    assert len(positive) == len(negative) == 2
    assert context.all()
    assert diagnostics["ln_transition_negative_examples_found"] >= 1
    assert diagnostics["ln_transition_search_trials"] >= 1
    for pos, neg in zip(positive, negative[:, 0]):
        assert not np.array_equal(pos, neg)
        pos_rows = pos[: 4 * 94].reshape(4, 94)[:, 90:]
        neg_rows = neg[: 4 * 94].reshape(4, 94)[:, 90:]
        np.testing.assert_array_equal(pos_rows, neg_rows)


def test_ln_interaction_schema_appends_features_and_requires_hard_negatives():
    notes = [
        Note(0, 900, lane=0, id="hold"),
        Note(500, 500, lane=4, id="tap-under-hold"),
        Note(1000, 1000, lane=5, id="tap-after-release"),
    ]
    positive, negative, context = chart_samples(
        notes,
        6,
        31,
        windows_per_chart=3,
        negatives=1,
        negative_policy=NEGATIVE_POLICY_LN_TRANSITION,
        feature_schema=LN_INTERACTION_SCHEMA,
    )

    assert positive.shape[1] == FEATURE_SCHEMA_COUNTS[LN_INTERACTION_SCHEMA]
    assert negative.shape == (len(positive), 1, positive.shape[1])
    assert context.all()
    base_positive, base_negative, base_context = chart_samples(
        notes,
        6,
        31,
        windows_per_chart=3,
        negatives=1,
        negative_policy=NEGATIVE_POLICY_LN_TRANSITION,
    )
    np.testing.assert_array_equal(positive[:, :FEATURE_COUNT], base_positive)
    np.testing.assert_array_equal(
        negative[:, :, :FEATURE_COUNT], base_negative
    )
    np.testing.assert_array_equal(context, base_context)

    with pytest.raises(ValueError, match="require the LN-transition"):
        chart_samples(
            notes, 6, 31, feature_schema=LN_INTERACTION_SCHEMA
        )


def test_phrase_context_samples_rolling_eight_beats_and_requires_hard_negatives():
    notes = [Note(0, 3000, lane=0, id="long-ln")]
    notes.extend(
        Note(start, start, lane=1 + (index % 5), id=f"tap-{index}")
        for index, start in enumerate(range(250, 5001, 250))
    )

    positive, negative, context = chart_samples(
        notes,
        6,
        37,
        windows_per_chart=10,
        negatives=1,
        negative_policy=NEGATIVE_POLICY_LN_TRANSITION,
        feature_schema=LN_PHRASE_CONTEXT_SCHEMA,
    )

    assert positive.shape[1] == FEATURE_SCHEMA_COUNTS[LN_PHRASE_CONTEXT_SCHEMA]
    assert negative.shape == (len(positive), 1, positive.shape[1])
    assert context.any()
    summary_start = FEATURE_COUNT + len(LN_INTERACTION_FEATURE_NAMES)
    phrase_width = len(LN_PHRASE_CONTEXT_FEATURE_NAMES)
    tap_index = LN_PHRASE_CONTEXT_FEATURES.index("tap_heads")
    phrase = positive[:, summary_start:].reshape(
        len(positive), 16, phrase_width // 16
    )
    assert np.any(phrase[:, :, tap_index].sum(axis=1) > 4 / 6)

    with pytest.raises(ValueError, match="require the LN-transition"):
        chart_samples(
            notes,
            6,
            37,
            feature_schema=LN_PHRASE_CONTEXT_SCHEMA,
        )


def test_phrase_hard_negative_changes_a_contiguous_multirow_layout():
    rows = [
        (1000, [Note(1000, 3000, lane=0, id="hold")]),
        (1250, [Note(1250, 1250, lane=2, id="tap-a")]),
        (1500, [Note(1500, 1500, lane=2, id="tap-b")]),
    ]
    options = [[(0,)], [(1,), (2,), (4,)], [(1,), (2,), (4,)]]
    actual = [(0,), (2,), (2,)]
    initial = np.full(6, -100000, dtype=np.int64)

    positive = window_features_with_ln_phrase_context(
        [
            {
                "start_ms": start,
                "notes": [
                    (int(lane), int(note.end))
                    for lane, note in zip(lanes, row_notes)
                ],
            }
            for (start, row_notes), lanes in zip(rows, actual)
        ],
        6,
        [500, 500, 500],
        initial_ends=initial,
        initial_last=initial,
    )
    diagnostics = {
        "phrase_multirow_search_trials": 0,
        "phrase_single_row_fallback_trials": 0,
        "phrase_multirow_negatives_found": 0,
        "phrase_single_row_negatives_found": 0,
    }
    found, _, _ = _find_ln_transition_negatives(
        rows,
        options,
        actual,
        6,
        [500, 500, 500],
        initial,
        initial,
        np.random.default_rng(41),
        1,
        feature_schema=LN_PHRASE_CONTEXT_SCHEMA,
        phrase_search_diagnostics=diagnostics,
    )

    assert len(found) == 1
    assert diagnostics["phrase_multirow_search_trials"] > 0
    assert diagnostics["phrase_multirow_negatives_found"] == 1
    assert diagnostics["phrase_single_row_fallback_trials"] == 0
    local_positive = positive[: 4 * 94].reshape(4, 94)
    local_negative = found[0][: 4 * 94].reshape(4, 94)
    changed_rows = np.any(local_positive != local_negative, axis=1)
    assert changed_rows.sum() >= 2


def test_ln_interaction_negatives_ignore_unencoded_pre_window_releases():
    rows = [(500, [Note(500, 500, lane=2, id="tap")])]
    options = [[(2,), (4,)]]
    actual = [(2,)]
    initial_ends = np.full(6, -100000, dtype=np.int64)
    initial_last = initial_ends.copy()
    initial_ends[0] = initial_last[0] = 460
    initial_lns = ((0, 450, 0),)

    legacy, _, _ = _find_ln_transition_negatives(
        rows, options, actual, 6, [500], initial_ends, initial_last,
        np.random.default_rng(3), 1, initial_lns,
    )
    interaction, _, _ = _find_ln_transition_negatives(
        rows, options, actual, 6, [500], initial_ends, initial_last,
        np.random.default_rng(3), 1, initial_lns, LN_INTERACTION_SCHEMA,
    )

    assert len(legacy) == 1
    assert interaction == []


def test_window_training_returns_empty_arrays_for_unperturbable_full_chords():
    notes = [Note(0, 0, lane, id=str(lane)) for lane in range(6)]
    positive, negative, context = chart_samples(
        notes, 6, 19, windows_per_chart=4, negatives=1)

    assert positive.shape == (0, FEATURE_COUNT)
    assert negative.shape == (0, 1, FEATURE_COUNT)
    assert context.shape == (0,)


def test_window_diagnostics_report_when_every_sampled_window_fails():
    notes = [Note(0, 0, lane, id=str(lane)) for lane in range(6)]
    positive, negative, context, diagnostics = chart_samples(
        notes, 6, 23, windows_per_chart=4, negatives=1,
        return_diagnostics=True,
    )

    assert positive.shape == (0, FEATURE_COUNT)
    assert negative.shape == (0, 1, FEATURE_COUNT)
    assert context.shape == (0,)
    assert diagnostics["candidate_event_windows"] == 1
    assert diagnostics["sampled_window_attempts"] == 1
    assert diagnostics["retained_windows"] == 0
    assert diagnostics["negative_failed_windows"] == 1
    assert diagnostics["negative_examples_found"] == 0
    assert sum(
        diagnostics["max_chord_size_histograms"]["negative_failed"].values()
    ) == 1


def test_window_diagnostics_report_partial_negative_failures():
    notes = [
        Note(0, 0, lane, id=f"chord-{lane}")
        for lane in range(6)
    ] + [Note(500, 500, 2, id="tap")]

    positive, negative, context, diagnostics = chart_samples(
        notes, 6, 29, windows_per_chart=2, negatives=1,
        return_diagnostics=True,
    )

    assert len(positive) == 1
    assert negative.shape == (1, 1, FEATURE_COUNT)
    assert diagnostics["candidate_event_windows"] == 2
    assert diagnostics["sampled_window_attempts"] == 2
    assert diagnostics["retained_windows"] == 1
    assert diagnostics["negative_failed_windows"] == 1
    assert diagnostics["negative_examples_found"] == 1
    assert diagnostics["sampled_ln_windows"] == 0
    assert diagnostics["sampled_tap_windows"] == 2
    assert diagnostics["retained_ln_windows"] == int(np.count_nonzero(context))
    assert sum(diagnostics["max_chord_size_histograms"]["sampled"].values()) == 2
    assert sum(diagnostics["max_chord_size_histograms"]["retained"].values()) == 1
    assert sum(
        diagnostics["max_chord_size_histograms"]["negative_failed"].values()
    ) == 1


def test_feature_schema_hash_changes_when_feature_names_change():
    names = ("lane_attack", "hold_duration")

    assert _feature_schema_sha256(names) == _feature_schema_sha256(names)
    assert _feature_schema_sha256(names) != _feature_schema_sha256(
        ("lane_attack", "hold_remaining")
    )


def test_sequence_beat_clock_ignores_implausible_red_lines_and_keeps_previous_period():
    timing_points = [
        "0,500,4,2,0,70,1,0",
        "1000,10001,4,2,0,70,1,0",
        "1250,-100,4,2,0,70,1,0",
        "1500,250,4,2,0,70,0,0",
        "2000,600,4,2,0,70,1,0",
    ]

    clock, ignored = _sequence_beat_clock(timing_points)

    assert ignored == 2
    assert clock(1200) == 500
    assert clock(1750) == 500
    assert clock(2000) == 600


def test_sequence_beat_clock_uses_fallback_when_no_valid_red_line_remains():
    clock, ignored = _sequence_beat_clock(
        ["0,1000000000,4,2,0,70,1,0"], fallback=480
    )

    assert ignored == 1
    assert clock(5000) == 480


def test_sequence_beat_clock_counts_nonfinite_and_nonpositive_red_lines():
    clock, ignored = _sequence_beat_clock([
        "0,nan,4,2,0,70,1,0",
        "100,-1,4,2,0,70,1,0",
        "200,inf,4,2,0,70,1,0",
    ])

    assert ignored == 3
    assert clock(500) == 500


def test_sequence_beat_clock_ignores_extremely_short_redline_periods():
    timing_points = [
        "-500,372.670807453416,44444,2,0,0,1,8",
        "-499.99,1E-61,44444,2,0,0,1,8",
        "-499.98,372.670807453416,44444,2,0,0,1,8",
        "137,372.670807453416,44444,2,0,0,1,8",
    ]

    clock, ignored = _sequence_beat_clock(timing_points)
    positions = _sequence_beat_positions(
        [-500, -499.98, 0, 137, 23987], timing_points
    )

    assert MIN_SEQUENCE_BEAT_MS == 10
    assert ignored == 1
    assert np.all(np.diff(positions) > 0)
    assert clock(-499.98) == pytest.approx(372.670807453416)


def test_sequence_timing_limit_stays_within_float16_feature_range():
    assert MAX_SEQUENCE_BEAT_MS / 1000 < np.finfo(np.float16).max


def test_chart_cache_rejects_old_or_changed_feature_schema(tmp_path):
    cache = tmp_path / "chart.npz"
    stored_config = {
        "cache_revision": CACHE_REVISION,
        "feature_count": FEATURE_COUNT,
        "feature_schema_sha256": _feature_schema_sha256(),
    }
    np.savez_compressed(
        cache,
        positive=np.empty((0, FEATURE_COUNT), np.float16),
        negative=np.empty((0, 1, FEATURE_COUNT), np.float16),
        context=np.empty((0,), bool),
        diagnostics=np.asarray("{}"),
        config=np.asarray(json.dumps(stored_config)),
    )

    with pytest.raises(ValueError, match="configuration mismatch"):
        _load_chart_cache(
            cache,
            {**stored_config, "feature_schema_sha256": "changed-schema"},
        )
    with pytest.raises(ValueError, match="configuration mismatch"):
        _load_chart_cache(
            cache,
            {**stored_config, "cache_revision": "sequence-window-cache-v1"},
        )
    with pytest.raises(ValueError, match="configuration mismatch"):
        _load_chart_cache(cache, {"feature_count": FEATURE_COUNT})


def test_ln_transition_policy_has_an_isolated_cache_revision():
    assert CACHE_REVISION == "sequence-window-cache-v3"
    assert LN_TRANSITION_CACHE_REVISION == "sequence-window-cache-v4-ln-transition"
    assert LN_INTERACTION_CACHE_REVISION == "sequence-window-cache-v5-ln-interactions"
    assert LN_PHRASE_CONTEXT_CACHE_REVISION == (
        "sequence-window-cache-v8-ln-phrase-context"
    )
    assert CACHE_FILE_SUFFIXES[LN_PHRASE_CONTEXT_CACHE_REVISION] == (
        "sequence-window-v8-ln-phrase-context.npz"
    )


def test_phrase_context_report_requires_integrated_timing_and_revision():
    feature_names = list(FEATURE_SCHEMA_NAMES[LN_PHRASE_CONTEXT_SCHEMA])
    report = {
        "revision": LN_PHRASE_CONTEXT_SCHEMA,
        "feature_names": feature_names,
        "prep": {"configs": {
            "cache_revision": LN_PHRASE_CONTEXT_CACHE_REVISION,
            "feature_count": FEATURE_SCHEMA_COUNTS[LN_PHRASE_CONTEXT_SCHEMA],
            "feature_schema_sha256": _feature_schema_sha256(feature_names),
            "feature_schema": LN_PHRASE_CONTEXT_SCHEMA,
            "beat_reference": "integrated_redline_beats",
            "phrase_context_beats": 8.0,
            "min_sequence_beat_ms": MIN_SEQUENCE_BEAT_MS,
        }},
    }

    assert _validate_report_config(report)["cache_revision"] == (
        LN_PHRASE_CONTEXT_CACHE_REVISION
    )

    changed = json.loads(json.dumps(report))
    changed["prep"]["configs"]["phrase_context_beats"] = 4.0
    with pytest.raises(ValueError, match="Phrase-context report/cache schema"):
        _validate_report_config(changed)


def test_training_key_selection_defaults_to_all_available_or_one_specialist():
    data = {
        (6, "train"): object(),
        (6, "validation"): object(),
        (10, "train"): object(),
        (10, "validation"): object(),
    }

    assert _training_keys(data) == [6, 10]
    assert _training_keys(data, [10]) == [10]


def test_training_key_selection_requires_validation_for_each_requested_key():
    data = {(10, "train"): object()}

    with pytest.raises(ValueError, match="training and validation"):
        _training_keys(data, [10])


def test_song_group_split_seed_preserves_legacy_default_and_is_deterministic():
    group = "group-1"
    legacy = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
    expected_legacy = (
        "test" if legacy == 0 else "validation" if legacy == 1 else "train"
    )

    assert split_for(group) == expected_legacy
    assert split_for(group, seed=20261112) == "test"
    assert split_for(group, seed=20261112) == split_for(group, seed=20261112)


def test_optional_test_evaluation_preserves_default_and_can_be_skipped():
    assert _evaluation_splits(False) == ("train", "validation", "test")
    assert _evaluation_splits(True) == ("train", "validation")


def _initial_model_artifact():
    feature_count = FEATURE_SCHEMA_COUNTS[LN_INTERACTION_SCHEMA]
    shapes = ((128, feature_count), (64, 128), (1, 64))
    return {
        "revision": LN_INTERACTION_SCHEMA,
        "features": list(FEATURE_SCHEMA_NAMES[LN_INTERACTION_SCHEMA]),
        "training_keys": [6, 7, 8, 9, 10],
        "training_manifest_sha256": _manifest_sha256(_initial_manifest()),
        "layers": [
            {
                "weight": np.full(shape, index + 1, dtype=np.float32).tolist(),
                "bias": np.full(shape[0], index + 1, dtype=np.float32).tolist(),
            }
            for index, shape in enumerate(shapes)
        ],
    }


def _initial_manifest():
    return [
        dict(group="song:train", keys=10, split="train", windows=1),
        dict(group="song:validation", keys=10, split="validation", windows=1),
        dict(group="song:test", keys=10, split="test", windows=1),
    ]


def _initial_report(artifact):
    return {
        "revision": artifact["revision"],
        "feature_names": artifact["features"],
        "training_keys": artifact["training_keys"],
        "training_manifest_sha256": artifact["training_manifest_sha256"],
        "manifest": _initial_manifest(),
    }


def test_initial_model_rejects_new_split_that_trained_on_evaluation_song():
    artifact = _initial_model_artifact()
    report = _initial_report(artifact)
    new_manifest = [dict(row) for row in _initial_manifest()]
    new_manifest[0]["split"] = "validation"
    new_manifest[1]["split"] = "train"

    with pytest.raises(ValueError, match="same frozen training manifest"):
        _validate_initial_model_report(artifact, report, new_manifest)


def test_initial_model_rejects_manifest_or_artifact_hash_mismatch():
    artifact = _initial_model_artifact()
    report = _initial_report(artifact)
    report["manifest"][0]["windows"] += 1

    with pytest.raises(ValueError, match="same frozen training manifest"):
        _validate_initial_model_report(artifact, report, _initial_manifest())

    report = _initial_report(artifact)
    artifact["training_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="same frozen training manifest"):
        _validate_initial_model_report(artifact, report, _initial_manifest())


@pytest.mark.parametrize(
    ("part", "marker"),
    [
        ("report", "initial_model"),
        ("report", "initial_model_sha256"),
        ("report", "initial_training_manifest_sha256"),
        ("model", "initial_training_manifest_sha256"),
    ],
)
def test_initial_model_rejects_untracked_warm_start_ancestors(part, marker):
    artifact = _initial_model_artifact()
    report = _initial_report(artifact)
    (report if part == "report" else artifact)[marker] = "older-split"

    with pytest.raises(ValueError, match="ancestor manifest proof"):
        _validate_initial_model_report(artifact, report, _initial_manifest())


def test_initial_model_accepts_tracked_ancestors_only_on_same_manifest():
    artifact = _initial_model_artifact()
    report = _initial_report(artifact)
    manifest_hash = _manifest_sha256(_initial_manifest())
    report.update(initial_model="same-split/model.json",
                  initial_training_manifest_sha256=manifest_hash)
    artifact["initial_training_manifest_sha256"] = manifest_hash
    assert _validate_initial_model_report(
        artifact, report, _initial_manifest()) == manifest_hash

    artifact["initial_training_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ancestor manifest proof"):
        _validate_initial_model_report(artifact, report, _initial_manifest())


@pytest.mark.parametrize(
    ("part", "field", "value"),
    [
        ("report", "initial_training_manifest_sha256", "0" * 64),
        ("model", "initial_training_manifest_sha256", "0" * 64),
    ],
)
def test_initial_model_rejects_one_sided_or_mismatched_ancestor_proofs(
        part, field, value):
    artifact = _initial_model_artifact()
    report = _initial_report(artifact)
    (report if part == "report" else artifact)[field] = value

    with pytest.raises(ValueError, match="ancestor manifest proof"):
        _validate_initial_model_report(artifact, report, _initial_manifest())


def _ln_interaction_model(torch):
    feature_count = FEATURE_SCHEMA_COUNTS[LN_INTERACTION_SCHEMA]
    return torch.nn.Sequential(
        torch.nn.Linear(feature_count, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1))


def test_initial_model_warm_start_copies_compatible_weights_and_hashes_source(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "initial-model.json"
    artifact = _initial_model_artifact()
    raw = json.dumps(artifact, separators=(",", ":"))
    path.write_text(raw, encoding="utf-8")
    source_report = _initial_report(artifact)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(source_report), encoding="utf-8")
    model = _ln_interaction_model(torch)

    provenance = _load_initial_model_weights(
        model, path, feature_schema=LN_INTERACTION_SCHEMA,
        train_keys=[10], manifest=_initial_manifest(), torch=torch)

    assert provenance["initial_model_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert provenance["initial_report_sha256"] == hashlib.sha256(
        report_path.read_bytes()).hexdigest()
    assert provenance["initial_training_manifest_sha256"] == _manifest_sha256(
        _initial_manifest())
    layers = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    assert torch.all(layers[0].weight == 1)
    assert torch.all(layers[1].weight == 2)
    assert torch.all(layers[2].weight == 3)

    source_report["model_sha256"] = "0" * 64
    report_path.write_text(json.dumps(source_report), encoding="utf-8")
    with pytest.raises(ValueError, match="model SHA256 does not match"):
        _load_initial_model_weights(
            _ln_interaction_model(torch), path,
            feature_schema=LN_INTERACTION_SCHEMA,
            train_keys=[10], manifest=_initial_manifest(), torch=torch)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda artifact: artifact.update(revision="sequence-window-v1"),
         "feature schema"),
        (lambda artifact: artifact.update(training_keys=[6, 7, 8, 9]),
         "requested training key"),
        (lambda artifact: artifact["layers"][0].update(weight=[[0.0]]),
         "weight shape"),
    ],
)
def test_initial_model_warm_start_rejects_incompatible_artifacts(
        tmp_path, update, message):
    torch = pytest.importorskip("torch")
    artifact = _initial_model_artifact()
    update(artifact)
    path = tmp_path / "initial-model.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    (tmp_path / "report.json").write_text(
        json.dumps(_initial_report(artifact)), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_initial_model_weights(
            _ln_interaction_model(torch), path,
            feature_schema=LN_INTERACTION_SCHEMA,
            train_keys=[10], manifest=_initial_manifest(), torch=torch)
