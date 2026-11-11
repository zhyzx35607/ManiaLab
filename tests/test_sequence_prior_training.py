import json

import numpy as np
import pytest

from manialab.domain import Note
from manialab.pattern_sequence import FEATURE_COUNT
from scripts.train_sequence_prior import (
    CACHE_REVISION,
    LN_TRANSITION_CACHE_REVISION,
    NEGATIVE_POLICY_LN_TRANSITION,
    MAX_SEQUENCE_BEAT_MS,
    _feature_schema_sha256,
    _ln_transition_signature,
    _load_chart_cache,
    _sequence_beat_clock,
    chart_samples,
)


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
