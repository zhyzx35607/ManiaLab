import json

import numpy as np
import pytest

from manialab.domain import Note
from manialab.pattern_sequence import FEATURE_COUNT
from scripts.train_sequence_prior import (
    CACHE_REVISION,
    MAX_SEQUENCE_BEAT_MS,
    _feature_schema_sha256,
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
