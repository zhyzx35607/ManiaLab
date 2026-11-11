import json

import numpy as np
import pytest

import manialab.layout as layout
from manialab.arrange import parameters
from manialab.domain import Note, invariant, validate
from manialab.pattern_sequence import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_COUNTS,
    FEATURE_SCHEMA_NAMES,
    LN_INTERACTION_SCHEMA,
    ROW_WIDTH,
    window_features,
    window_features_with_ln_interactions,
)
from manialab.sequence_prior import (
    SequencePrior,
    SequenceRow,
    load_sequence_prior,
    score_sequence_windows,
    sequence_beat_clock,
    sequence_window_features,
    sequence_windows_features,
)


def test_sequence_score_shortlist_keeps_every_possible_bounded_winner():
    rng = np.random.default_rng(11)
    for strength in (0.15, 0.35, 0.6, 1.0):
        for _ in range(100):
            values = rng.normal(size=80)
            valid = np.sort(rng.choice(80, size=55, replace=False))
            keep = 16
            shortlist = set(layout._sequence_score_shortlist(
                values, valid, keep, strength).tolist())
            adjustment = rng.uniform(-strength, strength, size=len(valid))
            noise = rng.uniform(0, 0.025, size=len(valid))
            winners = np.argpartition(
                values[valid] + adjustment + noise, keep - 1
            )[:keep]

            assert {int(valid[index]) for index in winners} <= shortlist


def test_sequence_score_candidate_cap_keeps_lowest_cost_candidates():
    values = np.arange(100, dtype=np.float64)[::-1]
    choices = np.arange(100, dtype=np.int64)

    limited = layout._limit_sequence_score_candidates(values, choices, 64)

    assert len(limited) == 64
    assert set(limited) == set(range(36, 100))


def test_sequence_score_candidate_cap_covers_worst_release_ordering():
    theoretical_maximum = 0
    for keys in range(6, 11):
        for size in range(1, keys + 1):
            theoretical_maximum = max(
                theoretical_maximum,
                4 * len(layout.chord_choices(keys, size)),
            )

    worst_case = layout.release_choices(10, np.array([0, 1, 2, 4, 3], dtype=float))
    assert theoretical_maximum == 1008
    assert len(worst_case) == theoretical_maximum
    assert len(layout._limit_sequence_score_candidates(
        np.arange(len(worst_case), dtype=float),
        np.arange(len(worst_case), dtype=np.int64),
        layout.SEQUENCE_SCORE_CANDIDATE_CAP,
    )) == len(worst_case)


def _row(start, beat, notes, keys=6, initial_ends=None, initial_last=None):
    sentinel = -100000
    return SequenceRow(
        start_ms=start,
        beat_ms=beat,
        notes=tuple(notes),
        initial_ends=tuple(initial_ends or [sentinel] * keys),
        initial_last=tuple(initial_last or [sentinel] * keys),
    )


def test_sequence_window_features_match_training_encoder_for_ln_and_variable_tempo():
    initial_ends = [-100000, -100000, 1200, -100000, -100000, -100000]
    initial_last = [-100000, -100000, 800, -100000, -100000, -100000]
    rows = (
        _row(1000, 500, [(0, 1250)], initial_ends=initial_ends, initial_last=initial_last),
        _row(1250, 250, [(0, 1250), (3, 1600)]),
        _row(1500, 250, [(2, 1500)]),
        _row(1750, 250, [(4, 2100)]),
    )

    actual = sequence_window_features(rows, 6)
    expected = window_features(
        [row.feature_row() for row in rows],
        6,
        [row.beat_ms for row in rows],
        initial_ends=initial_ends,
        initial_last=initial_last,
    )

    assert np.array_equal(actual, expected)


def test_ln_interaction_sequence_encoder_matches_training_for_single_and_batch():
    window = (
        _row(1000, 500, [(0, 1500)]),
        _row(1250, 500, [(1, 1250), (5, 1250)]),
        _row(1600, 400, [(0, 1900)]),
    )
    expected = window_features_with_ln_interactions(
        [row.feature_row() for row in window],
        6,
        [row.beat_ms for row in window],
    )

    actual = sequence_window_features(window, 6, LN_INTERACTION_SCHEMA)
    batch = sequence_windows_features(
        [window, window[:2]], 6, LN_INTERACTION_SCHEMA
    )

    assert actual.shape == (FEATURE_SCHEMA_COUNTS[LN_INTERACTION_SCHEMA],)
    assert np.array_equal(actual, expected)
    assert np.array_equal(batch[0], actual)
    assert np.array_equal(
        batch[1],
        sequence_window_features(window[:2], 6, LN_INTERACTION_SCHEMA),
    )


@pytest.mark.parametrize("keys", range(6, 11))
def test_batched_ln_interactions_match_single_window_encoder_for_all_shapes(keys):
    sentinel = -100000
    initial_ends = [sentinel] * keys
    initial_last = [sentinel] * keys
    for lane, last, end in (
        (0, 600, 1000),
        (2, 700, 1060),
        (4, 800, 1450),
        (5, 900, 1600),
    ):
        initial_last[lane] = last
        initial_ends[lane] = end

    starts = (1000, 1320, 1650, 1980)
    beats = (250.0, 333.0, 500.0, 750.0)
    notes = (
        ((0, 1000), (1, 1300)),
        ((2, 1620), (3, 1320)),
        ((2, 1650), (3, 2100), (5, 1650)),
        ((1, 1980),),
    )
    windows = []
    for row_count in range(1, 5):
        rows = []
        for offset in range(row_count):
            rows.append(SequenceRow(
                start_ms=starts[offset],
                beat_ms=beats[offset],
                notes=notes[offset],
                initial_ends=tuple(initial_ends if offset == 0 else [sentinel] * keys),
                initial_last=tuple(initial_last if offset == 0 else [sentinel] * keys),
            ))
        windows.append(tuple(rows))

    actual = sequence_windows_features(windows, keys, LN_INTERACTION_SCHEMA)
    expected = np.stack([
        sequence_window_features(window, keys, LN_INTERACTION_SCHEMA)
        for window in windows
    ])

    assert np.array_equal(actual[:, :FEATURE_COUNT], expected[:, :FEATURE_COUNT])
    np.testing.assert_allclose(
        actual[:, FEATURE_COUNT:],
        expected[:, FEATURE_COUNT:],
        rtol=0,
        atol=1e-7,
    )


def test_batched_sequence_features_match_training_encoder_for_mixed_windows():
    initial_ends = [-100000, -100000, 1200, -100000, -100000, -100000]
    initial_last = [-100000, -100000, 800, -100000, -100000, -100000]
    windows = [
        (
            _row(1000, 500, [(0, 1250)], initial_ends=initial_ends,
                 initial_last=initial_last),
            _row(1250, 250, [(0, 1250)]),
            _row(1500, 250, [(2, 1500)]),
            _row(1750, 250, [(4, 2100)]),
        ),
        (
            _row(1000, 500, [(1, 1000)]),
            _row(1250, 500, [(3, 1600)]),
        ),
        (_row(500, 400, [(4, 500)]),),
    ]

    actual = sequence_windows_features(windows, 6)
    expected = np.stack([
        sequence_window_features(window, 6) for window in windows
    ])

    assert np.array_equal(actual, expected)


def test_sequence_scoring_uses_the_training_mirror_average():
    weights = np.zeros((1, FEATURE_COUNT), dtype=np.float32)
    weights[0, 3 * ROW_WIDTH] = 1
    prior = SequencePrior(
        ({"weight": weights, "bias": np.zeros(1, dtype=np.float32)},),
        "fixture",
    )
    windows = [
        (_row(1000, 500, [(0, 1000)]),),
        (_row(1000, 500, [(5, 1000)]),),
    ]

    scores = score_sequence_windows(windows, 6, prior)

    assert scores.shape == (2,)
    assert scores[0] == pytest.approx(scores[1])


def test_beam_signature_distinguishes_sequence_window_history():
    notes_a = _row(1000, 500, [(0, 1000)])
    notes_b = _row(1000, 500, [(1, 1000)])
    state_args = (
        0.0,
        np.full(6, -100000.0),
        np.full(6, -100000.0),
        np.zeros(6),
        {},
        (),
        1000,
        None,
        (),
    )
    first = layout.State(*state_args, sequence_rows=(notes_a,))
    second = layout.State(*state_args, sequence_rows=(notes_b,))

    assert layout._state_signature(first) != layout._state_signature(second)


def test_sequence_prior_loader_validates_revision_features_and_shapes(tmp_path):
    payload = {
        "revision": "sequence-window-v1",
        "features": list(FEATURE_NAMES),
        "training_manifest_sha256": "0" * 64,
        "layers": [
            {
                "weight": np.zeros((1, FEATURE_COUNT), dtype=np.float32).tolist(),
                "bias": [0.0],
            }
        ],
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    prior = load_sequence_prior(path)

    assert len(prior.sha256) == 64
    assert prior.training_manifest_sha256 == "0" * 64
    assert prior.layers[0]["weight"].shape == (1, FEATURE_COUNT)

    for changed, match in (
        ({**payload, "revision": "sequence-window-v2"}, "revision"),
        ({**payload, "features": ["wrong"]}, "feature schema"),
        ({**payload, "training_manifest_sha256": "not-a-hash"}, "manifest hash"),
        ({**payload, "layers": [{"weight": [[float("nan")] * FEATURE_COUNT], "bias": [0]}]},
         "layer 0"),
        ({**payload, "layers": [{"weight": [[0.0]], "bias": [0]}]}, "layer 0"),
    ):
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_sequence_prior(path)


def test_sequence_prior_loader_accepts_explicit_ln_interaction_schema(tmp_path):
    feature_names = FEATURE_SCHEMA_NAMES[LN_INTERACTION_SCHEMA]
    feature_count = FEATURE_SCHEMA_COUNTS[LN_INTERACTION_SCHEMA]
    payload = {
        "revision": LN_INTERACTION_SCHEMA,
        "features": list(feature_names),
        "training_manifest_sha256": "1" * 64,
        "layers": [
            {
                "weight": np.zeros((1, feature_count), dtype=np.float32).tolist(),
                "bias": [0.0],
            }
        ],
    }
    path = tmp_path / "ln-interactions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    prior = load_sequence_prior(path)

    assert prior.feature_schema == LN_INTERACTION_SCHEMA
    assert prior.layers[0]["weight"].shape == (1, feature_count)


def test_sequence_timing_guard_is_shared_with_training():
    from scripts.train_sequence_prior import _sequence_beat_clock

    points = [
        "0,500,4,2,0,70,1,0",
        "1000,10001,4,2,0,70,1,0",
        "1500,250,4,2,0,70,0,0",
    ]
    runtime_clock, runtime_ignored = sequence_beat_clock(points)
    training_clock, training_ignored = _sequence_beat_clock(points)

    assert runtime_ignored == training_ignored == 1
    assert [runtime_clock(t) for t in (0, 1200, 1800)] == [
        training_clock(t) for t in (0, 1200, 1800)
    ]


def test_beam_sequence_context_retains_full_rows_and_scores_each_window(monkeypatch):
    prior = SequencePrior((), "fixture")
    captured = []

    def capture(windows, keys, model):
        captured.extend(windows)
        return np.zeros(len(windows), dtype=np.float32)

    monkeypatch.setattr(layout, "score_sequence_windows", capture)
    notes = [
        Note(1000, 1250, lane=-1, source="other", id="ln"),
        Note(1250, 1250, lane=-1, source="other", id="release-row"),
        Note(1500, 1500, lane=-1, source="other", id="tap-a"),
        Note(1750, 2100, lane=-1, source="other", id="tap-b"),
    ]
    settings = parameters({
        "keys": 6,
        "pitch_follow": 0,
        "balance": 0,
        "jack_avoid": 0,
        "pattern_prior": 0,
    })

    output = layout.search_layout(
        notes,
        settings,
        width=32,
        timing_points=[
            "0,500,4,2,0,70,1,0",
            "1250,250,4,2,0,70,1,0",
        ],
        sequence_prior=prior,
    )

    assert invariant(output) == invariant(notes)
    assert validate(output, 6)["valid"]
    four_row_windows = [window for window in captured if len(window) == 4]
    assert four_row_windows
    assert any(
        [row.start_ms for row in window] == [1000, 1250, 1500, 1750]
        and [row.beat_ms for row in window] == [500, 250, 250, 250]
        and window[0].notes[0][1] == 1250
        for window in four_row_windows
    )
    release_context = next(
        window for window in captured
        if len(window) >= 2
        if window[0].notes[0][0] == window[1].notes[0][0]
        and window[0].notes[0][1] == window[1].start_ms
    )
    features = sequence_window_features(release_context, 6)
    row_data = features[: 4 * ROW_WIDTH].reshape(4, ROW_WIDTH)
    release_row = 4 - len(release_context) + 1
    lane = release_context[1].notes[0][0]
    assert row_data[release_row, lane * 9 + 7] == 1


def test_beam_window_keeps_ln_occupancy_from_before_its_left_edge(monkeypatch):
    prior = SequencePrior((), "fixture")
    captured = []

    def capture(windows, keys, model):
        captured.extend(windows)
        return np.zeros(len(windows), dtype=np.float32)

    monkeypatch.setattr(layout, "score_sequence_windows", capture)
    notes = [
        Note(500, 3000, lane=-1, source="other", id="long-ln"),
        Note(750, 750, lane=-1, source="other", id="tap-1"),
        Note(1000, 1000, lane=-1, source="other", id="tap-2"),
        Note(1250, 1250, lane=-1, source="other", id="tap-3"),
        Note(1500, 1500, lane=-1, source="other", id="tap-4"),
    ]
    settings = parameters({
        "keys": 6,
        "pitch_follow": 0,
        "balance": 0,
        "jack_avoid": 0,
        "pattern_prior": 0,
    })

    layout.search_layout(notes, settings, width=4, sequence_prior=prior)

    crossing_windows = [
        window for window in captured
        if len(window) == 4 and window[0].start_ms == 750
    ]
    assert crossing_windows
    features = sequence_window_features(crossing_windows[0], 6)
    row_data = features[: 4 * ROW_WIDTH].reshape(4, ROW_WIDTH)
    active_lanes = [
        lane for lane, end in enumerate(crossing_windows[0][0].initial_ends)
        if end > crossing_windows[0][0].start_ms
    ]
    assert active_lanes
    assert any(
        crossing_windows[0][0].initial_last[lane] == 500
        and row_data[0, lane * 9 + 3] == 1
        for lane in active_lanes
    )


def test_sequence_prior_is_rejected_for_locked_or_classic_layouts():
    prior = SequencePrior((), "fixture")
    notes = [Note(1000, 1000, lane=-1, source="other", id="a")]
    locked = [Note(500, 800, lane=0, source="import", id="locked")]

    with pytest.raises(ValueError, match="full-chart"):
        layout.search_layout(notes, parameters({"keys": 6}), locked=locked,
                             sequence_prior=prior)
    with pytest.raises(ValueError, match="beam"):
        from manialab.arrange import assign_lanes
        assign_lanes(notes, {"keys": 6, "layout_engine": "classic"},
                     sequence_prior=prior)
