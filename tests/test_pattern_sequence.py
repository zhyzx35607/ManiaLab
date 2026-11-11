import numpy as np
import pytest

from manialab.pattern_sequence import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_COUNTS,
    GLOBAL_FEATURES,
    LANE_FEATURES,
    LN_INTERACTION_FEATURE_NAMES,
    LN_INTERACTION_ROW_FEATURES,
    LN_PHRASE_CONTEXT_BINS,
    LN_PHRASE_CONTEXT_FEATURES,
    LN_PHRASE_CONTEXT_FEATURE_NAMES,
    LN_PHRASE_CONTEXT_SCHEMA,
    ROW_FEATURES,
    ROW_WIDTH,
    WINDOW_ROWS,
    mirror_window_features,
    window_features,
    window_features_with_ln_interactions,
    window_features_with_ln_phrase_context,
)


LANE_WIDTH = len(LANE_FEATURES)
ROW_GLOBAL_BASE = 10 * LANE_WIDTH
GLOBAL_BASE = WINDOW_ROWS * ROW_WIDTH


@pytest.mark.parametrize("keys", range(6, 11))
def test_window_features_are_fixed_width_float32_and_right_aligned(keys):
    result = window_features(
        [{"start_ms": 1000, "notes": [(keys - 1, 1500)]}],
        keys,
        500,
    )

    assert result.shape == (FEATURE_COUNT,)
    assert result.dtype == np.float32
    assert len(FEATURE_NAMES) == FEATURE_COUNT
    assert np.isfinite(result).all()
    first_row = result[:ROW_WIDTH]
    last_row = result[3 * ROW_WIDTH : 4 * ROW_WIDTH]
    assert not first_row.any()
    assert last_row[ROW_GLOBAL_BASE] == 1
    assert last_row[ROW_GLOBAL_BASE + 1] == pytest.approx(1 / keys)
    assert result[GLOBAL_BASE + keys - 6] == 1
    assert result[GLOBAL_BASE + len(GLOBAL_FEATURES) - 1] == pytest.approx(0.5)


def test_mirror_reverses_only_valid_lanes_and_is_an_involution():
    keys = 7
    x = window_features(
        [
            {"start_ms": 1000, "notes": [(0, 1800), (5, 1000)]},
            {"start_ms": 1250, "notes": [(3, 1250)]},
        ],
        keys,
        500,
    )
    mirrored = mirror_window_features(x, keys)

    assert np.array_equal(mirror_window_features(mirrored, keys), x)
    assert np.array_equal(
        mirrored[GLOBAL_BASE:],
        x[GLOBAL_BASE:],
    )
    x_rows = x[:GLOBAL_BASE].reshape(WINDOW_ROWS, ROW_WIDTH)
    mirrored_rows = mirrored[:GLOBAL_BASE].reshape(WINDOW_ROWS, ROW_WIDTH)
    for row in range(WINDOW_ROWS):
        x_lanes = x_rows[row, : 10 * LANE_WIDTH].reshape(10, LANE_WIDTH)
        mirrored_lanes = mirrored_rows[row, : 10 * LANE_WIDTH].reshape(10, LANE_WIDTH)
        assert np.array_equal(mirrored_lanes[:keys], x_lanes[:keys][::-1])
        assert np.array_equal(mirrored_lanes[keys:], x_lanes[keys:])
        assert np.array_equal(
            mirrored_rows[row, 10 * LANE_WIDTH :],
            x_rows[row, 10 * LANE_WIDTH :],
        )

    batch = np.stack((x, mirrored))
    assert np.array_equal(mirror_window_features(batch, keys)[0], mirrored)


def test_ln_interaction_features_preserve_v1_prefix_and_mirror_symmetry():
    rows = [
        {"start_ms": 1000, "notes": [(0, 1500), (4, 1000)]},
        {"start_ms": 1250, "notes": [(1, 1250), (5, 1250)]},
        {"start_ms": 1600, "notes": [(0, 1900)]},
    ]
    base = window_features(rows, 6, 500)
    extended = window_features_with_ln_interactions(rows, 6, 500)

    assert extended.shape == (FEATURE_COUNT + len(LN_INTERACTION_FEATURE_NAMES),)
    assert np.array_equal(extended[:FEATURE_COUNT], base)
    assert np.isfinite(extended).all()
    mirrored = mirror_window_features(extended, 6)
    assert np.array_equal(mirror_window_features(mirrored, 6), extended)
    assert np.array_equal(mirrored[FEATURE_COUNT:], extended[FEATURE_COUNT:])

    interaction = extended[FEATURE_COUNT:].reshape(
        WINDOW_ROWS, len(LN_INTERACTION_ROW_FEATURES)
    )
    row_one = interaction[-3]
    row_two = interaction[-2]
    row_three = interaction[-1]
    index = {name: i for i, name in enumerate(LN_INTERACTION_ROW_FEATURES)}
    assert row_one[index["active_ln_density"]] == 0
    assert row_two[index["hybrid_same_hand_taps"]] == pytest.approx(1 / 6)
    assert row_two[index["hybrid_cross_hand_taps"]] == pytest.approx(1 / 6)
    assert row_three[index["inverse_ln_heads"]] == pytest.approx(1 / 6)
    assert row_three[index["near_release_press"]] == pytest.approx(1 / 6)
    assert row_three[index["release_press_same_hand"]] == pytest.approx(1 / 6)


def test_shared_thumb_hybrid_features_split_same_and_cross_hand_affinity():
    values = window_features_with_ln_interactions(
        [
            {"start_ms": 1000, "notes": [(3, 1800)]},
            {"start_ms": 1250, "notes": [(0, 1250)]},
        ],
        7,
        500,
    )
    interaction = values[FEATURE_COUNT:].reshape(
        WINDOW_ROWS, len(LN_INTERACTION_ROW_FEATURES)
    )
    index = {name: i for i, name in enumerate(LN_INTERACTION_ROW_FEATURES)}

    assert interaction[-1, index["hybrid_same_hand_taps"]] == pytest.approx(
        .5 / 7
    )
    assert interaction[-1, index["hybrid_cross_hand_taps"]] == pytest.approx(
        .5 / 7
    )


def test_phrase_context_appends_ordered_eight_beat_summary_and_is_mirror_invariant():
    rows = [
        {"start_ms": 1000, "beat_position": 0.0, "notes": [(0, 1800)]},
        {"start_ms": 1250, "beat_position": 0.5, "notes": [(4, 1250)]},
        {"start_ms": 1500, "beat_position": 1.0, "notes": [(2, 2500)]},
        {"start_ms": 1750, "beat_position": 1.5, "notes": [(5, 1750)]},
        {"start_ms": 2000, "beat_position": 2.0, "notes": [(1, 2000)]},
    ]
    result = window_features_with_ln_phrase_context(
        rows, 6, [500] * len(rows)
    )
    initial_ends = [1800, -100000, -100000, -100000, -100000, -100000]
    initial_last = [1000, -100000, -100000, -100000, -100000, -100000]
    local = window_features_with_ln_interactions(
        rows[-WINDOW_ROWS:], 6, [500] * WINDOW_ROWS,
        initial_ends=initial_ends,
        initial_last=initial_last,
    )
    phrase_start = FEATURE_COUNT + len(LN_INTERACTION_FEATURE_NAMES)
    summary = result[phrase_start:].reshape(
        LN_PHRASE_CONTEXT_BINS, -1
    )
    phrase_index = {
        name: index for index, name in enumerate(LN_PHRASE_CONTEXT_FEATURES)
    }

    assert result.shape == (
        FEATURE_SCHEMA_COUNTS[LN_PHRASE_CONTEXT_SCHEMA],
    )
    assert len(LN_PHRASE_CONTEXT_FEATURE_NAMES) == (
        LN_PHRASE_CONTEXT_BINS * len(phrase_index)
    )
    assert np.isfinite(result).all()
    assert np.array_equal(result[:len(local)], local)
    assert summary[12, phrase_index["ln_heads"]] == pytest.approx(1 / 6)
    assert summary[13, phrase_index["tap_heads"]] == pytest.approx(1 / 6)
    assert summary[14, phrase_index["ln_heads"]] == pytest.approx(1 / 6)
    assert summary[:, phrase_index["active_ln_fraction"]].sum() > 0
    assert np.array_equal(
        mirror_window_features(result, 6)[phrase_start:],
        result[phrase_start:],
    )


def test_phrase_context_distinguishes_same_counts_with_different_event_order():
    first = [
        {"start_ms": 1000, "beat_position": 0.0, "notes": [(0, 1800)]},
        {"start_ms": 1250, "beat_position": 0.5, "notes": [(4, 1250)]},
        {"start_ms": 1500, "beat_position": 1.0, "notes": [(2, 2200)]},
    ]
    reordered = [
        {"start_ms": 1000, "beat_position": 0.0, "notes": [(0, 1800)]},
        {"start_ms": 1250, "beat_position": 0.5, "notes": [(2, 2200)]},
        {"start_ms": 1500, "beat_position": 1.0, "notes": [(4, 1500)]},
    ]
    a = window_features_with_ln_phrase_context(first, 6, 500)
    b = window_features_with_ln_phrase_context(reordered, 6, 500)

    assert not np.array_equal(a[-len(LN_PHRASE_CONTEXT_FEATURE_NAMES):],
                              b[-len(LN_PHRASE_CONTEXT_FEATURE_NAMES):])


def test_phrase_context_uses_integrated_ln_release_beats_across_tempo_changes():
    rows = [
        {
            "start_ms": 0,
            "beat_position": 0.0,
            "note_end_positions": [5.0],
            "notes": [(0, 1500)],
        },
        {
            "start_ms": 500,
            "beat_position": 1.0,
            "note_end_positions": [1.0],
            "notes": [(4, 500)],
        },
        {
            "start_ms": 1000,
            "beat_position": 3.0,
            "note_end_positions": [3.0],
            "notes": [(2, 1000)],
        },
        {
            "start_ms": 1500,
            "beat_position": 5.0,
            "note_end_positions": [5.0],
            "notes": [(0, 1500)],
        },
    ]
    result = window_features_with_ln_phrase_context(
        rows,
        6,
        [500, 500, 250, 250],
        beat_positions=[0.0, 1.0, 3.0, 5.0],
    )
    summary = result[
        FEATURE_COUNT + len(LN_INTERACTION_FEATURE_NAMES):
    ].reshape(LN_PHRASE_CONTEXT_BINS, -1)
    index = {
        name: offset for offset, name in enumerate(LN_PHRASE_CONTEXT_FEATURES)
    }

    assert summary[6, index["ln_duration_long"]] == pytest.approx(1 / 6)
    assert summary[6, index["ln_duration_medium"]] == 0
    assert summary[15, index["ln_releases"]] == pytest.approx(1 / 6)
    assert summary[15, index["release_press_same_hand"]] == pytest.approx(1 / 6)


def test_phrase_context_keeps_pre_window_ln_release_and_release_press():
    rows = [
        {"start_ms": 1000, "beat_position": 2.0, "notes": [(2, 1000)]},
        {"start_ms": 1500, "beat_position": 3.0, "notes": [(0, 1500)]},
    ]
    initial_ends = [1500, None, None, None, None, None]
    initial_last = [0, None, None, None, None, None]
    result = window_features_with_ln_phrase_context(
        rows,
        6,
        500,
        initial_ends=initial_ends,
        initial_last=initial_last,
        beat_positions=[2.0, 3.0],
        initial_end_positions=[3.0, None, None, None, None, None],
        initial_last_positions=[0.0, None, None, None, None, None],
    )
    summary = result[
        FEATURE_COUNT + len(LN_INTERACTION_FEATURE_NAMES):
    ].reshape(LN_PHRASE_CONTEXT_BINS, -1)
    names = {
        name: offset for offset, name in enumerate(LN_PHRASE_CONTEXT_FEATURES)
    }

    assert summary[15, names["ln_releases"]] == pytest.approx(1 / 6)
    assert summary[15, names["release_press_same_hand"]] == pytest.approx(1 / 6)
    assert summary[:, names["active_ln_fraction"]].sum() > 0


def test_phrase_context_encodes_near_release_press_hand_relationship():
    rows = [
        {
            "start_ms": 1000,
            "beat_position": 0.0,
            "notes": [(0, 1450), (5, 1450)],
        },
        {"start_ms": 1500, "beat_position": 1.0, "notes": [(4, 1500)]},
    ]
    result = window_features_with_ln_phrase_context(
        rows, 6, [500, 500]
    )
    summary = result[
        FEATURE_COUNT + len(LN_INTERACTION_FEATURE_NAMES):
    ].reshape(LN_PHRASE_CONTEXT_BINS, -1)
    names = {
        name: offset for offset, name in enumerate(LN_PHRASE_CONTEXT_FEATURES)
    }

    assert summary[-1, names["release_press_same_hand"]] == 0
    assert summary[-1, names["near_release_press_same_hand"]] == pytest.approx(1 / 12)
    assert summary[-1, names["near_release_press_cross_hand"]] == pytest.approx(
        1 / 12
    )
    assert np.array_equal(
        result[
            FEATURE_COUNT + len(LN_INTERACTION_FEATURE_NAMES):
        ].reshape(LN_PHRASE_CONTEXT_BINS, -1)[:, names["near_release_press_same_hand"]],
        mirror_window_features(result, 6)[
            FEATURE_COUNT + len(LN_INTERACTION_FEATURE_NAMES):
        ].reshape(LN_PHRASE_CONTEXT_BINS, -1)[:, names["near_release_press_same_hand"]],
    )

    same_hand = window_features_with_ln_phrase_context(
        [
            {"start_ms": 1000, "beat_position": 0.0, "notes": [(0, 1450)]},
            {"start_ms": 1500, "beat_position": 1.0, "notes": [(1, 1500)]},
        ],
        6,
        [500, 500],
    )
    same_summary = same_hand[
        FEATURE_COUNT + len(LN_INTERACTION_FEATURE_NAMES):
    ].reshape(LN_PHRASE_CONTEXT_BINS, -1)
    assert same_summary[-1, names["near_release_press_same_hand"]] == pytest.approx(
        1 / 6
    )


def test_phrase_context_rejects_missing_initial_ln_positions():
    rows = [
        {"start_ms": 1000, "beat_position": 2.0, "notes": [(2, 1000)]},
    ]

    with pytest.raises(ValueError, match="initial_end_positions"):
        window_features_with_ln_phrase_context(
            rows,
            6,
            500,
            initial_ends=[1500, None, None, None, None, None],
            initial_last=[0, None, None, None, None, None],
            beat_positions=[2.0],
            initial_end_positions=[None, None, None, None, None, None],
            initial_last_positions=[0.0, None, None, None, None, None],
        )
    with pytest.raises(ValueError, match="initial_last_positions"):
        window_features_with_ln_phrase_context(
            rows,
            6,
            500,
            initial_ends=[1500, None, None, None, None, None],
            initial_last=[0, None, None, None, None, None],
            beat_positions=[2.0],
            initial_end_positions=[3.0, None, None, None, None, None],
            initial_last_positions=[None, None, None, None, None, None],
        )


def test_hold_state_duration_and_release_between_rows_are_encoded():
    x = window_features(
        [
            {"start_ms": 1000, "notes": [(2, 2000)]},
            {"start_ms": 1500, "notes": [(0, 1500)]},
            {"start_ms": 2250, "notes": [(2, 2250)]},
        ],
        6,
        500,
    )
    rows = x[:GLOBAL_BASE].reshape(WINDOW_ROWS, ROW_WIDTH)
    lane_two = rows[2, 2 * LANE_WIDTH : 3 * LANE_WIDTH]
    released_row_lane_two = rows[3, 2 * LANE_WIDTH : 3 * LANE_WIDTH]
    assert lane_two[LANE_FEATURES.index("occupied_before")] == 1
    assert lane_two[LANE_FEATURES.index("hold_remaining")] == pytest.approx(1 / 32)
    assert released_row_lane_two[LANE_FEATURES.index("release_between_rows")] == 1
    assert released_row_lane_two[LANE_FEATURES.index("since_release")] == pytest.approx(0.5 / 8)
    assert released_row_lane_two[LANE_FEATURES.index("attack")] == 1
    assert released_row_lane_two[LANE_FEATURES.index("hold_head")] == 0


def test_variable_tempo_uses_each_rows_local_beat_period():
    x = window_features(
        [
            {"start_ms": 1000, "notes": [(0, 1500)]},
            {"start_ms": 1500, "notes": [(1, 1500)]},
        ],
        6,
        [500, 250],
    )
    rows = x[:GLOBAL_BASE].reshape(WINDOW_ROWS, ROW_WIDTH)
    first_lane = rows[2, :LANE_WIDTH]

    assert first_lane[LANE_FEATURES.index("hold_duration")] == pytest.approx(
        np.log1p(1) / np.log1p(32)
    )
    assert rows[3, ROW_GLOBAL_BASE + ROW_FEATURES.index("gap_beats")] == pytest.approx(
        2 / 8
    )
    assert x[-1] == pytest.approx(.25)


def test_empty_row_is_accepted_only_at_an_exact_known_release():
    x = window_features(
        [
            {"start_ms": 1000, "notes": [(1, 1500)]},
            {"start_ms": 1500, "notes": []},
        ],
        6,
        500,
    )
    row = x[:GLOBAL_BASE].reshape(WINDOW_ROWS, ROW_WIDTH)[-1]
    lane_one = row[LANE_WIDTH : 2 * LANE_WIDTH]
    assert lane_one[LANE_FEATURES.index("release_between_rows")] == 1

    with pytest.raises(ValueError, match="no LN release"):
        window_features(
            [
                {"start_ms": 1000, "notes": [(1, 1500)]},
                {"start_ms": 1600, "notes": []},
            ],
            6,
            500,
        )


@pytest.mark.parametrize(
    "rows,kwargs,match",
    [
        ([{"start_ms": 1000, "notes": [(6, 1000)]}], {}, "invalid lane"),
        ([{"start_ms": 1000.5, "notes": [(0, 1000)]}], {}, "integer millisecond"),
        ([{"start_ms": -1, "notes": [(0, 0)]}], {}, "cannot be negative"),
        (
            [
                {"start_ms": 1000, "notes": [(0, 2000)]},
                {"start_ms": 1500, "notes": [(0, 1500)]},
            ],
            {},
            "overlapping LN",
        ),
        ([{"start_ms": 1000, "notes": [(0, 1000), (0, 1000)]}], {}, "duplicate"),
        ([{"start_ms": 1000, "notes": [(0, 999)]}], {}, "ends before"),
        (
            [
                {"start_ms": 1500, "notes": [(0, 1500)]},
                {"start_ms": 1400, "notes": [(1, 1400)]},
            ],
            {},
            "strictly increasing",
        ),
    ],
)
def test_invalid_rows_are_rejected(rows, kwargs, match):
    with pytest.raises(ValueError, match=match):
        window_features(rows, 6, 500, **kwargs)


def test_initial_hold_and_attack_history_are_used_and_validated():
    x = window_features(
        [{"start_ms": 1000, "notes": [(0, 1000)]}],
        6,
        500,
        initial_ends=[None, 1500, None, None, None, None],
        initial_last=[None, 500, None, None, None, None],
    )
    lanes = x[:GLOBAL_BASE].reshape(WINDOW_ROWS, ROW_WIDTH)[-1, : 10 * LANE_WIDTH].reshape(
        10, LANE_WIDTH
    )
    assert lanes[1, LANE_FEATURES.index("occupied_before")] == 1
    assert lanes[1, LANE_FEATURES.index("since_release")] == 1
    assert lanes[1, LANE_FEATURES.index("since_attack")] == pytest.approx(1 / 8)

    with pytest.raises(ValueError, match="overlapping LN"):
        window_features(
            [{"start_ms": 1000, "notes": [(1, 1000)]}],
            6,
            500,
            initial_ends={1: 1200},
        )

    with pytest.raises(ValueError, match="exactly 6"):
        window_features(
            [{"start_ms": 1000, "notes": []}],
            6,
            500,
            initial_last=[100],
        )


def test_mirroring_rejects_invalid_shapes_and_key_counts():
    with pytest.raises(ValueError, match="6 through 10"):
        mirror_window_features(np.zeros(FEATURE_COUNT), 5)
    with pytest.raises(ValueError, match="must have shape"):
        mirror_window_features(np.zeros(FEATURE_COUNT - 1), 6)
