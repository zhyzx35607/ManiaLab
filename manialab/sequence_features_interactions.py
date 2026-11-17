"""LN interaction feature encoder for sequence windows."""
from __future__ import annotations

from numbers import Real

import numpy as np

from .sequence_schema import LN_INTERACTION_ROW_FEATURES, WINDOW_ROWS
from .sequence_features_common import _normalize_rows, _positive_keys, _state_times
from .sequence_features_base import _UNKNOWN_TIME, window_features


def window_features_with_ln_interactions(
        rows, keys, beat_ms, initial_ends=None, initial_last=None):
    """Encode v1 features plus mirror-invariant LN interaction summaries.

    Interaction features use only the supplied event rows and per-lane state at
    the window edge, so training and generation observe the same history.
    """
    keys = _positive_keys(keys)
    base = window_features(
        rows, keys, beat_ms, initial_ends=initial_ends, initial_last=initial_last
    )
    events = _normalize_rows(rows, keys)
    if isinstance(beat_ms, Real) and not isinstance(beat_ms, bool):
        beat_lengths = [float(beat_ms)] * len(events)
    else:
        beat_lengths = [float(value) for value in beat_ms]

    end_values = _state_times(initial_ends, keys, "initial_ends")
    last_values = _state_times(initial_last, keys, "initial_last")
    ends = np.asarray(
        [_UNKNOWN_TIME if value is None else value for value in end_values],
        dtype=np.int64,
    )
    last = np.asarray(
        [_UNKNOWN_TIME if value is None else value for value in last_values],
        dtype=np.int64,
    )

    hand_weights = np.zeros((keys, 2), dtype=np.float64)
    middle = keys // 2
    for lane in range(keys):
        if keys % 2 and lane == middle:
            hand_weights[lane] = (0.5, 0.5)
        elif lane < middle:
            hand_weights[lane, 0] = 1.0
        else:
            hand_weights[lane, 1] = 1.0
    same_hand = hand_weights @ hand_weights.T
    pair_denominator = max(1, keys * (keys - 1) // 2)

    interaction = np.zeros(
        (WINDOW_ROWS, len(LN_INTERACTION_ROW_FEATURES)), dtype=np.float32
    )
    feature_index = {
        name: index for index, name in enumerate(LN_INTERACTION_ROW_FEATURES)
    }
    past_lns = [
        (int(last[lane]), int(ends[lane]), lane)
        for lane in range(keys)
        if last[lane] != _UNKNOWN_TIME and ends[lane] > last[lane]
    ]
    prior_start = _UNKNOWN_TIME

    def set_count(row_index, name, count, denominator=keys):
        interaction[row_index, feature_index[name]] = count / max(1, denominator)

    for offset, ((start, notes), row_beat_ms) in enumerate(
            zip(events, beat_lengths)):
        row_index = WINDOW_ROWS - len(events) + offset
        prior_hold = (last != _UNKNOWN_TIME) & (ends > last)
        active = prior_hold & (ends > start)
        released = prior_hold & (
            (ends == start)
            | (
                (prior_start != _UNKNOWN_TIME)
                & (prior_start < ends)
                & (ends < start)
            )
        )
        attacked_lanes = [lane for lane, _ in notes]
        ln_lanes = [lane for lane, end in notes if end > start]
        tap_lanes = [lane for lane, end in notes if end == start]
        inverse_window = min(150.0, row_beat_ms * 0.3)
        release_window = max(40.0, min(150.0, row_beat_ms * 0.25))

        set_count(row_index, "active_ln_density", int(np.count_nonzero(active)))
        active_lanes = np.flatnonzero(active)
        active_pairs = [
            (left, right)
            for index, left in enumerate(active_lanes)
            for right in active_lanes[index + 1:]
        ]
        set_count(
            row_index,
            "active_ln_same_hand_pairs",
            sum(same_hand[left, right] for left, right in active_pairs),
            pair_denominator,
        )
        set_count(
            row_index,
            "active_ln_cross_hand_pairs",
            sum(1.0 - same_hand[left, right]
                for left, right in active_pairs),
            pair_denominator,
        )
        set_count(row_index, "ln_head_density", len(ln_lanes))
        set_count(row_index, "tap_head_density", len(tap_lanes))
        set_count(row_index, "release_density", int(np.count_nonzero(released)))

        same_hand_taps = cross_hand_taps = 0
        for lane in tap_lanes:
            if len(active_lanes):
                affinity = float(np.mean(same_hand[active_lanes, lane]))
                same_hand_taps += affinity
                cross_hand_taps += 1.0 - affinity
        set_count(row_index, "hybrid_same_hand_taps", same_hand_taps)
        set_count(row_index, "hybrid_cross_hand_taps", cross_hand_taps)

        same_hand_lns = cross_hand_lns = 0
        for lane in ln_lanes:
            if len(active_lanes):
                affinity = float(np.mean(same_hand[active_lanes, lane]))
                same_hand_lns += affinity
                cross_hand_lns += 1.0 - affinity
        set_count(row_index, "hybrid_same_hand_ln_heads", same_hand_lns)
        set_count(row_index, "hybrid_cross_hand_ln_heads", cross_hand_lns)

        same_hand_release_press = cross_hand_release_press = 0
        released_lanes = np.flatnonzero(released)
        for lane in attacked_lanes:
            if len(released_lanes):
                affinity = float(np.mean(same_hand[released_lanes, lane]))
                same_hand_release_press += affinity
                cross_hand_release_press += 1.0 - affinity
        set_count(
            row_index, "release_press_same_hand", same_hand_release_press
        )
        set_count(
            row_index, "release_press_cross_hand", cross_hand_release_press
        )

        split_release_hands = 0
        for hand in range(2):
            in_hand = hand_weights[:, hand] > 0
            if np.any(released & in_hand) and np.any(active & in_hand):
                split_release_hands += 1
        set_count(row_index, "split_release_hands", split_release_hands, 2)

        inverse_heads = reverse_shield_taps = shield_heads = 0
        for lane, end in notes:
            gap = start - int(ends[lane])
            if prior_hold[lane] and 0 < gap <= inverse_window:
                if end > start:
                    inverse_heads += 1
                else:
                    reverse_shield_taps += 1
            elif (
                not prior_hold[lane]
                and end > start
                and last[lane] != _UNKNOWN_TIME
                and 0 < start - int(last[lane]) <= inverse_window
            ):
                shield_heads += 1
        set_count(row_index, "inverse_ln_heads", inverse_heads)
        set_count(row_index, "reverse_shield_taps", reverse_shield_taps)
        set_count(row_index, "shield_ln_heads", shield_heads)

        nearby_lanes = {
            lane
            for ln_start, ln_end, lane in past_lns
            if ln_start < start
            and 1 < abs(ln_end - start) < release_window
        }
        near_release_press = sum(
            any(same_hand[prior_lane, lane] > 0 for prior_lane in nearby_lanes)
            for lane in attacked_lanes
        )
        set_count(row_index, "near_release_press", near_release_press)

        for lane, end in notes:
            ends[lane] = end
            last[lane] = start
            if end > start:
                past_lns.append((start, end, lane))
        prior_start = start

    return np.concatenate((base, interaction.reshape(-1)))
