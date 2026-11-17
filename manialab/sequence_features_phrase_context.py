"""Long-context LN phrase feature encoder."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np

from .sequence_schema import (
    LN_PHRASE_CONTEXT_BEATS,
    LN_PHRASE_CONTEXT_BINS,
    LN_PHRASE_CONTEXT_FEATURES,
    WINDOW_ROWS,
)
from .sequence_features_common import _positive_keys, _state_times, _time
from .sequence_features_interactions import window_features_with_ln_interactions


def window_features_with_ln_phrase_context(
        rows, keys, beat_ms, initial_ends=None, initial_last=None,
        beat_positions=None, initial_end_positions=None,
        initial_last_positions=None):
    """Encode the recent four rows plus an eight-beat LN interaction summary.

    ``rows`` may contain more than four event rows. Rows can optionally carry
    ``beat_position`` values; otherwise their positions are integrated from
    each preceding row's beat length. The phrase summary is mirror-invariant,
    so it can be appended to the existing LN-interaction schema unchanged.
    """
    keys = _positive_keys(keys)
    events = _normalize_phrase_rows(rows, keys)
    if isinstance(beat_ms, Real) and not isinstance(beat_ms, bool):
        if not np.isfinite(beat_ms) or beat_ms <= 0:
            raise ValueError("beat_ms must be finite and positive")
        beat_lengths = [float(beat_ms)] * len(events)
    elif (isinstance(beat_ms, (Sequence, np.ndarray))
          and not isinstance(beat_ms, (str, bytes))):
        if len(beat_ms) != len(events):
            raise ValueError("beat_ms sequence must match the event row count")
        beat_lengths = []
        for value in beat_ms:
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError("beat_ms values must be finite and positive")
            beat_lengths.append(float(value))
    else:
        raise ValueError("beat_ms must be a finite positive number or row sequence")

    if beat_positions is None:
        positions = []
        for index, (start, _) in enumerate(events):
            supplied = rows[index].get("beat_position")
            if supplied is not None:
                if (isinstance(supplied, bool)
                        or not isinstance(supplied, Real)
                        or not np.isfinite(supplied)):
                    raise ValueError("beat_position values must be finite numbers")
                positions.append(float(supplied))
            elif not positions:
                positions.append(0.0)
            else:
                previous_start = events[index - 1][0]
                positions.append(
                    positions[-1]
                    + (start - previous_start) / beat_lengths[index - 1]
                )
    else:
        if len(beat_positions) != len(events):
            raise ValueError("beat_positions must match the event row count")
        positions = []
        for value in beat_positions:
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError("beat_positions must contain finite numbers")
            positions.append(float(value))
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError("beat_positions must strictly increase")

    first_start = events[0][0]
    ends = _state_times(initial_ends, keys, "initial_ends")
    last = _state_times(initial_last, keys, "initial_last")
    base_offset = max(0, len(events) - WINDOW_ROWS)
    base_initial_ends, base_initial_last = list(ends), list(last)
    for (start, notes) in events[:base_offset]:
        for lane, end in notes:
            base_initial_ends[lane] = end if end > start else None
            base_initial_last[lane] = start

    local_rows = [
        {"start_ms": start, "notes": notes}
        for start, notes in events[base_offset:]
    ]
    local_base = window_features_with_ln_interactions(
        local_rows,
        keys,
        beat_lengths[base_offset:],
        initial_ends=base_initial_ends,
        initial_last=base_initial_last,
    )
    summary = _ln_phrase_context_features(
        events, positions, beat_lengths, keys, ends, last,
        rows=rows,
        initial_end_positions=initial_end_positions,
        initial_last_positions=initial_last_positions,
    )
    return np.concatenate((local_base, summary))

def _normalize_phrase_rows(rows, keys):
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("rows must be a sequence of event rows")
    if not rows:
        raise ValueError("rows must contain at least one event row")
    normalized = []
    previous_start = None
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or "start_ms" not in row or "notes" not in row:
            raise ValueError(f"row {row_index} must contain start_ms and notes")
        start = _time(row["start_ms"], f"row {row_index} start_ms")
        if start < 0 or (previous_start is not None and start <= previous_start):
            raise ValueError("row start_ms values must be non-negative and increasing")
        previous_start = start
        raw_notes = row["notes"]
        if isinstance(raw_notes, (str, bytes)) or not isinstance(raw_notes, Sequence):
            raise ValueError(f"row {row_index} notes must be a sequence")
        notes = []
        lanes = set()
        for note_index, note in enumerate(raw_notes):
            if (isinstance(note, (str, bytes)) or not isinstance(note, Sequence)
                    or len(note) != 2):
                raise ValueError(f"row {row_index} note {note_index} must be (lane, end_ms)")
            lane, end = note
            if (isinstance(lane, bool) or not isinstance(lane, Integral)
                    or not 0 <= lane < keys):
                raise ValueError(f"row {row_index} note {note_index} has an invalid lane")
            end = _time(end, f"row {row_index} note {note_index} end_ms")
            if end < start or int(lane) in lanes:
                raise ValueError(f"row {row_index} contains an invalid or duplicate note")
            lanes.add(int(lane))
            notes.append((int(lane), end))
        normalized.append((start, notes))
    return normalized

def _phrase_hand_weights(keys):
    hand_weights = np.zeros((keys, 2), dtype=np.float64)
    middle = keys // 2
    for lane in range(keys):
        if keys % 2 and lane == middle:
            hand_weights[lane] = (0.5, 0.5)
        elif lane < middle:
            hand_weights[lane, 0] = 1.0
        else:
            hand_weights[lane, 1] = 1.0
    return hand_weights

def _ln_phrase_context_features(events, positions, beat_lengths, keys,
                                initial_ends, initial_last, *, rows,
                                initial_end_positions=None,
                                initial_last_positions=None):
    features = np.zeros(
        (LN_PHRASE_CONTEXT_BINS, len(LN_PHRASE_CONTEXT_FEATURES)),
        dtype=np.float32,
    )
    feature_index = {
        name: index for index, name in enumerate(LN_PHRASE_CONTEXT_FEATURES)
    }
    hand_weights = _phrase_hand_weights(keys)
    same_hand = hand_weights @ hand_weights.T
    pair_denominator = max(1, keys * (keys - 1) // 2)
    window_start = positions[-1] - LN_PHRASE_CONTEXT_BEATS
    bin_width = LN_PHRASE_CONTEXT_BEATS / LN_PHRASE_CONTEXT_BINS
    intervals = []
    release_history = []
    for name, values in (
        ("initial_end_positions", initial_end_positions),
        ("initial_last_positions", initial_last_positions),
    ):
        if values is not None and (
            len(values) != keys
            or any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                )
                for value in values
            )
        ):
            raise ValueError(
                f"{name} must contain one finite beat position or None per lane"
            )
    initial_end_positions = (
        None if initial_end_positions is None else tuple(
            None if value is None else float(value)
            for value in initial_end_positions
        )
    )
    initial_last_positions = (
        None if initial_last_positions is None else tuple(
            None if value is None else float(value)
            for value in initial_last_positions
        )
    )
    row_end_positions = []
    for row in rows:
        supplied = row.get("note_end_positions")
        if supplied is None:
            row_end_positions.append(None)
            continue
        raw_notes = row["notes"]
        if (isinstance(supplied, (str, bytes))
                or not isinstance(supplied, Sequence)
                or len(supplied) != len(raw_notes)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                    for value in supplied
                )):
            raise ValueError(
                "note_end_positions must match the row notes and be finite"
            )
        row_end_positions.append(tuple(float(value) for value in supplied))

    def bin_index(position):
        if position < window_start or position > positions[-1]:
            return None
        return min(
            LN_PHRASE_CONTEXT_BINS - 1,
            max(0, int((position - window_start) / bin_width)),
        )

    def add(index, name, amount=1.0, denominator=keys):
        if index is not None:
            features[index, feature_index[name]] += amount / max(1, denominator)

    for lane, end in enumerate(initial_ends):
        start_time = initial_last[lane]
        if (end is None or start_time is None
                or end <= start_time):
            continue
        if initial_end_positions is not None:
            end_position = initial_end_positions[lane]
        else:
            end_position = positions[0] + (
                end - events[0][0]
            ) / beat_lengths[0]
        if initial_last_positions is not None:
            start_position = initial_last_positions[lane]
        else:
            start_position = positions[0] - (
                events[0][0] - start_time
            ) / beat_lengths[0]
        if end_position is None:
            raise ValueError(
                "initial_end_positions must include every active LN"
            )
        if start_position is None:
            raise ValueError(
                "initial_last_positions must include every active LN"
            )
        if end_position < start_position:
            raise ValueError("initial LN beat positions must be ordered")
        release_history.append((int(start_time), int(end), lane))
        if end_position < window_start:
            continue
        intervals.append((start_position, end_position, lane, False))
        add(bin_index(end_position), "ln_releases")

    for row_index, ((start, notes), position, beat_length) in enumerate(
            zip(events, positions, beat_lengths)):
        current = []
        releases = []
        for prior_start, prior_end, lane, is_head in intervals:
            if prior_start < position < prior_end:
                current.append(lane)
            if prior_end == position:
                releases.append(lane)

        current = sorted(set(current))
        releases = sorted(set(releases))
        event_bin = bin_index(position)
        if event_bin is not None:
            tap_lanes = [lane for lane, end in notes if end == start]
            ln_notes = [
                (note_index, lane, end)
                for note_index, (lane, end) in enumerate(notes)
                if end > start
            ]
            add(event_bin, "tap_heads", len(tap_lanes))
            add(event_bin, "ln_heads", len(ln_notes))
            for lane in tap_lanes:
                if current:
                    affinity = float(np.mean(same_hand[current, lane]))
                    add(event_bin, "hybrid_same_hand_taps", affinity)
                    add(event_bin, "hybrid_cross_hand_taps", 1.0 - affinity)
            for _, lane, _ in ln_notes:
                if current:
                    affinity = float(np.mean(same_hand[current, lane]))
                    add(event_bin, "hybrid_same_hand_ln_heads", affinity)
                    add(event_bin, "hybrid_cross_hand_ln_heads", 1.0 - affinity)
            attacked = [lane for lane, _ in notes]
            for lane in attacked:
                if releases:
                    affinity = float(np.mean(same_hand[releases, lane]))
                    add(event_bin, "release_press_same_hand", affinity)
                    add(event_bin, "release_press_cross_hand", 1.0 - affinity)
                release_window = max(40.0, min(150.0, beat_length * 0.25))
                nearby_releases = [
                    prior_lane
                    for prior_start, prior_end, prior_lane in release_history
                    if prior_start < start
                    and 1 < abs(prior_end - start) < release_window
                ]
                if nearby_releases:
                    affinity = float(np.mean(same_hand[nearby_releases, lane]))
                    add(event_bin, "near_release_press_same_hand", affinity)
                    add(
                        event_bin,
                        "near_release_press_cross_hand",
                        1.0 - affinity,
                    )
            for note_index, _, end in ln_notes:
                supplied_end_positions = row_end_positions[row_index]
                duration = (
                    supplied_end_positions[note_index] - position
                    if supplied_end_positions is not None
                    else (end - start) / beat_length
                )
                duration_kind = (
                    "ln_duration_short" if duration < 1.0
                    else "ln_duration_medium" if duration < 4.0
                    else "ln_duration_long"
                )
                add(event_bin, duration_kind)

        for note_index, (lane, end) in enumerate(notes):
            if end <= start:
                continue
            supplied_end_positions = row_end_positions[row_index]
            end_position = (
                supplied_end_positions[note_index]
                if supplied_end_positions is not None
                else position + (end - start) / beat_length
            )
            if end_position < position:
                raise ValueError("note LN beat positions must be ordered")
            intervals.append((position, end_position, lane, True))
            release_history.append((start, end, lane))
            release_bin = bin_index(end_position)
            add(release_bin, "ln_releases")

    for bin_index_value in range(LN_PHRASE_CONTEXT_BINS):
        bin_start = window_start + bin_index_value * bin_width
        bin_end = bin_start + bin_width
        occupancy = 0.0
        active_intervals = []
        for start_position, end_position, lane, _ in intervals:
            overlap = max(
                0.0,
                min(bin_end, end_position) - max(bin_start, start_position),
            )
            if overlap > 0:
                occupancy += overlap / bin_width
                active_intervals.append(
                    (max(bin_start, start_position), min(bin_end, end_position), lane)
                )
        add(
            bin_index_value,
            "active_ln_fraction",
            occupancy,
            denominator=keys,
        )
        same_pair_time = cross_pair_time = 0.0
        for index, (left_start, left_end, left_lane) in enumerate(active_intervals):
            for right_start, right_end, right_lane in active_intervals[index + 1:]:
                simultaneous = max(
                    0.0, min(left_end, right_end) - max(left_start, right_start)
                )
                if simultaneous <= 0:
                    continue
                same_pair_time += same_hand[left_lane, right_lane] * simultaneous
                cross_pair_time += (
                    1.0 - same_hand[left_lane, right_lane]
                ) * simultaneous
        features[
            bin_index_value, feature_index["active_ln_same_hand_pairs"]
        ] += same_pair_time / bin_width / pair_denominator
        features[
            bin_index_value, feature_index["active_ln_cross_hand_pairs"]
        ] += cross_pair_time / bin_width / pair_denominator

    return features.reshape(-1)
