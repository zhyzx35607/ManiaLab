"""Fixed-width, inspectable features for short mania note windows.

This module only describes a supplied sequence. It does not choose note times,
lanes, or LN lengths.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np

WINDOW_ROWS = 4
MAX_LANES = 10
LANE_FEATURES = (
    "attack",
    "hold_head",
    "hold_duration",
    "occupied_before",
    "hold_remaining",
    "since_attack",
    "since_release",
    "release_between_rows",
    "valid_lane",
)
ROW_FEATURES = ("event", "chord_share", "hold_share", "gap_beats")
KEY_FEATURES = tuple(f"keys_{keys}" for keys in range(6, 11))
GLOBAL_FEATURES = KEY_FEATURES + ("beat_seconds",)
ROW_WIDTH = MAX_LANES * len(LANE_FEATURES) + len(ROW_FEATURES)
FEATURE_COUNT = WINDOW_ROWS * ROW_WIDTH + len(GLOBAL_FEATURES)
FEATURE_NAMES = tuple(
    name
    for row in range(WINDOW_ROWS)
    for name in (
        *(f"row_{row}_lane_{lane}_{feature}"
          for lane in range(MAX_LANES) for feature in LANE_FEATURES),
        *(f"row_{row}_{feature}" for feature in ROW_FEATURES),
    )
) + GLOBAL_FEATURES
LN_INTERACTION_ROW_FEATURES = (
    "active_ln_density",
    "active_ln_same_hand_pairs",
    "active_ln_cross_hand_pairs",
    "ln_head_density",
    "tap_head_density",
    "release_density",
    "hybrid_same_hand_taps",
    "hybrid_cross_hand_taps",
    "hybrid_same_hand_ln_heads",
    "hybrid_cross_hand_ln_heads",
    "release_press_same_hand",
    "release_press_cross_hand",
    "split_release_hands",
    "inverse_ln_heads",
    "reverse_shield_taps",
    "shield_ln_heads",
    "near_release_press",
)
LN_INTERACTION_FEATURE_NAMES = tuple(
    f"interaction_row_{row}_{feature}"
    for row in range(WINDOW_ROWS)
    for feature in LN_INTERACTION_ROW_FEATURES
)
LN_INTERACTION_FEATURE_COUNT = len(LN_INTERACTION_FEATURE_NAMES)
LN_INTERACTION_SCHEMA = "sequence-window-ln-interactions-v1"
LN_PHRASE_CONTEXT_SCHEMA = "sequence-window-ln-phrase-context-v2"
LN_PHRASE_CONTEXT_BEATS = 8.0
LN_PHRASE_CONTEXT_BINS = 16
LN_PHRASE_CONTEXT_FEATURES = (
    "tap_heads",
    "ln_heads",
    "ln_releases",
    "active_ln_fraction",
    "active_ln_same_hand_pairs",
    "active_ln_cross_hand_pairs",
    "hybrid_same_hand_taps",
    "hybrid_cross_hand_taps",
    "hybrid_same_hand_ln_heads",
    "hybrid_cross_hand_ln_heads",
    "release_press_same_hand",
    "release_press_cross_hand",
    "near_release_press_same_hand",
    "near_release_press_cross_hand",
    "ln_duration_short",
    "ln_duration_medium",
    "ln_duration_long",
)
LN_PHRASE_CONTEXT_FEATURE_NAMES = tuple(
    f"phrase_bin_{row}_{feature}"
    for row in range(LN_PHRASE_CONTEXT_BINS)
    for feature in LN_PHRASE_CONTEXT_FEATURES
)
FEATURE_SCHEMA_NAMES = {
    "sequence-window-v1": FEATURE_NAMES,
    LN_INTERACTION_SCHEMA: FEATURE_NAMES + LN_INTERACTION_FEATURE_NAMES,
    LN_PHRASE_CONTEXT_SCHEMA: (
        FEATURE_NAMES
        + LN_INTERACTION_FEATURE_NAMES
        + LN_PHRASE_CONTEXT_FEATURE_NAMES
    ),
}
FEATURE_SCHEMA_COUNTS = {
    schema: len(names) for schema, names in FEATURE_SCHEMA_NAMES.items()
}

_LANE_INDEX = {name: index for index, name in enumerate(LANE_FEATURES)}
_ROW_INDEX = {name: index for index, name in enumerate(ROW_FEATURES)}
_MAX_HISTORY_BEATS = 8.0
_MAX_REMAINING_BEATS = 32.0
_MAX_HOLD_BEATS = 32.0
_UNKNOWN_TIME = -100000


def _positive_keys(keys: int) -> int:
    if isinstance(keys, bool) or not isinstance(keys, Integral) or not 6 <= keys <= 10:
        raise ValueError("keys must be an integer from 6 through 10")
    return int(keys)


def _time(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer millisecond time")
    return int(value)


def _state_times(value: object, keys: int, name: str) -> list[int | None]:
    """Read per-lane state supplied as a sequence or lane-to-time mapping."""
    if value is None:
        return [None] * keys
    if isinstance(value, Mapping):
        result: list[int | None] = [None] * keys
        for lane, timestamp in value.items():
            if isinstance(lane, bool) or not isinstance(lane, Integral) or not 0 <= lane < keys:
                raise ValueError(f"{name} contains an invalid lane")
            result[int(lane)] = None if timestamp is None else _time(timestamp, name)
        return result
    if isinstance(value, (str, bytes)) or not (
        isinstance(value, Sequence) or isinstance(value, np.ndarray)
    ):
        raise ValueError(f"{name} must be a per-lane sequence or mapping")
    if len(value) != keys:
        raise ValueError(f"{name} must contain exactly {keys} lane values")
    return [None if timestamp is None else _time(timestamp, name) for timestamp in value]


def _beat_duration(beat_ms: float, amount_ms: int | float, maximum: float) -> float:
    beats = max(0.0, float(amount_ms) / beat_ms)
    return min(beats, maximum) / maximum


def _normalize_rows(rows: object, keys: int) -> list[tuple[int, list[tuple[int, int]]]]:
    if isinstance(rows, (str, bytes)) or not (
        isinstance(rows, Sequence) or isinstance(rows, np.ndarray)
    ):
        raise ValueError("rows must be a sequence of event rows")
    if not 1 <= len(rows) <= WINDOW_ROWS:
        raise ValueError(f"rows must contain between 1 and {WINDOW_ROWS} events")

    normalized: list[tuple[int, list[tuple[int, int]]]] = []
    previous_start: int | None = None
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or "start_ms" not in row or "notes" not in row:
            raise ValueError(f"row {row_index} must contain start_ms and notes")
        start = _time(row["start_ms"], f"row {row_index} start_ms")
        if start < 0:
            raise ValueError(f"row {row_index} start_ms cannot be negative")
        if previous_start is not None and start <= previous_start:
            raise ValueError("row start_ms values must be strictly increasing")
        previous_start = start

        raw_notes = row["notes"]
        if isinstance(raw_notes, (str, bytes)) or not (
            isinstance(raw_notes, Sequence) or isinstance(raw_notes, np.ndarray)
        ):
            raise ValueError(f"row {row_index} notes must be a sequence")
        notes: list[tuple[int, int]] = []
        lanes: set[int] = set()
        for note_index, note in enumerate(raw_notes):
            if isinstance(note, (str, bytes)) or not (
                isinstance(note, Sequence) or isinstance(note, np.ndarray)
            ) or len(note) != 2:
                raise ValueError(f"row {row_index} note {note_index} must be (lane, end_ms)")
            lane, end = note
            if isinstance(lane, bool) or not isinstance(lane, Integral) or not 0 <= lane < keys:
                raise ValueError(f"row {row_index} note {note_index} has an invalid lane")
            lane = int(lane)
            end = _time(end, f"row {row_index} note {note_index} end_ms")
            if end < start:
                raise ValueError(f"row {row_index} note {note_index} ends before it starts")
            if lane in lanes:
                raise ValueError(f"row {row_index} contains duplicate notes in lane {lane}")
            lanes.add(lane)
            notes.append((lane, end))
        normalized.append((start, notes))
    return normalized


def window_features(rows, keys, beat_ms, initial_ends=None, initial_last=None):
    """Encode one to four ordered event rows as a fixed-width float32 vector.

    Each row is ``{"start_ms": int, "notes": [(lane, end_ms), ...]}``. A note
    with ``end_ms > start_ms`` is an LN. An empty ``notes`` list is accepted
    only for an event row at which a known LN ends.

    ``initial_ends`` and ``initial_last`` optionally provide per-lane prior LN
    end times and prior attack times. They may be sequences of exactly ``keys``
    integer millisecond values or ``None`` values, or mappings from lane index
    to such values. Unspecified state is encoded as old/unknown history.

    ``beat_ms`` may be one positive period for the whole window or one period
    per supplied event row; row-local periods preserve variable-tempo context.
    Rows are right-aligned in a four-row window. Each row contains nine values
    per lane, then four row-level values. The final six values encode key count
    as one-hot and beat length in seconds.
    """
    keys = _positive_keys(keys)
    events = _normalize_rows(rows, keys)
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
    initial_ends = _state_times(initial_ends, keys, "initial_ends")
    last_attack = _state_times(initial_last, keys, "initial_last")
    first_start = events[0][0]
    if any(timestamp is not None and timestamp > first_start for timestamp in last_attack):
        raise ValueError("initial_last values cannot be later than the first row")

    output = np.zeros(FEATURE_COUNT, dtype=np.float32)
    lane_block = len(LANE_FEATURES)
    active_ends = list(initial_ends)
    last_release = [
        end if end is not None and end <= first_start else None
        for end in initial_ends
    ]
    prior_start: int | None = None

    for row_offset, (start, notes) in enumerate(events):
        row_beat_ms = beat_lengths[row_offset]
        row_position = WINDOW_ROWS - len(events) + row_offset
        row_base = row_position * ROW_WIDTH
        lane_base = row_base

        releases_between: set[int] = set()
        releases_at_event: set[int] = set()
        for lane, end in enumerate(active_ends):
            if end is None or end > start:
                continue
            if prior_start is not None and prior_start < end <= start:
                releases_between.add(lane)
            if end == start:
                releases_at_event.add(lane)
                releases_between.add(lane)
            if last_release[lane] is None or end > last_release[lane]:
                last_release[lane] = end

        if not notes and not releases_at_event:
            raise ValueError(
                f"row {row_offset} has no notes and no LN release at its event time"
            )

        note_lanes = {lane for lane, _ in notes}
        for lane in note_lanes:
            end = active_ends[lane]
            if end is not None and end > start:
                raise ValueError(f"lane {lane} has overlapping LN occupancy at {start} ms")

        for lane in range(keys):
            base = lane_base + lane * lane_block
            output[base + _LANE_INDEX["valid_lane"]] = 1.0
            output[base + _LANE_INDEX["occupied_before"]] = float(
                active_ends[lane] is not None and active_ends[lane] > start
            )
            remaining = (
                active_ends[lane] - start
                if active_ends[lane] is not None and active_ends[lane] > start
                else 0
            )
            output[base + _LANE_INDEX["hold_remaining"]] = _beat_duration(
                row_beat_ms, remaining, _MAX_REMAINING_BEATS
            )
            prior_attack = last_attack[lane]
            output[base + _LANE_INDEX["since_attack"]] = (
                _beat_duration(row_beat_ms, start - prior_attack, _MAX_HISTORY_BEATS)
                if prior_attack is not None
                else 1.0
            )
            prior_release = last_release[lane]
            output[base + _LANE_INDEX["since_release"]] = (
                _beat_duration(row_beat_ms, start - prior_release, _MAX_HISTORY_BEATS)
                if prior_release is not None
                else 1.0
            )
            output[base + _LANE_INDEX["release_between_rows"]] = float(
                lane in releases_between
            )

        hold_count = 0
        for lane, end in notes:
            base = lane_base + lane * lane_block
            output[base + _LANE_INDEX["attack"]] = 1.0
            last_attack[lane] = start
            if end > start:
                hold_count += 1
                output[base + _LANE_INDEX["hold_head"]] = 1.0
                output[base + _LANE_INDEX["hold_duration"]] = min(
                    np.log1p((end - start) / row_beat_ms) / np.log1p(_MAX_HOLD_BEATS),
                    1.0,
                )
                active_ends[lane] = end
            else:
                active_ends[lane] = None

        global_base = row_base + MAX_LANES * lane_block
        output[global_base + _ROW_INDEX["event"]] = 1.0
        output[global_base + _ROW_INDEX["chord_share"]] = len(notes) / keys
        output[global_base + _ROW_INDEX["hold_share"]] = (
            hold_count / len(notes) if notes else 0.0
        )
        if prior_start is not None:
            gap_beats = (start - prior_start) / row_beat_ms
            output[global_base + _ROW_INDEX["gap_beats"]] = min(
                gap_beats, _MAX_HISTORY_BEATS
            ) / _MAX_HISTORY_BEATS
        prior_start = start

    global_base = WINDOW_ROWS * ROW_WIDTH
    output[global_base + keys - 6] = 1.0
    output[-1] = beat_lengths[-1] / 1000.0
    return output


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


def mirror_window_features(x, keys):
    """Mirror lane-indexed features without changing row order or global data."""
    keys = _positive_keys(keys)
    values = np.asarray(x, dtype=np.float32)
    valid_counts = tuple(FEATURE_SCHEMA_COUNTS.values())
    if values.ndim not in (1, 2) or values.shape[-1] not in valid_counts:
        raise ValueError(
            "x must have shape (width,) or (batch, width) for a supported "
            f"feature schema with width in {valid_counts}"
        )
    mirrored = np.array(values, copy=True)
    row_data = mirrored[..., : WINDOW_ROWS * ROW_WIDTH].reshape(
        -1, WINDOW_ROWS, ROW_WIDTH
    )
    lanes = row_data[:, :, : MAX_LANES * len(LANE_FEATURES)].reshape(
        -1, WINDOW_ROWS, MAX_LANES, len(LANE_FEATURES)
    )
    lanes[:, :, :keys] = lanes[:, :, :keys][:, :, ::-1].copy()
    return mirrored
