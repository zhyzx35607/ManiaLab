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

_LANE_INDEX = {name: index for index, name in enumerate(LANE_FEATURES)}
_ROW_INDEX = {name: index for index, name in enumerate(ROW_FEATURES)}
_MAX_HISTORY_BEATS = 8.0
_MAX_REMAINING_BEATS = 32.0
_MAX_HOLD_BEATS = 32.0


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


def mirror_window_features(x, keys):
    """Mirror lane-indexed features without changing row order or global data."""
    keys = _positive_keys(keys)
    values = np.asarray(x, dtype=np.float32)
    if values.ndim not in (1, 2) or values.shape[-1] != FEATURE_COUNT:
        raise ValueError(f"x must have shape ({FEATURE_COUNT},) or (batch, {FEATURE_COUNT})")
    mirrored = np.array(values, copy=True)
    row_data = mirrored[..., : WINDOW_ROWS * ROW_WIDTH].reshape(
        -1, WINDOW_ROWS, ROW_WIDTH
    )
    lanes = row_data[:, :, : MAX_LANES * len(LANE_FEATURES)].reshape(
        -1, WINDOW_ROWS, MAX_LANES, len(LANE_FEATURES)
    )
    lanes[:, :, :keys] = lanes[:, :, :keys][:, :, ::-1].copy()
    return mirrored
