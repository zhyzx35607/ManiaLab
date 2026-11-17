"""Shared input validation for versioned sequence feature encoders."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

from .sequence_schema import WINDOW_ROWS


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
