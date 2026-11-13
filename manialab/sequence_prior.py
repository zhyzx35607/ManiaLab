"""Runtime support for the research-only four-row mania sequence prior."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .pattern_prior import score_network
from .pattern_sequence import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_COUNTS,
    FEATURE_SCHEMA_NAMES,
    LN_INTERACTION_SCHEMA,
    LN_INTERACTION_ROW_FEATURES,
    LN_PHRASE_CONTEXT_SCHEMA,
    LANE_FEATURES,
    MAX_LANES,
    ROW_FEATURES,
    ROW_WIDTH,
    WINDOW_ROWS,
    mirror_window_features,
    window_features,
    window_features_with_ln_interactions,
    window_features_with_ln_phrase_context,
)
from .timing import BeatClock

MIN_SEQUENCE_BEAT_MS = 10.0
MAX_SEQUENCE_BEAT_MS = 10_000.0
SEQUENCE_PRIOR_REVISION = "sequence-window-v1"


@dataclass(frozen=True, slots=True)
class SequenceRow:
    start_ms: int
    beat_ms: float
    notes: tuple[tuple[int, int], ...]
    initial_ends: tuple[int, ...]
    initial_last: tuple[int, ...]
    beat_position: float | None = None
    note_end_positions: tuple[float, ...] | None = None
    initial_end_positions: tuple[float | None, ...] | None = None
    initial_last_positions: tuple[float | None, ...] | None = None

    def feature_row(self):
        return {
            "start_ms": self.start_ms,
            "notes": self.notes,
            "beat_position": self.beat_position,
            "note_end_positions": self.note_end_positions,
        }


@dataclass(frozen=True, slots=True)
class SequencePrior:
    layers: tuple[dict[str, np.ndarray], ...]
    sha256: str
    training_manifest_sha256: str = ""
    feature_schema: str = "sequence-window-v1"


def sequence_beat_clock(timing_points=(), fallback=500.0):
    """Apply the training-time red-line guard before encoding local beats."""
    filtered = []
    ignored = 0
    for line in timing_points:
        fields = line.split(",")
        try:
            period = float(fields[1])
            red = len(fields) < 7 or fields[6].strip() == "1"
        except (ValueError, IndexError):
            filtered.append(line)
            continue
        if red and (
            not np.isfinite(period)
            or period < MIN_SEQUENCE_BEAT_MS
            or period > MAX_SEQUENCE_BEAT_MS
        ):
            ignored += 1
            continue
        filtered.append(line)
    return BeatClock(filtered, fallback), ignored


def sequence_beat_positions(times, timing_points=(), fallback=500.0):
    """Integrate red-line timing points into continuous beat positions."""
    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError("fallback beat period must be finite and positive")
    clock, _ = sequence_beat_clock(timing_points, fallback)
    offsets = np.asarray(clock.offsets, dtype=np.float64)
    periods = np.asarray(clock.periods, dtype=np.float64)
    if not len(offsets):
        offsets = np.asarray([0.0], dtype=np.float64)
        periods = np.asarray([fallback], dtype=np.float64)
    origins = np.zeros(len(offsets), dtype=np.float64)
    for index in range(1, len(offsets)):
        origins[index] = (
            origins[index - 1]
            + (offsets[index] - offsets[index - 1]) / periods[index - 1]
        )
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("times must be a one-dimensional finite sequence")
    indices = np.maximum(0, np.searchsorted(offsets, values, side="right") - 1)
    result = origins[indices] + (
        values - offsets[indices]
    ) / periods[indices]
    if not np.isfinite(result).all():
        raise ValueError("Could not integrate sequence beat positions")
    return result


def load_sequence_prior(path):
    """Load and validate an explicitly selected research sequence model."""
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    feature_schema = value.get("revision")
    if (not isinstance(feature_schema, str)
            or feature_schema not in FEATURE_SCHEMA_NAMES):
        raise ValueError("Unsupported sequence prior revision")
    feature_names = FEATURE_SCHEMA_NAMES[feature_schema]
    if value.get("features") != list(feature_names):
        raise ValueError("Sequence prior feature schema mismatch")
    manifest_hash = value.get("training_manifest_sha256")
    if not isinstance(manifest_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
        raise ValueError("Sequence prior training manifest hash is invalid")
    exported = value.get("layers")
    if not isinstance(exported, list) or not exported:
        raise ValueError("Sequence prior must contain exported layers")

    layers = []
    width = len(feature_names)
    for index, layer in enumerate(exported):
        weights = np.asarray(layer["weight"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        if (
            weights.ndim != 2
            or weights.shape[1] != width
            or weights.shape[0] == 0
            or bias.shape != (weights.shape[0],)
            or not np.isfinite(weights).all()
            or not np.isfinite(bias).all()
        ):
            raise ValueError(f"Invalid sequence prior layer {index}")
        layers.append({"weight": weights, "bias": bias})
        width = weights.shape[0]
    if width != 1:
        raise ValueError("Sequence prior must produce one score per window")
    return SequencePrior(
        tuple(layers),
        hashlib.sha256(raw).hexdigest(),
        manifest_hash,
        feature_schema,
    )


def sequence_window_features(rows, keys, feature_schema="sequence-window-v1"):
    """Encode the exact right-aligned window and left-edge state used in training."""
    if not rows:
        raise ValueError("A sequence window must contain at least one event row")
    if (not isinstance(feature_schema, str)
            or feature_schema not in FEATURE_SCHEMA_NAMES):
        raise ValueError(f"Unsupported sequence feature schema: {feature_schema}")
    first = rows[0]
    encoded_rows = [row.feature_row() for row in rows]
    beat_lengths = [row.beat_ms for row in rows]
    if feature_schema == LN_INTERACTION_SCHEMA:
        return window_features_with_ln_interactions(
            encoded_rows,
            keys,
            beat_lengths,
            initial_ends=first.initial_ends,
            initial_last=first.initial_last,
        )
    if feature_schema == LN_PHRASE_CONTEXT_SCHEMA:
        return window_features_with_ln_phrase_context(
            encoded_rows,
            keys,
            beat_lengths,
            initial_ends=first.initial_ends,
            initial_last=first.initial_last,
            initial_end_positions=first.initial_end_positions,
            initial_last_positions=first.initial_last_positions,
        )
    return window_features(
        encoded_rows, keys, beat_lengths,
        initial_ends=first.initial_ends,
        initial_last=first.initial_last,
    )


def _sequence_windows_ln_interactions(windows, keys):
    """Batch-encode v5 LN interaction features for runtime candidate windows."""
    output = np.zeros(
        (len(windows), WINDOW_ROWS, len(LN_INTERACTION_ROW_FEATURES)),
        dtype=np.float32,
    )
    feature_index = {
        name: index for index, name in enumerate(LN_INTERACTION_ROW_FEATURES)
    }
    unknown = -100000

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
    connected_hand = same_hand > 0
    pair_left, pair_right = np.triu_indices(keys, 1)
    pair_denominator = max(1, len(pair_left))
    grouped = {}
    for index, rows in enumerate(windows):
        grouped.setdefault(len(rows), []).append(index)

    for row_count, indices in grouped.items():
        selected = [windows[index] for index in indices]
        batch_size = len(selected)
        starts = np.asarray(
            [[row.start_ms for row in rows] for rows in selected],
            dtype=np.int64,
        )
        beats = np.asarray(
            [[row.beat_ms for row in rows] for rows in selected],
            dtype=np.float64,
        )
        if not np.isfinite(beats).all() or np.any(beats <= 0):
            raise ValueError("Sequence row beat_ms values must be finite and positive")

        initial_ends = np.asarray(
            [rows[0].initial_ends for rows in selected], dtype=np.int64
        )
        last_attack = np.asarray(
            [rows[0].initial_last for rows in selected], dtype=np.int64
        )
        if initial_ends.shape != (batch_size, keys) or last_attack.shape != (
                batch_size, keys):
            raise ValueError("Sequence row state width does not match key count")
        if np.any(last_attack > starts[:, :1]):
            raise ValueError("initial_last values cannot be later than the first row")

        attacks = np.zeros((batch_size, row_count, keys), dtype=bool)
        note_ends = np.full(
            (batch_size, row_count, keys), unknown, dtype=np.int64
        )
        for batch_index, rows in enumerate(selected):
            previous_start = None
            for offset, row in enumerate(rows):
                start = int(row.start_ms)
                if previous_start is not None and start <= previous_start:
                    raise ValueError("Sequence row start_ms values must increase")
                previous_start = start
                for lane, end in row.notes:
                    lane = int(lane)
                    end = int(end)
                    if not 0 <= lane < keys or end < start:
                        raise ValueError("Sequence row contains an invalid note")
                    if attacks[batch_index, offset, lane]:
                        raise ValueError("Sequence row contains duplicate lane notes")
                    attacks[batch_index, offset, lane] = True
                    note_ends[batch_index, offset, lane] = end

        active_ends = initial_ends.copy()
        prior_start = np.full(batch_size, unknown, dtype=np.int64)
        past_starts = np.full(
            (batch_size, (WINDOW_ROWS + 1) * keys), unknown, dtype=np.int64
        )
        past_ends = np.full_like(past_starts, unknown)
        past_lanes = np.broadcast_to(
            np.tile(np.arange(keys), WINDOW_ROWS + 1),
            past_starts.shape,
        )
        initial_past = (
            (last_attack != unknown) & (initial_ends > last_attack)
        )
        past_starts[:, :keys] = np.where(
            initial_past, last_attack, unknown
        )
        past_ends[:, :keys] = np.where(
            initial_past, initial_ends, unknown
        )
        group_output = np.zeros(
            (batch_size, WINDOW_ROWS, len(LN_INTERACTION_ROW_FEATURES)),
            dtype=np.float32,
        )

        for offset in range(row_count):
            start = starts[:, offset]
            beat_ms = beats[:, offset]
            row_position = WINDOW_ROWS - row_count + offset
            prior_hold = (
                (last_attack != unknown) & (active_ends > last_attack)
            )
            active = prior_hold & (active_ends > start[:, None])
            released = prior_hold & (
                (active_ends == start[:, None])
                | (
                    (prior_start[:, None] != unknown)
                    & (prior_start[:, None] < active_ends)
                    & (active_ends < start[:, None])
                )
            )

            current_ends = note_ends[:, offset]
            current_attacks = attacks[:, offset]
            ln_heads = current_attacks & (current_ends > start[:, None])
            taps = current_attacks & (current_ends == start[:, None])
            occupied = active_ends > start[:, None]
            if np.any(current_attacks & occupied):
                raise ValueError("Sequence row overlaps an active LN")
            if np.any(
                ~current_attacks.any(axis=1)
                & ~np.any(active_ends == start[:, None], axis=1)
            ):
                raise ValueError("Sequence row has no notes or LN release")

            values = group_output[:, row_position]

            def set_count(name, count, denominator=keys):
                values[:, feature_index[name]] = (
                    count / max(1, denominator)
                )

            active_count = active.sum(axis=1)
            set_count("active_ln_density", active_count)
            if len(pair_left):
                active_pairs = (
                    active[:, pair_left] & active[:, pair_right]
                )
                same_pairs = np.sum(
                    active_pairs * same_hand[pair_left, pair_right], axis=1
                )
                cross_pairs = np.sum(
                    active_pairs * (1.0 - same_hand[pair_left, pair_right]),
                    axis=1,
                )
            else:
                same_pairs = cross_pairs = np.zeros(batch_size)
            set_count(
                "active_ln_same_hand_pairs", same_pairs, pair_denominator
            )
            set_count(
                "active_ln_cross_hand_pairs", cross_pairs, pair_denominator
            )
            set_count("ln_head_density", ln_heads.sum(axis=1))
            set_count("tap_head_density", taps.sum(axis=1))
            set_count("release_density", released.sum(axis=1))

            active_affinity = active @ same_hand
            active_affinity = np.divide(
                active_affinity,
                active_count[:, None],
                out=np.zeros_like(active_affinity),
                where=active_count[:, None] > 0,
            )
            same_hand_taps = np.sum(active_affinity * taps, axis=1)
            same_hand_lns = np.sum(active_affinity * ln_heads, axis=1)
            cross_hand_taps = np.sum(
                (1.0 - active_affinity) * taps, axis=1
            )
            cross_hand_lns = np.sum(
                (1.0 - active_affinity) * ln_heads, axis=1
            )
            cross_hand_taps = np.where(active_count > 0, cross_hand_taps, 0)
            cross_hand_lns = np.where(active_count > 0, cross_hand_lns, 0)
            set_count("hybrid_same_hand_taps", same_hand_taps)
            set_count("hybrid_cross_hand_taps", cross_hand_taps)
            set_count("hybrid_same_hand_ln_heads", same_hand_lns)
            set_count("hybrid_cross_hand_ln_heads", cross_hand_lns)

            release_count = released.sum(axis=1)
            release_affinity = released @ same_hand
            release_affinity = np.divide(
                release_affinity,
                release_count[:, None],
                out=np.zeros_like(release_affinity),
                where=release_count[:, None] > 0,
            )
            same_hand_release_press = np.sum(
                release_affinity * current_attacks, axis=1
            )
            cross_hand_release_press = np.sum(
                (1.0 - release_affinity) * current_attacks, axis=1
            )
            cross_hand_release_press = np.where(
                release_count > 0, cross_hand_release_press, 0
            )
            set_count(
                "release_press_same_hand", same_hand_release_press
            )
            set_count(
                "release_press_cross_hand", cross_hand_release_press
            )

            split_release_hands = np.zeros(batch_size, dtype=np.float64)
            for hand in range(2):
                in_hand = hand_weights[:, hand] > 0
                split_release_hands += (
                    released[:, in_hand].any(axis=1)
                    & active[:, in_hand].any(axis=1)
                )
            set_count("split_release_hands", split_release_hands, 2)

            inverse_window = np.minimum(150.0, beat_ms * 0.3)
            gaps = start[:, None] - active_ends
            near_inverse = (
                prior_hold
                & (gaps > 0)
                & (gaps <= inverse_window[:, None])
            )
            set_count("inverse_ln_heads", (near_inverse & ln_heads).sum(axis=1))
            set_count(
                "reverse_shield_taps", (near_inverse & taps).sum(axis=1)
            )
            shield = (
                ~prior_hold
                & ln_heads
                & (last_attack != unknown)
                & ((start[:, None] - last_attack) > 0)
                & (
                    (start[:, None] - last_attack)
                    <= inverse_window[:, None]
                )
            )
            set_count("shield_ln_heads", shield.sum(axis=1))

            release_window = np.maximum(
                40.0, np.minimum(150.0, beat_ms * 0.25)
            )
            past_valid = past_starts != unknown
            near_past = (
                past_valid
                & (past_starts < start[:, None])
                & (np.abs(past_ends - start[:, None]) > 1)
                & (np.abs(past_ends - start[:, None]) < release_window[:, None])
            )
            nearby_by_lane = np.zeros((batch_size, keys), dtype=bool)
            for lane in range(keys):
                nearby_by_lane[:, lane] = np.any(
                    near_past & (past_lanes == lane), axis=1
                )
            has_nearby_hand = (nearby_by_lane @ connected_hand) > 0
            near_release_press = np.sum(
                has_nearby_hand & current_attacks, axis=1
            )
            set_count("near_release_press", near_release_press)

            past_slot = slice(
                (offset + 1) * keys, (offset + 2) * keys
            )
            past_starts[:, past_slot] = np.where(
                ln_heads, start[:, None], unknown
            )
            past_ends[:, past_slot] = np.where(
                ln_heads, current_ends, unknown
            )
            last_attack = np.where(current_attacks, start[:, None], last_attack)
            active_ends = np.where(
                ln_heads,
                current_ends,
                np.where(current_attacks, unknown, active_ends),
            )
            prior_start = start

        output[indices] = group_output

    return output.reshape(len(windows), -1)


def sequence_windows_features(windows, keys, feature_schema="sequence-window-v1"):
    """Batch-encode candidate windows while matching the training encoder."""
    if not windows:
        if (not isinstance(feature_schema, str)
                or feature_schema not in FEATURE_SCHEMA_COUNTS):
            raise ValueError(f"Unsupported sequence feature schema: {feature_schema}")
        return np.empty((0, FEATURE_SCHEMA_COUNTS[feature_schema]), dtype=np.float32)
    if feature_schema == LN_PHRASE_CONTEXT_SCHEMA:
        return np.stack([
            sequence_window_features(rows, keys, feature_schema)
            for rows in windows
        ])
    if any(not 1 <= len(rows) <= WINDOW_ROWS for rows in windows):
        raise ValueError("A sequence window must contain one to four event rows")
    if feature_schema == LN_INTERACTION_SCHEMA:
        base = sequence_windows_features(
            windows, keys, "sequence-window-v1"
        )
        interactions = _sequence_windows_ln_interactions(windows, keys)
        return np.concatenate((base, interactions), axis=1)
    if feature_schema != "sequence-window-v1":
        raise ValueError(f"Unsupported sequence feature schema: {feature_schema}")

    output = np.zeros((len(windows), FEATURE_COUNT), dtype=np.float32)
    grouped = {}
    for index, rows in enumerate(windows):
        grouped.setdefault(len(rows), []).append(index)
    lane_index = {name: index for index, name in enumerate(LANE_FEATURES)}
    row_index = {name: index for index, name in enumerate(ROW_FEATURES)}
    initial_unknown = -100000

    for row_count, indices in grouped.items():
        batch_size = len(indices)
        batch_indices = np.asarray(indices, dtype=np.int64)
        selected = [windows[index] for index in indices]
        starts = np.asarray(
            [[row.start_ms for row in rows] for rows in selected],
            dtype=np.int64,
        )
        beats = np.asarray(
            [[row.beat_ms for row in rows] for rows in selected],
            dtype=np.float64,
        )
        if not np.isfinite(beats).all() or np.any(beats <= 0):
            raise ValueError("Sequence row beat_ms values must be finite and positive")

        initial_ends = np.asarray(
            [rows[0].initial_ends for rows in selected], dtype=np.int64
        )
        last_attack = np.asarray(
            [rows[0].initial_last for rows in selected], dtype=np.int64
        )
        if initial_ends.shape != (batch_size, keys) or last_attack.shape != (
                batch_size, keys):
            raise ValueError("Sequence row state width does not match key count")
        if np.any(last_attack > starts[:, :1]):
            raise ValueError("initial_last values cannot be later than the first row")

        active_ends = initial_ends.copy()
        last_release = np.where(
            (active_ends != initial_unknown)
            & (active_ends <= starts[:, :1]),
            active_ends,
            initial_unknown,
        )
        prior_start = np.full(batch_size, initial_unknown, dtype=np.int64)

        for offset in range(row_count):
            row_starts = starts[:, offset]
            row_beats = beats[:, offset]
            row_position = WINDOW_ROWS - row_count + offset
            row_base = row_position * ROW_WIDTH
            features = np.zeros((batch_size, ROW_WIDTH), dtype=np.float32)
            lane_features = features[:, :MAX_LANES * len(LANE_FEATURES)].reshape(
                batch_size, MAX_LANES, len(LANE_FEATURES)
            )
            lane_features[:, :keys, lane_index["valid_lane"]] = 1.0

            known_end = active_ends != initial_unknown
            expired = known_end & (active_ends <= row_starts[:, None])
            has_prior = prior_start != initial_unknown
            released = expired & (
                ((has_prior[:, None])
                 & (prior_start[:, None] < active_ends)
                 & (active_ends <= row_starts[:, None]))
                | (active_ends == row_starts[:, None])
            )
            update_release = expired & (
                (last_release == initial_unknown)
                | (active_ends > last_release)
            )
            last_release = np.where(update_release, active_ends, last_release)

            occupied = known_end & (active_ends > row_starts[:, None])
            remaining = np.maximum(
                0, active_ends - row_starts[:, None]
            )
            lane_features[:, :keys, lane_index["occupied_before"]] = occupied
            remaining_beats = remaining / row_beats[:, None]
            lane_features[:, :keys, lane_index["hold_remaining"]] = (
                np.minimum(remaining_beats, 32.0) / 32.0
            )
            has_attack = last_attack != initial_unknown
            since_attack = np.maximum(
                0, row_starts[:, None] - last_attack
            ) / row_beats[:, None]
            lane_features[:, :keys, lane_index["since_attack"]] = np.where(
                has_attack,
                np.minimum(since_attack, 8.0) / 8.0,
                1.0,
            )
            has_release = last_release != initial_unknown
            since_release = np.maximum(
                0, row_starts[:, None] - last_release
            ) / row_beats[:, None]
            lane_features[:, :keys, lane_index["since_release"]] = np.where(
                has_release,
                np.minimum(since_release, 8.0) / 8.0,
                1.0,
            )
            lane_features[:, :keys, lane_index["release_between_rows"]] = released

            note_lanes = np.zeros((batch_size, keys), dtype=bool)
            note_ends = np.full(
                (batch_size, keys), initial_unknown, dtype=np.int64
            )
            for batch_index, rows in enumerate(selected):
                for lane, end in rows[offset].notes:
                    if not 0 <= lane < keys or end < row_starts[batch_index]:
                        raise ValueError("Sequence row contains an invalid note")
                    if note_lanes[batch_index, lane]:
                        raise ValueError("Sequence row contains duplicate lane notes")
                    if occupied[batch_index, lane]:
                        raise ValueError("Sequence row overlaps an active LN")
                    note_lanes[batch_index, lane] = True
                    note_ends[batch_index, lane] = end
            if np.any(~note_lanes.any(axis=1) & ~np.any(
                    active_ends == row_starts[:, None], axis=1)):
                raise ValueError("Sequence row has no notes or LN release")

            lane_features[:, :keys, lane_index["attack"]] = note_lanes
            hold_heads = note_lanes & (note_ends > row_starts[:, None])
            lane_features[:, :keys, lane_index["hold_head"]] = hold_heads
            hold_duration = np.maximum(
                0, note_ends - row_starts[:, None]
            ) / row_beats[:, None]
            lane_features[:, :keys, lane_index["hold_duration"]] = np.where(
                hold_heads,
                np.minimum(
                    np.log1p(hold_duration) / np.log1p(32.0), 1.0
                ),
                0.0,
            )
            last_attack = np.where(note_lanes, row_starts[:, None], last_attack)
            active_ends = np.where(
                note_lanes & hold_heads,
                note_ends,
                np.where(note_lanes, initial_unknown, active_ends),
            )

            global_features = features[:, MAX_LANES * len(LANE_FEATURES):]
            global_features[:, row_index["event"]] = 1.0
            note_count = note_lanes.sum(axis=1)
            hold_count = hold_heads.sum(axis=1)
            global_features[:, row_index["chord_share"]] = note_count / keys
            global_features[:, row_index["hold_share"]] = np.divide(
                hold_count,
                note_count,
                out=np.zeros(batch_size, dtype=np.float64),
                where=note_count > 0,
            )
            gap = np.maximum(0, row_starts - prior_start) / row_beats
            global_features[:, row_index["gap_beats"]] = np.where(
                has_prior,
                np.minimum(gap, 8.0) / 8.0,
                0.0,
            )
            output[
                batch_indices,
                row_base:row_base + ROW_WIDTH,
            ] = features
            prior_start = row_starts

        key_base = WINDOW_ROWS * ROW_WIDTH
        output[indices, key_base + keys - 6] = 1.0
        output[indices, -1] = beats[:, -1] / 1000.0

    return output


def score_sequence_windows(windows, keys, prior):
    """Score candidate windows with the training-time left/right mirror average."""
    if not windows:
        return np.empty(0, dtype=np.float32)
    encoded = sequence_windows_features(windows, keys, prior.feature_schema)
    mirrored = mirror_window_features(encoded, keys)
    batch_size = len(encoded)
    scores = score_network(
        np.concatenate((encoded, mirrored), axis=0), prior.layers)
    return (scores[:batch_size] + scores[batch_size:]) / 2
