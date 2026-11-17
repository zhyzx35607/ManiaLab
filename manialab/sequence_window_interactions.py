"""Batch encoding for LN interaction sequence features."""
from __future__ import annotations

import numpy as np

from .sequence_schema import LN_INTERACTION_ROW_FEATURES, WINDOW_ROWS


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
