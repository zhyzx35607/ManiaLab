"""LN-aware beam search for 6–10K, preserving every input object.

Torosu's explicit hand layout and multi-row search informed the design. This
implementation uses ordered chord subsets, vectorized costs and strict interval
constraints: it never copies, drops, shortens or retimes a note to fit a lane.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
import math

import numpy as np

from .domain import invariant, validate
from .pattern_prior import preference_cost
from .arrange import settings_at
from .sequence_prior import (
    SequencePrior,
    SequenceRow,
    score_sequence_windows,
    sequence_beat_clock,
    sequence_beat_positions,
)
from .pattern_sequence import LN_PHRASE_CONTEXT_BEATS, LN_PHRASE_CONTEXT_SCHEMA
from .timing import BeatClock
from .ln_patterns import ln_row_costs

# Distinct LN-release orderings can expand the 10K five-note subset space to 1008.
SEQUENCE_SCORE_CANDIDATE_CAP = 1024


@lru_cache(maxsize=5)
def fingers(keys):
    """(hand, digit), with digit 1=thumb .. 5=pinky; hand 2=shared thumb."""
    profiles = {
        6: ((0, 4), (0, 3), (0, 2), (1, 2), (1, 3), (1, 4)),
        7: ((0, 4), (0, 3), (0, 2), (2, 1), (1, 2), (1, 3), (1, 4)),
        8: ((0, 5), (0, 4), (0, 3), (0, 2), (1, 2), (1, 3), (1, 4), (1, 5)),
        9: ((0, 5), (0, 4), (0, 3), (0, 2), (2, 1), (1, 2), (1, 3), (1, 4), (1, 5)),
        10: ((0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)),
    }
    return profiles[keys]


@lru_cache(maxsize=50)
def chord_choices(keys, size):
    return np.array(list(combinations(range(keys), size)), dtype=np.int16).reshape(-1, size)


def release_choices(keys, ends):
    """Bounded alternative voice orders for chords with different release times.

    Ordered lane subsets alone force pitch order even when a different voice
    needs the free hand. Include mirrored and release-ordered assignments; the
    existing pitch preference still scores each actual voice-to-lane mapping.
    """
    base = chord_choices(keys, len(ends))
    if len(ends) < 2 or np.all(ends == ends[0]):
        return base
    order = np.argsort(ends, kind='stable')
    inverse = np.argsort(order)
    alternatives = [base, base[:, ::-1], base[:, inverse], base[:, ::-1][:, inverse]]
    return np.unique(np.concatenate(alternatives), axis=0)


@dataclass(slots=True)
class State:
    cost: float
    ends: np.ndarray
    last: np.ndarray
    load: np.ndarray
    source: dict
    recent: tuple
    time: int
    parent: object
    notes: tuple
    sequence_rows: tuple = ()


def _freeze_state_value(value):
    if isinstance(value, np.ndarray):
        return tuple(_freeze_state_value(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(sorted(
            ((_freeze_state_value(key), _freeze_state_value(item))
             for key, item in value.items()),
            key=lambda pair: repr(pair[0]),
        ))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_state_value(item) for item in value)
    return value


def _state_signature(state):
    """Capture every field that can change a future beam transition."""
    return (
        _freeze_state_value(state.ends),
        _freeze_state_value(state.last),
        _freeze_state_value(state.load),
        _freeze_state_value(state.source),
        _freeze_state_value(state.recent),
        _freeze_state_value(state.time),
        _freeze_state_value(state.sequence_rows),
    )


def _sequence_score_shortlist(values, valid_choices, keep, strength):
    """Skip model inference for candidates that cannot enter the beam winners.

    The learned adjustment is bounded by ``strength`` in either direction and
    deterministic noise is in [0, .025). A fixed reference model score is
    taken from the lowest-cost baseline candidate. Any candidate more than
    twice the learned-adjustment range above the unadjusted kth score cannot
    overtake the kth baseline candidate.
    """
    valid_choices = np.asarray(valid_choices, dtype=np.int64)
    if len(valid_choices) <= keep:
        return valid_choices
    baseline = values[valid_choices]
    cutoff = np.partition(baseline, keep - 1)[keep - 1]
    margin = 2 * strength + .025
    return valid_choices[baseline <= cutoff + margin]


def _limit_sequence_score_candidates(values, choices, limit):
    """Guard candidate scoring while retaining the bounded lowest-cost rows."""
    choices = np.asarray(choices, dtype=np.int64)
    if len(choices) <= limit:
        return choices
    best = np.argpartition(values[choices], limit - 1)[:limit]
    return np.sort(choices[best])


def source_key(note):
    voice = getattr(note, "voice", "mix")
    return (voice, note.instrument_id, note.pitch) if note.source == "pitch" else (note.source, note.lane)


def _conflicts(a, b):
    return a.start == b.start or a.start <= b.start < a.end or b.start <= a.start < b.end


def search_layout(notes, settings, locked=(), beat_ms=500, width=8, progress=None,
                  cancelled=lambda: False, segment_preferences=(), timing_points=(),
                  sequence_prior=None, sequence_strength=.35):
    keys = settings["keys"]
    notes, locked = list(notes), list(locked)
    if sequence_prior is not None:
        if not isinstance(sequence_prior, SequencePrior):
            raise TypeError("sequence_prior must be loaded with load_sequence_prior()")
        if (
            isinstance(sequence_strength, bool)
            or not isinstance(sequence_strength, (int, float))
            or not math.isfinite(sequence_strength)
            or not 0 <= sequence_strength <= 1
        ):
            raise ValueError("sequence_strength must be between 0 and 1")
        if locked:
            raise ValueError("sequence prior currently requires a full-chart layout")
    use_sequence_prior = sequence_prior is not None and sequence_strength > 0
    sequence_clock = (
        sequence_beat_clock(timing_points, 500)[0]
        if use_sequence_prior else None
    )
    phrase_context_prior = (
        use_sequence_prior
        and sequence_prior.feature_schema == LN_PHRASE_CONTEXT_SCHEMA
    )
    if not notes:
        return sorted(locked, key=lambda n: (n.start, n.lane))
    if locked and not validate(locked, keys)["valid"]:
        raise ValueError("锁定区域已有轨道冲突")
    by_time = defaultdict(list)
    for note in notes:
        by_time[note.start].append(note)
    fixed = sorted(locked, key=lambda n: n.start)
    fixed_lanes = [[n for n in fixed if n.lane == lane] for lane in range(keys)]
    profile = fingers(keys)
    hands = np.array([[.5, .5] if h == 2 else [1., 0.] if h == 0 else [0., 1.] for h, _ in profile])
    digits = np.array([d for _, d in profile])
    beam = [State(0., np.full(keys, -100000.), np.full(keys, -100000.),
                  np.zeros(keys), {}, (), 0, None, ())]
    fixed_index = 0
    groups = sorted(by_time.items())
    phrase_position_by_time = None
    phrase_positions = None
    if phrase_context_prior:
        phrase_times = sorted({
            int(value)
            for note in notes
            for value in (note.start, note.end)
        })
        phrase_position_by_time = dict(zip(
            phrase_times,
            sequence_beat_positions(phrase_times, timing_points, 500),
        ))
        phrase_positions = np.asarray([
            phrase_position_by_time[start] for start, _ in groups
        ], dtype=np.float64)
    pitch_values = [n.pitch for n in notes if n.source == "pitch"]
    low, high = np.percentile(pitch_values, [10, 90]) if pitch_values else (36., 84.)
    high = max(low + 12, high)
    clock = BeatClock(timing_points, beat_ms)
    for row_index, (start, chord) in enumerate(groups):
        beat_ms = clock(start)
        sequence_beat_ms = sequence_clock(start) if use_sequence_prior else None
        sequence_beat_position = (
            float(phrase_positions[row_index])
            if phrase_context_prior else None
        )
        local = settings_at(settings, start, segment_preferences)
        if cancelled():
            raise InterruptedError("已取消键型搜索")
        chord = sorted(chord, key=lambda n: (n.pitch if n.source == "pitch" else n.lane, n.id, n.end))
        size = len(chord)
        if size > keys:
            raise ValueError(f"{start}ms 有 {size} 个同时起音，超过 {keys}K")
        slot = np.arange(size)[None, :]
        ends = np.array([n.end for n in chord], float)
        choices = release_choices(keys, ends)
        sources = [source_key(n) for n in chord]
        fixed_since = []
        while fixed_index < len(fixed) and fixed[fixed_index].start <= start:
            fixed_since.append(fixed[fixed_index]);fixed_index += 1
        forbidden = np.array([[any(_conflicts(n, f) for f in fixed_lanes[lane]) for lane in range(keys)] for n in chord])
        preferred = np.array([np.clip((n.pitch - low) / (high - low), 0, 1) * (keys - 1)
                              if n.source == "pitch" else n.lane if n.lane >= 0 else (keys - 1) / 2 for n in chord])
        duration = np.maximum(0, ends - start)
        chord_hands = hands[choices].sum(axis=1)
        hold_hands = (hands[choices] * (duration > 0)[None, :, None]).sum(axis=1)
        # Tiny deterministic noise changes ties, never hides a strong conflict cost.
        seed = np.uint64((settings["seed"] + row_index * 104729) & 0xffffffff)
        noise = ((np.arange(len(choices), dtype=np.uint64) * 2654435761 + seed) % 65521).astype(float) / 65521 * .025
        candidates = []
        parent_records = []
        sequence_windows = []
        for parent_index, state in enumerate(beam):
            previous_ends, previous_last = state.ends.copy(), state.last.copy()
            load = state.load * math.exp(-(start - state.time) / max(350, beat_ms * 3))
            for f in fixed_since:
                previous_ends[f.lane] = max(previous_ends[f.lane], f.end)
                previous_last[f.lane] = max(previous_last[f.lane], f.start)
                load[f.lane] += math.exp(-(start - f.start) / max(350, beat_ms * 3))
            blocked = (previous_ends > start) | (previous_last == start)
            valid = ~(blocked[choices] | forbidden[slot, choices]).any(axis=1)
            if not np.any(valid):
                continue
            last_gap = start - previous_last
            release_gap = start - previous_ends
            jack_window = min(220., max(85., beat_ms * .42))
            lane_cost = np.maximum(0, 1 - last_gap / jack_window) * local["jack_avoid"] * 4
            was_hold = previous_ends > previous_last
            inverse_ok = (release_gap >= max(1, local['release_gap'])) & (release_gap <= min(150., beat_ms * .3))
            release_weight = 1 - inverse_ok * local.get('ln_inverse', 0.) * .9
            lane_cost += np.where(was_hold, np.maximum(0, 1 - release_gap / max(80, local["release_gap"])) * 6 * release_weight, 0)
            lane_cost += load / max(1., load.mean()) * local["balance"] * .25
            cost = np.broadcast_to(lane_cost, (size, keys)).copy()
            cost += abs(np.arange(keys)[None, :] - preferred[:, None]) / keys * local["pitch_follow"] * 1.6
            for i, (note, source) in enumerate(zip(chord, sources)):
                last_source = state.source.get(source)
                if last_source:
                    old_lane, old_time, old_end = last_source
                    gap = start - old_time
                    if gap <= beat_ms * 2:
                        cost[i] += abs(np.arange(keys) - old_lane) / keys * .35
                    # Preserve real imported jacks instead of converting every
                    # repeated source row into random alternation.
                    if note.source == "import" and gap <= beat_ms * .55 and old_end <= old_time:
                        cost[i] += np.arange(keys) != old_lane
                        cost[i, old_lane] -= lane_cost[old_lane] * .75
            values = cost[slot, choices].sum(axis=1)
            values += ln_row_costs(choices, start, ends, previous_ends, previous_last, hands, beat_ms, local)
            held = hands[previous_ends > start].sum(axis=0)
            hand_load = load @ hands
            values += np.sum(chord_hands * (hand_load + held * 1.8), axis=1) * local["balance"] * .1
            # A chord filling one hand while that hand holds LNs needs more
            # coordination than a split row. Keep this a preference.
            capacity = hands.sum(axis=0)
            pressure = chord_hands + held
            values += np.sum(np.maximum(0, pressure - capacity + 1) ** 2, axis=1) * .45
            values += np.abs((hold_hands + held)[:, 0] - (hold_hands + held)[:, 1]) * .12 * local["balance"]
            values += (np.maximum(0, digits[choices] - 3) * np.minimum(duration / 3000, 2)[None, :]).sum(axis=1) * .06
            if state.recent and start - state.recent[-1][0] < beat_ms * 2:
                old_time, old_mask, old_size = state.recent[-1]
                if size == old_size and size >= 2:
                    old_cols = np.array([i for i in range(keys) if old_mask & (1 << i)])
                    values += np.mean(abs(np.sort(choices, axis=1) - old_cols), axis=1) / keys * .15
            strength=local.get('pattern_prior',0)
            if strength:
                values += preference_cost(choices,keys,start,previous_ends,previous_last,
                    state.recent,duration,beat_ms,strength)
            sequence_rows_by_choice = {}
            score_choices = np.empty(0, dtype=np.int64)
            score_window_start = score_window_end = 0
            reference_choice = None
            if use_sequence_prior:
                valid_choices = np.flatnonzero(valid)
                keep = min(width * 2, len(valid_choices))
                score_choices = _sequence_score_shortlist(
                    values, valid_choices, keep, sequence_strength)
                score_choices = _limit_sequence_score_candidates(
                    values, score_choices, SEQUENCE_SCORE_CANDIDATE_CAP)
                reference_choice = int(valid_choices[
                    np.argmin(values[valid_choices])])
                initial_ends = tuple(int(value) for value in state.ends)
                initial_last = tuple(int(value) for value in state.last)
                windows = []
                for choice_index in score_choices:
                    assigned_notes = tuple(sorted(
                        (int(lane), int(end))
                        for lane, end in zip(choices[choice_index], ends)
                    ))
                    row = SequenceRow(
                        start_ms=int(start),
                        beat_ms=float(sequence_beat_ms),
                        notes=assigned_notes,
                        initial_ends=initial_ends,
                        initial_last=initial_last,
                        beat_position=sequence_beat_position,
                        note_end_positions=(
                            tuple(
                                float(phrase_position_by_time[end])
                                for _, end in assigned_notes
                            )
                            if phrase_context_prior else None
                        ),
                        initial_end_positions=(
                            tuple(
                                float(phrase_position_by_time[int(end)])
                                if int(end) in phrase_position_by_time else None
                                for end in initial_ends
                            )
                            if phrase_context_prior else None
                        ),
                        initial_last_positions=(
                            tuple(
                                float(phrase_position_by_time[int(last_time)])
                                if int(last_time) in phrase_position_by_time else None
                                for last_time in initial_last
                            )
                            if phrase_context_prior else None
                        ),
                    )
                    sequence_rows_by_choice[int(choice_index)] = row
                    if phrase_context_prior:
                        history = tuple(
                            previous for previous in state.sequence_rows
                            if sequence_beat_position - previous.beat_position
                            <= LN_PHRASE_CONTEXT_BEATS
                        )
                    else:
                        history = state.sequence_rows[-3:]
                    windows.append((*history, row))
                score_window_start = len(sequence_windows)
                sequence_windows.extend(windows)
                score_window_end = len(sequence_windows)
            parent_records.append(
                dict(
                    parent_index=parent_index,
                    state=state,
                    values=values,
                    valid=valid,
                    previous_ends=previous_ends,
                    previous_last=previous_last,
                    load=load,
                    sequence_rows_by_choice=sequence_rows_by_choice,
                    score_choices=score_choices,
                    score_window_start=score_window_start,
                    score_window_end=score_window_end,
                    reference_choice=reference_choice,
                )
            )
        if use_sequence_prior and sequence_windows:
            learned_scores = score_sequence_windows(
                sequence_windows, keys, sequence_prior)
            for record in parent_records:
                score_choices = record["score_choices"]
                start_index = record["score_window_start"]
                end_index = record["score_window_end"]
                scores = learned_scores[start_index:end_index]
                reference_score = scores[
                    int(np.searchsorted(
                        score_choices, record["reference_choice"]
                    ))
                ]
                record["values"][score_choices] += -sequence_strength * np.tanh(
                    scores - reference_score
                )
                scored_mask = np.zeros(len(choices), dtype=bool)
                scored_mask[score_choices] = True
                record["valid"] &= scored_mask
        for record in parent_records:
            values = record["values"]
            values += noise
            values[~record["valid"]] = np.inf
            keep = min(width * 2, int(record["valid"].sum()))
            winners = np.argpartition(values, keep - 1)[:keep]
            candidates.extend(
                (record["state"].cost + float(values[c]),
                 record["parent_index"], int(c),
                 record["previous_ends"], record["previous_last"],
                 record["load"],
                 record["sequence_rows_by_choice"].get(int(c)))
                for c in winners
            )
        if not candidates:
            raise ValueError(f"{start}ms 的 LN 占轨后搜索未找到保时定轨；未删音或缩短 LN")
        candidates.sort(key=lambda item: item[0])
        next_beam, seen = [], set()
        for (score, parent_index, choice_index, previous_ends, previous_last,
             load, sequence_row) in candidates:
            parent = beam[parent_index]
            columns = choices[choice_index]
            new_ends, new_last, new_load = previous_ends.copy(), previous_last.copy(), load.copy()
            new_ends[columns], new_last[columns] = ends, start
            new_load[columns] += 1 + np.minimum(duration / 2500, 2)
            mask = sum(1 << int(c) for c in columns)
            mapping = {k: v for k, v in parent.source.items() if start - v[1] < beat_ms * 8}
            for source, col, end in zip(sources, columns, ends):
                mapping[source] = (int(col), start, end)
            recent = (*parent.recent[-2:], (start, mask, size))
            decision = tuple(replace(n, lane=int(col)) for n, col in zip(chord, columns))
            if sequence_row is None:
                sequence_history = parent.sequence_rows
            elif phrase_context_prior:
                sequence_history = tuple(
                    previous for previous in (
                        *parent.sequence_rows, sequence_row
                    )
                    if sequence_row.beat_position - previous.beat_position
                    <= LN_PHRASE_CONTEXT_BEATS
                )
            else:
                sequence_history = (*parent.sequence_rows[-2:], sequence_row)
            next_state = State(score, new_ends, new_last, new_load, mapping,
                recent, start, parent, decision,
                sequence_history)
            signature = _state_signature(next_state)
            if signature in seen:
                continue
            seen.add(signature)
            next_beam.append(next_state)
            if len(next_beam) >= width:
                break
        beam = next_beam
        if progress and (row_index % 24 == 0 or row_index + 1 == len(groups)):
            progress(45 + int(40 * (row_index + 1) / len(groups)),
                f"连续键型搜索 {row_index + 1}/{len(groups)} 排 · 保留 {len(beam)} 条路径")
    state = min(beam, key=lambda s: s.cost)
    result = list(locked)
    while state.parent is not None:
        result.extend(state.notes)
        state = state.parent
    result.sort(key=lambda n: (n.start, n.lane))
    if invariant(result) != invariant(notes + locked):
        raise AssertionError("键型搜索改变了输入物件")
    if not validate(result, keys)["valid"]:
        raise AssertionError("键型搜索产生轨道冲突")
    return result
