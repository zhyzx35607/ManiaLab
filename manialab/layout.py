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
from .timing import BeatClock
from .ln_patterns import ln_row_costs


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


def source_key(note):
    voice = getattr(note, "voice", "mix")
    return (voice, note.pitch) if note.source == "pitch" else (note.source, note.lane)


def _conflicts(a, b):
    return a.start == b.start or a.start <= b.start < a.end or b.start <= a.start < b.end


def search_layout(notes, settings, locked=(), beat_ms=500, width=8, progress=None, cancelled=lambda: False, segment_preferences=(), timing_points=()):
    keys = settings["keys"]
    notes, locked = list(notes), list(locked)
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
    beam = [State(0., np.full(keys, -100000.), np.full(keys, -100000.), np.zeros(keys), {}, (), 0, None, ())]
    fixed_index = 0
    groups = sorted(by_time.items())
    pitch_values = [n.pitch for n in notes if n.source == "pitch"]
    low, high = np.percentile(pitch_values, [10, 90]) if pitch_values else (36., 84.)
    high = max(low + 12, high)
    clock = BeatClock(timing_points, beat_ms)
    for row_index, (start, chord) in enumerate(groups):
        beat_ms = clock(start)
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
            values += noise
            values[~valid] = np.inf
            # Keep several branches from each state, then select globally.
            keep = min(width * 2, int(valid.sum()))
            winners = np.argpartition(values, keep - 1)[:keep]
            candidates.extend((state.cost + float(values[c]), parent_index, int(c), previous_ends,
                               previous_last, load) for c in winners)
        if not candidates:
            raise ValueError(f"{start}ms 的 LN 占轨后搜索未找到保时定轨；未删音或缩短 LN")
        candidates.sort(key=lambda item: item[0])
        next_beam, seen = [], set()
        for score, parent_index, choice_index, previous_ends, previous_last, load in candidates:
            parent = beam[parent_index]
            columns = choices[choice_index]
            new_ends, new_last, new_load = previous_ends.copy(), previous_last.copy(), load.copy()
            new_ends[columns], new_last[columns] = ends, start
            new_load[columns] += 1 + np.minimum(duration / 2500, 2)
            mask = sum(1 << int(c) for c in columns)
            signature = (tuple(new_ends), tuple(new_last), mask)
            if signature in seen:
                continue
            seen.add(signature)
            mapping = {k: v for k, v in parent.source.items() if start - v[1] < beat_ms * 8}
            for source, col, end in zip(sources, columns, ends):
                mapping[source] = (int(col), start, end)
            decision = tuple(replace(n, lane=int(col)) for n, col in zip(chord, columns))
            next_beam.append(State(score, new_ends, new_last, new_load, mapping,
                (*parent.recent[-2:], (start, mask, size)), start, parent, decision))
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
