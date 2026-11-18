from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
import random

import numpy as np
from scipy.optimize import linear_sum_assignment

from .domain import UNOBSERVED_RELEASES, Note, invariant, validate
from .timing import BeatClock
from .preferences import row_settings, settings_at
from .solver import solver_lock as _solver_lock, solve_binary as _solve_binary
from .arrange_config import BOUNDS, DEFAULTS, INTEGER, parameters
from .candidate_selection import _choose_holds as _choose_candidate_holds
from .candidate_selection import _select_notes as _select_candidate_notes


def hand_weights(keys, lane):
    if keys % 2 and lane == keys // 2:
        return (.5, .5)
    return (1., 0.) if lane < keys // 2 else (0., 1.)


def layout_metrics(notes, keys, beat_ms=500, timing_points=()):
    """Comparable ergonomic costs, not a learned or player-validated quality grade."""
    last = [-100000] * keys
    released = [-100000] * keys
    pressure = [0., 0.]
    previous_time = 0
    cost, jacks, releases, peak = 0., 0, 0, 0.
    clock = BeatClock(timing_points, beat_ms)
    for n in sorted(notes, key=lambda n: (n.start, n.lane)):
        beat_ms = clock(n.start)
        decay = math.exp(-(n.start - previous_time) / max(250, beat_ms * 2))
        pressure = [v * decay for v in pressure]
        hands = hand_weights(keys, n.lane)
        delta = n.start - last[n.lane]
        gap = n.start - released[n.lane]
        if delta < beat_ms * .4:
            cost += 4 * (1 - delta / (beat_ms * .4))
            jacks += 1
        if released[n.lane] > last[n.lane] and gap < 80:
            cost += 4 * (1 - max(0, gap) / 80)
            releases += 1
        for h in (0, 1):
            held = sum(hand_weights(keys, k)[h] for k in range(keys) if released[k] > n.start)
            cost += hands[h] * (pressure[h] * .3 + held * .6)
            pressure[h] += hands[h]
        if keys >= 8 and n.lane in (0, keys - 1) and n.hold:
            cost += min(6, (n.end - n.start) / 2000) * .2
        peak = max(peak, *pressure)
        last[n.lane], released[n.lane] = n.start, n.end
        previous_time = n.start
    return dict(cost=round(cost, 3), short_jacks=jacks, release_repeats=releases,
                peak_hand_pressure=round(peak, 3))


def assign_lanes(notes, settings=None, locked=(), beat_ms=500, progress=None,
                 cancelled=lambda: False, segment_preferences=(), timing_points=(),
                 sequence_prior=None, sequence_strength=.35):
    """Compare complete assignments against hand/LN-release costs."""
    p = parameters(settings)
    if sequence_prior is not None and p["layout_engine"] != "beam":
        raise ValueError("sequence prior requires the beam layout engine")
    notes, locked = list(notes), list(locked)
    if p["layout_engine"] == "beam":
        from .layout import search_layout
        try:
            return search_layout(notes, p, locked, beat_ms, progress=progress, cancelled=cancelled,
                                 segment_preferences=segment_preferences,
                                 timing_points=timing_points,
                                 sequence_prior=sequence_prior,
                                 sequence_strength=sequence_strength)
        except ValueError:
            if sequence_prior is not None:
                raise
            # Ordered subsets deliberately limit the search space. The exact
            # per-chord assignment remains a fallback for feasible irregular
            # intervals and local rewrites with many locked future objects.
            if progress:
                progress(45, "连续搜索未找到路径，尝试完整和弦匹配")
    best, best_cost, failure = None, float("inf"), None
    for attempt in range(6):
        if cancelled():
            raise InterruptedError("已取消定轨搜索")
        trial = p | {"seed": (p["seed"] + attempt * 104729) % 2147483648}
        try:
            output = _assign_once(notes, trial, locked, beat_ms, cancelled, segment_preferences, timing_points)
        except ValueError as error:
            failure = error
            continue
        cost = layout_metrics(output, p["keys"], beat_ms, timing_points)["cost"]
        if cost < best_cost:
            best, best_cost = output, cost
        if progress:
            progress(45 + int(38 * (attempt + 1) / 6), f"比较完整定轨方案 {attempt + 1}/6 · 检查双手负担和 LN 接续")
    if best is None:
        raise failure or ValueError("未找到可行定轨")
    return best


def _assign_once(notes, settings=None, locked=(), beat_ms=500, cancelled=lambda: False, segment_preferences=(), timing_points=()):
    """Interval coloring with Hungarian chord assignment. Never edits event times/types."""
    p = parameters(settings)
    keys, rng = p["keys"], random.Random(p["seed"])
    fixed = list(locked)
    if fixed and not validate(fixed, keys)["valid"]:
        raise ValueError("锁定区域已有冲突，无法进行局部定轨")
    groups = defaultdict(list)
    for n in notes:
        groups[n.start].append(n)
    occupied, last, load = [-1] * keys, [-100000] * keys, [0.0] * keys
    last_pitch = [None] * keys
    fixed_by_lane = [[n for n in fixed if n.lane == lane] for lane in range(keys)]
    fixed_history = sorted(fixed, key=lambda n: n.start)
    fixed_index, previous_start = 0, 0
    source_lane = {}
    result = []
    clock = BeatClock(timing_points, beat_ms)
    for start, chord in sorted(groups.items()):
        beat_ms = clock(start)
        local = row_settings(p, chord, segment_preferences)
        if cancelled():
            raise InterruptedError("已取消定轨搜索")
        load = [v * math.exp(-(start - previous_start) / max(500, beat_ms * 4)) for v in load]
        previous_start = start
        while fixed_index < len(fixed_history) and fixed_history[fixed_index].start <= start:
            f = fixed_history[fixed_index]
            last[f.lane], occupied[f.lane] = max(last[f.lane], f.start), max(occupied[f.lane], f.end)
            load[f.lane] += math.exp(-(start - f.start) / max(500, beat_ms * 4))
            fixed_index += 1
        chord = sorted(chord, key=lambda n: (n.pitch, n.id))
        if len(chord) > keys:
            raise ValueError(f"{start}ms 有 {len(chord)} 个同时起音，超过 {keys}K；保时定轨不能删音")
        costs = np.full((len(chord), keys), 1e12)
        total_load = sum(load) + 1
        for i, n in enumerate(chord):
            preferred = (n.pitch - 36) / 48 * (keys - 1) if n.source == "pitch" else (n.lane + .5) * keys / max(keys, 1) - .5
            preferred = min(keys - 1, max(0, preferred))
            for lane in range(keys):
                if occupied[lane] > start or last[lane] == start:
                    continue
                if any(n.start == f.start or (n.start < f.end and f.start < n.end) or
                       (f.start <= n.start < f.end) or (n.start <= f.start < n.end) for f in fixed_by_lane[lane]):
                    continue
                elapsed = start - max(last[lane], occupied[lane])
                hands = hand_weights(keys, lane)
                hand_load = sum(hands[h] * sum(load[j] * hand_weights(keys, j)[h] for j in range(keys)) for h in (0, 1))
                held_hand = sum(hands[h] * sum(hand_weights(keys, j)[h] for j in range(keys) if occupied[j] > start) for h in (0, 1))
                follow = abs(lane - preferred) / keys
                if last_pitch[lane] == n.pitch and n.source == "pitch":
                    follow *= .35
                source = (n.source, n.voice, n.instrument_id, n.pitch if n.source == "pitch" else n.lane)
                continuity = .3 if source in source_lane and source_lane[source] != lane else 0
                jack_window = min(220, max(90, beat_ms * .4))
                costs[i, lane] = (follow * local["pitch_follow"] * 2 + load[lane] / total_load * local["balance"] * 5 +
                    hand_load / total_load * local["balance"] * 2 + held_hand * .35 +
                    max(0, jack_window - elapsed) / jack_window * local["jack_avoid"] * 3 + continuity +
                    (2 if elapsed < local["release_gap"] else 0) + rng.random() * .65)
        rows, cols = linear_sum_assignment(costs)
        if len(rows) != len(chord) or any(costs[r, c] > 1e11 for r, c in zip(rows, cols)):
            raise ValueError(f"{start}ms 的 LN 占轨后没有足够轨道（{keys}K）；未删除或缩短任何音符")
        for i, lane in zip(rows, cols):
            n = replace(chord[i], lane=int(lane))
            result.append(n)
            occupied[lane], last[lane], last_pitch[lane] = n.end, n.start, n.pitch
            load[lane] += 1 + min(5, (n.end - n.start) / 1000)
            source_lane[(n.source, n.voice, n.instrument_id, n.pitch if n.source == "pitch" else chord[i].lane)] = int(lane)
    if invariant(notes) != invariant(result):
        raise AssertionError("定轨改变了原始物件")
    combined = sorted(fixed + result, key=lambda n: (n.start, n.lane))
    if not validate(combined, keys)["valid"]:
        raise AssertionError("定轨结果存在结构冲突")
    return combined


def choose_holds(notes, local, settings, segment_preferences=()):
    return _choose_candidate_holds(notes, local, settings, segment_preferences, solve_binary=_solve_binary)


def select_notes(candidates, settings=None, segment_preferences=()):
    return _select_candidate_notes(candidates, settings, segment_preferences,
                                   solve_binary=_solve_binary, hold_selector=choose_holds)
