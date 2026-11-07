from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left
from dataclasses import replace
import math
import random

import numpy as np
from scipy.optimize import linear_sum_assignment, milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix, vstack

from .domain import Note, invariant, validate


DEFAULTS = dict(keys=10, density=7.0, ln_ratio=.55, min_ln=240, max_held=5, release_gap=55,
                chord_size=4, pitch_follow=.55, balance=.8, jack_avoid=.8, seed=42,
                onset_threshold=.45, frame_threshold=.3, focus="balanced")
BOUNDS = dict(keys=(6, 10), density=(1, 24), ln_ratio=(0, 1), min_ln=(100, 2000), max_held=(1, 10),
              release_gap=(0, 250), chord_size=(1, 10), pitch_follow=(0, 1), balance=(0, 1),
              jack_avoid=(0, 1), seed=(0, 2147483647), onset_threshold=(.1, .9), frame_threshold=(.1, .8))
INTEGER = {"keys", "min_ln", "max_held", "release_gap", "chord_size", "seed"}


def parameters(raw=None):
    result = DEFAULTS.copy()
    for k, v in (raw or {}).items():
        if k in BOUNDS:
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError(f"参数 {k} 必须是有限数字")
            lo, hi = BOUNDS[k]
            if not lo <= v <= hi or (k in INTEGER and int(v) != v):
                raise ValueError(f"参数 {k} 必须在 {lo}–{hi} 范围内")
            result[k] = int(v) if k in INTEGER else float(v)
        elif k == "focus":
            if v not in ("balanced", "pitched", "transient"):
                raise ValueError("采音侧重无效")
            result[k] = v
        else:
            raise ValueError(f"未知参数：{k}")
    result["max_held"] = min(result["keys"], result["max_held"])
    result["chord_size"] = min(result["keys"], result["chord_size"])
    return result


def hand_weights(keys, lane):
    if keys % 2 and lane == keys // 2:
        return (.5, .5)
    return (1., 0.) if lane < keys // 2 else (0., 1.)


def layout_metrics(notes, keys, beat_ms=500):
    """Comparable ergonomic costs, not a learned or player-validated quality grade."""
    last = [-100000] * keys
    released = [-100000] * keys
    pressure = [0., 0.]
    previous_time = 0
    cost, jacks, releases, peak = 0., 0, 0, 0.
    for n in sorted(notes, key=lambda n: (n.start, n.lane)):
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


def assign_lanes(notes, settings=None, locked=(), beat_ms=500, progress=None, cancelled=lambda: False):
    """Compare complete assignments against hand/LN-release costs."""
    p = parameters(settings)
    notes, locked = list(notes), list(locked)
    best, best_cost, failure = None, float("inf"), None
    for attempt in range(6):
        if cancelled():
            raise InterruptedError("已取消定轨搜索")
        trial = p | {"seed": (p["seed"] + attempt * 104729) % 2147483648}
        try:
            output = _assign_once(notes, trial, locked, beat_ms, cancelled)
        except ValueError as error:
            failure = error
            continue
        cost = layout_metrics(output, p["keys"], beat_ms)["cost"]
        if cost < best_cost:
            best, best_cost = output, cost
        if progress:
            progress(45 + int(38 * (attempt + 1) / 6), f"比较完整定轨方案 {attempt + 1}/6 · 检查双手负担和 LN 接续")
    if best is None:
        raise failure or ValueError("未找到可行定轨")
    return best


def _assign_once(notes, settings=None, locked=(), beat_ms=500, cancelled=lambda: False):
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
    for start, chord in sorted(groups.items()):
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
                source = (n.source, n.pitch if n.source == "pitch" else n.lane)
                continuity = .3 if source in source_lane and source_lane[source] != lane else 0
                jack_window = min(220, max(90, beat_ms * .4))
                costs[i, lane] = (follow * p["pitch_follow"] * 2 + load[lane] / total_load * p["balance"] * 5 +
                    hand_load / total_load * p["balance"] * 2 + held_hand * .35 +
                    max(0, jack_window - elapsed) / jack_window * p["jack_avoid"] * 3 + continuity +
                    (2 if elapsed < p["release_gap"] else 0) + rng.random() * .65)
        rows, cols = linear_sum_assignment(costs)
        if len(rows) != len(chord) or any(costs[r, c] > 1e11 for r, c in zip(rows, cols)):
            raise ValueError(f"{start}ms 的 LN 占轨后没有足够轨道（{keys}K）；未删除或缩短任何音符")
        for i, lane in zip(rows, cols):
            n = replace(chord[i], lane=int(lane))
            result.append(n)
            occupied[lane], last[lane], last_pitch[lane] = n.end, n.start, n.pitch
            load[lane] += 1 + min(5, (n.end - n.start) / 1000)
            source_lane[(n.source, n.pitch if n.source == "pitch" else chord[i].lane)] = int(lane)
    if invariant(notes) != invariant(result):
        raise AssertionError("定轨改变了原始物件")
    combined = sorted(fixed + result, key=lambda n: (n.start, n.lane))
    if not validate(combined, keys)["valid"]:
        raise AssertionError("定轨结果存在结构冲突")
    return combined


def select_notes(candidates, settings=None, segment_preferences=()):
    p = parameters(settings)
    grouped = []
    for n in sorted(candidates, key=lambda n: (n.start, n.pitch)):
        if not grouped or n.start - grouped[-1][0].start > 18:
            grouped.append([])
        grouped[-1].append(n)
    result, recent, skipped = [], [], 0
    local_preferences = []
    for chord in grouped:
        start = min(n.start for n in chord)
        local = p.copy()
        for segment in segment_preferences:
            if segment["start_ms"] <= start < segment["end_ms"]:
                local.update(segment["preferences"])
        recent = [t for t in recent if t > start - 1000]
        budget = max(0, min(local["chord_size"], p["keys"], math.ceil(local["density"]) - len(recent)))
        def priority(n):
            weight = 1.0
            if local["focus"] == "pitched" and n.source == "pitch":
                weight = 1.6
            if local["focus"] == "transient" and n.source == "transient":
                weight = 1.6
            return n.strength * weight
        used_pitches = set()
        for n in sorted(chord, key=priority, reverse=True):
            if budget <= 0 or (n.pitch, n.source) in used_pitches:
                skipped += 1
                continue
            result.append(n)
            local_preferences.append(local)
            recent.append(n.start)
            used_pitches.add((n.pitch, n.source))
            budget -= 1
    selected, report = choose_holds(result, local_preferences, p)
    return selected, {"candidates": len(candidates), "selected": len(result), "omitted": skipped, **report}


def choose_holds(notes, local, settings):
    """Choose a target count of evidence-backed LNs, keeping the attack set fixed.

    Binary interval constraints reserve room at *every future attack*, including
    the release gap. They also enforce the simultaneous-hold limit. First solve
    for the achievable count, then optimize sustain quality at the target count.
    """
    count = len(notes)
    wanted = int(math.floor(sum(p["ln_ratio"] for p in local) + .5))
    eligible = [i for i, n in enumerate(notes) if n.source == "pitch" and
                n.end - n.start >= local[i]["min_ln"] and local[i]["ln_ratio"] > 0]
    target_ratio = sum(p["ln_ratio"] for p in local) / max(1, count)
    chosen, attainable, optimal = set(), 0, True
    if eligible:
        starts = sorted(set(n.start for n in notes))
        chord_sizes = {t: 0 for t in starts}
        limits = {t: settings["max_held"] for t in starts}
        for n, pref in zip(notes, local):
            chord_sizes[n.start] += 1
            limits[n.start] = min(limits[n.start], pref["max_held"])
        rows, columns, values = [], [], []
        for col, i in enumerate(eligible):
            n, pref = notes[i], local[i]
            a = bisect_left(starts, n.start)
            b = bisect_left(starts, n.end)
            c = bisect_left(starts, n.end + pref["release_gap"])
            for row in range(a, b):
                rows.append(row * 2); columns.append(col); values.append(1.)
            # An LN consumes an available lane after its own starting row.
            for row in range(a + 1, c):
                rows.append(row * 2 + 1); columns.append(col); values.append(1.)
        matrix = csc_matrix((values, (rows, columns)), shape=(len(starts) * 2, len(eligible)))
        upper = np.array([v for t in starts for v in (limits[t], settings["keys"] - chord_sizes[t])], float)
        constraint = LinearConstraint(matrix, -np.inf, upper)
        options = {"time_limit": 10., "mip_rel_gap": 0.}
        maximum = milp(-np.ones(len(eligible)), integrality=np.ones(len(eligible)),
                       bounds=Bounds(0, 1), constraints=constraint, options=options)
        if maximum.x is None:
            raise ValueError("LN 约束求解失败，请降低密度或和弦上限")
        pool = np.flatnonzero(maximum.x > .5)
        attainable, optimal = len(pool), maximum.status == 0
        requested = min(wanted, attainable)
        if requested:
            quality = np.array([notes[i].strength + math.log1p((notes[i].end - notes[i].start) / 500) +
                                local[i]["ln_ratio"] * 2 for i in eligible])
            matrix2 = vstack([matrix, np.ones((1, len(eligible)))], format="csc")
            best = milp(-quality, integrality=np.ones(len(eligible)), bounds=Bounds(0, 1),
                        constraints=LinearConstraint(matrix2, np.r_[np.full(len(upper), -np.inf), requested],
                                                     np.r_[upper, requested]), options=options)
            if best.x is not None:
                chosen = {eligible[j] for j in np.flatnonzero(best.x > .5)}
            else:
                # Any subset of a feasible interval solution stays feasible.
                chosen = {eligible[j] for j in sorted(pool, key=lambda j: quality[j], reverse=True)[:requested]}
    result = [replace(n, end=n.end if i in chosen else n.start) for i, n in enumerate(notes)]
    actual = len(chosen) / max(1, count)
    message = f"LN 目标 {target_ratio:.0%}，实际 {actual:.1%}（{len(chosen)}/{count}）。"
    if len(chosen) < wanted:
        message += "持续音证据或占轨空间不足，保留起音，未虚构长条尾端。"
    return result, dict(target_ln_ratio=round(target_ratio, 4), target_ln_count=wanted,
                        actual_ln_ratio=round(actual, 4), eligible_ln_count=len(eligible),
                        feasible_ln_count=attainable, feasible_ln_ratio=round(attainable / max(1, count), 4),
                        maximum_proven=optimal, message=message)
