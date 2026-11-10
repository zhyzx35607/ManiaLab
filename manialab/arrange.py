from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left, bisect_right
from dataclasses import replace
import math
import random
import threading
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment, milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix, vstack

from .domain import UNOBSERVED_RELEASES, Note, invariant, validate
from .timing import BeatClock


_solver_lock = threading.Lock()


def _solve_binary(*args, **kwargs):
    # HiGHS owns a process-wide scheduler. Flask jobs run on worker threads;
    # one solver thread avoids reusing a native thread pool across those jobs.
    options = dict(kwargs.pop("options", {}), threads=1, parallel=False)
    with _solver_lock, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*", category=RuntimeWarning)
        return milp(*args, options=options, **kwargs)


DEFAULTS = dict(keys=10, density=7.0, ln_ratio=.55, min_ln=240, max_held=5, release_gap=55,
                chord_size=4, pitch_follow=.55, balance=.8, jack_avoid=.8, seed=42,
                onset_threshold=.45, frame_threshold=.3, focus="balanced", phrase_ln=.65, layout_engine="beam", pattern_prior=.35,
                drums_weight=1., bass_weight=1., vocals_weight=1., other_weight=1., guitar_weight=1., piano_weight=1.,
                ln_coordination=.35, ln_release=.65, ln_inverse=0.)
BOUNDS = dict(keys=(6, 10), density=(1, 24), ln_ratio=(0, 1), min_ln=(100, 2000), max_held=(1, 10),
              release_gap=(0, 250), chord_size=(1, 10), pitch_follow=(0, 1), balance=(0, 1),
              jack_avoid=(0, 1), seed=(0, 2147483647), onset_threshold=(.1, .9), frame_threshold=(.1, .8), phrase_ln=(0, 1), pattern_prior=(0,1),
              drums_weight=(0,2), bass_weight=(0,2), vocals_weight=(0,2), other_weight=(0,2), guitar_weight=(0,2), piano_weight=(0,2),
              ln_coordination=(0,1), ln_release=(0,1), ln_inverse=(0,1))
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
        elif k == "layout_engine":
            if v not in ("beam", "classic"):
                raise ValueError("定轨引擎无效")
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


def settings_at(settings, time, segments):
    local = settings.copy()
    for segment in segments:
        if segment["start_ms"] <= time < segment["end_ms"]:
            local.update(segment["preferences"])
    return local


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


def assign_lanes(notes, settings=None, locked=(), beat_ms=500, progress=None, cancelled=lambda: False, segment_preferences=(), timing_points=()):
    """Compare complete assignments against hand/LN-release costs."""
    p = parameters(settings)
    notes, locked = list(notes), list(locked)
    if p["layout_engine"] == "beam":
        from .layout import search_layout
        try:
            return search_layout(notes, p, locked, beat_ms, progress=progress, cancelled=cancelled,
                                 segment_preferences=segment_preferences, timing_points=timing_points)
        except ValueError:
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
        local = settings_at(p, start, segment_preferences)
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


def select_notes(candidates, settings=None, segment_preferences=()):
    p = parameters(settings)
    grouped = []
    for n in sorted(candidates, key=lambda n: (n.start, n.pitch)):
        if not grouped or n.start != grouped[-1][0].start:
            grouped.append([])
        grouped[-1].append(n)
    pool, preferences, scores, chord_ranges = [], [], [], []
    muted = 0
    for chord in grouped:
        start = min(n.start for n in chord)
        local = settings_at(p, start, segment_preferences)
        def priority(n):
            weight = local.get(n.voice + "_weight", 1.)
            if local["focus"] == "pitched" and n.source == "pitch":
                weight *= 1.6
            if local["focus"] == "transient" and n.source == "transient":
                weight *= 1.6
            # A supported vocal phrase is one optional sustained gesture.
            # Give its existing attack a small preference; never create an
            # attack or extend a tail here just to meet an LN percentage.
            phrase = .35 * local["phrase_ln"] * n.support_coverage if n.phrase else 0.
            shift = abs(n.start - n.evidence_start) if n.evidence_start is not None else 0.
            # Quantization does not turn an uncertain off-beat detection into
            # strong evidence. Prefer closer attacks when the density budget
            # forces a choice, retaining the raw timestamp for inspection.
            alignment = .4 + .6 * math.exp(-.5 * (shift / 20) ** 2)
            return .15 + (max(0., n.strength) * weight + phrase) * alignment
        used_pitches = set()
        first = len(pool)
        for n in sorted(chord, key=priority, reverse=True):
            if local.get(n.voice + "_weight", 1.) <= 0:
                muted += 1
                continue
            if (n.pitch, n.source, n.voice, n.instrument_id) in used_pitches:
                continue
            pool.append(n)
            preferences.append(local)
            scores.append(priority(n))
            used_pitches.add((n.pitch, n.source, n.voice, n.instrument_id))
        chord_ranges.append((first, len(pool), min(local["chord_size"], p["keys"])))
    if not pool:
        result, local_preferences, attack_report = [], [], {"attack_selection": "global", "attack_optimal": True}
    else:
        # The old chronological budget spent the next second's allowance on
        # early weak sounds before seeing a stronger musical attack. Optimize
        # the complete candidate set under the same rolling density/chord caps.
        order = sorted(range(len(pool)), key=lambda i: (pool[i].start, i))
        time_values = [pool[i].start for i in order]
        rows, cols, values, upper = [], [], [], []
        for first, last, cap in chord_ranges:
            row = len(upper); upper.append(cap)
            rows.extend([row] * (last - first)); cols.extend(range(first, last)); values.extend([1.] * (last - first))
        for time in sorted(set(time_values)):
            a, b = bisect_right(time_values, time - 1000), bisect_right(time_values, time)
            at = order[b - 1]
            cap = math.ceil(preferences[at]["density"])
            indices = order[a:b]
            row = len(upper); upper.append(cap)
            rows.extend([row] * len(indices)); cols.extend(indices); values.extend([1.] * len(indices))
        matrix = csc_matrix((values, (rows, cols)), shape=(len(upper), len(pool)))
        solved = _solve_binary(-np.asarray(scores), integrality=np.ones(len(pool)), bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix, -np.inf, np.asarray(upper)), options={"time_limit": 10., "mip_rel_gap": 0.})
        if solved.x is None:
            raise ValueError("采音选择未找到满足密度上限的方案；请降低候选数量后重试")
        chosen = {i for i in np.flatnonzero(solved.x > .5)}
        if np.any(matrix @ np.array([i in chosen for i in range(len(pool))], float) > np.asarray(upper) + 1e-6):
            raise ValueError("采音选择结果未通过密度与和弦约束检查")
        result = [pool[i] for i in order if i in chosen]
        local_preferences = [preferences[i] for i in order if i in chosen]
        attack_report = dict(attack_selection="global", attack_optimal=solved.status == 0,
                             selected_voices={v: sum(n.voice == v for n in result) for v in sorted({n.voice for n in pool})},
                             phrase_attacks=sum(n.phrase for n in result),
                             selected_large_attack_shifts=sum(abs(n.start-n.evidence_start)>30 for n in result if n.evidence_start is not None))
    selected, report = choose_holds(result, local_preferences, p)
    return selected, {"candidates": len(candidates), "selected": len(result), "omitted": len(candidates) - len(result),
                      "muted_voice_candidates": muted, **attack_report, **report}


def choose_holds(notes, local, settings):
    """Choose a target count of evidence-backed LNs, keeping the attack set fixed.

    Binary interval constraints reserve room at *every future attack*, including
    the release gap. They also enforce the simultaneous-hold limit. First solve
    for the achievable count, then optimize sustain quality at the target count.
    """
    count = len(notes)
    wanted = int(math.floor(sum(p["ln_ratio"] for p in local) + .5))
    eligible = [i for i, n in enumerate(notes) if n.source == "pitch" and
                n.end - n.start >= local[i]["min_ln"] and local[i]["ln_ratio"] > 0
                and n.release_reason not in UNOBSERVED_RELEASES]
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
        maximum = _solve_binary(-np.ones(len(eligible)), integrality=np.ones(len(eligible)),
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
            best = _solve_binary(-quality, integrality=np.ones(len(eligible)), bounds=Bounds(0, 1),
                        constraints=LinearConstraint(matrix2, np.r_[np.full(len(upper), -np.inf), requested],
                                                     np.r_[upper, requested]), options=options)
            if best.x is not None:
                chosen = {eligible[j] for j in np.flatnonzero(best.x > .5)}
            else:
                # Any subset of a feasible interval solution stays feasible.
                chosen = {eligible[j] for j in sorted(pool, key=lambda j: quality[j], reverse=True)[:requested]}
    result = [replace(n, end=n.end if i in chosen else n.start,
                      phrase=n.phrase and i in chosen) for i, n in enumerate(notes)]
    actual = len(chosen) / max(1, count)
    message = f"LN 目标 {target_ratio:.0%}，实际 {actual:.1%}（{len(chosen)}/{count}）。"
    if len(chosen) < wanted:
        message += "持续音证据或占轨空间不足，保留起音，未虚构长条尾端。"
    return result, dict(target_ln_ratio=round(target_ratio, 4), target_ln_count=wanted,
                        actual_ln_ratio=round(actual, 4), eligible_ln_count=len(eligible),
                        feasible_ln_count=attainable, feasible_ln_ratio=round(attainable / max(1, count), 4),
                        maximum_proven=optimal, message=message)
