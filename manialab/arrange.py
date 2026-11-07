from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
import random

import numpy as np
from scipy.optimize import linear_sum_assignment

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


def assign_lanes(notes, settings=None, locked=()):
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
    result = []
    for start, chord in sorted(groups.items()):
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
                hand = 0 if lane < keys / 2 else 1
                hand_load = sum(load[j] for j in range(keys) if (0 if j < keys / 2 else 1) == hand)
                held_hand = sum(occupied[j] > start for j in range(keys) if (0 if j < keys / 2 else 1) == hand)
                follow = abs(lane - preferred) / keys
                if last_pitch[lane] == n.pitch and n.source == "pitch":
                    follow *= .35
                costs[i, lane] = (follow * p["pitch_follow"] * 2 + load[lane] / total_load * p["balance"] * 5 +
                    hand_load / total_load * p["balance"] * 2 + held_hand * .35 +
                    max(0, 180 - elapsed) / 180 * p["jack_avoid"] * 3 +
                    (2 if elapsed < p["release_gap"] else 0) + rng.random() * .28)
        rows, cols = linear_sum_assignment(costs)
        if len(rows) != len(chord) or any(costs[r, c] > 1e11 for r, c in zip(rows, cols)):
            raise ValueError(f"{start}ms 的 LN 占轨后没有足够轨道（{keys}K）；未删除或缩短任何音符")
        for i, lane in zip(rows, cols):
            n = replace(chord[i], lane=int(lane))
            result.append(n)
            occupied[lane], last[lane], last_pitch[lane] = n.end, n.start, n.pitch
            load[lane] += 1 + min(5, (n.end - n.start) / 1000)
    if invariant(notes) != invariant(result):
        raise AssertionError("定轨改变了原始物件")
    combined = sorted(fixed + result, key=lambda n: (n.start, n.lane))
    if not validate(combined, keys)["valid"]:
        raise AssertionError("定轨结果存在结构冲突")
    return combined


def select_notes(candidates, settings=None, segment_preferences=()):
    p = parameters(settings)
    grouped = defaultdict(list)
    # Group nearby attacks for selection only; preserve their original integer timestamps.
    for n in candidates:
        grouped[n.start // 35].append(n)
    result, held, recent, skipped = [], [], [], 0
    for _, chord in sorted(grouped.items()):
        start = min(n.start for n in chord)
        local = p.copy()
        for segment in segment_preferences:
            if segment["start_ms"] <= start < segment["end_ms"]:
                local.update(segment["preferences"])
        held = [t for t in held if t + local["release_gap"] > start]
        recent = [t for t in recent if t > start - 1000]
        budget = max(0, min(local["chord_size"], p["keys"] - len(held), math.ceil(local["density"]) - len(recent)))
        def priority(n):
            weight = 1.0
            if local["focus"] == "pitched" and n.source == "pitch":
                weight = 1.6
            if local["focus"] == "transient" and n.source == "transient":
                weight = 1.6
            return n.strength * weight + min(3, (n.end - n.start) / 1000) * local["ln_ratio"] * .12
        used_pitches = set()
        for n in sorted(chord, key=priority, reverse=True):
            if budget <= 0 or (n.pitch, n.source) in used_pitches:
                skipped += 1
                continue
            # LN preference selects supported sustains, never lengthens a detected event.
            duration = n.end - n.start
            threshold = local["min_ln"] + (1 - local["ln_ratio"]) ** 2 * 1200
            hold = n.source == "pitch" and local["ln_ratio"] > 0 and duration >= threshold and len(held) < local["max_held"]
            chosen = replace(n, end=n.end if hold else n.start)
            result.append(chosen)
            if hold:
                held.append(n.end)
            recent.append(n.start)
            used_pitches.add((n.pitch, n.source))
            budget -= 1
    return result, {"candidates": len(candidates), "selected": len(result), "omitted": skipped,
                    "message": "LN 末端来自持续音候选；偏好值不保证精确 LN 比例。"}
