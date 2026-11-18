"""Select evidence-backed attacks and holds before lane assignment."""
from bisect import bisect_left, bisect_right
from dataclasses import replace
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csc_matrix, vstack

from .arrange_config import parameters
from .domain import UNOBSERVED_RELEASES
from .preferences import settings_at, row_settings
from .solver import solve_binary as _solve_binary


def _select_notes(candidates, settings=None, segment_preferences=(), *, solve_binary, hold_selector):
    p = parameters(settings)
    grouped = []
    for n in sorted(candidates, key=lambda n: (n.start, n.pitch)):
        if not grouped or n.start != grouped[-1][0].start:
            grouped.append([])
        grouped[-1].append(n)
    pool, preferences, scores, chord_ranges = [], [], [], []
    muted = 0
    for chord in grouped:
        local = row_settings(p, chord, segment_preferences)

        def priority(n):
            local = settings_at(p, n, segment_preferences)
            weight = local.get(n.voice + "_weight", 1.)
            if local["focus"] == "pitched" and n.source == "pitch":
                weight *= 1.6
            if local["focus"] == "transient" and n.source == "transient":
                weight *= 1.6
            # Prefer supported phrase attacks without inventing attacks or tails.
            phrase = .35 * local["phrase_ln"] * n.support_coverage if n.phrase else 0.
            shift = abs(n.start - n.evidence_start) if n.evidence_start is not None else 0.
            # Keep uncertain off-beat detections visible to the scoring decision.
            alignment = .4 + .6 * math.exp(-.5 * (shift / 20) ** 2)
            return .15 + (max(0., n.strength) * weight + phrase) * alignment

        used_pitches = set()
        first = len(pool)
        for n in sorted(chord, key=priority, reverse=True):
            note_preferences = settings_at(p, n, segment_preferences)
            if note_preferences.get(n.voice + "_weight", 1.) <= 0:
                muted += 1
                continue
            if (n.pitch, n.source, n.voice, n.instrument_id) in used_pitches:
                continue
            pool.append(n)
            preferences.append(note_preferences)
            scores.append(priority(n))
            used_pitches.add((n.pitch, n.source, n.voice, n.instrument_id))
        chord_ranges.append((first, len(pool), min(local["chord_size"], p["keys"])))
    if not pool:
        result, local_preferences, attack_report = [], [], {"attack_selection": "global", "attack_optimal": True}
    else:
        # Score the full set so early weak sounds cannot consume later capacity.
        order = sorted(range(len(pool)), key=lambda i: (pool[i].start, i))
        time_values = [pool[i].start for i in order]
        rows, cols, values, upper = [], [], [], []
        for first, last, cap in chord_ranges:
            row = len(upper)
            upper.append(cap)
            rows.extend([row] * (last - first))
            cols.extend(range(first, last))
            values.extend([1.] * (last - first))
        for time in sorted(set(time_values)):
            a, b = bisect_right(time_values, time - 1000), bisect_right(time_values, time)
            row_start = bisect_left(time_values, time)
            cap = math.ceil(min(preferences[order[i]]["density"] for i in range(row_start, b)))
            indices = order[a:b]
            row = len(upper)
            upper.append(cap)
            rows.extend([row] * len(indices))
            cols.extend(indices)
            values.extend([1.] * len(indices))
        matrix = csc_matrix((values, (rows, cols)), shape=(len(upper), len(pool)))
        solved = solve_binary(-np.asarray(scores), integrality=np.ones(len(pool)), bounds=Bounds(0, 1),
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
    selected, report = hold_selector(result, local_preferences, p, segment_preferences)
    return selected, {"candidates": len(candidates), "selected": len(result), "omitted": len(candidates) - len(result),
                      "muted_voice_candidates": muted, **attack_report, **report}


def select_notes(candidates, settings=None, segment_preferences=()):
    return _select_notes(candidates, settings, segment_preferences,
                         solve_binary=_solve_binary, hold_selector=choose_holds)


def _choose_holds(notes, local, settings, segment_preferences, *, solve_binary):
    """Choose a target count of evidence-backed LNs, keeping attacks fixed.

    Interval constraints reserve room at future attacks and section boundaries.
    First solve the attainable count, then optimize sustain quality at target.
    """
    count = len(notes)
    wanted = int(math.floor(sum(p["ln_ratio"] for p in local) + .5))
    eligible = [i for i, n in enumerate(notes) if n.source == "pitch" and
                n.end - n.start >= local[i]["min_ln"] and local[i]["ln_ratio"] > 0
                and n.release_reason not in UNOBSERVED_RELEASES]
    target_ratio = sum(p["ln_ratio"] for p in local) / max(1, count)
    chosen, attainable, optimal = set(), 0, True
    if eligible:
        starts = sorted({n.start for n in notes} |
                        {s[k] for s in segment_preferences for k in ('start_ms', 'end_ms')})
        chord_sizes = {t: 0 for t in starts}
        limits = {t: settings_at(settings, t, segment_preferences)["max_held"] for t in starts}
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
                rows.append(row * 2)
                columns.append(col)
                values.append(1.)
            # Reserve a lane through the release gap, after the starting row.
            for row in range(a + 1, c):
                rows.append(row * 2 + 1)
                columns.append(col)
                values.append(1.)
        matrix = csc_matrix((values, (rows, columns)), shape=(len(starts) * 2, len(eligible)))
        upper = np.array([v for t in starts for v in (limits[t], settings["keys"] - chord_sizes[t])], float)
        constraint = LinearConstraint(matrix, -np.inf, upper)
        options = {"time_limit": 10., "mip_rel_gap": 0.}
        maximum = solve_binary(-np.ones(len(eligible)), integrality=np.ones(len(eligible)),
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
            best = solve_binary(-quality, integrality=np.ones(len(eligible)), bounds=Bounds(0, 1),
                        constraints=LinearConstraint(matrix2, np.r_[np.full(len(upper), -np.inf), requested],
                                                     np.r_[upper, requested]), options=options)
            if best.x is not None:
                chosen = {eligible[j] for j in np.flatnonzero(best.x > .5)}
            else:
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


def choose_holds(notes, local, settings, segment_preferences=()):
    return _choose_holds(notes, local, settings, segment_preferences, solve_binary=_solve_binary)
