"""Research-only source gaps from frozen piano activity, not release labels.

Never promote an unknown release or infer silence from missing event candidates.
The application does not call this module.
"""
from __future__ import annotations

from bisect import bisect_left
from statistics import median
from unittest.mock import patch

import numpy as np

from manialab.domain import UNOBSERVED_RELEASES
from manialab.expression_articulation import _overlap_pairs
import manialab.expression_articulation as articulation


def inactive_intervals(frame, onset):
    """Consecutive low-activity samples across all 88 pitches, on a 100 Hz clock.

    End at the final sampled frame, never extrapolate a low frame into future
    audio. Thresholds match the existing piano decoder; they are not silence
    probabilities. Only intervals at least 80 ms long may inform grouping.
    """
    frame, onset = np.asarray(frame), np.asarray(onset)
    if (frame.ndim != 2 or frame.shape[1] != 88 or onset.shape != frame.shape
            or not np.isfinite(frame).all() or not np.isfinite(onset).all()
            or np.any((frame < 0) | (frame > 1)) or np.any((onset < 0) | (onset > 1))):
        raise ValueError('Expected finite 100 Hz piano probabilities with 88 pitches')
    low = (frame.max(axis=1) <= .1) & (onset.max(axis=1) <= .3)
    changes = np.diff(np.r_[False, low, False].astype(np.int8))
    return [(int(a)*10, (int(b)-1)*10) for a, b in
            zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
            if b-a >= 9]


def gap_support(intervals, begin, end):
    """Require 80 ms inside the candidate gap, rather than a nearby low frame."""
    for left, right in intervals:
        if left >= end:
            break
        a, b = max(begin, left), min(end, right)
        if b-a >= 80:
            return (a, b)
    return None


def contextual_runs(notes, full_source, intervals):
    """Same source grouping with an additional, explicitly model-based gap.

    Earlier raw tails and all weak candidates veto a proposed empty interval.
    Only already-articulated onset groups may split; existing sustained groups
    remain unchanged. Returns runs, the unchanged pause rule and gap evidence.
    """
    groups = []
    for note in sorted(notes, key=lambda n: (n.start, n.pitch, n.id)):
        if not groups or note.start-groups[-1][0].start > 30:
            groups.append([note])
        else:
            groups[-1].append(note)
    gaps = [b[0].start-a[0].start for a, b in zip(groups, groups[1:])]
    pause = max(500., min(1600., median(gaps)*2.5)) if gaps else 500.
    original = []
    for group in groups:
        if not original or group[0].start-original[-1][-1][0].start >= pause:
            original.append([group])
        else:
            original[-1].append(group)
    identities = {(n.voice, n.instrument_id) for n in notes}
    if len(identities) > 1:
        raise ValueError('An activity context may only describe one model source')
    raw = sorted((n for n in full_source if (n.voice, n.instrument_id) in identities),
                 key=lambda n: n.start)
    raw_starts = [n.start for n in raw]
    raw_ends = np.maximum.accumulate([n.end for n in raw]) if raw else []
    runs, additions = [], []
    latest_end = None
    for source_run in original:
        articulated = len(source_run) >= 3 and _overlap_pairs(source_run)*2 >= len(source_run)-1
        complete = True
        for index, group in enumerate(source_run):
            start = group[0].start
            existing_gap = (articulated and latest_end is not None and complete
                            and start-latest_end >= 80)
            support = None
            if index > 0 and articulated and not existing_gap:
                prior = bisect_left(raw_starts, start)-1
                left = int(raw_ends[prior]) if prior >= 0 else start
                support = gap_support(intervals, left, start)
            if index == 0 or existing_gap or support is not None:
                if support is not None:
                    additions.append(dict(before_attack_ms=start, candidate_gap_start_ms=left,
                        inactive_start_ms=support[0], inactive_end_ms=support[1],
                        previous_attack_ms=runs[-1][-1][0].start))
                runs.append([group])
            else:
                runs[-1].append(group)
            latest_end = max(latest_end if latest_end is not None else start, max(n.end for n in group))
            complete = complete and all(n.source == 'pitch' and n.end > n.start
                and n.release_reason not in UNOBSERVED_RELEASES for n in group)
    return runs, pause, additions


def activity_policy(notes, full_source, intervals, *, observed_until_ms):
    """Scoped comparison adapter; other sources retain the production policy."""
    baseline = articulation.articulation_policy(notes, observed_until_ms=observed_until_ms)
    original_runs = articulation.attack_runs
    gaps = []
    def with_context(local):
        local = list(local)
        if {(n.voice, n.instrument_id) for n in local} != {('piano', '')}:
            return original_runs(local)
        runs, pause, additions = contextual_runs(local, full_source, intervals)
        gaps.extend(additions)
        return runs, pause
    with patch.object(articulation, 'attack_runs', with_context):
        result = articulation.articulation_policy(notes, observed_until_ms=observed_until_ms)
    for note in notes:
        if note.voice != 'piano' or note.instrument_id:
            continue
        rule = result[note.id]
        assert baseline[note.id]['tap'] or not rule['tap'], 'Never add short-key expressions'
        after = next((g for g in gaps if g['previous_attack_ms'] == rule['run_end_attack']), None)
        if after and rule['continuation'] == 'observed_interval_gap':
            rule['continuation'] = 'model_inactive_gap'
        rule['source_activity_context'] = dict(baseline_tap=baseline[note.id]['tap'],
            changed=baseline[note.id]['tap'] != rule['tap'], next_gap=after,
            evidence='piano_model_low_activity', physical_silence_verified=False)
    return result, gaps
