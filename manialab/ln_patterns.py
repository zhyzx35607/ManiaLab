"""Describe LN interactions and score lane choices without changing sound times.

Torosu's held-column and inverse-gap analysis informed these descriptors. They
are measurable pattern demands, not an osu! difficulty rating or quality grade.
Every release is visited, including releases without a simultaneous note head.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from bisect import bisect_left, bisect_right

import numpy as np

from .timing import BeatClock


def hand_matrix(keys):
    from .layout import fingers
    return np.array([[.5, .5] if h == 2 else [1., 0.] if h == 0 else [0., 1.]
                     for h, _ in fingers(keys)])


def ln_row_costs(choices, start, ends, previous_ends, previous_last, hands, beat_ms, prefs):
    """Cost the heads AND future releases of these unmodified intervals.

    0.5 coordination is neutral; lower values separate taps from holding hands,
    higher values prefer playing a tap with a hand already holding an LN.
    Release control discourages nearby asynchronous actions within one hand.
    Inverse prefers actual positive release/repress gaps, never fabricates tails.
    """
    count, size = choices.shape
    costs = np.zeros(count)
    coordination = prefs.get('ln_coordination', .35)
    release_control = prefs.get('ln_release', .65)
    inverse = prefs.get('ln_inverse', 0.)
    selected_hands = hands[choices]
    active = previous_ends > start
    holding = hands[active].sum(axis=0)
    taps = ends == start
    if np.any(taps):
        paired = (selected_hands[:, taps] * (holding > 0)).sum(axis=(1, 2))
        costs += paired * (0.5 - coordination) * 3
    was_hold = previous_ends > previous_last
    gaps = start - previous_ends[choices]
    inverse_window = min(150., beat_ms * .3)
    usable_gap = (gaps >= max(1, prefs['release_gap'])) & (gaps <= inverse_window)
    costs -= (usable_gap * was_hold[choices] * (ends > start)).sum(axis=1) * inverse * 3

    if not release_control:
        return costs
    window = max(40., min(150., beat_ms * .25))
    # Heads near the release of another finger of the same hand. A release at
    # the exact same timestamp is synchronous, so it is reported, not penalized.
    separation = abs(previous_ends - start)
    nearby = was_hold & (separation > 1) & (separation < window)
    weights = np.maximum(0, 1 - separation / window) * nearby
    release_load = np.zeros(2)
    for time in np.unique(previous_ends[nearby]):
        group = nearby & (previous_ends == time)
        release_load += weights[group].max() * hands[group].max(axis=0)
    # One same-hand chord release is one timing action, not five independent
    # asynchronous actions. Finger occupancy is scored separately by layout.
    pressed_hands = np.minimum(1, selected_hands.sum(axis=1))
    costs += (pressed_hands * release_load).sum(axis=1) * release_control * 1.4

    # Compare a new LN's release to already placed active holds. This sees
    # releases even if no future head happens at that time.
    new_releases = np.unique(ends[ends > start])
    for i, end in enumerate(new_releases):
        released_hands = np.minimum(1, selected_hands[:, ends == end].sum(axis=1))
        delta = abs(previous_ends - end)
        asynchronous = active & (delta > 1)
        close = np.maximum(0, 1 - delta / window) * asynchronous
        longer = (previous_ends > end + 1) & active
        demand = np.minimum(1, longer @ hands) * .22
        for time in np.unique(previous_ends[close > 0]):
            group = (close > 0) & (previous_ends == time)
            demand += close[group].max() * hands[group].max(axis=0)
        costs += (released_hands * demand).sum(axis=1) * release_control
        for earlier in new_releases[:i]:
            if end - earlier <= 1:
                continue
            earlier_hands = np.minimum(1, selected_hands[:, ends == earlier].sum(axis=1))
            same_hand = (released_hands * earlier_hands).sum(axis=1)
            close_release = max(0., 1 - (end - earlier) / window)
            costs += same_hand * (.22 + close_release) * release_control
    return costs


def analyze_ln_patterns(notes, keys, beat_ms=500., timing_points=()):
    """Complete endpoint sweep, with bounded location samples for the UI."""
    notes = sorted(notes, key=lambda n: (n.start, n.lane, n.end))
    hands = hand_matrix(keys)
    clock = BeatClock(timing_points, beat_ms)
    heads, tails = defaultdict(list), defaultdict(list)
    for n in notes:
        heads[n.start].append(n)
        if n.hold:
            tails[n.end].append(n)
    release_notes = sorted((n for n in notes if n.hold),key=lambda n:n.end)
    release_times = [n.end for n in release_notes]
    counts, sections = Counter(), defaultdict(Counter)
    first_time = min((n.start for n in notes), default=0)
    last_time = max((n.end for n in notes), default=0)
    span = max(1, last_time - first_time)
    # One sample per category in each of 20 song-wide buckets. A long intro
    # must not consume all locations before the difficult ending is reached.
    event_samples = {}
    section_hold_ms, section_peak = Counter(), Counter()
    active, previous = {}, {}
    held_ms = np.zeros(keys)
    max_held_hand = np.zeros(2)
    releases_in_second = deque()
    peak_releases = 0
    previous_time = notes[0].start if notes else 0

    def add(kind, at, text, columns=(), amount=1):
        counts[kind] += amount
        sections[at // 15000][kind] += amount
        bucket = min(19, int((at - first_time) * 20 / span))
        event_samples.setdefault((kind, bucket), dict(kind=kind, start_ms=at, text=text, columns=list(columns)))

    for at in sorted(heads.keys() | tails.keys()):
        for lane in active:
            held_ms[lane] += at - previous_time
        position = previous_time
        while position < at:
            index = position // 15000
            boundary = min(at, (index + 1) * 15000)
            section_hold_ms[index] += (boundary - position) * len(active)
            section_peak[index] = max(section_peak[index], len(active))
            position = boundary
        previous_time = at
        row, released = heads[at], tails[at]
        if row:
            window=max(40.,min(150.,clock(at)*.25))
            nearby=release_notes[bisect_left(release_times,at-window):bisect_right(release_times,at+window)]
            nearby=[n for n in nearby if n.start<at and 1<abs(n.end-at)<window]
            for h in (0,1):
                if any(hands[n.lane,h]>0 for n in row) and any(hands[n.lane,h]>0 for n in nearby):
                    add('near_release_press_hands',at,f'{"左" if h==0 else "右"}手放手附近接不同步起音',
                        [n.lane for n in nearby if hands[n.lane,h]>0])
        continuing = [n for n in active.values() if n.end > at]
        if released:
            add('release_rows', at, f'{len(released)} 轨放手', [n.lane for n in released])
            counts['releases'] += len(released)
            sections[at // 15000]['releases'] += len(released)
            if not row:
                add('release_only_rows', at, '只有放手、没有新起音', [n.lane for n in released])
            while releases_in_second and at - releases_in_second[0] >= 1000:
                releases_in_second.popleft()
            releases_in_second.extend([at] * len(released))
            peak_releases = max(peak_releases, len(releases_in_second))
            for h in (0, 1):
                r = [n for n in released if hands[n.lane, h] > 0]
                kept = [n for n in continuing if hands[n.lane, h] > 0]
                if r and kept:
                    add('split_release_hands', at, f'{"左" if h == 0 else "右"}手部分放开、其他轨继续长按',
                        [n.lane for n in r + kept])
                press = [n for n in row if hands[n.lane, h] > 0]
                if r and press:
                    add('release_press_hands', at, f'{"左" if h == 0 else "右"}手同时放手与起音',
                        [n.lane for n in r + press])
        for n in released:
            active.pop(n.lane, None)
        held_hands = hands[list(active)].sum(axis=0) if active else np.zeros(2)
        for n in row:
            old = previous.get(n.lane)
            if old and old.hold:
                gap = at - old.end
                if 0 < gap <= min(150., clock(at) * .3):
                    kind = 'inverse_heads' if n.hold else 'reverse_shield_heads'
                    add(kind, at, f'第 {n.lane + 1} 轨放手后 {gap}ms 接{"LN" if n.hold else "短键"}', [n.lane])
                if 0 <= gap < 80:
                    add('tight_repress_heads', at, f'第 {n.lane + 1} 轨短间隔接续 {gap}ms', [n.lane])
            elif old and n.hold and 0 < at - old.start <= min(150., clock(at) * .3):
                add('shield_heads',at,f'第 {n.lane + 1} 轨短键接 LN',[n.lane])
            if not n.hold and active:
                counts['hybrid_taps'] += 1
                sections[at // 15000]['hybrid_taps'] += 1
                if np.any((hands[n.lane] > 0) & (held_hands > 0)):
                    add('same_hand_hybrid_taps', at, '同手长按中打短键', [n.lane])
            previous[n.lane] = n
        # Do not count same-row holds as pre-existing hybrid; they begin
        # together, which has different coordination from a tap under an LN.
        for n in row:
            if n.hold:
                active[n.lane] = n
        held = hands[list(active)].sum(axis=0) if active else np.zeros(2)
        max_held_hand = np.maximum(max_held_hand, held)

    holds = sum(n.hold for n in notes)
    taps = len(notes) - holds
    density_sections = []
    for i in sorted(sections.keys() | section_hold_ms.keys()):
        start, end = max(first_time, i*15000), min(last_time, (i+1)*15000)
        density_sections.append(dict(start_ms=start, end_ms=end, counts=dict(sections[i]),
            occupied_key_fraction=round(section_hold_ms[i] / (max(1, end-start)*keys),4),
            peak_held=section_peak[i]))
    return dict(revision=4, counts=dict(counts),
        ln_ratio=round(holds / max(1, len(notes)), 4),
        occupied_key_fraction=round(float(held_ms.sum()) / (span * keys), 4),
        held_seconds_by_lane=[round(float(t) / 1000, 3) for t in held_ms],
        peak_held_by_hand=max_held_hand.tolist(),
        hybrid_tap_fraction=round(counts['hybrid_taps'] / max(1, taps), 4),
        same_hand_hybrid_fraction=round(counts['same_hand_hybrid_taps'] / max(1, counts['hybrid_taps']), 4),
        inverse_head_fraction=round(counts['inverse_heads'] / max(1, len(notes)), 4),
        peak_releases_per_second=peak_releases,
        locations=sorted(event_samples.values(),key=lambda e:(e['start_ms'],e['kind'])),
        sections=density_sections,
        notice='描述实际键型，不是质量分或星级。7K/9K 中键按左右手各半估计；同手事件包含中键的可能配合。反键计 LN→LN 正间隙≤min(0.3拍,150ms)，尾接短键另计；此门槛是本工具的近邻统计定义。')
