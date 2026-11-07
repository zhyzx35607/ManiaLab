"""Locate hand/LN coordination demands without calling them mapping mistakes."""
from collections import Counter, defaultdict

from .arrange import hand_weights
from .timing import BeatClock


def review_chart(notes, keys, beat_ms=500, timing_points=()):
    rows = defaultdict(list)
    for note in sorted(notes, key=lambda n: (n.start, n.lane)):
        rows[note.start].append(note)
    previous = [None] * keys
    counts, events, sections = Counter(), [], defaultdict(Counter)
    last_location = {}
    capacity = [sum(hand_weights(keys, lane)[hand] for lane in range(keys)) for hand in (0, 1)]
    clock = BeatClock(timing_points, beat_ms)

    def add(kind, start, text, lane=None, hand=None, **details):
        counts[kind] += 1
        sections[start // 15000][kind] += 1
        identity = (kind, lane, hand)
        # Count all events but avoid listing every row of the same passage.
        if start - last_location.get(identity, -100000) < 750:
            return
        last_location[identity] = start
        events.append(dict(kind=kind, start_ms=start, text=text, lane=lane, hand=hand, **details))

    for start, chord in sorted(rows.items()):
        beat_ms = clock(start)
        held = [n for n in previous if n is not None and n.end > start]
        for hand in (0, 1):
            holding = sum(hand_weights(keys, n.lane)[hand] for n in held)
            pressing = sum(hand_weights(keys, n.lane)[hand] for n in chord)
            if holding >= 2 and pressing >= 1 and holding + pressing >= capacity[hand] - .5:
                add('held_hand_load', start, f"{'左' if hand == 0 else '右'}手多轨长按中接起音", hand=hand,
                    held_columns=[n.lane for n in held if hand_weights(keys, n.lane)[hand]],
                    attack_columns=[n.lane for n in chord if hand_weights(keys, n.lane)[hand]])
        for note in chord:
            old = previous[note.lane]
            if old:
                if old.hold and 0 <= start - old.end < 80:
                    gap = start - old.end
                    add('short_release', start, f"第 {note.lane + 1} 轨放键后 {gap}ms 再按", lane=note.lane, gap_ms=gap)
                gap = start - old.start
                if 0 < gap < min(120, max(60, beat_ms * .3)):
                    add('fast_repeat', start, f"第 {note.lane + 1} 轨间隔 {gap}ms 连续起音", lane=note.lane, gap_ms=gap)
            previous[note.lane] = note
    return dict(counts=dict(counts), locations=events[:80],
                sections=[dict(start_ms=i*15000, end_ms=(i+1)*15000, counts=dict(c)) for i,c in sorted(sections.items())],
                notice="这些位置需要结合目标难度试打，不表示键型错误，也不是星级评分。")
