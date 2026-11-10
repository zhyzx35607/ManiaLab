"""Research-only continuity costs for selecting existing rhythmic groups.

These links express an arrangement preference. A source/class label is not
verified instrument identity, and adjacency does not establish continuous sound.
"""
from collections import defaultdict
import math


def validate_weight(weight, beat_ms):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError('节奏组连贯权重必须在0至1之间')
    if weight and (isinstance(beat_ms, bool) or not isinstance(beat_ms, (int, float))
                   or not math.isfinite(beat_ms) or beat_ms <= 0):
        raise ValueError('节奏组连贯需要有效拍长')


def group_links(families, metadata, beat_ms, weight, score, boundaries=()):
    """Link adjacent pitched families within the existing 1.25-beat horizon.

Charge at most weight times one conservative attack value at each boundary.
No rewards for both-selected or both-omitted pairs; no chord-depth incentive.
"""
    validate_weight(weight, beat_ms)
    if weight == 0:
        return []
    streams = defaultdict(list)
    for index, (notes, _) in enumerate(families):
        if not notes or any(n.source != 'pitch' for n in notes):
            continue
        keys = {(metadata[n.id]['section'], n.voice, n.instrument_id, metadata[n.id]['role']) for n in notes}
        if len(keys) != 1:
            raise ValueError('节奏组跨越声部、类别或段落')
        key = next(iter(keys))
        values = defaultdict(list)
        for note in notes:
            values[note.start].append(score(note)*(2 if key[3] == 'primary' else 1))
        # Every offered layer includes one note per attack. A discarded strong
        # pitch must not lend its score to a retained weaker representative.
        floor = sum(min(row) for row in values.values())/len(values)
        streams[key].append((min(values), max(values), index, floor))
    links = []
    for key, rows in streams.items():
        rows.sort()
        for left, right in zip(rows, rows[1:]):
            gap = right[0]-left[1]
            if not 0 < gap <= beat_ms*1.25 or any(left[1] < t <= right[0] for t in boundaries):
                continue
            penalty = weight*min(left[3], right[3])
            if penalty > 0:
                links.append(dict(left=left[2], right=right[2], section=key[0], voice=key[1],
                    instrument_id=key[2], role=key[3], left_last_attack=left[1],
                    right_first_attack=right[0], penalty=penalty))
    return links


def add_costs(links, families, utility, constraint):
    """Linearize the absolute difference of two at-most-one family choices."""
    for link in links:
        left, right = families[link['left']][1], families[link['right']][1]
        difference = len(utility)
        utility.append(-link['penalty'])
        constraint({**{i: 1 for i in left}, **{i: -1 for i in right}, difference: -1}, 0)
        constraint({**{i: -1 for i in left}, **{i: 1 for i in right}, difference: -1}, 0)


def selection_report(links, families, chosen, weight):
    rows = []
    for link in links:
        left = any(chosen[i] for i in families[link['left']][1])
        right = any(chosen[i] for i in families[link['right']][1])
        rows.append(dict(link, left_selected=bool(left), right_selected=bool(right),
                         transition=bool(left != right)))
    return dict(weight=weight, research_only=True, semantics='rhythmic_selection_not_physical_identity',
        transitions=sum(r['transition'] for r in rows),
        penalty=sum(r['penalty'] for r in rows if r['transition']), links=rows)
