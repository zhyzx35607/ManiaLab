"""Separate deterministic onset assignments from unavoidable coverage changes.

Counts use the existing maximum-cardinality, minimum-time-error assignment.
Coverage considers every maximum-cardinality assignment at the same tolerance;
it does not claim that a selected MIDI identity is the only possible match.
"""
from collections import defaultdict, deque

from scripts.evaluate_piano_acoustics import pairs


def coverage(reference, predicted, *, strict=False):
    tolerance = .04 if strict else .05
    assignments = pairs(reference, predicted, strict=strict)
    matched = dict(assignments)
    inverse = {j: i for i, j in assignments}
    by_pitch = defaultdict(list)
    for j, note in enumerate(predicted):
        by_pitch[note['pitch']].append(j)
    adjacent = [
        [j for j in by_pitch[note['pitch']]
         if abs(note['onset'] - predicted[j]['onset']) <= tolerance]
        for note in reference
    ]

    # Alternating paths starting at unmatched reference vertices identify the
    # matched references that could instead become unmatched at equal size.
    reachable = set(range(len(reference))) - matched.keys()
    pending = deque(reachable)
    while pending:
        i = pending.popleft()
        for j in adjacent[i]:
            if matched.get(i) == j:
                continue
            if j not in inverse:
                raise ValueError('An augmenting path contradicts maximum matching')
            other = inverse[j]
            if other not in reachable:
                reachable.add(other)
                pending.append(other)

    possible = {i for i, neighbors in enumerate(adjacent) if neighbors}
    necessary = matched.keys() - reachable
    return dict(selected=set(matched), possible=possible, necessary=necessary,
                predicted_selected=set(inverse))


def changes(reference, before, after, *, strict=False):
    old = coverage(reference, before, strict=strict)
    new = coverage(reference, after, strict=strict)
    selected = dict(recovered=new['selected'] - old['selected'],
                    new_missing=old['selected'] - new['selected'])
    unavoidable = dict(recovered=new['necessary'] - old['possible'],
                       new_missing=old['necessary'] - new['possible'])
    return dict(before=old, after=new, selected=selected, unavoidable=unavoidable)


def summarize_changes(result, predicted_count):
    return dict(
        recovered=len(result['selected']['recovered']),
        new_missing=len(result['selected']['new_missing']),
        unmatched_predictions=predicted_count - len(result['after']['selected']),
        unavoidable_recovered=len(result['unavoidable']['recovered']),
        unavoidable_new_missing=len(result['unavoidable']['new_missing']),
        before_ambiguous_coverage=len(result['before']['possible'] - result['before']['necessary']),
        after_ambiguous_coverage=len(result['after']['possible'] - result['after']['necessary']),
    )
