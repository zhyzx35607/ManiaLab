"""Simulate longer observed piano tails without changing score supervision.

The added tail is an input perturbation, not an inferred pedal or acoustic
release. It never moves an attack, merges unisons, or changes an event label.
"""
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from numbers import Integral


EXTRA_QUARTERS = (1, 2)
OVERLAP_SCHEMA = 'piano-score-event-overlap-v1'


def augment_events(source: dict, extra_quarters: int) -> list[dict]:
    """Extend every end by exact quarters, capped at the next same-pitch attack.

    A real source interval is never shortened, even if a later unison already
    overlaps it. The cap uses pitch/time across the entire piece and never
    reads part/staff/voice or supervision eligibility. Simultaneous unisons
    remain separate and do not cap one another.
    """
    if (isinstance(extra_quarters, bool) or not isinstance(extra_quarters, Integral)
            or extra_quarters not in EXTRA_QUARTERS):
        raise ValueError('Expected exactly +1 or +2 quarter input augmentation')
    unit = source['ticks_per_quarter']
    if isinstance(unit, bool) or not isinstance(unit, Integral) or unit <= 0:
        raise ValueError('Expected a positive integral score clock')
    events = source['representation']['events']
    if len({event['id'] for event in events}) != len(events):
        raise ValueError('Duplicate score-event identity')
    attacks = defaultdict(set)
    for event in events:
        if any(isinstance(event[name], bool) or not isinstance(event[name], Integral)
               for name in ('start_tick', 'end_tick', 'pitch')):
            raise ValueError('Expected exact integer event geometry')
        if event['end_tick'] <= event['start_tick']:
            raise ValueError('Expected a positive source interval')
        attacks[event['pitch']].add(event['start_tick'])
    clocks = {pitch: sorted(values) for pitch, values in attacks.items()}
    result = deepcopy(list(events))
    extra = int(extra_quarters) * int(unit)
    for event in result:
        times = clocks[event['pitch']]
        position = bisect_right(times, event['start_tick'])
        end = event['end_tick'] + extra
        if position < len(times):
            end = max(event['end_tick'], min(end, times[position]))
        event['end_tick'] = end
    return result


def summarize_augmentation(source: dict, observed: list[dict]) -> dict:
    before = {event['id']: event for event in source['representation']['events']}
    if len(observed) != len(before) or {e['id'] for e in observed} != set(before):
        raise ValueError('Observed event identity differs from source')
    increments = [event['end_tick']-before[event['id']]['end_tick'] for event in observed]
    if any(delta < 0 for delta in increments):
        raise ValueError('Input augmentation shortened source events')
    return dict(events=len(observed), extended=sum(delta > 0 for delta in increments),
                unchanged=sum(delta == 0 for delta in increments),
                max_added_ticks=max(increments, default=0),
                total_added_ticks=sum(increments), onset_pitch_id_unchanged=all(
                    all(event[key] == before[event['id']][key] for key in ('start_tick','pitch','id'))
                    for event in observed))
