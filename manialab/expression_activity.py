"""Describe section activity without confusing duration with musical priority."""
from __future__ import annotations

from .domain import UNOBSERVED_RELEASES
from .expression_articulation import source_activity


def section_activity(notes, begin, end):
    """Separate local attacks from eligible intervals continuing across a boundary.

    Candidate strength retains the existing admission threshold; it is not a
    cross-model confidence or a salience probability. Carrying intervals describe
    context only: their expression still belongs to the section of their onset.
    """
    local = [n for n in notes if begin <= n.start < end and n.strength >= .35]
    carried = [n for n in notes if n.strength >= .35 and n.start < begin < n.end
               and n.source == 'pitch' and n.release_reason not in UNOBSERVED_RELEASES]
    activity = source_activity(local + carried, begin, end)
    attacks = sorted(n.start for n in local)
    groups, anchor = 0, None
    for time in attacks:
        if anchor is None or time-anchor > 30:
            groups += 1
            anchor = time
    activity.update(local_candidate_count=len(local), local_attack_groups=groups,
                    carried_candidate_count=len(carried),
                    carried_instrument_ids=sorted({n.instrument_id for n in carried if n.instrument_id}),
                    present=bool(local or carried))
    return activity
