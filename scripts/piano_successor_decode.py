"""Select one observed target attack group per source from raw positive edges.

This is a deletion-only diagnostic decoder, not a melody or LN assignment.
Callers aggregate uniquely owned edges from all windows before calling it.
It never reads supervision, staff, bars, source durations or target identities.
"""
from numbers import Integral

import numpy as np


def _vector(values, name):
    # Object conversion preserves mixed scalar types, so strings, bools and
    # integers cannot silently become valid IDs, ticks or floating logits.
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(name + ' must be one-dimensional')
    return array


def select_successor_groups(source_ids, target_ticks, logits) -> np.ndarray:
    """Keep positive edges at the best-scoring target tick of each source.

    The group score is its maximum edge logit, with earlier target ticks
    winning exact ties. Every nonnegative edge in that group is retained;
    sources with only negative scores keep no edges. The returned boolean
    mask has the input order and length. Inputs are never changed.
    """
    sources = _vector(source_ids, 'source_ids')
    ticks = _vector(target_ticks, 'target_ticks')
    scores = _vector(logits, 'logits')
    if not len(sources) == len(ticks) == len(scores):
        raise ValueError('Source IDs, target ticks and logits must have equal lengths')
    if any(not isinstance(value, str) or not value.strip() for value in sources):
        raise ValueError('Source IDs must be nonempty strings')
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           for value in ticks):
        raise ValueError('Target ticks must be exact non-boolean integers')
    if any(not isinstance(value, (float, np.floating)) or not np.isfinite(value)
           for value in scores):
        raise ValueError('Logits must be finite floating-point values')

    best = {}
    for source, tick, score in zip(sources, ticks, scores):
        if score < 0:
            continue
        # Python integers retain unsigned/large tick values during tie breaks.
        tick = int(tick)
        previous = best.get(source)
        if previous is None or score > previous[0] or (score == previous[0] and tick < previous[1]):
            best[source] = (score, tick)
    return np.fromiter((score >= 0 and int(tick) == best[source][1]
                        for source, tick, score in zip(sources, ticks, scores)),
                       dtype=bool, count=len(sources))
