"""Add explicitly proved later negatives to the frozen successor objective.

The caller establishes source provenance, complete positive groups and unique
window ownership. This module only accepts that proof as a separate boolean
mask; it never derives labels from model outputs or changes inference inputs.
Known negatives at the positive tick remain excluded, as in the old objective.
"""
from collections import defaultdict

import numpy as np

from scripts.piano_successor_ranking import (RankingGroup,
    prepare_successor_ranking, ranking_loss_from_groups)


def prepare_later_ranking(source_ids, target_ticks, targets, supervision_mask,
                          derived_later_mask):
    """Merge proved unknown later edges into the original negative tick groups.

    A marked edge must have an unsupervised zero label and occur strictly after
    its source's unique supervised positive tick. Missing positives and proof
    conflicts raise errors. Unmarked unknowns and known same-tick negatives are
    not added. Complete-positive and owner checks belong to the proof builder.
    """
    previous = prepare_successor_ranking(source_ids, target_ticks, targets, supervision_mask)
    derived = np.asarray(derived_later_mask, dtype=object)
    if derived.ndim != 1:
        raise ValueError('Derived later mask must be one-dimensional')
    if len(derived) != len(source_ids):
        raise ValueError('Derived later mask must have the same length as ranking arrays')
    if any(not isinstance(value, (bool, np.bool_)) for value in derived):
        raise ValueError('Derived later mask must contain boolean values')

    sources = np.asarray(source_ids, dtype=object)
    ticks = np.asarray(target_ticks, dtype=object)
    labels = np.asarray(targets, dtype=object)
    mask = np.asarray(supervision_mask, dtype=bool)
    added_indices = np.flatnonzero(derived.astype(bool))
    if any(mask[index] or labels[index] != 0 for index in added_indices):
        raise ValueError('Derived later edges must be unsupervised zero-label edges')
    if not len(added_indices):
        return previous

    positives = defaultdict(list)
    for index in np.flatnonzero(mask & (labels == 1)):
        positives[str(sources[index])].append(int(index))
    additions = defaultdict(lambda: defaultdict(list))
    for index in added_indices:
        source = str(sources[index])
        if not positives[source]:
            raise ValueError('A derived later edge requires a supervised positive source')
        if ticks[index] <= ticks[positives[source][0]]:
            raise ValueError('A derived later edge must be strictly later than the positive tick')
        additions[source][int(ticks[index])].append(int(index))

    original = {group.source_id: group for group in previous}
    result = []
    for source in sorted(original.keys() | additions.keys()):
        if source not in additions:
            result.append(original[source])
            continue
        old = original.get(source)
        by_tick = {int(ticks[group[0]]): list(group)
                   for group in (() if old is None else old.negative_groups)}
        for tick, indices in additions[source].items():
            by_tick.setdefault(tick, []).extend(indices)
        result.append(RankingGroup(source, tuple(positives[source]),
            tuple(tuple(sorted(by_tick[tick])) for tick in sorted(by_tick))))
    return tuple(result)


def later_ranking_loss(logits, source_ids, target_ticks, targets, supervision_mask,
                       derived_later_mask):
    """Use the unchanged mean-source / mean-tick / mean-all-positives loss."""
    groups = prepare_later_ranking(source_ids, target_ticks, targets,
                                   supervision_mask, derived_later_mask)
    return ranking_loss_from_groups(logits, groups, expected_edges=len(source_ids))
