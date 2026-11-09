"""Include known wrong-voice edges at the correct successor tick in ranking.

This is a separate objective from the frozen other-tick-only experiment. It
adds one group of supervised negative edges at the positive tick, preserving
all positive chord members, every earlier negative group and the same means.
Unknown edges remain unlabelled and receive no direct auxiliary gradient.
"""
from collections import defaultdict

import numpy as np

from scripts.piano_successor_ranking import (RankingGroup,
    prepare_successor_ranking, ranking_loss_from_groups)


def prepare_same_tick_ranking(source_ids, target_ticks, targets, supervision_mask):
    """Prepare the original negative groups plus known same-tick negatives.

    The frozen preparation function validates all metadata and rejects multiple
    positive ticks. Its groups are reused unchanged unless that source also
    has a supervised negative at its positive tick. Sources with positives and
    only same-tick negatives now qualify; sources lacking either side do not.
    """
    previous = prepare_successor_ranking(source_ids, target_ticks, targets, supervision_mask)
    by_name = {group.source_id: group for group in previous}
    sources = np.asarray(source_ids, dtype=object)
    ticks = np.asarray(target_ticks, dtype=object)
    labels = np.asarray(targets, dtype=object)
    mask = np.asarray(supervision_mask, dtype=bool)
    supervised = defaultdict(list)
    for index in np.flatnonzero(mask):
        supervised[str(sources[index])].append(int(index))

    result = []
    for source, indices in sorted(supervised.items()):
        positive = tuple(index for index in indices if labels[index] == 1)
        if not positive:
            continue
        positive_tick = ticks[positive[0]]
        same_tick = tuple(index for index in indices
                          if labels[index] == 0 and ticks[index] == positive_tick)
        earlier = by_name.get(source)
        if not same_tick:
            if earlier is not None:
                result.append(earlier)
            continue
        negative_groups = (() if earlier is None else earlier.negative_groups) + (same_tick,)
        negative_groups = tuple(sorted(negative_groups, key=lambda group: int(ticks[group[0]])))
        result.append(RankingGroup(source, positive, negative_groups))
    return tuple(result)


def same_tick_ranking_loss(logits, source_ids, target_ticks, targets, supervision_mask):
    """Use the unchanged mean-source / mean-group / mean-positive loss.

    Adding a same-tick negative group increases that source's group denominator
    by one. There is no separate coefficient or margin for the added group.
    """
    groups = prepare_same_tick_ranking(source_ids, target_ticks, targets, supervision_mask)
    return ranking_loss_from_groups(logits, groups, expected_edges=len(source_ids))
