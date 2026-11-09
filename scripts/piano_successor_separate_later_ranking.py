"""Keep original and proved-later successor ranking separately normalized.

The caller supplies the same proved-later mask as the merged objective, but
new edges never enter an original negative group or its averaging denominator.
These are two auxiliary losses; neither labels nor inference are changed here.
"""
from collections import defaultdict

import numpy as np

from scripts.piano_successor_later_ranking import prepare_later_ranking
from scripts.piano_successor_ranking import (RankingGroup,
    prepare_successor_ranking, ranking_loss_from_groups)


def prepare_separate_later_ranking(source_ids, target_ticks, targets,
                                   supervision_mask, derived_later_mask):
    """Return immutable original groups and proved-only added groups.

    Existing validation enforces explicit boolean proof, unsupervised zero
    labels, a supervised positive source and strictly later target ticks.
    Complete-positive and owner checks remain the proof builder's duty.
    Each added group uses every original supervised positive of its source.
    Original and added edges at the same tick belong to separate groups.
    """
    # Reuse the established proof checks, never its merged negative groups.
    prepare_later_ranking(source_ids, target_ticks, targets,
                          supervision_mask, derived_later_mask)
    original = prepare_successor_ranking(source_ids, target_ticks, targets,
                                         supervision_mask)
    sources = np.asarray(source_ids, dtype=object)
    ticks = np.asarray(target_ticks, dtype=object)
    labels = np.asarray(targets, dtype=object)
    mask = np.asarray(supervision_mask, dtype=bool)
    derived = np.asarray(derived_later_mask, dtype=bool)

    positives = defaultdict(list)
    for index in np.flatnonzero(mask & (labels == 1)):
        positives[str(sources[index])].append(int(index))
    additions = defaultdict(lambda: defaultdict(list))
    for index in np.flatnonzero(derived):
        additions[str(sources[index])][int(ticks[index])].append(int(index))
    added = tuple(
        RankingGroup(source, tuple(positives[source]),
            tuple(tuple(by_tick[tick]) for tick in sorted(by_tick)))
        for source, by_tick in sorted(additions.items()))
    return original, added


def separate_later_ranking_losses(logits, source_ids, target_ticks, targets,
                                  supervision_mask, derived_later_mask):
    """Return unweighted original and added losses, each independently meaned.

    Each term uses the frozen mean-source / mean-tick / mean-all-P formula.
    Only their positive indices may overlap. Missing terms return None, after
    logit validation. The caller combines these terms with the primary loss.
    Shared model parameters can still change predictions of unlabelled edges.
    """
    original, added = prepare_separate_later_ranking(
        source_ids, target_ticks, targets, supervision_mask, derived_later_mask)
    original_loss = ranking_loss_from_groups(logits, original,
                                              expected_edges=len(source_ids))
    added_loss = ranking_loss_from_groups(logits, added,
                                           expected_edges=len(source_ids))
    return original_loss, added_loss
