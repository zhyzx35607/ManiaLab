"""Masked multi-positive successor ranking, separate from model inputs.

Unknown edges never become negative examples. This auxiliary loss compares
known positive successors with known negative edges at other attack ticks.
It cannot supervise or identify unknown cross-bar successors.
"""
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class RankingGroup:
    source_id: str
    positive_indices: tuple[int, ...]
    negative_groups: tuple[tuple[int, ...], ...]


def _vector(values, name):
    result = np.asarray(values, dtype=object)
    if result.ndim != 1:
        raise ValueError(name + ' must be one-dimensional')
    return result


def prepare_successor_ranking(source_ids, target_ticks, targets, supervision_mask):
    """Prepare immutable index groups using only explicit supervised labels.

    Each eligible source has one positive attack tick and at least one other
    tick with known negative edges. Same-tick negative edges remain the BCE
    loss's responsibility. A source with multiple positive ticks is rejected.
    """
    sources = _vector(source_ids, 'source_ids')
    ticks = _vector(target_ticks, 'target_ticks')
    labels = _vector(targets, 'targets')
    mask = _vector(supervision_mask, 'supervision_mask')
    if not len(sources) == len(ticks) == len(labels) == len(mask):
        raise ValueError('Ranking arrays must have equal lengths')
    if any(not isinstance(value, str) or not value.strip() for value in sources):
        raise ValueError('Source IDs must be nonempty strings')
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           for value in ticks):
        raise ValueError('Target ticks must be exact non-boolean integers')
    if any(not isinstance(value, (Real, np.bool_)) or value not in (0, 1) for value in labels):
        raise ValueError('Targets must contain only finite binary labels')
    if any(not isinstance(value, (bool, np.bool_)) for value in mask):
        raise ValueError('Supervision mask must contain boolean values')

    by_source = defaultdict(list)
    for index in np.flatnonzero(mask.astype(bool)):
        by_source[str(sources[index])].append(int(index))
    result = []
    for source, indices in sorted(by_source.items()):
        positive = tuple(index for index in indices if labels[index] == 1)
        if not positive:
            continue
        positive_ticks = {int(ticks[index]) for index in positive}
        if len(positive_ticks) != 1:
            raise ValueError('A supervised source has multiple positive target ticks')
        tick = next(iter(positive_ticks))
        negatives = defaultdict(list)
        for index in indices:
            if labels[index] == 0 and ticks[index] != tick:
                negatives[int(ticks[index])].append(index)
        if negatives:
            result.append(RankingGroup(source, positive,
                tuple(tuple(negatives[key]) for key in sorted(negatives))))
    return tuple(result)


def ranking_loss_from_groups(logits, groups, *, expected_edges):
    """Apply mean-source / mean-negative-tick / mean-positive softplus loss.

    Only indexed supervised logits receive a direct auxiliary gradient.
    The shared encoder may still change unlabelled outputs during training.
    """
    import torch
    from torch.nn import functional as F

    if (isinstance(expected_edges, (bool, np.bool_)) or not isinstance(expected_edges, Integral)
            or expected_edges < 0):
        raise ValueError('Expected edge count must be a nonnegative integer')
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 1
            or len(logits) != expected_edges or not logits.is_floating_point()):
        raise ValueError('Logits must be a floating vector with the expected edge count')
    if not torch.isfinite(logits).all().item():
        raise ValueError('Ranking logits must be finite')
    groups = tuple(groups)
    sources, used = set(), set()
    for group in groups:
        if (not isinstance(group, RankingGroup) or not isinstance(group.source_id, str)
                or not group.source_id.strip() or group.source_id in sources
                or not group.positive_indices or not group.negative_groups):
            raise ValueError('Invalid or duplicate prepared ranking group')
        sources.add(group.source_id)
        for indices in (group.positive_indices,) + group.negative_groups:
            if not indices:
                raise ValueError('Prepared index groups cannot be empty')
            for index in indices:
                if (isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
                        or not 0 <= index < expected_edges or index in used):
                    raise ValueError('Prepared edge indices must be unique and in range')
                used.add(index)
    if not groups:
        return None

    losses = []
    for group in groups:
        positive = logits.index_select(0, torch.tensor(group.positive_indices,
                                                       dtype=torch.long, device=logits.device))
        per_tick = []
        for indices in group.negative_groups:
            negative = logits.index_select(0, torch.tensor(indices, dtype=torch.long, device=logits.device))
            per_tick.append(F.softplus(negative.amax() - positive).mean())
        losses.append(torch.stack(per_tick).mean())
    return torch.stack(losses).mean()


def successor_ranking_loss(logits, source_ids, target_ticks, targets, supervision_mask):
    """Validate source metadata and logits, then evaluate the auxiliary loss."""
    groups = prepare_successor_ranking(source_ids, target_ticks, targets, supervision_mask)
    return ranking_loss_from_groups(logits, groups, expected_edges=len(source_ids))
