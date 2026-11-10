"""Pure controls for recovering fixed Tsumugi interval instrument evidence."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np


def rebuild_intervals(constructions, indices, *, slots, classes, valid_frames):
    """Preserve original per-track interval order and its construction mapping."""
    tracks = [[] for _ in range(88 * slots)]
    keys, entries = {}, []
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate construction index')
    last_track = -1
    for index in indices:
        row = constructions[index]
        args, slot = row['arguments'], row['v1_slot_index']
        if row['constructed_note'] is None or type(slot) is not int or not 0 <= slot < slots:
            raise ValueError('Cannot recover rejected construction or missing slot')
        pair, begin, end = args['pair_id'], args['begin_frame'], args['end_frame']
        if (not all(type(v) is int for v in (pair, begin, end))
                or not 0 <= pair < classes * 88 or not 0 <= begin <= end < valid_frames
                or args['valid_model_frames'] != valid_frames):
            raise ValueError('Invalid saved interval')
        track = (pair % 88) * slots + slot
        if track < last_track:
            raise ValueError('Construction order differs from V1 track order')
        last_track = track
        entry = (0, track, len(tracks[track]), begin, end)
        entries.append(entry)
        keys[entry[:3]] = index
        tracks[track].append((begin, end))
    return [tracks], entries, keys


def describe_logits(logits, *, preferred_class):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite class vector')
    if type(preferred_class) is not int or not 0 <= preferred_class < len(values):
        raise ValueError('Invalid preferred class')
    best = np.flatnonzero(values == values.max()).tolist()
    chosen = preferred_class if preferred_class in best else best[0]
    top = float(values[chosen])
    sigmoid = 1 / (1 + math.exp(-top)) if top >= 0 else math.exp(top) / (1 + math.exp(top))
    ordered = np.sort(values)
    return dict(class_id=chosen, top_sigmoid=sigmoid,
                logit_margin=float(ordered[-1] - ordered[-2]),
                positive_classes=int(np.count_nonzero(values >= 0)), tied_top=len(best))


def note_decisions(lineage, logits, rankings):
    """Keep all source indices; compare first-source and equal mean logits only."""
    values = np.asarray(logits)
    rankings = np.asarray(rankings)
    if values.ndim != 2 or rankings.shape != values.shape or not np.isfinite(values).all():
        raise ValueError('Invalid interval evidence arrays')
    classes = values.shape[1]
    if not np.array_equal(np.sort(rankings, axis=1), np.tile(np.arange(classes), (len(values), 1))):
        raise ValueError('Rankings must contain each class once')
    result = []
    for index, item in enumerate(lineage):
        sources, note = item['sources'], item['note']
        if (not sources or len(set(sources)) != len(sources)
                or any(type(i) is not int or not 0 <= i < len(values) for i in sources)):
            raise ValueError('Invalid construction lineage')
        original = note['instrument_id']
        if (rankings[sources[0]].tolist() != note['instrument_candidates']
                or original != int(rankings[sources[0], 0])):
            raise ValueError('First source no longer reproduces original classification')
        first = describe_logits(values[sources[0]], preferred_class=original)
        if first['class_id'] != original:
            raise ValueError('Logits do not support original top classification')
        mean = describe_logits(values[sources].mean(axis=0, dtype=np.float64), preferred_class=original)
        source_classes = rankings[sources, 0].astype(int).tolist()
        result.append(dict(original_index=index, sources=sources, first=first, mean=mean,
                           source_classes=source_classes,
                           sources_agree=len(set(source_classes)) == 1,
                           changed=mean['class_id'] != original))
    return result


def score_bin(value, edges):
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid score for binning')
    return sum(value >= edge for edge in edges)


def ambiguity_bins(reference, predicted, decisions, mapping):
    """Every prediction belongs to one category, including ambiguous/unmatched."""
    if len(predicted) != len(decisions):
        raise ValueError('Missing prediction decisions')
    neighbors = [[i for i, ref in enumerate(reference)
                  if not pred['is_drum'] and ref['pitch'] == pred['pitch']
                  and abs(ref['onset'] - pred['onset']) <= .05] for pred in predicted]
    degree = Counter(i for matches in neighbors for i in matches)
    results = {}
    for mode in ('first', 'mean'):
        bins = {name: [Counter() for _ in range(4)] for name in ('top_sigmoid', 'logit_margin')}
        categories = Counter()
        for pred, decision, matches in zip(predicted, decisions, neighbors):
            if pred['is_drum']:
                category = 'original_drum_prediction'
            elif not matches:
                category = 'no_compatible_reference'
            elif len(matches) != 1 or degree[matches[0]] != 1:
                category = 'ambiguous_correspondence'
            else:
                family = mapping[reference[matches[0]]['program']]
                category = 'unique_correct_family' if decision[mode]['class_id'] == family else 'unique_wrong_family'
            categories[category] += 1
            for name, edges in (('top_sigmoid', (.5, .8, .95)), ('logit_margin', (1., 2., 4.))):
                bins[name][score_bin(decision[mode][name], edges)][category] += 1
        results[mode] = dict(categories=dict(categories), bins={k: list(map(dict, v)) for k, v in bins.items()})
    return results
