"""Per-reference development gates for the fixed joint acoustic-head experiment.

Matching uncertainty is evaluated over every maximum-cardinality assignment,
ignoring the secondary minimum-onset-error tie break. No neural inference here.
"""
from collections import defaultdict, deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from scripts.evaluate_piano_acoustics import pairs
from scripts.train_piano_onset_head import pooled


def matching(reference, predicted, *, strict=False, offsets=False, fixed_pairs=False):
    assigned = dict(pairs(reference, predicted, strict=strict, offsets=offsets))
    inverse = {j: i for i, j in assigned.items()}
    by_pitch = defaultdict(list)
    for j, note in enumerate(predicted):
        by_pitch[note['pitch']].append(j)
    adjacent = []
    for note in reference:
        tail = .08 if strict else max(.05, .2 * (note['offset'] - note['onset']))
        adjacent.append([j for j in by_pitch[note['pitch']]
                         if abs(note['onset'] - predicted[j]['onset']) <= (.04 if strict else .05)
                         and (not offsets or abs(note['offset'] - predicted[j]['offset']) <= tail)])
    reachable = set(range(len(reference))) - assigned.keys()
    pending = deque(reachable)
    while pending:
        i = pending.popleft()
        for j in adjacent[i]:
            if assigned.get(i) == j:
                continue
            if j not in inverse:
                raise ValueError('Assignment was not maximum cardinality')
            other = inverse[j]
            if other not in reachable:
                reachable.add(other)
                pending.append(other)
    result = dict(selected=sorted(assigned), possible=[i for i, js in enumerate(adjacent) if js],
                  necessary=sorted(assigned.keys() - reachable))
    if fixed_pairs:
        # A selected edge is fixed exactly when deleting it reduces maximum
        # cardinality. Work per pitch so most real singleton graphs are cheap.
        forced = {}
        for pitch in sorted({n['pitch'] for n in reference}):
            ri = [i for i, n in enumerate(reference) if n['pitch'] == pitch]
            pi = by_pitch[pitch]
            if not pi:
                continue
            pindex = {j: k for k, j in enumerate(pi)}
            graph = np.zeros((len(ri), len(pi)), dtype=bool)
            for k, i in enumerate(ri):
                for j in adjacent[i]:
                    graph[k, pindex[j]] = True
            size = sum(i in assigned for i in ri)
            degrees = graph.sum(axis=0)
            for k, i in enumerate(ri):
                if i not in assigned:
                    continue
                j = assigned[i]
                column = pindex[j]
                if graph[k].sum() == 1 and degrees[column] == 1:
                    forced[i] = j
                    continue
                graph[k, column] = False
                alternative = maximum_bipartite_matching(csr_matrix(graph), perm_type='column')
                graph[k, column] = True
                if int(np.count_nonzero(alternative >= 0)) < size:
                    forced[i] = j
        result['fixed_pairs'] = [[i, forced[i]] for i in sorted(forced)]
        result['ambiguous_pair_references'] = sorted(assigned.keys() - forced.keys())
    return result


def compare_recording(key_reference, sustain_reference, before, after):
    if [(n['pitch'], n['onset']) for n in key_reference] != [
            (n['pitch'], n['onset']) for n in sustain_reference]:
        raise ValueError('Key and sustain reference identities differ')
    result = {}
    for label, strict in [('50ms', False), ('40ms', True)]:
        old = matching(key_reference, before, strict=strict, fixed_pairs=True)
        new = matching(key_reference, after, strict=strict, fixed_pairs=True)
        old_full = matching(sustain_reference, before, strict=strict, offsets=True)
        new_full = matching(sustain_reference, after, strict=strict, offsets=True)
        old_pairs, new_pairs = dict(old['fixed_pairs']), dict(new['fixed_pairs'])
        tails = []
        for i in sorted(old_pairs.keys() & new_pairs.keys()):
            ref, a, b = sustain_reference[i], before[old_pairs[i]], after[new_pairs[i]]
            old_error, new_error = abs(a['offset'] - ref['offset']), abs(b['offset'] - ref['offset'])
            increase = new_error - old_error
            tails.append(dict(reference_index=i, pitch=ref['pitch'], onset=ref['onset'],
                              reference_end=ref['offset'], before_index=old_pairs[i], after_index=new_pairs[i],
                              before_end=a['offset'], after_end=b['offset'],
                              before_error_seconds=old_error, after_error_seconds=new_error,
                              error_increase_seconds=increase, exceeds_80ms=increase > .08))
        result[label] = dict(
            onset_before=old, onset_after=new, sustain_before=old_full, sustain_after=new_full,
            baseline_onsets_not_guaranteed=sorted(set(old['selected']) - set(new['necessary'])),
            baseline_sustain_not_guaranteed=sorted(set(old_full['selected']) - set(new_full['necessary'])),
            sustain_selected_lost=sorted(set(old_full['selected']) - set(new_full['selected'])),
            sustain_selected_gained=sorted(set(new_full['selected']) - set(old_full['selected'])),
            tail_changes=tails, tail_regressions=[t['reference_index'] for t in tails if t['exceeds_80ms']],
            incomparable_onset_references=sorted((set(old['selected']) | set(new['selected'])) -
                                                (old_pairs.keys() & new_pairs.keys())))
    return result


def eligible(baseline, current, comparisons):
    old = {row['id']: row for row in baseline}
    new = {row['id']: row for row in current}
    if len(old) != len(baseline) or len(new) != len(current) or old.keys() != new.keys() or old.keys() != comparisons.keys():
        raise ValueError('Selection requires the same unique recordings')
    checks = {}
    for label in ('50ms', '40ms'):
        details = [comparisons[name][label] for name in old]
        checks[label] = dict(
            baseline_onsets_guaranteed=all(not d['baseline_onsets_not_guaranteed'] for d in details),
            extra_not_increased=all(
                (new[name]['metrics']['key'][label]['onsets']['predicted'] -
                 new[name]['metrics']['key'][label]['onsets']['matched']) <=
                (old[name]['metrics']['key'][label]['onsets']['predicted'] -
                 old[name]['metrics']['key'][label]['onsets']['matched']) for name in old),
            reattack_coverage_not_lower=all(
                new[name]['metrics']['key'][label]['close_repeats']['both_onsets_matched'] >=
                old[name]['metrics']['key'][label]['close_repeats']['both_onsets_matched'] for name in old),
            baseline_sustain_intervals_guaranteed=all(not d['baseline_sustain_not_guaranteed'] for d in details),
            no_tail_error_increase_over_80ms=all(not d['tail_regressions'] for d in details))
    before_onset, after_onset = pooled(baseline), pooled(current)
    before_sustain = pooled(baseline, 'sustain', field='intervals')
    after_sustain = pooled(current, 'sustain', field='intervals')
    improved = after_onset['f1'] > before_onset['f1'] or after_sustain['f1'] > before_sustain['f1']
    return dict(checks=checks, strict_improvement=improved,
                eligible=improved and all(all(v.values()) for v in checks.values()),
                pooled_onsets=after_onset, pooled_sustain_intervals=after_sustain)


def select_epoch(history):
    if [r['epoch'] for r in history] != [0, 1, 2]:
        raise ValueError('Selection requires all three fixed checkpoints')
    candidates = [row for row in history if row['epoch'] == 0 or row['selection']['eligible']]
    return max(candidates, key=lambda r: (r['selection']['pooled_onsets']['f1'],
                                         r['selection']['pooled_sustain_intervals']['f1'], -r['epoch']))['epoch']
