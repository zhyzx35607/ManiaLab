"""Derive auxiliary later negatives in the original positive owner's graph.

Source notation proves only this local successor task. These annotations must
never be passed to model features, prediction ownership or the decoder.
"""
from collections import Counter, defaultdict

import numpy as np

from scripts.summarize_piano_successor_ranking import require


PROOF_VERSION = 'complete-local-successor-later-same-part-v1'


def inspect_window(payload, context, piece_id):
    """Verify original supervision and prove later relationships independently.

    This proof is valid even in a window without P. Training eligibility is a
    separate decision made across all windows by derive_piece_sidecars().
    """
    ids = payload['event_ids']
    require(ids.ndim == 1 and len(set(map(str, ids))) == len(ids), 'Duplicate/invalid node identity')
    prefix = piece_id + ':'
    require(all(str(value).startswith(prefix) for value in ids), 'Wrong piece identity')
    local_ids = [str(value)[len(prefix):] for value in ids]
    require(all(value in context['events'] for value in local_ids), 'Unknown source event')
    notes, edges = payload['notes'], payload['potential_edges']
    require(edges.ndim == 2 and edges.shape[0] == 2 and edges.dtype.kind in 'iu', 'Invalid candidate edges')
    require(len(notes) == len(ids) and (not edges.size or (edges.min() >= 0 and edges.max() < len(ids))),
            'Invalid candidate node index')
    require(len(set(map(tuple, edges.T))) == edges.shape[1], 'Duplicate candidate pair')
    a, b = edges
    mask, labels, core = payload['voice_mask'], payload['voice'], payload['core_mask']
    require(mask.dtype == bool and mask.shape == (len(a),) and labels.shape == mask.shape
            and np.isin(labels, [0, 1]).all(), 'Invalid original labels/mask')
    require(core.dtype == bool and core.shape == (len(ids),), 'Invalid original owner mask')
    events = [context['events'][value] for value in local_ids]
    ticks = np.asarray([event['start_tick'] for event in events], dtype=np.int64)
    ends = np.asarray([event['end_tick'] for event in events], dtype=np.int64)
    require(np.array_equal(notes['onset_div'], ticks), 'Changed source onset')
    require(np.array_equal(notes['pitch'], [event['pitch'] for event in events]), 'Changed source pitch')
    require(np.all(notes['duration_div'] >= ends-ticks), 'Observation shortened source duration')
    require(np.all(ticks[a] < ticks[b]), 'Candidate is not a strictly later attack')
    first = [context['first'][value] for value in local_ids]
    last = [context['last'][value] for value in local_ids]
    fpart = np.asarray([note['part'] for note in first])
    lpart = np.asarray([note['part'] for note in last])
    fbar = np.asarray([note['measure'] for note in first])
    lbar = np.asarray([note['measure'] for note in last])
    eligible = np.asarray([context['eligible'][value] for value in local_ids], dtype=bool)
    fclean = np.asarray([(n['part'], n['measure']) not in context['bad'] for n in first], dtype=bool)
    lclean = np.asarray([(n['part'], n['measure']) not in context['bad'] for n in last], dtype=bool)
    expected_mask = eligible[a] & eligible[b] & core[b] & (lpart[a] == fpart[b]) & (lbar[a] == fbar[b]) & fclean[b]
    require(np.array_equal(mask, expected_mask), 'Original supervision mask differs from source')
    truth = context['truth']
    expected_actual = np.asarray([local_ids[j] in truth.get(local_ids[i], set()) for i, j in edges.T], dtype=bool) & mask
    require(np.array_equal(labels.astype(bool), expected_actual), 'Original truth differs from source')
    next_tick = np.zeros(len(ids), dtype=np.int64)
    knows = np.zeros(len(ids), dtype=bool)
    for index, identifier in enumerate(local_ids):
        targets = truth.get(identifier, set())
        if not targets:
            continue
        clocks = {context['events'][value]['start_tick'] for value in targets}
        require(len(clocks) == 1 and eligible[index] and lclean[index], 'Invalid positive successor proof')
        next_tick[index] = next(iter(clocks))
        knows[index] = True
    proof = (~mask & eligible[a] & eligible[b] & lclean[a] & fclean[b] &
             (lpart[a] == fpart[b]) & knows[a] & (ticks[b] > next_tick[a]))
    require(not np.any(proof & labels.astype(bool)), 'Proof overlaps original positive')
    positives = defaultdict(set)
    for index in np.flatnonzero(expected_actual):
        positives[local_ids[a[index]]].add(local_ids[b[index]])
    return dict(source_ids=np.asarray(local_ids)[a], target_ids=np.asarray(local_ids)[b],
                target_ticks=ticks[b], proof=proof, owner=core[b], positives=dict(positives))


def derive_piece_sidecars(items, context, piece_id):
    """Anchor every auxiliary source once, where its complete original P lives.

    Missing positives skip the source. Contradictory or duplicated original
    supervision is rejected. Existing candidate geometry is not changed.
    """
    require(len({item['chunk'] for item in items}) == len(items), 'Duplicate chunk identity')
    windows, owner, partial, any_positive = {}, {}, set(), {}
    for item in items:
        require(item['metadata']['piece_id'] == piece_id, 'Mismatched window piece')
        chunk = item['chunk']
        data = inspect_window(item['payload'], context, piece_id)
        windows[chunk] = data
        for source, positives in data['positives'].items():
            require(source not in any_positive, 'Positive source has multiple owner windows')
            any_positive[source] = chunk
            if positives == context['truth'][source]:
                owner[source] = chunk
            else:
                partial.add(source)
    owned_pairs, all_visible, trainable, proof_values = {}, set(), {}, {}
    result, histogram = {}, Counter()
    for item in items:
        chunk = item['chunk']; data = windows[chunk]; payload = item['payload']
        derived = np.asarray([bool(proved and owner.get(str(source)) == chunk)
                              for source, proved in zip(data['source_ids'], data['proof'])], dtype=bool)
        source_groups = []
        for source in sorted(set(map(str, data['source_ids'][derived]))):
            subset = data['source_ids'] == source
            existing_ticks = set(map(int, data['target_ticks'][subset & payload['voice_mask'] & (payload['voice'] == 0)]))
            true_tick = context['events'][next(iter(context['truth'][source]))]['start_tick']
            existing_ticks.discard(true_tick)
            new_ticks = set(map(int, data['target_ticks'][subset & derived]))
            histogram[str(len(existing_ticks)) + '->' + str(len(existing_ticks | new_ticks))] += 1
            source_groups.append(dict(source_id=piece_id+':'+source,
                positive_target_ids=[piece_id+':'+value for value in sorted(context['truth'][source])],
                true_tick=true_tick, added_edges=int((subset & derived).sum()),
                previous_negative_ticks=len(existing_ticks), added_distinct_negative_ticks=len(new_ticks-existing_ticks)))
        for index, (source, target) in enumerate(zip(data['source_ids'], data['target_ids'])):
            pair = (str(source), str(target)); value = bool(data['proof'][index])
            # A halo edge can be unscored here yet scored in its own core. Only
            # compare canonical proof truth, independent of the core mask.
            canonical = value or bool(payload['voice_mask'][index] and not payload['voice'][index]
                and source in context['truth']
                and int(data['target_ticks'][index]) > context['events'][next(iter(context['truth'][source]))]['start_tick'])
            if pair in proof_values:
                require(proof_values[pair] == canonical, 'Proof changes with window ownership')
            proof_values[pair] = canonical
            if data['owner'][index]:
                require(pair not in owned_pairs, 'Candidate has duplicate prediction ownership')
                owned_pairs[pair] = value
            if value:
                all_visible.add(pair)
            if derived[index]:
                require(pair not in trainable, 'Derived pair has duplicate training ownership')
                trainable[pair] = (chunk, bool(data['owner'][index]))
        result[chunk] = dict(mask=derived, proof_mask=data['proof'], positive_groups=source_groups)
    global_proved = {pair for pair, flag in owned_pairs.items() if flag}
    trained_pairs = set(trainable)
    no_positive = {pair for pair in global_proved if pair[0] not in owner}
    no_candidate = {pair for pair in global_proved if pair[0] in owner and pair not in trained_pairs}
    covered = global_proved & trained_pairs
    require(len(global_proved) == len(no_positive)+len(no_candidate)+len(covered), 'Coverage partition mismatch')
    tick_group_count = lambda pairs: len({(s, context['events'][t]['start_tick']) for s, t in pairs})
    coverage = dict(windows=len(items), complete_positive_sources=len(owner), incomplete_positive_sources=len(partial),
        global_owned_provable_edges=len(global_proved), global_owned_provable_sources=len({s for s,t in global_proved}),
        global_owned_provable_tick_groups=tick_group_count(global_proved),
        global_owned_covered_in_positive_window=len(covered),
        global_owned_without_complete_positive_window=len(no_positive),
        global_owned_not_visible_in_positive_window=len(no_candidate),
        trainable_derived_edges=len(trainable), trainable_derived_sources=len({s for s,t in trainable}),
        trainable_derived_tick_groups=tick_group_count(trainable),
        trainable_target_core_edges=sum(core for chunk,core in trainable.values()),
        trainable_target_halo_edges=sum(not core for chunk,core in trainable.values()),
        visible_provable_pairs=len(all_visible), negative_tick_group_histogram=dict(sorted(histogram.items())))
    return result, coverage, proof_values
