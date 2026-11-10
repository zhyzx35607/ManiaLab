"""Hand-written local scores exercise proof and owner boundaries without fitting."""
import copy

import numpy as np
import pytest

from scripts.piano_later_negative_proof import derive_piece_sidecars, inspect_window
from scripts.summarize_piano_same_tick_errors import source_context


PIECE = 'manual'
IDS = ('s', 'p1', 'p2', 'late', 'same', 'early', 'cross', 'known')
PAIRS = tuple(('s', target) for target in IDS[1:])
LABELS = (1, 1, 0, 0, 0, 0, 0)
MASK = (True, True, False, True, True, False, True)
LOCAL_CORE = (True, True, True, False, True, True, False, True)
LATER_CORE = (False, False, False, True, False, False, True, False)


def score():
    # s ends at 120. Its next voice-1 attack is the complete {p1,p2} chord.
    # same/early/known are voice 2 in the same bar. late is in the next bar;
    # its voice/staff differ intentionally and do not establish its identity.
    rows = [
        ('s', 'p', 60, 0, 120, 1, 1, '1'),
        ('p1', 'p', 62, 120, 240, 1, 1, '1'),
        ('p2', 'p', 65, 120, 240, 1, 1, '1'),
        ('late', 'p', 67, 240, 360, 2, 2, '7'),
        ('same', 'p', 69, 120, 180, 1, 1, '2'),
        ('early', 'p', 71, 60, 120, 1, 1, '2'),
        ('cross', 'q', 40, 240, 360, 2, 1, '1'),
        ('known', 'p', 72, 180, 240, 1, 1, '2'),
    ]
    notes = [dict(zip(('id', 'part', 'pitch', 'start_tick', 'end_tick', 'measure', 'staff', 'voice'), row))
             for row in rows]
    return dict(written=[dict(note=note) for note in notes], excluded=[],
        representation=dict(events=[dict(id=note['id'], part=note['part'], pitch=note['pitch'],
            start_tick=note['start_tick'], end_tick=note['end_tick'],
            written_ids=[note['id']], supervision_eligible=True) for note in notes]))


def window(context, *, chunk=0, ids=IDS, pairs=PAIRS, labels=LABELS,
           mask=MASK, core=LOCAL_CORE):
    # Geometry is copied from the hand-written notes. Labels and ownership are
    # explicit fixture arguments, never inferred by the helper under test.
    notes = np.zeros(len(ids), dtype=[('pitch', 'i8'), ('onset_div', 'i8'), ('duration_div', 'i8')])
    for index, identifier in enumerate(ids):
        event = context['events'][identifier]
        notes[index] = (event['pitch'], event['start_tick'], event['end_tick']-event['start_tick'])
    index_of = {identifier: index for index, identifier in enumerate(ids)}
    edges = np.asarray([(index_of[a], index_of[b]) for a, b in pairs], dtype=np.int64).reshape(-1, 2).T
    return dict(chunk=chunk, metadata=dict(piece_id=PIECE), payload=dict(
        event_ids=np.asarray([PIECE+':'+identifier for identifier in ids]), notes=notes,
        potential_edges=edges, voice=np.asarray(labels, dtype=np.float32),
        voice_mask=np.asarray(mask, dtype=bool), core_mask=np.asarray(core, dtype=bool)))


def context_and_local_window():
    context = source_context(score())
    assert context['truth']['s'] == {'p1', 'p2'}
    return context, window(context)


def add_bad_bar(source, measure):
    note = dict(id='unverified', part='p', pitch=50, start_tick=30 if measure == 1 else 270,
                end_tick=60 if measure == 1 else 300, measure=measure, staff=1, voice='9')
    source['written'].append(dict(note=note))
    source['representation']['events'].append(dict(id=note['id'], part=note['part'], pitch=note['pitch'],
        start_tick=note['start_tick'], end_tick=note['end_tick'], written_ids=[note['id']],
        supervision_eligible=False))


def test_complete_chord_proves_later_same_part_target_and_keeps_both_positives():
    context, item = context_and_local_window()
    result, coverage, canonical = derive_piece_sidecars([item], context, PIECE)
    assert result[0]['mask'].tolist() == [False, False, True, False, False, False, False]
    assert result[0]['proof_mask'].tolist() == [False, False, True, False, False, False, False]
    assert result[0]['positive_groups'] == [dict(source_id='manual:s',
        positive_target_ids=['manual:p1', 'manual:p2'], true_tick=120, added_edges=1,
        previous_negative_ticks=2, added_distinct_negative_ticks=1)]
    assert coverage['complete_positive_sources'] == 1
    assert coverage['incomplete_positive_sources'] == 0
    assert coverage['trainable_derived_edges'] == coverage['trainable_derived_tick_groups'] == 1
    assert coverage['trainable_target_halo_edges'] == 1
    assert coverage['trainable_target_core_edges'] == 0
    assert coverage['global_owned_provable_edges'] == 0
    assert canonical == {('s', 'p1'): False, ('s', 'p2'): False, ('s', 'late'): True,
                         ('s', 'same'): False, ('s', 'early'): False, ('s', 'cross'): False,
                         ('s', 'known'): True}


@pytest.mark.parametrize('omit_node', [False, True])
def test_missing_any_positive_member_skips_whole_source_even_when_later_target_is_owned(omit_node):
    context, _ = context_and_local_window()
    ids = tuple(identifier for identifier in IDS if not (omit_node and identifier == 'p2'))
    pairs = tuple(pair for pair in PAIRS if pair[1] != 'p2')
    item = window(context, ids=ids, pairs=pairs,
        labels=[1, 0, 0, 0, 0, 0], mask=[True, False, True, True, False, True],
        core=[True] * len(ids))
    result, coverage, _ = derive_piece_sidecars([item], context, PIECE)
    assert not result[0]['mask'].any() and result[0]['positive_groups'] == []
    assert result[0]['proof_mask'].tolist() == [False, True, False, False, False, False]
    assert coverage['incomplete_positive_sources'] == 1
    assert coverage['complete_positive_sources'] == coverage['trainable_derived_edges'] == 0
    assert coverage['global_owned_provable_edges'] == 1
    assert coverage['global_owned_without_complete_positive_window'] == 1


def test_no_visible_positive_edges_does_not_use_complete_global_truth_as_training_positives():
    context, _ = context_and_local_window()
    item = window(context, ids=['s', 'late'], pairs=[('s', 'late')],
                  labels=[0], mask=[False], core=[False, True])
    result, coverage, _ = derive_piece_sidecars([item], context, PIECE)
    assert result[0]['proof_mask'].tolist() == [True]
    assert result[0]['mask'].tolist() == [False]
    assert coverage['complete_positive_sources'] == coverage['incomplete_positive_sources'] == 0
    assert coverage['global_owned_without_complete_positive_window'] == 1


def test_later_window_halo_positives_do_not_repeat_source_training_and_canonical_is_owner_invariant():
    context, first = context_and_local_window()
    second = window(context, chunk=1, labels=[0] * 7, mask=[False] * 7, core=LATER_CORE)
    before = [copy.deepcopy(first['payload']), copy.deepcopy(second['payload'])]
    result, coverage, canonical = derive_piece_sidecars([first, second], context, PIECE)
    assert result[0]['mask'].tolist() == [False, False, True, False, False, False, False]
    assert not result[1]['mask'].any() and result[1]['positive_groups'] == []
    assert result[1]['proof_mask'].tolist() == [False, False, True, False, False, False, True]
    assert coverage['complete_positive_sources'] == 1
    assert coverage['global_owned_provable_edges'] == coverage['global_owned_covered_in_positive_window'] == 1
    assert coverage['global_owned_without_complete_positive_window'] == 0
    assert coverage['global_owned_not_visible_in_positive_window'] == 0
    assert coverage['trainable_derived_edges'] == coverage['trainable_derived_sources'] == 1
    assert coverage['trainable_target_halo_edges'] == 1
    assert coverage['visible_provable_pairs'] == 2
    # known is a supervised negative in window 0, a proved halo relation in 1.
    assert canonical[('s', 'known')] and canonical[('s', 'late')]
    assert not any(canonical['s', target] for target in ('same', 'early', 'cross', 'p1', 'p2'))
    for item, saved in zip((first, second), before):
        for key, value in saved.items():
            assert np.array_equal(item['payload'][key], value)


def test_global_provable_candidate_outside_positive_window_is_counted_but_not_trained():
    context, _ = context_and_local_window()
    first = window(context, ids=['s', 'p1', 'p2'], pairs=[('s', 'p1'), ('s', 'p2')],
                   labels=[1, 1], mask=[True, True], core=[True] * 3)
    second = window(context, chunk=1, ids=['s', 'late'], pairs=[('s', 'late')],
                    labels=[0], mask=[False], core=[False, True])
    result, coverage, _ = derive_piece_sidecars([first, second], context, PIECE)
    assert not result[0]['mask'].any() and not result[1]['mask'].any()
    assert coverage['complete_positive_sources'] == 1
    assert coverage['global_owned_provable_edges'] == 1
    assert coverage['global_owned_not_visible_in_positive_window'] == 1
    assert coverage['global_owned_without_complete_positive_window'] == 0
    assert coverage['global_owned_covered_in_positive_window'] == coverage['trainable_derived_edges'] == 0


@pytest.mark.parametrize('partial', [False, True])
def test_one_positive_source_cannot_be_owned_by_two_windows_even_if_each_is_partial(partial):
    context, first = context_and_local_window()
    second = window(context, chunk=1)
    if partial:
        first = window(context, ids=['s', 'p1'], pairs=[('s', 'p1')],
                       labels=[1], mask=[True], core=[True, True])
        second = window(context, chunk=1, ids=['s', 'p2'], pairs=[('s', 'p2')],
                        labels=[1], mask=[True], core=[True, True])
    with pytest.raises(ValueError, match='multiple owner windows'):
        derive_piece_sidecars([first, second], context, PIECE)


def test_duplicate_candidate_prediction_ownership_is_rejected_without_positive_sources():
    context, _ = context_and_local_window()
    first = window(context, ids=['s', 'late'], pairs=[('s', 'late')],
                   labels=[0], mask=[False], core=[False, True])
    second = copy.deepcopy(first); second['chunk'] = 1
    with pytest.raises(ValueError, match='duplicate prediction ownership'):
        derive_piece_sidecars([first, second], context, PIECE)


@pytest.mark.parametrize('field', ['mask', 'label', 'prefix', 'onset', 'pitch', 'shortened_duration'])
def test_original_supervision_and_source_identity_changes_are_rejected(field):
    context, item = context_and_local_window(); payload = item['payload']
    if field == 'mask': payload['voice_mask'][2] = True
    elif field == 'label': payload['voice'][0] = 0
    elif field == 'prefix': payload['event_ids'][0] = 'wrong:s'
    elif field == 'onset': payload['notes']['onset_div'][1] += 1
    elif field == 'pitch': payload['notes']['pitch'][1] += 1
    else: payload['notes']['duration_div'][0] -= 1
    with pytest.raises(ValueError):
        inspect_window(payload, context, PIECE)


@pytest.mark.parametrize('endpoint', ['source', 'source_bar', 'target', 'target_bar'])
def test_unreliable_endpoints_and_bad_local_bars_cannot_prove_later_negatives(endpoint):
    source = score()
    if endpoint in ('source', 'target'):
        identifier = 's' if endpoint == 'source' else 'late'
        next(event for event in source['representation']['events'] if event['id'] == identifier)['supervision_eligible'] = False
    else:
        add_bad_bar(source, 1 if endpoint == 'source_bar' else 2)
    context = source_context(source)
    if endpoint in ('source', 'source_bar'):
        assert 's' not in context['truth']
        item = window(context, labels=[0] * 7, mask=[False] * 7)
    else:
        assert context['truth']['s'] == {'p1', 'p2'}
        item = window(context)
    inspected = inspect_window(item['payload'], context, PIECE)
    assert not inspected['proof'].any()
    result, coverage, _ = derive_piece_sidecars([item], context, PIECE)
    assert not result[0]['mask'].any() and coverage['trainable_derived_edges'] == 0


def test_verified_tie_uses_final_written_measure_staff_and_voice_for_positive_proof():
    source = score()
    first = source['written'][0]['note']
    first.update(end_tick=60, measure=0, staff=2, voice='9')
    last = dict(first, id='s-tail', start_tick=60, end_tick=120, measure=1, staff=1, voice='1')
    source['written'].append(dict(note=last))
    source['representation']['events'][0]['written_ids'].append('s-tail')
    context = source_context(source)
    assert context['truth']['s'] == {'p1', 'p2'}
    result, coverage, _ = derive_piece_sidecars([window(context)], context, PIECE)
    assert result[0]['mask'].tolist() == [False, False, True, False, False, False, False]
    assert result[0]['positive_groups'][0]['positive_target_ids'] == ['manual:p1', 'manual:p2']
    assert coverage['complete_positive_sources'] == 1


def test_observed_tail_extension_does_not_move_true_tick_or_change_shared_proof():
    context, item = context_and_local_window()
    extended = copy.deepcopy(item)
    extended['payload']['notes']['duration_div'] += 480
    old, old_coverage, old_canonical = derive_piece_sidecars([item], context, PIECE)
    new, new_coverage, new_canonical = derive_piece_sidecars([extended], context, PIECE)
    assert old_canonical == new_canonical and old_coverage == new_coverage
    assert old[0]['positive_groups'] == new[0]['positive_groups']
    assert new[0]['positive_groups'][0]['true_tick'] == 120
    for key in ('mask', 'proof_mask'):
        assert np.array_equal(old[0][key], new[0][key])
    assert extended['payload']['notes']['duration_div'][0] == 600


def test_variant_can_add_visible_candidate_without_changing_shared_pair_proofs():
    source = score()
    note = dict(id='later2', part='p', pitch=75, start_tick=480, end_tick=600, measure=3, staff=2, voice='4')
    source['written'].append(dict(note=note))
    source['representation']['events'].append(dict(id='later2', part='p', pitch=75,
        start_tick=480, end_tick=600, written_ids=['later2'], supervision_eligible=True))
    context = source_context(source)
    original = window(context)
    changed = window(context, ids=IDS+('later2',), pairs=PAIRS+(('s', 'later2'),),
        labels=LABELS+(0,), mask=MASK+(False,), core=LOCAL_CORE+(False,))
    changed['payload']['notes']['duration_div'][0] += 480
    first, first_coverage, first_canonical = derive_piece_sidecars([original], context, PIECE)
    second, second_coverage, second_canonical = derive_piece_sidecars([changed], context, PIECE)
    assert first[0]['positive_groups'][0]['positive_target_ids'] == second[0]['positive_groups'][0]['positive_target_ids']
    assert {pair: second_canonical[pair] for pair in first_canonical} == first_canonical
    assert second_canonical['s', 'later2']
    assert first_coverage['trainable_derived_edges'] == 1
    assert second_coverage['trainable_derived_edges'] == 2
    assert second[0]['mask'].tolist() == [False, False, True, False, False, False, False, True]


@pytest.mark.parametrize('damage', ['duplicate_chunk', 'piece_metadata', 'duplicate_node',
    'unknown_event', 'duplicate_edge', 'backward_edge', 'float_edge', 'nonboolean_mask', 'nonbinary_label'])
def test_malformed_window_identity_or_graph_is_not_silently_repaired(damage):
    context, item = context_and_local_window(); payload = item['payload']; items = [item]
    if damage == 'duplicate_chunk': items.append(copy.deepcopy(item))
    elif damage == 'piece_metadata': item['metadata']['piece_id'] = 'different'
    elif damage == 'duplicate_node': payload['event_ids'][1] = payload['event_ids'][0]
    elif damage == 'unknown_event': payload['event_ids'][0] = 'manual:nope'
    elif damage == 'duplicate_edge': payload['potential_edges'][:, 1] = payload['potential_edges'][:, 0]
    elif damage == 'backward_edge': payload['potential_edges'][:, 0] = [1, 0]
    elif damage == 'float_edge': payload['potential_edges'] = payload['potential_edges'].astype(float)
    elif damage == 'nonboolean_mask': payload['voice_mask'] = payload['voice_mask'].astype(int)
    else: payload['voice'][0] = .5
    with pytest.raises(ValueError):
        derive_piece_sidecars(items, context, PIECE)
