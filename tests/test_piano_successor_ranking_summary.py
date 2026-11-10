"""Paired audit counts cannot hide lost truth behind net improvements."""
import numpy as np
import pytest

from scripts.summarize_piano_successor_ranking import (align_baseline,
    decoding_diagnostics, paired_transitions, truth_group_retention)


def test_edge_transitions_count_both_directions_and_ignore_unscored():
    actual = np.asarray([1,1,0,0,1], dtype=bool)
    before = np.asarray([1,0,1,0,1], dtype=bool)
    after = ~before
    mask = np.asarray([1,1,1,1,0], dtype=bool)
    assert paired_transitions(actual,mask,before,after) == dict(tp_to_fn=1,fn_to_tp=1,fp_to_tn=1,tn_to_fp=1)


def test_all_chord_targets_distinguishes_a_partial_group_from_a_complete_group():
    result = truth_group_retention(np.asarray(['a','a','b','b']), np.ones(4,dtype=bool),
        np.ones(4,dtype=bool), np.asarray([1,1,0,0],dtype=bool), np.asarray([1,0,1,1],dtype=bool))
    assert result['sources_with_truth'] == result['multiple_target_sources'] == 2
    assert result['true_edges'] == 4
    assert result['before_at_least_one'] == 1 and result['after_at_least_one'] == 2
    assert result['before_all'] == result['after_all'] == 1
    assert result['all_lost'] == result['all_gained'] == 1
    assert result['at_least_one_lost'] == 0 and result['at_least_one_gained'] == 1


def test_groups_without_truth_are_excluded_and_unknown_true_placeholder_is_ignored():
    result = truth_group_retention(np.asarray(['a','a','b']), np.asarray([0,0,1],dtype=bool),
        np.asarray([1,1,0],dtype=bool), np.ones(3,dtype=bool), np.ones(3,dtype=bool))
    assert all(value == 0 for value in result.values())


def sample():
    return dict(source_ids=np.asarray(['a','a']),target_ids=np.asarray(['b','c']),
        target_ticks=np.asarray([120,240]),actual=np.asarray([True,False]),
        score_mask=np.asarray([True,False]),logits=np.asarray([.3,.8]),selected=np.asarray([False,True]))


def test_pair_alignment_uses_both_ids_and_preserves_new_order():
    old = sample(); new = {key:value[::-1] for key,value in old.items() if key!='selected'}
    new['logits'] = np.asarray([.4,.5])
    aligned = align_baseline(old,new)
    assert aligned['logits'].tolist() == [.8,.3]
    assert aligned['selected'].tolist() == [True,False]


@pytest.mark.parametrize('field', ['target_ids','target_ticks','actual','score_mask'])
def test_changed_identity_or_label_fails_pairing(field):
    old = sample(); new = {key:value.copy() for key,value in old.items()}
    if field == 'target_ids':new[field][0]='z'
    elif field == 'target_ticks':new[field][0]=999
    else:new[field][0]=~new[field][0]
    with pytest.raises(ValueError):align_baseline(old,new)


def test_duplicate_pair_is_rejected():
    old=sample();old['target_ids'][1]='b'
    with pytest.raises(ValueError,match='Duplicate'):align_baseline(old,old)


def test_unscored_competition_loss_is_counted_without_changing_decision():
    data=sample();result=decoding_diagnostics(data,data['selected'])
    assert result['removed_true_edges'] == result['removed_true_with_unscored_maximum_only'] == 1
    assert result['removed_true_with_only_unscored_winning_edges'] == 1
    assert result['raw']['tp'] == result['decoded']['fn'] == 1


def test_decoder_diagnostic_rejects_added_negative_edge():
    data=sample();data['logits'][0]=-.3
    with pytest.raises(ValueError,match='added'):decoding_diagnostics(data,np.asarray([True,False]))


def test_decoder_diagnostic_rejects_multiple_selected_ticks():
    data=sample()
    with pytest.raises(ValueError,match='multiple'):decoding_diagnostics(data,np.asarray([True,True]))
