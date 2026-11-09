import numpy as np
import pytest
from scripts.summarize_piano_later_context import compare_contexts


def test_same_winner_already_outranks_positive_group_locally():
    r = compare_contexts([1.,2.], [True,True], [3.], [4.])
    assert r['local_outranks_positive_group']
    assert r['local_ranked_above_lost_members'] == 2


def test_context_change_can_invert_pairwise_ranking():
    r = compare_contexts([1.,2.], [True,True], [.5], [4.])
    assert not r['local_outranks_positive_group']
    assert r['local_ranked_at_or_below_lost_members'] == 2
    assert r['output_minus_local_max'] == 3.5


def test_local_tie_with_best_positive_does_not_strictly_outrank_group():
    r = compare_contexts([1.,2.], [True,True], [2.], [4.])
    assert not r['local_outranks_positive_group']
    assert r['local_ranked_above_lost_members'] == 1
    assert r['local_ranked_at_or_below_lost_members'] == 1


def test_all_global_tied_maxima_must_be_compared_in_local_context():
    r = compare_contexts([1.,2.,-1.], [True,True,False], [.5,3.], [4.,4.])
    assert r['local_outranks_positive_group']
    assert r['lost_true_edges'] == 2


@pytest.mark.parametrize('values', [
    ([],[],[1.],[2.]), ([1.],[True],[],[]), ([1.],[1],[0.],[2.]),
    ([1.],[False],[0.],[2.]), ([-1.],[True],[0.],[2.]),
    ([1.],[True],[0.],[1.]), ([1.],[True],[np.nan],[2.]),
    ([1.],[True],[0.,1.],[2.,3.]), ([1.],[True],[0.,1.],[3.])])
def test_invalid_contexts_are_rejected(values):
    with pytest.raises(ValueError): compare_contexts(*values)
