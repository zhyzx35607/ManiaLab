"""Proved later negatives, frozen baseline equivalence and gradient isolation."""
import copy

import numpy as np
import pytest

from scripts.piano_successor_later_ranking import prepare_later_ranking, later_ranking_loss
from scripts.piano_successor_ranking import (RankingGroup, prepare_successor_ranking,
    ranking_loss_from_groups, successor_ranking_loss)


def test_later_edges_merge_by_tick_and_keep_all_positive_chord_members():
    args = (['a'] * 8, [120, 120, 120, 240, 240, 360, 360, 480],
            [1, 1, 0, 0, 0, 0, 0, 1],
            [True, True, True, True, False, False, False, False])
    assert prepare_successor_ranking(*args) == (RankingGroup('a', (0, 1), ((3,),)),)
    groups = prepare_later_ranking(*args, [False] * 4 + [True, True, True, False])
    assert groups == (RankingGroup('a', (0, 1), ((3, 4), (5, 6))),)


def test_added_negative_can_qualify_source_without_old_other_tick_negatives():
    args = (['a'] * 4, [120, 120, 120, 240], [1, 1, 0, 0], [True, True, True, False])
    assert prepare_successor_ranking(*args) == ()
    assert prepare_later_ranking(*args, [False, False, False, True]) == (
        RankingGroup('a', (0, 1), ((3,),)),)


def test_sources_ticks_and_member_indices_are_sorted_with_old_earlier_groups_retained():
    groups = prepare_later_ranking(
        ['b', 'a', 'b', 'a', 'a', 'a', 'b', 'a'],
        [120, 240, 240, 120, 480, 480, 360, 360],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [True, True, True, True, False, True, False, False],
        [False, False, False, False, True, False, True, True])
    assert groups == (RankingGroup('a', (1,), ((3,), (7,), (4, 5))),
                      RankingGroup('b', (0,), ((2,), (6,))))


@pytest.mark.parametrize('args', [
    ([], [], [], []),
    (['a', 'a'], [120, 120], [1, 0], [True, True]),
    (['a', 'a'], [120, 240], [0, 0], [True, True]),
    (['a', 'a'], [120, 240], [1, 0], [False, False]),
    (['a', 'a'], [120, 240], [1, 1], [True, False]),
    (['b', 'a', 'b', 'a'], [120, 120, 360, 240], [1, 1, 0, 0], [True] * 4),
])
def test_all_false_proof_returns_frozen_prepared_groups(args):
    assert prepare_later_ranking(*args, [False] * len(args[0])) == prepare_successor_ranking(*args)


def test_preparation_does_not_mutate_inputs_or_keep_mutable_aliases():
    args = [np.asarray(['a'] * 5), np.asarray([120, 120, 240, 240, 360]),
            np.asarray([1., 1., 0., 0., 0.]), np.asarray([True, True, True, False, False]),
            np.asarray([False, False, False, True, True])]
    before = copy.deepcopy(args)
    result = prepare_later_ranking(*args)
    for actual, expected in zip(args, before):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
    for array in args:
        array[:] = array[0]
    assert result == (RankingGroup('a', (0, 1), ((2, 3), (4,))),)


@pytest.mark.parametrize('derived', [True, [[False, True]], [False], [False, True, False],
    [0, 1], [False, 1], [False, 'true'], [False, None], [False, np.nan], [False, 0.]])
def test_derived_mask_rejects_wrong_shape_length_and_non_boolean_values(derived):
    with pytest.raises(ValueError, match='Derived later mask'):
        prepare_later_ranking(['a', 'a'], [120, 240], [1, 0], [True, False], derived)


def test_numpy_boolean_mask_is_accepted_without_numeric_coercion():
    assert prepare_later_ranking(np.asarray(['a', 'a']), np.asarray([120, 240]),
        np.asarray([1, 0]), np.asarray([True, False]), np.asarray([False, True])) == (
            RankingGroup('a', (0,), ((1,),)),)


@pytest.mark.parametrize(('labels', 'mask', 'derived'), [
    ([1, 0], [True, True], [False, True]),
    ([1, 1], [True, False], [False, True]),
    ([1, 0], [True, False], [True, False]),
])
def test_proof_cannot_overlap_existing_supervision_or_positive_label(labels, mask, derived):
    with pytest.raises(ValueError, match='unsupervised zero-label'):
        prepare_later_ranking(['a', 'a'], [120, 240], labels, mask, derived)


@pytest.mark.parametrize('tick', [0, 119, 120])
def test_proved_negative_must_be_strictly_after_positive_tick(tick):
    with pytest.raises(ValueError, match='strictly later'):
        prepare_later_ranking(['a', 'a'], [120, tick], [1, 0],
                              [True, False], [False, True])


@pytest.mark.parametrize(('sources', 'labels', 'mask'), [
    (['a', 'a'], [0, 0], [True, False]),
    (['a', 'a'], [1, 0], [False, False]),
    (['a', 'b'], [1, 0], [True, False]),
])
def test_proof_requires_positive_for_its_own_source(sources, labels, mask):
    with pytest.raises(ValueError, match='requires a supervised positive'):
        prepare_later_ranking(sources, [120, 240], labels, mask, [False, True])


@pytest.mark.parametrize('args', [
    (['a'], [120.], [1], [True]),
    (['a'], [True], [1], [True]),
    (['a'], [120], [np.nan], [True]),
    (['a'], [120], [1], [1]),
    (['a'], [120], [1], [True, False]),
    ([['a']], [120], [1], [True]),
    ([''], [120], [1], [True]),
])
def test_original_metadata_checks_are_not_bypassed(args):
    with pytest.raises(ValueError):
        prepare_later_ranking(*args, [False] * len(args[0]))


def test_multiple_positive_ticks_remain_an_error_even_with_a_later_proof():
    with pytest.raises(ValueError, match='multiple positive'):
        prepare_later_ranking(['a'] * 3, [120, 240, 360], [1, 1, 0],
                              [True, True, False], [False, False, True])


@pytest.fixture
def torch_module():
    return pytest.importorskip('torch')


def test_all_false_proof_exactly_preserves_loss_and_every_gradient(torch_module):
    torch = torch_module
    args = (['a'] * 6 + ['b', 'b'], [120, 120, 120, 240, 360, 480, 120, 240],
            [1, 1, 0, 0, 0, 1, 1, 0], [True, True, True, True, False, False, True, True])
    old = torch.tensor([.3, -.1, 50., .8, 100., 90., -.2, 1.4],
                       dtype=torch.float64, requires_grad=True)
    new = old.detach().clone().requires_grad_()
    old_loss = successor_ranking_loss(old, *args)
    new_loss = later_ranking_loss(new, *args, [False] * len(old))
    old_loss.backward(); new_loss.backward()
    assert torch.equal(old_loss, new_loss)
    assert torch.equal(old.grad, new.grad)


def test_proved_negatives_get_gradient_but_unknown_and_same_tick_known_edges_do_not(torch_module):
    torch = torch_module
    logits = torch.tensor([.2, .3, 100., -.4, .5, .8, 90., 80.],
                          dtype=torch.float64, requires_grad=True)
    loss = later_ranking_loss(logits, ['a'] * 8,
        [120, 120, 120, 60, 240, 360, 480, 360], [1, 1, 0, 0, 0, 0, 0, 1],
        [True, True, True, True, True, False, False, False],
        [False, False, False, False, False, True, False, False])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert (logits.grad[:2] < 0).all() and (logits.grad[3:6] > 0).all()
    assert torch.equal(logits.grad[[2, 6, 7]], torch.zeros(3, dtype=torch.float64))


@pytest.mark.parametrize('positive_count', [1, 2, 3, 4])
def test_every_positive_chord_member_receives_gradient(torch_module, positive_count):
    torch = torch_module
    logits = torch.zeros(positive_count+1, dtype=torch.float64, requires_grad=True)
    loss = later_ranking_loss(logits, ['a'] * len(logits), [120] * positive_count + [240],
        [1] * positive_count + [0], [True] * positive_count + [False],
        [False] * positive_count + [True])
    loss.backward()
    assert float(loss.detach()) == pytest.approx(np.log(2))
    assert logits.grad[:-1].tolist() == pytest.approx([-.5/positive_count] * positive_count)
    assert float(logits.grad[-1]) == pytest.approx(.5)


def test_duplicate_equal_negative_members_do_not_multiply_group_contribution(torch_module):
    torch = torch_module
    single = torch.tensor([.2, .8], dtype=torch.float64, requires_grad=True)
    repeated = torch.tensor([.2, .8, .8, .8], dtype=torch.float64, requires_grad=True)
    one = later_ranking_loss(single, ['a'] * 2, [120, 240], [1, 0],
                             [True, False], [False, True])
    many = later_ranking_loss(repeated, ['a'] * 4, [120, 240, 240, 240], [1, 0, 0, 0],
                              [True, False, False, False], [False, True, True, True])
    one.backward(); many.backward()
    assert torch.equal(one, many)
    assert torch.equal(single.grad[0], repeated.grad[0])
    assert float(repeated.grad[1:].sum()) == pytest.approx(float(single.grad[1]))


def test_existing_and_derived_negatives_at_one_tick_use_one_maximum(torch_module):
    torch = torch_module
    logits = torch.tensor([0., -.4, .8, 0.], dtype=torch.float64, requires_grad=True)
    loss = later_ranking_loss(logits, ['a'] * 4, [120, 240, 240, 360], [1, 0, 0, 0],
        [True, True, False, False], [False, False, True, True])
    assert float(loss.detach()) == pytest.approx((np.logaddexp(0, .8)+np.log(2))/2)
    loss.backward()
    assert logits.grad[1] == 0 and (logits.grad[2:] > 0).all()


def test_sources_ticks_and_all_positives_keep_the_frozen_nested_means(torch_module):
    torch = torch_module
    values = np.asarray([.3, -.1, .8, .2, -.4, -.2, 1.4])
    logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    loss = later_ranking_loss(logits, ['a'] * 5 + ['b', 'b'],
        [120, 120, 240, 240, 360, 120, 240], [1, 1, 0, 0, 0, 1, 0],
        [True, True, True, False, False, True, False],
        [False, False, False, True, True, False, True])
    a = np.mean([np.mean(np.logaddexp(0, .8-values[:2])),
                 np.mean(np.logaddexp(0, -.4-values[:2]))])
    assert float(loss.detach()) == pytest.approx(np.mean([a, np.logaddexp(0, 1.6)]))
    loss.backward()
    assert logits.grad[2] > 0 and logits.grad[3] == 0


def test_wrapper_uses_frozen_loss_and_preserves_no_loss_for_unqualified_sources(torch_module):
    torch = torch_module
    logits = torch.tensor([.3, .8], requires_grad=True)
    args = (['a', 'a'], [120, 240], [1, 0], [True, False], [False, True])
    groups = prepare_later_ranking(*args)
    assert torch.equal(later_ranking_loss(logits, *args),
                       ranking_loss_from_groups(logits, groups, expected_edges=2))
    assert later_ranking_loss(logits, *args[:-1], [False, False]) is None
    assert later_ranking_loss(torch.zeros(0), [], [], [], [], []) is None


def test_edge_permutation_preserves_loss_and_gradient_mapping(torch_module):
    torch = torch_module
    sources = np.asarray(['a', 'a', 'b', 'a', 'b', 'a', 'a'])
    ticks = np.asarray([120, 240, 120, 360, 240, 120, 120])
    labels = np.asarray([1, 0, 1, 0, 0, 1, 0])
    mask = np.asarray([True, True, True, False, False, True, True])
    derived = np.asarray([False, False, False, True, True, False, False])
    logits = torch.tensor([.3, .8, .7, -.1, .4, .2, 90.], dtype=torch.float64, requires_grad=True)
    loss = later_ranking_loss(logits, sources, ticks, labels, mask, derived); loss.backward()
    order = np.asarray([4, 2, 0, 5, 3, 1, 6])
    changed = logits.detach()[order].clone().requires_grad_()
    moved = later_ranking_loss(changed, sources[order], ticks[order], labels[order], mask[order], derived[order])
    moved.backward()
    assert float(loss.detach()) == pytest.approx(float(moved.detach()))
    assert torch.allclose(changed.grad, logits.grad[order])


@pytest.mark.parametrize('values', [[float('nan'), 0.], [float('inf'), 0.], [0.], [[0., 0.]]])
def test_logits_validation_still_runs_through_frozen_loss(torch_module, values):
    torch = torch_module
    with pytest.raises(ValueError):
        later_ranking_loss(torch.tensor(values), ['a', 'a'], [120, 240], [1, 0],
                           [True, False], [False, True])
