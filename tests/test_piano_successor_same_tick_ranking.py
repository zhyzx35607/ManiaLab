"""Known same-tick negatives, retained chord positives and isolated gradients."""
import copy

import numpy as np
import pytest

from scripts.piano_successor_ranking import (RankingGroup, prepare_successor_ranking,
    ranking_loss_from_groups, successor_ranking_loss)
from scripts.piano_successor_same_tick_ranking import (prepare_same_tick_ranking,
    same_tick_ranking_loss)


def test_same_tick_negatives_form_one_extra_group_and_preserve_all_positives():
    args = (['a'] * 8, [120, 120, 120, 120, 240, 240, 360, 120],
            [1, 1, 0, 0, 0, 0, 1, 0], [True, True, True, True, True, True, False, False])
    old = prepare_successor_ranking(*args)
    assert old == (RankingGroup('a', (0, 1), ((4, 5),)),)
    new = prepare_same_tick_ranking(*args)
    assert new == (RankingGroup('a', (0, 1), ((2, 3), (4, 5))),)
    assert old[0].negative_groups[0] in new[0].negative_groups


def test_same_tick_only_negatives_make_previously_skipped_source_eligible():
    args = (['a'] * 3, [120] * 3, [1, 1, 0], [True] * 3)
    assert prepare_successor_ranking(*args) == ()
    assert prepare_same_tick_ranking(*args) == (RankingGroup('a', (0, 1), ((2,),)),)


def test_group_order_is_source_then_target_tick_with_existing_groups_retained():
    groups = prepare_same_tick_ranking(['b', 'a', 'b', 'a', 'a', 'a', 'a'],
        [120, 240, 120, 120, 240, 360, 120], [1, 1, 0, 0, 0, 0, 0], [True] * 7)
    assert groups == (RankingGroup('a', (1,), ((3, 6), (4,), (5,))),
                      RankingGroup('b', (0,), ((2,),)))


@pytest.mark.parametrize(('labels', 'mask'), [
    ([0, 0], [True, True]),
    ([1, 1], [True, True]),
    ([1, 0], [False, False]),
    ([1, 0], [True, False]),
    ([1, 0], [False, True]),
])
def test_no_known_positive_or_no_known_negative_skips_source(labels, mask):
    assert prepare_same_tick_ranking(['a', 'a'], [120, 120], labels, mask) == ()


def test_empty_inputs_skip():
    assert prepare_same_tick_ranking([], [], [], []) == ()


def test_no_input_mutation_and_no_prepared_aliases():
    args = [np.asarray(['a'] * 4), np.asarray([120, 120, 240, 120]),
            np.asarray([1., 0., 0., 0.]), np.asarray([True, True, True, False])]
    before = copy.deepcopy(args)
    groups = prepare_same_tick_ranking(*args)
    for actual, expected in zip(args, before):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
    args[1][:] = 0
    args[2][:] = 1
    assert groups == (RankingGroup('a', (0,), ((1,), (2,))),)


@pytest.mark.parametrize('args', [
    (['a'], [120.], [1], [True]),
    (['a'], [120], [np.nan], [True]),
    (['a'], [120], [1], [1]),
    (['a'], [120], [1], [True, False]),
    ([['a']], [120], [1], [True]),
])
def test_frozen_metadata_validation_is_not_bypassed(args):
    with pytest.raises(ValueError):
        prepare_same_tick_ranking(*args)


def test_multiple_positive_ticks_are_not_merged_or_relabelled():
    with pytest.raises(ValueError, match='multiple positive'):
        prepare_same_tick_ranking(['a'] * 3, [120, 240, 120], [1, 1, 0], [True] * 3)


@pytest.fixture
def torch_module():
    return pytest.importorskip('torch')


def test_unknowns_have_zero_gradient_while_same_tick_negative_is_penalized(torch_module):
    torch = torch_module
    logits = torch.tensor([.2, .3, .8, -.4, 100., 90., 80.],
                          dtype=torch.float64, requires_grad=True)
    loss = same_tick_ranking_loss(logits, ['a'] * 7,
        [120, 120, 120, 240, 120, 360, 120], [1, 1, 0, 0, 0, 1, 1],
        [True, True, True, True, False, False, False])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert (logits.grad[:2] < 0).all() and (logits.grad[2:4] > 0).all()
    assert torch.equal(logits.grad[4:], torch.zeros(3, dtype=torch.float64))


@pytest.mark.parametrize('positive_count', [1, 2, 3, 4])
def test_every_chord_positive_participates_against_same_tick_negatives(torch_module, positive_count):
    torch = torch_module
    logits = torch.zeros(positive_count+1, dtype=torch.float64, requires_grad=True)
    loss = same_tick_ranking_loss(logits, ['a'] * len(logits), [120] * len(logits),
        [1] * positive_count + [0], [True] * len(logits))
    loss.backward()
    assert float(loss.detach()) == pytest.approx(np.log(2))
    assert logits.grad[:-1].tolist() == pytest.approx([-.5/positive_count] * positive_count)
    assert float(logits.grad[-1]) == pytest.approx(.5)


def test_added_group_changes_mean_denominator_without_an_extra_coefficient(torch_module):
    torch = torch_module
    logits = torch.tensor([0., np.log(3), 0.], dtype=torch.float64, requires_grad=True)
    args = (['a'] * 3, [120, 120, 240], [1, 0, 0], [True] * 3)
    old = successor_ranking_loss(logits, *args)
    loss = same_tick_ranking_loss(logits, *args)
    assert float(old.detach()) == pytest.approx(np.log(2))
    assert float(loss.detach()) == pytest.approx((np.log(4)+np.log(2))/2)
    loss.backward()
    assert logits.grad.tolist() == pytest.approx([-.625, .375, .25])


def test_same_tick_group_uses_maximum_negative_and_preserves_mean_over_sources(torch_module):
    torch = torch_module
    scores = np.asarray([.3, -.1, .8, .2, -.4, -.2, 1.4])
    logits = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    loss = same_tick_ranking_loss(logits, ['a'] * 5 + ['b', 'b'],
        [120, 120, 120, 120, 240, 120, 120], [1, 1, 0, 0, 0, 1, 0], [True] * 7)
    a = np.mean([np.mean(np.logaddexp(0, .8-scores[:2])),
                 np.mean(np.logaddexp(0, -.4-scores[:2]))])
    expected = np.mean([a, np.logaddexp(0, 1.4-(-.2))])
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert logits.grad[2] > 0 and logits.grad[3] == 0


def test_without_same_tick_negatives_loss_and_all_gradients_equal_frozen_objective(torch_module):
    torch = torch_module
    args = (['a'] * 5 + ['b', 'b'], [120, 120, 240, 360, 120, 120, 240],
            [1, 1, 0, 0, 0, 1, 0], [True, True, True, True, False, True, True])
    assert prepare_same_tick_ranking(*args) == prepare_successor_ranking(*args)
    scores = [.3, -.1, .8, .2, 100., -.2, 1.4]
    old = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    new = old.detach().clone().requires_grad_()
    old_loss = successor_ranking_loss(old, *args)
    new_loss = same_tick_ranking_loss(new, *args)
    old_loss.backward(); new_loss.backward()
    assert torch.equal(old_loss, new_loss)
    assert torch.equal(old.grad, new.grad)


def test_prepared_groups_use_frozen_loss_and_skipped_sources_have_no_loss(torch_module):
    torch = torch_module
    logits = torch.tensor([.3, .8], requires_grad=True)
    args = (['a', 'a'], [120, 120], [1, 0], [True, True])
    groups = prepare_same_tick_ranking(*args)
    assert torch.equal(same_tick_ranking_loss(logits, *args),
                       ranking_loss_from_groups(logits, groups, expected_edges=2))
    assert same_tick_ranking_loss(logits, ['a', 'a'], [120, 120], [1, 1], [True, True]) is None
    assert same_tick_ranking_loss(logits, ['a', 'a'], [120, 120], [0, 0], [True, True]) is None


def test_edge_permutation_preserves_loss_and_gradient_correspondence(torch_module):
    torch = torch_module
    sources = np.asarray(['a', 'a', 'b', 'a', 'b', 'a'])
    ticks = np.asarray([120, 120, 120, 360, 120, 120])
    labels = np.asarray([1, 0, 1, 0, 0, 1]); mask = np.ones(6, dtype=bool)
    logits = torch.tensor([.3, .8, .7, -.1, .4, .2], dtype=torch.float64, requires_grad=True)
    loss = same_tick_ranking_loss(logits, sources, ticks, labels, mask); loss.backward()
    order = np.asarray([4, 2, 0, 5, 3, 1]); changed = logits.detach()[order].clone().requires_grad_()
    moved_loss = same_tick_ranking_loss(changed, sources[order], ticks[order], labels[order], mask[order])
    moved_loss.backward()
    assert float(loss.detach()) == pytest.approx(float(moved_loss.detach()))
    assert torch.allclose(changed.grad, logits.grad[order])
