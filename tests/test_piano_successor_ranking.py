"""Mask-safe grouping and direct gradients for local successor ranking."""
import copy

import numpy as np
import pytest

from scripts.piano_successor_ranking import (RankingGroup, prepare_successor_ranking,
    ranking_loss_from_groups, successor_ranking_loss)


def test_groups_ignore_unknown_edges_and_known_negative_edges_at_positive_tick():
    groups = prepare_successor_ranking(['a'] * 6, [120, 120, 120, 240, 360, 240],
        [1., 1., 0., 0., 1., 0.], [True, True, True, True, False, False])
    assert groups == (RankingGroup('a', (0, 1), ((3,),)),)


def test_group_order_is_by_source_then_negative_tick():
    groups = prepare_successor_ranking(['b', 'a', 'b', 'a', 'a', 'a'],
        [120, 120, 360, 360, 240, 240], [1, 1, 0, 0, 0, 0], [True] * 6)
    assert groups == (RankingGroup('a', (1,), ((4, 5), (3,))),
                      RankingGroup('b', (0,), ((2,),)))


@pytest.mark.parametrize(('targets', 'ticks', 'mask'), [
    ([0, 0], [120, 240], [True, True]),
    ([1, 1], [120, 120], [True, True]),
    ([1, 0], [120, 120], [True, True]),
    ([1, 0], [120, 240], [False, False]),
    ([1, 0], [120, 240], [True, False]),
])
def test_no_positive_or_other_tick_known_negative_skips_source(targets, ticks, mask):
    assert prepare_successor_ranking(['a', 'a'], ticks, targets, mask) == ()


def test_no_input_changes_and_prepared_groups_do_not_alias_input():
    values = [np.asarray(['a', 'a']), np.asarray([120, 240]),
              np.asarray([1., 0.]), np.asarray([True, True])]
    before = copy.deepcopy(values)
    prepared = prepare_successor_ranking(*values)
    for actual, expected in zip(values, before):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
    values[2][:] = 0
    assert prepared == (RankingGroup('a', (0,), ((1,),)),)


def test_multiple_known_positive_ticks_rejects_instead_of_relabelling():
    with pytest.raises(ValueError, match='multiple positive'):
        prepare_successor_ranking(['a', 'a'], [120, 240], [1, 1], [True, True])


@pytest.mark.parametrize('args', [
    ('a', [120], [1], [True]),
    ([['a']], [120], [1], [True]),
    (['a'], [[120]], [1], [True]),
    (['a'], [120], [[1]], [True]),
    (['a'], [120], [1], [[True]]),
    (['a'], [120], [1], [True, False]),
    ([''], [120], [1], [True]),
    (['a', 1], [120, 240], [1, 0], [True, True]),
    (['a'], [True], [1], [True]),
    (['a'], [120.], [1], [True]),
    (['a'], [120], [np.nan], [True]),
    (['a'], [120], [np.inf], [True]),
    (['a'], [120], [-1], [True]),
    (['a'], [120], [.5], [True]),
    (['a'], [120], [complex(1, 0)], [True]),
    (['a'], [120], ['1'], [True]),
    (['a'], [120], [1], [1]),
])
def test_invalid_metadata_shape_types_and_nonfinite_targets_are_rejected(args):
    with pytest.raises(ValueError):
        prepare_successor_ranking(*args)


def test_empty_metadata_has_no_groups():
    assert prepare_successor_ranking([], [], [], []) == ()


@pytest.fixture
def torch_module():
    return pytest.importorskip('torch')


def test_unknown_logits_and_same_tick_negative_have_zero_direct_gradient(torch_module):
    torch = torch_module
    logits = torch.tensor([.2, .3, 100., -.4, 90.], dtype=torch.float64, requires_grad=True)
    loss = successor_ranking_loss(logits, ['a'] * 5, [120, 120, 120, 240, 360],
                                  [1, 1, 0, 0, 1], [True, True, True, True, False])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0] < 0 and logits.grad[1] < 0 and logits.grad[3] > 0
    assert logits.grad[2] == 0 and logits.grad[4] == 0


@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_every_positive_chord_member_is_ranked_without_chord_size_weighting(torch_module, count):
    torch = torch_module
    logits = torch.tensor([0.] * count + [0.], requires_grad=True)
    loss = successor_ranking_loss(logits, ['a'] * (count+1), [120] * count + [240],
                                  [1] * count + [0], [True] * (count+1))
    loss.backward()
    assert float(loss.detach()) == pytest.approx(np.log(2))
    assert logits.grad[:-1].tolist() == pytest.approx([-.5/count] * count)
    assert float(logits.grad[-1]) == pytest.approx(.5)


def test_three_level_mean_and_maximum_match_independent_scalar_formula(torch_module):
    torch = torch_module
    scores = np.asarray([.3, -.1, .8, .2, -.4, -.2, 1.4])
    logits = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    loss = successor_ranking_loss(logits, ['a'] * 5 + ['b', 'b'],
        [120, 120, 240, 240, 360, 120, 240], [1, 1, 0, 0, 0, 1, 0], [True] * 7)
    first = np.mean([np.mean(np.logaddexp(0, .8-scores[:2])),
                     np.mean(np.logaddexp(0, -.4-scores[:2]))])
    expected = np.mean([first, np.logaddexp(0, 1.4-(-.2))])
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert logits.grad[2] > 0 and logits.grad[3] == 0


def test_prepared_entry_and_complete_entry_agree_and_preserve_labels(torch_module):
    torch = torch_module
    sources, ticks = np.asarray(['a', 'a']), np.asarray([120, 240])
    targets, mask = np.asarray([1., 0.]), np.asarray([True, True])
    before = (targets.copy(), mask.copy())
    groups = prepare_successor_ranking(sources, ticks, targets, mask)
    logits = torch.tensor([.3, .8], requires_grad=True)
    direct = successor_ranking_loss(logits, sources, ticks, targets, mask)
    prepared = ranking_loss_from_groups(logits, groups, expected_edges=2)
    assert torch.equal(direct, prepared)
    assert np.array_equal(targets, before[0]) and np.array_equal(mask, before[1])


def test_noneligible_sources_have_no_auxiliary_loss(torch_module):
    torch = torch_module
    logits = torch.tensor([.3, .8], requires_grad=True)
    assert successor_ranking_loss(logits, ['a', 'a'], [120, 240], [0, 0], [True, True]) is None
    assert ranking_loss_from_groups(torch.tensor([]), (), expected_edges=0) is None


def test_edge_permutation_preserves_loss_and_permuted_gradients(torch_module):
    torch = torch_module
    sources = np.asarray(['a', 'a', 'b', 'a', 'b'])
    ticks = np.asarray([120, 240, 120, 360, 240])
    labels = np.asarray([1, 0, 1, 0, 0]); mask = np.ones(5, dtype=bool)
    logits = torch.tensor([.3, .8, .7, -.1, .4], dtype=torch.float64, requires_grad=True)
    loss = successor_ranking_loss(logits, sources, ticks, labels, mask); loss.backward()
    order = np.asarray([4, 2, 0, 3, 1]); altered = logits.detach()[order].clone().requires_grad_()
    changed = successor_ranking_loss(altered, sources[order], ticks[order], labels[order], mask[order])
    changed.backward()
    assert float(loss.detach()) == pytest.approx(float(changed.detach()))
    assert torch.allclose(altered.grad, logits.grad[order])


@pytest.mark.parametrize('values', [[float('nan'), 0.], [0., float('inf')], [-float('inf'), 0.]])
def test_nonfinite_logits_are_rejected_even_for_no_groups(torch_module, values):
    with pytest.raises(ValueError, match='finite'):
        ranking_loss_from_groups(torch_module.tensor(values), (), expected_edges=2)


def test_wrong_logit_shape_dtype_and_edge_count_are_rejected(torch_module):
    torch = torch_module
    for logits, count in [(torch.tensor([[.1, .2]]), 2), (torch.tensor([1, 2]), 2),
                          (torch.tensor([.1, .2]), 1), (torch.tensor([.1]), True)]:
        with pytest.raises(ValueError):
            ranking_loss_from_groups(logits, (), expected_edges=count)


@pytest.mark.parametrize('groups', [
    (RankingGroup('a', (0,), ()),),
    (RankingGroup('a', (), ((1,),)),),
    (RankingGroup('a', (0,), ((0,),)),),
    (RankingGroup('a', (0,), ((2,),)),),
    (RankingGroup('a', (0,), ((True,),)),),
    (RankingGroup('a', (0,), ((1,),)), RankingGroup('a', (0,), ((1,),))),
])
def test_invalid_prepared_indices_do_not_silently_double_weight_edges(torch_module, groups):
    with pytest.raises(ValueError):
        ranking_loss_from_groups(torch_module.tensor([.1, .2]), groups, expected_edges=2)
