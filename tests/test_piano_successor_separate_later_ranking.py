"""Separate proved-only ranking must not dilute the frozen original term."""
import copy

import numpy as np
import pytest

from scripts.piano_successor_separate_later_ranking import (
    prepare_separate_later_ranking, separate_later_ranking_losses)
from scripts.piano_successor_ranking import (RankingGroup,
    prepare_successor_ranking, ranking_loss_from_groups, successor_ranking_loss)


def test_added_groups_contain_only_proved_edges_and_all_original_positives():
    args = (['a'] * 9, [120, 120, 120, 60, 240, 240, 360, 480, 360],
            [1, 1, 0, 0, 0, 0, 0, 0, 1],
            [True, True, True, True, True, False, False, False, False])
    original, added = prepare_separate_later_ranking(*args,
        [False, False, False, False, False, True, True, False, False])
    assert original == prepare_successor_ranking(*args) == (
        RankingGroup('a', (0, 1), ((3,), (4,))),)
    assert added == (RankingGroup('a', (0, 1), ((5,), (6,))),)


def test_new_source_without_old_negatives_only_enters_added_groups():
    groups = prepare_separate_later_ranking(
        ['a', 'a', 'b', 'b', 'b', 'b'], [120, 240, 120, 120, 120, 240],
        [1, 0, 1, 1, 0, 0], [True, True, True, True, True, False],
        [False, False, False, False, False, True])
    assert groups == ((RankingGroup('a', (0,), ((1,),)),),
                      (RankingGroup('b', (2, 3), ((5,),)),))


def test_sources_ticks_and_members_have_stable_order_in_each_term():
    original, added = prepare_separate_later_ranking(
        ['b', 'a', 'b', 'a', 'a', 'a', 'b', 'a', 'a', 'a'],
        [120, 240, 240, 120, 480, 480, 360, 360, 480, 480],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [True, True, True, True, False, True, False, False, True, False],
        [False, False, False, False, True, False, True, True, False, True])
    assert original == (RankingGroup('a', (1,), ((3,), (5, 8))),
                        RankingGroup('b', (0,), ((2,),)))
    assert added == (RankingGroup('a', (1,), ((7,), (4, 9))),
                     RankingGroup('b', (0,), ((6,),)))


@pytest.mark.parametrize('args', [
    ([], [], [], []),
    (['a', 'a'], [120, 120], [1, 0], [True, True]),
    (['a', 'a'], [120, 240], [0, 0], [True, True]),
    (['a', 'a'], [120, 240], [1, 0], [False, False]),
    (['a', 'a'], [120, 240], [1, 1], [True, False]),
    (['b', 'a', 'b', 'a'], [120, 120, 360, 240], [1, 1, 0, 0], [True] * 4),
])
def test_all_false_proof_preserves_original_and_has_no_added_groups(args):
    assert prepare_separate_later_ranking(*args, [False] * len(args[0])) == (
        prepare_successor_ranking(*args), ())


def test_preparation_does_not_mutate_inputs_or_keep_mutable_aliases():
    args = [np.asarray(['a'] * 6), np.asarray([120, 120, 240, 240, 360, 360]),
            np.asarray([1., 1., 0., 0., 0., 0.]),
            np.asarray([True, True, True, False, False, False]),
            np.asarray([False, False, False, True, True, True])]
    before = copy.deepcopy(args)
    result = prepare_separate_later_ranking(*args)
    for actual, expected in zip(args, before):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
    for array in args:
        array[:] = array[0]
    assert result == ((RankingGroup('a', (0, 1), ((2,),)),),
                      (RankingGroup('a', (0, 1), ((3,), (4, 5))),))


@pytest.mark.parametrize('derived', [True, [[False, True]], [False],
    [False, True, False], [0, 1], [False, 1], [False, 'true'], [False, None],
    [False, np.nan], [False, 0.]])
def test_derived_mask_rejects_wrong_shape_length_and_non_boolean_values(derived):
    with pytest.raises(ValueError, match='Derived later mask'):
        prepare_separate_later_ranking(['a', 'a'], [120, 240], [1, 0],
                                       [True, False], derived)


def test_numpy_boolean_arrays_are_accepted():
    assert prepare_separate_later_ranking(np.asarray(['a', 'a']),
        np.asarray([120, 240]), np.asarray([True, False]),
        np.asarray([True, False]), np.asarray([False, True])) == (
            (), (RankingGroup('a', (0,), ((1,),)),))


@pytest.mark.parametrize(('labels', 'mask', 'derived'), [
    ([1, 0], [True, True], [False, True]),
    ([1, 1], [True, False], [False, True]),
    ([1, 0], [True, False], [True, False]),
])
def test_proof_cannot_overlap_existing_supervision_or_positive_label(labels, mask, derived):
    with pytest.raises(ValueError, match='unsupervised zero-label'):
        prepare_separate_later_ranking(['a', 'a'], [120, 240], labels, mask, derived)


@pytest.mark.parametrize('tick', [0, 119, 120])
def test_added_negative_must_be_strictly_after_positive_tick(tick):
    with pytest.raises(ValueError, match='strictly later'):
        prepare_separate_later_ranking(['a', 'a'], [120, tick], [1, 0],
                                       [True, False], [False, True])


@pytest.mark.parametrize(('sources', 'labels', 'mask'), [
    (['a', 'a'], [0, 0], [True, False]),
    (['a', 'a'], [1, 0], [False, False]),
    (['a', 'b'], [1, 0], [True, False]),
])
def test_proof_requires_original_positive_for_its_own_source(sources, labels, mask):
    with pytest.raises(ValueError, match='requires a supervised positive'):
        prepare_separate_later_ranking(sources, [120, 240], labels, mask, [False, True])


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
def test_original_metadata_checks_are_not_bypassed(args):
    with pytest.raises(ValueError):
        prepare_separate_later_ranking(*args, [False] * len(args[0]))


@pytest.mark.parametrize('derived', [[False, False, False], [False, False, True]])
def test_multiple_original_positive_ticks_remain_an_error(derived):
    with pytest.raises(ValueError, match='multiple positive'):
        prepare_separate_later_ranking(['a'] * 3, [120, 240, 360], [1, 1, 0],
                                       [True, True, False], derived)


@pytest.fixture
def torch_module():
    return pytest.importorskip('torch')


def test_added_groups_do_not_change_original_loss_or_any_original_gradient(torch_module):
    torch = torch_module
    args = (['a'] * 8 + ['b', 'b', 'c', 'c'],
            [120, 120, 120, 240, 240, 360, 480, 480, 120, 240, 120, 240],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
            [True, True, True, True, False, False, False, False, True, True, True, False])
    derived = [False, False, False, False, True, True, False, False,
               False, False, False, True]
    old = torch.tensor([.3, -.1, 50., .8, 1.2, -.2, 100., 90., -.2, 1.4, .6, .9],
                       dtype=torch.float64, requires_grad=True)
    new = old.detach().clone().requires_grad_()
    old_loss = successor_ranking_loss(old, *args)
    original, added = separate_later_ranking_losses(new, *args, derived)
    old_loss.backward()
    original_gradient = torch.autograd.grad(original, new, retain_graph=True)[0]
    added_gradient = torch.autograd.grad(added, new, retain_graph=True)[0]
    (original + added).backward()
    assert torch.equal(old_loss, original)
    assert torch.equal(old.grad, original_gradient)
    assert torch.equal(new.grad[[3, 9]], old.grad[[3, 9]])
    assert torch.equal(added_gradient[[2, 3, 6, 7, 8, 9]],
                       torch.zeros(6, dtype=torch.float64))
    assert (added_gradient[[0, 1, 10]] < 0).all()
    assert (added_gradient[[4, 5, 11]] > 0).all()
    assert (new.grad[[0, 1]] < old.grad[[0, 1]]).all()


@pytest.mark.parametrize('positive_count', [1, 2, 3, 4])
def test_both_terms_average_every_positive_chord_member(torch_module, positive_count):
    torch = torch_module
    logits = torch.zeros(positive_count + 2, dtype=torch.float64, requires_grad=True)
    original, added = separate_later_ranking_losses(logits, ['a'] * len(logits),
        [120] * positive_count + [240, 240], [1] * positive_count + [0, 0],
        [True] * (positive_count + 1) + [False],
        [False] * (positive_count + 1) + [True])
    for loss, negative_index in ((original, positive_count), (added, positive_count + 1)):
        gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        assert float(loss.detach()) == pytest.approx(np.log(2))
        assert gradient[:positive_count].tolist() == pytest.approx([-.5 / positive_count] * positive_count)
        assert float(gradient[negative_index]) == pytest.approx(.5)
        assert gradient[positive_count + (negative_index == positive_count)] == 0


def test_original_and_added_terms_use_independent_source_tick_and_positive_means(torch_module):
    torch = torch_module
    values = np.asarray([.3, -.1, .8, .2, -.4, .5, .1, 1.2, -.2, 1.4, .7, -.5, .9])
    logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    original, added = separate_later_ranking_losses(logits,
        ['a'] * 8 + ['b', 'b', 'c', 'c', 'c'],
        [120, 120, 240, 240, 60, 240, 240, 360, 120, 240, 120, 240, 360],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [True, True, True, True, True, False, False, False, True, True, True, False, False],
        [False, False, False, False, False, True, True, True, False, False, False, True, True])
    old_a = np.mean([np.mean(np.logaddexp(0, -.4 - values[:2])),
                     np.mean(np.logaddexp(0, .8 - values[:2]))])
    new_a = np.mean([np.mean(np.logaddexp(0, .5 - values[:2])),
                     np.mean(np.logaddexp(0, 1.2 - values[:2]))])
    new_c = np.mean(np.logaddexp(0, np.asarray([-.5, .9]) - .7))
    assert float(original.detach()) == pytest.approx(np.mean([old_a, np.logaddexp(0, 1.6)]))
    assert float(added.detach()) == pytest.approx(np.mean([new_a, new_c]))
    old_gradient = torch.autograd.grad(original, logits, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(added, logits)[0]
    assert old_gradient[2] > 0 and old_gradient[3] == 0
    assert new_gradient[5] > 0 and new_gradient[6] == 0
    assert torch.equal(old_gradient[[5, 6, 7, 10, 11, 12]], torch.zeros(6, dtype=torch.float64))
    assert torch.equal(new_gradient[[2, 3, 4, 8, 9]], torch.zeros(5, dtype=torch.float64))


def test_new_sources_change_only_added_source_denominator(torch_module):
    torch = torch_module
    logits = torch.tensor([0., .3, .8, -.2, 1.4], dtype=torch.float64, requires_grad=True)
    args = (['a', 'a', 'a', 'b', 'b'], [120, 240, 360, 120, 240],
            [1, 0, 0, 1, 0], [True, True, False, True, False])
    old_before, new_before = separate_later_ranking_losses(logits, *args,
        [False, False, True, False, False])
    old_after, new_after = separate_later_ranking_losses(logits, *args,
        [False, False, True, False, True])
    assert torch.equal(old_before, old_after)
    assert torch.equal(torch.autograd.grad(old_before, logits, retain_graph=True)[0],
                       torch.autograd.grad(old_after, logits, retain_graph=True)[0])
    before_gradient = torch.autograd.grad(new_before, logits, retain_graph=True)[0]
    after_gradient = torch.autograd.grad(new_after, logits)[0]
    assert float(new_after.detach()) == pytest.approx(
        (float(new_before.detach()) + np.logaddexp(0, 1.6)) / 2)
    assert torch.equal(after_gradient[[0, 2]], before_gradient[[0, 2]] / 2)
    assert before_gradient[3] == before_gradient[4] == 0
    assert after_gradient[3] < 0 and after_gradient[4] > 0


def test_many_added_ticks_do_not_dilute_original_tick_or_source_mean(torch_module):
    torch = torch_module
    count = 31
    logits = torch.tensor([.2, -.4] + [float(i) / 10 for i in range(count)],
                          dtype=torch.float64, requires_grad=True)
    args = (['a'] * len(logits), [120, 60] + list(range(240, 240 + count)),
            [1] + [0] * (count + 1), [True, True] + [False] * count)
    base_original, _ = separate_later_ranking_losses(logits, *args,
        [False] * len(logits))
    original, added = separate_later_ranking_losses(logits, *args,
        [False, False] + [True] * count)
    assert torch.equal(original, base_original)
    assert torch.equal(torch.autograd.grad(original, logits, retain_graph=True)[0],
                       torch.autograd.grad(base_original, logits, retain_graph=True)[0])
    assert float(added.detach()) == pytest.approx(np.logaddexp(0, np.arange(count) / 10 - .2).mean())


@pytest.mark.parametrize(('old_scores', 'new_scores'), [([-.4, -.7], [.8, .2]),
                                                        ([.8, .2], [-.4, -.7])])
def test_original_and_added_members_at_same_tick_keep_their_own_maxima(torch_module, old_scores, new_scores):
    torch = torch_module
    logits = torch.tensor([0.] + old_scores + new_scores,
                          dtype=torch.float64, requires_grad=True)
    original, added = separate_later_ranking_losses(logits, ['a'] * 5,
        [120, 240, 240, 240, 240], [1, 0, 0, 0, 0],
        [True, True, True, False, False], [False, False, False, True, True])
    assert float(original.detach()) == pytest.approx(np.logaddexp(0, max(old_scores)))
    assert float(added.detach()) == pytest.approx(np.logaddexp(0, max(new_scores)))
    (original + added).backward()
    assert logits.grad[1] > 0 and logits.grad[3] > 0
    assert logits.grad[2] == logits.grad[4] == 0


def test_unknown_and_same_tick_known_edges_have_no_direct_ranking_gradient(torch_module):
    torch = torch_module
    logits = torch.tensor([.2, .3, 100., -.4, .5, 90., 80.],
                          dtype=torch.float64, requires_grad=True)
    losses = separate_later_ranking_losses(logits, ['a'] * 7,
        [120, 120, 120, 60, 240, 360, 240], [1, 1, 0, 0, 0, 0, 1],
        [True, True, True, True, False, False, False],
        [False, False, False, False, True, False, False])
    for loss in losses:
        gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        assert torch.equal(gradient[[2, 5, 6]], torch.zeros(3, dtype=torch.float64))
        assert (gradient[:2] < 0).all()


def test_duplicate_equal_negatives_do_not_multiply_either_term(torch_module):
    torch = torch_module
    single = torch.tensor([.2, .8, -.4], dtype=torch.float64, requires_grad=True)
    repeated = torch.tensor([.2, .8, .8, .8, -.4, -.4, -.4],
                            dtype=torch.float64, requires_grad=True)
    small = separate_later_ranking_losses(single, ['a'] * 3, [120, 240, 240],
        [1, 0, 0], [True, True, False], [False, False, True])
    large = separate_later_ranking_losses(repeated, ['a'] * 7, [120] + [240] * 6,
        [1] + [0] * 6, [True] * 4 + [False] * 3, [False] * 4 + [True] * 3)
    for index, (one, many) in enumerate(zip(small, large)):
        one_gradient = torch.autograd.grad(one, single, retain_graph=True)[0]
        many_gradient = torch.autograd.grad(many, repeated, retain_graph=True)[0]
        assert torch.equal(one, many)
        assert torch.equal(one_gradient[0], many_gradient[0])
        start = 1 + 3 * index
        assert float(many_gradient[start:start + 3].sum()) == pytest.approx(float(one_gradient[index + 1]))
        assert many_gradient[start:start + 3].tolist() == pytest.approx([float(one_gradient[index + 1]) / 3] * 3)


def test_no_hard_mining_drops_easy_added_groups(torch_module):
    torch = torch_module
    logits = torch.tensor([0., 1., -12.], dtype=torch.float64, requires_grad=True)
    original, added = separate_later_ranking_losses(logits, ['a'] * 3,
        [120, 240, 360], [1, 0, 0], [True, False, False], [False, True, True])
    assert original is None
    assert float(added.detach()) == pytest.approx(
        (np.logaddexp(0, 1.) + np.logaddexp(0, -12.)) / 2)
    added.backward()
    assert logits.grad[2] > 0


@pytest.mark.parametrize(('mask', 'derived', 'present'), [
    ([True, False, False], [False, False, False], (False, False)),
    ([True, True, False], [False, False, False], (True, False)),
    ([True, False, False], [False, False, True], (False, True)),
    ([True, True, False], [False, False, True], (True, True)),
])
def test_each_absent_term_returns_none_independently(torch_module, mask, derived, present):
    torch = torch_module
    logits = torch.tensor([.2, .8, -.4], dtype=torch.float64)
    args = (['a'] * 3, [120, 240, 360], [1, 0, 0], mask, derived)
    losses = separate_later_ranking_losses(logits, *args)
    groups = prepare_separate_later_ranking(*args)
    for loss, prepared, expected in zip(losses, groups, present):
        assert (loss is not None) == expected
        if expected:
            assert torch.equal(loss, ranking_loss_from_groups(logits, prepared, expected_edges=3))
        else:
            assert prepared == ()


def test_empty_inputs_return_two_absent_terms(torch_module):
    assert separate_later_ranking_losses(torch_module.zeros(0), [], [], [], [], []) == (None, None)


def test_permuting_edges_preserves_each_loss_and_gradient_mapping(torch_module):
    torch = torch_module
    sources = np.asarray(['a', 'a', 'b', 'a', 'b', 'a', 'a', 'a'])
    ticks = np.asarray([120, 240, 120, 360, 240, 120, 120, 240])
    labels = np.asarray([1, 0, 1, 0, 0, 1, 0, 0])
    mask = np.asarray([True, True, True, False, False, True, True, False])
    derived = np.asarray([False, False, False, True, True, False, False, True])
    logits = torch.tensor([.3, .8, .7, -.1, .4, .2, 90., 1.1],
                          dtype=torch.float64, requires_grad=True)
    losses = separate_later_ranking_losses(logits, sources, ticks, labels, mask, derived)
    order = np.asarray([4, 2, 0, 5, 3, 1, 7, 6])
    changed = logits.detach()[order].clone().requires_grad_()
    moved = separate_later_ranking_losses(changed, sources[order], ticks[order],
                                         labels[order], mask[order], derived[order])
    for original, reordered in zip(losses, moved):
        gradient = torch.autograd.grad(original, logits, retain_graph=True)[0]
        moved_gradient = torch.autograd.grad(reordered, changed, retain_graph=True)[0]
        assert float(original.detach()) == pytest.approx(float(reordered.detach()))
        assert torch.allclose(moved_gradient, gradient[order])


@pytest.mark.parametrize('values', [
    [float('nan'), 0.], [float('inf'), 0.], [0., -float('inf')],
    [0.], [[0., 0.]], [0, 0], [True, False],
])
@pytest.mark.parametrize('has_groups', [False, True])
def test_invalid_logits_are_rejected_even_when_both_terms_are_empty(torch_module, values, has_groups):
    torch = torch_module
    with pytest.raises(ValueError):
        separate_later_ranking_losses(torch.tensor(values), ['a', 'a'], [120, 240],
            [1, 0], [True, False], [False, has_groups])


def test_non_tensor_logits_are_rejected_even_with_no_groups(torch_module):
    with pytest.raises(ValueError):
        separate_later_ranking_losses(np.asarray([.2, .8]), ['a', 'a'], [120, 240],
                                      [1, 0], [True, False], [False, False])


@pytest.mark.parametrize('term', ['original', 'added', 'weighted_sum'])
def test_autograd_matches_independent_finite_differences(torch_module, term):
    torch = torch_module
    values = np.asarray([.3, -.1, .8, .2, -.4, .5, .1, 1.2, -.2, 1.4, .7, -.5, .9, 99.])
    logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    losses = separate_later_ranking_losses(logits,
        ['a'] * 8 + ['b', 'b', 'c', 'c', 'c', 'a'],
        [120, 120, 240, 240, 60, 240, 240, 360, 120, 240, 120, 240, 360, 120],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [True, True, True, True, True, False, False, False, True, True, True, False, False, True],
        [False, False, False, False, False, True, True, True, False, False, False, True, True, False])

    def independent(vector):
        old_a = np.mean([np.mean(np.logaddexp(0, vector[4] - vector[:2])),
                        np.mean(np.logaddexp(0, max(vector[2:4]) - vector[:2]))])
        old = np.mean([old_a, np.logaddexp(0, vector[9] - vector[8])])
        new_a = np.mean([np.mean(np.logaddexp(0, max(vector[5:7]) - vector[:2])),
                        np.mean(np.logaddexp(0, vector[7] - vector[:2]))])
        new = np.mean([new_a, np.mean(np.logaddexp(0, vector[11:13] - vector[10]))])
        return {'original': old, 'added': new, 'weighted_sum': .1 * old + .1 * new}[term]

    result = {'original': losses[0], 'added': losses[1],
              'weighted_sum': .1 * losses[0] + .1 * losses[1]}[term]
    result.backward()
    step = 1e-6
    numerical = []
    for index in range(len(values)):
        plus, minus = values.copy(), values.copy()
        plus[index] += step
        minus[index] -= step
        numerical.append((independent(plus) - independent(minus)) / (2 * step))
    assert float(result.detach()) == pytest.approx(independent(values), abs=1e-12)
    assert np.allclose(logits.grad.numpy(), numerical, rtol=1e-6, atol=2e-10)
