"""Finite decoding selects an attack group without inventing edges or labels."""
import inspect

import numpy as np
import pytest

from scripts.piano_successor_decode import select_successor_groups


def test_best_tick_keeps_all_its_positive_chord_edges_without_promoting_negative():
    result = select_successor_groups(['a'] * 5, [120, 240, 240, 240, 360],
                                     [.4, 1.5, .1, -.1, .5])
    assert result.tolist() == [False, True, True, False, False]
    assert result.dtype == np.dtype(bool) and result.shape == (5,)


def test_group_score_is_maximum_not_sum_or_number_of_edges():
    result = select_successor_groups(['a'] * 4, [120, 120, 120, 240],
                                     [.5, .5, .5, .6])
    assert result.tolist() == [False, False, False, True]


def test_source_ids_separate_independent_choices():
    result = select_successor_groups(['a', 'b', 'a', 'b'], [120, 120, 240, 240],
                                     [.1, .9, .8, .2])
    assert result.tolist() == [False, True, True, False]


def test_edges_for_same_source_from_different_windows_compete_together():
    sources = np.concatenate([['a', 'b'], ['a', 'b']])
    ticks = np.concatenate([[120, 120], [960, 960]])
    scores = np.concatenate([[.5, .9], [.8, .1]])
    assert select_successor_groups(sources, ticks, scores).tolist() == [False, True, True, False]


def test_unlabelled_high_score_competes_and_can_remove_supervised_positive():
    # No label or mask reaches the decoder: the later unlabelled edge wins.
    assert select_successor_groups(['a', 'a'], [120, 960], [.4, .8]).tolist() == [False, True]


def test_sources_without_positive_scores_do_not_get_forced_successors():
    assert not select_successor_groups(['a', 'a', 'b'], [120, 240, 120],
                                      [-.1, -2., -1.]).any()


def test_zero_and_negative_zero_are_positive_at_the_fixed_threshold():
    result = select_successor_groups(['a'] * 4, [240, 120, 120, 120],
                                     [0., -0., 0., -1e-15])
    assert result.tolist() == [False, True, True, False]


def test_equal_scores_choose_earlier_tick_then_keep_its_positive_edges():
    result = select_successor_groups(['a'] * 4, [360, 120, 240, 120],
                                     [1., 1., 1., .01])
    assert result.tolist() == [False, True, False, True]


def test_permutations_preserve_edge_identity_decisions_and_return_input_order():
    sources = np.asarray(['a', 'b', 'a', 'b', 'a', 'b', 'c'])
    ticks = np.asarray([120, 360, 240, 120, 240, 120, 0])
    scores = np.asarray([.8, .9, 1., .9, .01, -.5, -1.])
    expected = np.asarray([False, False, True, True, True, False, False])
    random = np.random.default_rng(83)
    for _ in range(40):
        order = random.permutation(len(sources))
        actual = select_successor_groups(sources[order], ticks[order], scores[order])
        assert np.array_equal(actual, expected[order])


def test_input_arrays_remain_unchanged_and_may_be_read_only():
    sources = np.asarray(['a', 'a', 'b'])
    ticks = np.asarray([240, 120, 0], dtype=np.int32)
    scores = np.asarray([1., 1., -2.], dtype=np.float32)
    snapshots = [value.copy() for value in (sources, ticks, scores)]
    for value in (sources, ticks, scores):
        value.flags.writeable = False
    assert select_successor_groups(sources, ticks, scores).tolist() == [False, True, False]
    for value, before in zip((sources, ticks, scores), snapshots):
        assert value.dtype == before.dtype and np.array_equal(value, before)


def test_selection_can_only_delete_existing_positive_edges():
    random = np.random.default_rng(29)
    sources = np.asarray(['source-' + str(i % 17) for i in range(300)])
    ticks = random.integers(-10, 30, size=len(sources))
    scores = random.standard_normal(len(sources))
    selected = select_successor_groups(sources, ticks, scores)
    assert not (selected & (scores < 0)).any()
    for source in set(sources):
        assert len(set(ticks[selected & (sources == source)])) <= 1
    assert select_successor_groups(sources[selected], ticks[selected], scores[selected]).all()


def test_empty_input_returns_empty_boolean_mask():
    result = select_successor_groups([], [], [])
    assert result.shape == (0,) and result.dtype == np.dtype(bool)


def test_large_unsigned_tick_ties_do_not_overflow_or_lose_integer_precision():
    ticks = np.asarray([2**64 - 1, 2**64 - 2], dtype=np.uint64)
    assert select_successor_groups(['a', 'a'], ticks, [.2, .2]).tolist() == [False, True]


@pytest.mark.parametrize('args', [
    ('a', [1], [.1]),
    (['a'], 1, [.1]),
    (['a'], [1], .1),
    ([['a']], [1], [.1]),
    (['a'], [[1]], [.1]),
    (['a'], [1], [[.1]]),
    (['a', 'b'], [1], [.1]),
    (['a'], [1, 2], [.1]),
    (['a'], [1], [.1, .2]),
    (['a', 2], [1, 2], [.1, .2]),
    ([''], [1], [.1]),
    (['  '], [1], [.1]),
    ([b'a'], [1], [.1]),
    ([None], [1], [.1]),
    (['a'], [True], [.1]),
    (['a'], [np.bool_(False)], [.1]),
    (['a', 'a'], [1, 2.], [.1, .2]),
    (['a'], ['1'], [.1]),
    (['a'], [1], [1]),
    (['a'], [1], [True]),
    (['a'], [1], ['.1']),
    (['a'], [1], [complex(.1, 0)]),
    (['a'], [1], [np.nan]),
    (['a'], [1], [np.inf]),
    (['a'], [1], [-np.inf]),
])
def test_invalid_shapes_types_or_nonfinite_scores_are_rejected(args):
    with pytest.raises(ValueError):
        select_successor_groups(*args)


def test_api_exposes_no_supervision_or_notation_inputs():
    assert list(inspect.signature(select_successor_groups).parameters) == [
        'source_ids', 'target_ticks', 'logits']
    with pytest.raises(TypeError):
        select_successor_groups(['a'], [120], [.5], voice_mask=[True])
