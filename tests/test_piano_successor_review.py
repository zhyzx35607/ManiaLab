"""Independent checks of ownership and label-free finite-decoder scoring."""
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from scripts.piano_successor_decode import select_successor_groups


@pytest.fixture
def helpers():
    path = Path(__file__).resolve().parents[1] / 'scripts/check_piano_successor_decode.py'
    wanted = {'score', 'owned_arrays', 'join_owned', 'diagnose', 'aggregate'}
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in wanted]
    assert {node.name for node in selected} == wanted
    namespace = {'np': np, 'defaultdict': defaultdict,
                 'select_successor_groups': select_successor_groups}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def data(sources, ticks, logits, actual, mask, targets=None):
    return dict(source_ids=np.asarray(sources), target_ticks=np.asarray(ticks),
                target_ids=np.asarray(targets if targets is not None else
                                      [f'target-{i}' for i in range(len(sources))]),
                logits=np.asarray(logits, dtype=float), actual=np.asarray(actual, dtype=bool),
                score_mask=np.asarray(mask, dtype=bool))


def payload():
    notes = np.zeros(3, dtype=[('onset_div', np.int64)])
    notes['onset_div'] = [0, 120, 240]
    return dict(potential_edges=np.asarray([[0, 0, 1], [1, 2, 2]]),
                notes=notes, event_ids=np.asarray(['a', 'b', 'c']),
                voice=np.asarray([1., 0., 1.]), voice_mask=np.asarray([True, False, True]),
                core_mask=np.asarray([False, False, True]))


def test_owned_arrays_selects_target_core_and_retains_unscored_candidates(helpers):
    result = helpers['owned_arrays'](payload(), np.asarray([.2, .8, .5]))
    assert result['source_ids'].tolist() == ['a', 'b']
    assert result['target_ids'].tolist() == ['c', 'c']
    assert result['target_ticks'].tolist() == [240, 240]
    assert result['logits'].tolist() == [.8, .5]
    assert result['score_mask'].tolist() == [False, True]
    assert result['actual'].tolist() == [False, True]


def test_audio_output_owner_is_used_without_supervision_or_core_mask(helpers):
    p = payload()
    del p['core_mask']
    p['voice_output_owner'] = np.asarray([True, False, False])
    p['voice_mask'][:] = False
    result = helpers['owned_arrays'](p, np.asarray([.2, .8, .5]))
    assert result['source_ids'].tolist() == ['a']
    assert result['target_ids'].tolist() == ['b']
    assert result['target_ticks'].tolist() == [120]
    assert not result['score_mask'].any()


def test_join_owned_aggregates_same_source_across_windows_before_selection(helpers):
    first = data(['a'], [120], [.3], [True], [True], ['b'])
    second = data(['a'], [240], [.8], [False], [False], ['c'])
    joined = helpers['join_owned']([first, second])
    result, selected = helpers['diagnose'](joined)
    assert selected.tolist() == [False, True]
    assert result['removed_true_edges'] == 1
    assert result['removed_true_with_only_unscored_winning_edges'] == 1
    assert result['removed_true_sources_with_only_unscored_winning_edges'] == 1
    assert result['decoded']['fn'] == 1


def test_duplicate_owned_pair_is_rejected_even_if_logits_differ(helpers):
    first = data(['a'], [120], [.3], [True], [True], ['b'])
    second = data(['a'], [120], [.8], [True], [True], ['b'])
    with pytest.raises(ValueError, match='Duplicate owned'):
        helpers['join_owned']([first, second])


def test_selection_is_invariant_to_truth_and_mask_changes(helpers):
    original = data(['a', 'a', 'b', 'b'], [120, 240, 120, 240],
                    [.3, .8, -.1, .4], [True, False, False, True], [True, False, True, True])
    altered = {key: value.copy() for key, value in original.items()}
    altered['actual'] = ~altered['actual']
    altered['score_mask'] = ~altered['score_mask']
    _, first = helpers['diagnose'](original)
    _, second = helpers['diagnose'](altered)
    assert np.array_equal(first, second)


def test_removed_true_and_false_edges_reconcile_exactly_with_score_changes(helpers):
    value = data(['a', 'a', 'b', 'b', 'c'], [120, 240, 120, 240, 120],
                 [.5, .8, .1, .8, -.5], [True, False, False, True, True], [True] * 5)
    result, selected = helpers['diagnose'](value)
    assert selected.tolist() == [False, True, False, True, False]
    assert result['removed_true_edges'] == result['removed_false_edges'] == 1
    assert {k: result['raw'][k] for k in ('tp', 'fp', 'fn', 'tn')} == dict(tp=2, fp=2, fn=1, tn=0)
    assert {k: result['decoded'][k] for k in ('tp', 'fp', 'fn', 'tn')} == dict(tp=1, fp=1, fn=2, tn=1)
    assert result['raw']['recall'] == pytest.approx(2/3)
    assert result['decoded']['recall'] == pytest.approx(1/3)


def test_unscored_audio_diagnostics_need_no_actual_or_mask(helpers):
    value = data(['a', 'a'], [120, 240], [.3, .8], [False, False], [False, False])
    del value['actual']
    del value['score_mask']
    result, selected = helpers['diagnose'](value, scored=False)
    assert selected.tolist() == [False, True]
    assert 'raw' not in result and 'decoded' not in result
    assert result['raw_positive'] == 2 and result['selected_positive'] == 1


def test_multiple_targets_at_one_onset_do_not_count_as_a_branch(helpers):
    value = data(['a', 'a', 'a'], [120, 120, 240], [.8, .1, .3],
                 [True, True, False], [True, True, True])
    result, selected = helpers['diagnose'](value)
    assert selected.tolist() == [True, True, False]
    assert result['raw_sources_with_multiple_target_ticks'] == 1
    assert result['selected_sources_with_multiple_target_ticks'] == 0
    assert result['decoded']['tp'] == 2 and result['decoded']['fp'] == 0


def test_no_positive_edges_preserves_no_successor_and_reports_false_negatives(helpers):
    value = data(['a', 'a'], [120, 240], [-.3, -.8], [True, False], [True, True])
    result, selected = helpers['diagnose'](value)
    assert not selected.any()
    assert result['removed_positive'] == result['removed_true_edges'] == 0
    assert result['raw'] == result['decoded']
    assert result['decoded']['fn'] == result['decoded']['tn'] == 1


def test_macro_excludes_pieces_without_positive_truth_but_micro_counts_their_fp(helpers):
    positive = data(['a'], [120], [.3], [True], [True])
    negative = data(['b'], [120], [.3], [False], [True])
    cases = [helpers['diagnose'](value)[0] for value in (positive, negative)]
    result = helpers['aggregate'](cases)
    assert result['pieces'] == 2
    assert result['raw_macro_f1'] == result['decoded_macro_f1'] == 1.
    assert result['raw']['f1'] == result['decoded']['f1'] == pytest.approx(2/3)
    assert result['raw']['tp'] == result['raw']['fp'] == 1


@pytest.mark.parametrize(('winner_scores', 'winner_mask', 'expected'), [
    ([.9, .8], [False, True], (1, 0, 1, 0)),
    ([.9, .9], [False, True], (0, 1, 1, 0)),
    ([.9, .95], [False, True], (0, 0, 1, 0)),
    ([.9, .8], [False, False], (1, 0, 0, 1)),
    ([.9, .8], [True, True], (0, 0, 0, 0)),
])
def test_true_edge_loss_distinguishes_unscored_maxima_ties_and_mixed_groups(
        helpers, winner_scores, winner_mask, expected):
    value = data(['a'] * 3, [120, 240, 240], [.4] + winner_scores,
                 [True, False, False], [True] + winner_mask)
    result, selected = helpers['diagnose'](value)
    assert selected.tolist() == [False, True, True]
    assert result['removed_true_edges'] == 1
    fields = ('removed_true_with_unscored_maximum_only',
              'removed_true_with_tied_scored_unscored_maximum',
              'removed_true_with_mixed_winning_group',
              'removed_true_with_only_unscored_winning_edges')
    assert tuple(result[name] for name in fields) == expected


def test_competition_diagnostics_count_removed_edges_not_only_removed_sources(helpers):
    value = data(['a'] * 3, [120, 120, 240], [.4, .3, .9],
                 [True, True, False], [True, True, False])
    result, selected = helpers['diagnose'](value)
    assert selected.tolist() == [False, False, True]
    assert result['removed_true_edges'] == 2
    assert result['removed_true_with_unscored_maximum_only'] == 2
    assert result['removed_true_with_only_unscored_winning_edges'] == 2
    assert result['removed_true_sources_with_only_unscored_winning_edges'] == 1
