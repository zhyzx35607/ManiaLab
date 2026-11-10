import copy

import numpy as np
import pytest

from scripts.check_piano_separate_later_acceptance import complete_multiple_groups, judge


def passing_inputs():
    before = dict(decoded_macro_f1=.8, complete_truth_groups=100, complete_multiple_groups=40,
                  true_edges_lost_to_proved_maximum=5)
    after = dict(before, true_edges_lost_to_proved_maximum=4)
    variants = {kind: dict(before=before.copy(), after=after.copy()) for kind in ('clean', 'plus1', 'plus2')}
    measures = dict(voice=dict(tp=10, fp=1), chord=dict(tp=10, fp=0), staff_layer_pairs=dict(tp=20, fp=0))
    controls = {name: dict(before=copy.deepcopy(measures), after=copy.deepcopy(measures))
                for name in ('lower_hold', 'upper_hold', 'wide_arpeggio', 'crossing', 'pedal', 'chords')}
    return variants, controls


def test_passing_gate_only_authorizes_further_audio_research():
    result = judge(.8, .81, *passing_inputs())
    assert result['audio_connection_experiment_allowed'] and not result['failed']
    assert result['application_enabled'] is result['release_allowed'] is result['musical_acceptance'] is False


def test_improving_average_does_not_hide_chord_loss_or_control_regression():
    variants, controls = passing_inputs()
    variants['plus1']['after']['complete_multiple_groups'] = 39
    controls['crossing']['after']['voice']['tp'] = 9
    controls['wide_arpeggio']['after']['staff_layer_pairs']['tp'] = 19
    result = judge(.8, .9, variants, controls)
    assert not result['audio_connection_experiment_allowed']
    assert set(result['failed']) == {'plus1/complete_multiple_groups', 'crossing/voice/tp',
                                     'wide_arpeggio/staff_layer_pairs/tp'}


def test_competition_requires_no_regression_and_at_least_one_strict_gain():
    variants, controls = passing_inputs()
    for kind in variants:
        variants[kind]['after']['true_edges_lost_to_proved_maximum'] = 5
    result = judge(.8, .81, variants, controls)
    assert result['failed'] == ['overlap_competition_at_least_one_strict_improvement']
    variants['plus1']['after']['true_edges_lost_to_proved_maximum'] = 4
    variants['plus2']['after']['true_edges_lost_to_proved_maximum'] = 6
    assert judge(.8, .81, variants, controls)['failed'] == ['plus2/true_edges_lost_to_proved_maximum']


def test_multi_target_count_requires_every_member_and_excludes_single_target_sources():
    data = dict(source_ids=np.asarray(['s', 's', 'single', 'unknown']),
                actual=np.asarray([True, True, True, True]), score_mask=np.asarray([True, True, True, False]),
                before_selected=np.asarray([True, False, True, True]), after_selected=np.asarray([True, True, False, False]))
    assert complete_multiple_groups(data) == dict(groups=1, before=0, after=1)


def test_identical_raw_score_cannot_count_as_improvement():
    assert judge(.8, .8, *passing_inputs())['failed'] == ['raw_macro_mean']


@pytest.mark.parametrize('missing', ['variant', 'control'])
def test_incomplete_suite_cannot_pass(missing):
    variants, controls = passing_inputs()
    if missing == 'variant':
        del variants['clean']
    else:
        del controls['pedal']
    with pytest.raises(ValueError):
        judge(.8, .81, variants, controls)
