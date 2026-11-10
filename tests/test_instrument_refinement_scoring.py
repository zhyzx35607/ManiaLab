from copy import deepcopy

import pytest

from scripts.check_instrument_refinement import family_mapping, family_metrics


def note(onset=1., pitch=60, program=40, drum=False):
    return dict(onset=onset, offset=onset+.5, pitch=pitch, program=program, is_drum=drum)


def test_family_scoring_separates_correct_pitch_from_wrong_family_and_keeps_extra_detections():
    reference = [note(), note(onset=2, program=56)]
    predicted = [note(), note(onset=2), note(onset=3, drum=True)]
    before = deepcopy((reference, predicted))
    result, _ = family_metrics(reference, predicted, [0, 0, 2], {40:0, 56:1}, 40)
    assert result['standard']['pitched_onsets']['matched'] == 2
    assert result['standard']['instrument_onsets']['matched'] == 1
    assert result['standard']['instrument_onsets']['predicted'] == 3
    assert (reference, predicted) == before


def test_same_pitch_unison_families_are_matched_jointly_without_double_counting():
    reference = [note(), note(program=56)]
    predicted = [note(), note()]
    good, _ = family_metrics(reference, predicted, [1, 0], {40:0, 56:1}, 40)
    bad, _ = family_metrics(reference, predicted, [0, 0], {40:0, 56:1}, 40)
    assert good['standard']['instrument_onsets']['matched'] == 2
    assert bad['standard']['instrument_onsets']['matched'] == 1
    single, _ = family_metrics(reference, predicted[:1], [0], {40:0, 56:1}, 40)
    assert single['standard']['instrument_onsets']['matched'] == 1


def test_family_scores_do_not_remove_long_predictions_or_change_the_scoring_window():
    ref = [note(onset=29.5), note(onset=30)]
    pred = [dict(note(onset=29.5), offset=41), note(onset=30)]
    result, scope = family_metrics(ref, pred, [0, 0], {40:0}, 40)
    assert result['standard']['instrument_onsets']['matched'] == 1
    assert result['standard']['complete_instrument_intervals']['matched'] == 0
    assert result['standard']['complete_instrument_intervals']['predicted'] == 1
    assert scope['predicted_crosses_context_end'] == 1


def test_family_taxonomy_uses_declared_program_members_without_a_default_guess():
    taxonomy = {'labels': {'strings': {'members': [{'program_number': 40}, {'program_number': 41}]},
                           'drums': {'members': [{'program_number': None}]}}}
    classes, mapping = family_mapping(taxonomy)
    assert classes[mapping[40]] == classes[mapping[41]] == 'strings'
    with pytest.raises(ValueError, match='Unknown reference'):
        family_metrics([note(program=120)], [note()], [0], mapping, 40)
    with pytest.raises(ValueError, match='Every original'):
        family_metrics([note()], [note()], [], mapping, 40)
