import itertools

import pytest

from scripts.evaluate_adt_drums import match_times, score, boundary_counts, author_scores


def test_matching_keeps_duplicates_and_outside_predictions():
    result = match_times([0., 10.], [-100., 0., 0., 10., 999.], 2.)
    assert (result['matched'], result['extra']) == (2, 3)
    assert len(result['unmatched_predicted']) == 3


def test_maximum_matching_beats_greedy_nearest():
    result = match_times([0., 40.], [-30., 10.], 30.)
    assert result['pairs'] == [[0, 0], [1, 1]]


def test_small_matching_agrees_with_exhaustive_cardinality():
    for r in itertools.combinations_with_replacement(range(4), 3):
        for p in itertools.combinations_with_replacement(range(4), 3):
            expected = max(sum(abs(r[i]-p[j]) <= 1 for i, j in enumerate(order))
                           for order in itertools.permutations(range(3)))
            assert match_times(list(r), list(p), 1)['matched'] == expected


def test_wrong_class_does_not_match():
    result = score([dict(time_ms=10., kind='kick')], [dict(time_ms=10., kind='snare')], 40)
    assert result['kick']['missing'] == 1
    assert result['snare']['extra'] == 1


def test_boundary_diagnosis_uses_original_labels_and_matching():
    result = score([dict(time_ms=t, kind='kick') for t in (10., 1280., 2550., 2560.)],
                   [dict(time_ms=2550., kind='kick')], 20)
    assert boundary_counts(result)['kick'] == dict(near_reference=3, near_missing=2,
                                                  interior_reference=1, interior_missing=1)


def test_author_all_retains_shifted_events_before_zero_and_beyond_end():
    result = author_scores([dict(time_ms=100., kind='hi-hat')],
                           [dict(time_ms=t, kind='hi-hat') for t in (0., 122.5, 400000.)], 283840.)
    row = result['all']['chart_lead_22_5ms']['hi-hat']
    assert (row['predicted'], row['matched'], row['extra']) == (3, 1, 2)


def test_union_diagnostic_does_not_hide_primary_duplicates():
    result = author_scores([dict(time_ms=100., kind='hi-hat')],
                           [dict(time_ms=100., kind=k) for k in ('hi-hat', 'hi-hat', 'tom')], 283840.)
    row = result['all']['raw']
    assert row['hi-hat']['predicted'] == 2
    assert row['lane4_hh_tt']['predicted'] == 1


@pytest.mark.parametrize('reference,predicted,tolerance', [([float('nan')], [], 1), ([], [float('inf')], 1), ([], [], 0)])
def test_reject_invalid_clock(reference, predicted, tolerance):
    with pytest.raises(ValueError):
        match_times(reference, predicted, tolerance)
