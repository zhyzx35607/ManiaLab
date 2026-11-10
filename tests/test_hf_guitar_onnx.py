"""Symbolic checks for runtime migration; no model loading or inference."""
import pytest
import numpy as np

from scripts.evaluate_guitar_vocal_benchmark import window_metrics
from scripts.verify_hf_guitar_onnx import compare_events, compare_activations


def test_unused_context_difference_is_reported_separately_without_changing_tolerance():
    expected = np.full((1001, 88), .8, dtype=np.float32)
    actual = expected.copy()
    actual[980:, 84] += .025
    whole = compare_activations(actual, expected)
    physical = compare_activations(actual[:733], expected[:733])
    assert whole['preserved'] is False and whole['violations'] == 21
    assert physical['preserved'] is True and physical['violations'] == 0
    actual[500, 84] += .025
    assert compare_activations(actual[:733], expected[:733])['preserved'] is False


def test_activation_reporting_rejects_invalid_arrays_and_keeps_original_tolerance():
    expected = np.full((5, 88), .8, dtype=np.float32)
    assert compare_activations(expected + .0008, expected)['preserved'] is True
    assert compare_activations(expected + .001, expected)['preserved'] is False
    for actual in (expected[:2], np.full_like(expected, np.nan), np.full_like(expected, np.inf)):
        with pytest.raises(ValueError, match='Invalid activation'):
            compare_activations(actual, expected)


def note(**changes):
    result = dict(onset=1.0, offset=1.1, pitch=60, velocity=100, onset_score=.7,
                  release_reason='frame_decay', release_observed=True,
                  program=-1, is_drum=False, voice='guitar_gaps')
    result.update(changes)
    return result


def scored(reference, predictions, end=20.):
    return window_metrics(reference, predictions, begin=0., end=end,
                          context_start=0., context_end=22.)


def test_runtime_inventory_cannot_gain_or_lose_events():
    expected = [note()]
    with pytest.raises(ValueError, match='event count'):
        compare_events([], expected)
    with pytest.raises(ValueError, match='event count'):
        compare_events(expected * 2, expected)


@pytest.mark.parametrize('changes', [
    {'pitch': 61}, {'velocity': 99},
    {'release_reason': 'release_regression'}, {'release_observed': True},
])
def test_runtime_cannot_change_note_identity_or_promote_an_unknown_tail(changes):
    expected = note(release_reason='reattack', release_observed=False)
    with pytest.raises(ValueError, match='release provenance'):
        compare_events([dict(expected, **changes)], [expected])


def test_runtime_reports_small_endpoint_and_score_drift_but_rejects_large_drift():
    expected = note()
    actual = note(onset=1.0005, offset=1.10025, onset_score=.70004)
    report = compare_events([actual], [expected])
    assert report['maximum_endpoint_error_ms'] == pytest.approx(.5)
    assert report['maximum_onset_score_error'] == pytest.approx(.00004)
    with pytest.raises(ValueError, match='more than 1 ms'):
        compare_events([note(offset=1.1011)], [expected])


def test_submillisecond_onset_change_can_cross_the_scoring_tolerance():
    reference = [note()]
    before = [note(onset=1.04975)]
    after = [note(onset=1.05025)]
    assert compare_events(after, before)['maximum_endpoint_error_ms'] < 1
    previous, _ = scored(reference, before)
    current, _ = scored(reference, after)
    assert previous['standard']['pitched_onsets']['matched'] == 1
    assert current['standard']['pitched_onsets']['matched'] == 0
    assert previous['standard']['pitched_onsets']['predicted'] == 1
    assert current['standard']['pitched_onsets']['predicted'] == 1


def test_submillisecond_tail_change_can_cross_the_interval_tolerance():
    reference = [note()]
    before = [note(offset=1.14975)]
    after = [note(offset=1.15025)]
    assert compare_events(after, before)['maximum_endpoint_error_ms'] < 1
    previous, _ = scored(reference, before)
    current, _ = scored(reference, after)
    assert previous['standard']['pitched_onsets']['matched'] == 1
    assert current['standard']['pitched_onsets']['matched'] == 1
    assert previous['standard']['complete_intervals']['matched'] == 1
    assert current['standard']['complete_intervals']['matched'] == 0


def test_submillisecond_change_across_scoring_end_changes_prediction_scope():
    reference = [note(onset=19.99, offset=20.1)]
    before = [note(onset=19.99975, offset=20.1)]
    after = [note(onset=20.00025, offset=20.1)]
    assert compare_events(after, before)['maximum_endpoint_error_ms'] < 1
    previous, old_scope = scored(reference, before)
    current, new_scope = scored(reference, after)
    assert previous['standard']['pitched_onsets']['predicted'] == 1
    assert current['standard']['pitched_onsets']['predicted'] == 0
    assert old_scope['predicted_indices'] == [0]
    assert new_scope['predicted_indices'] == []
    assert old_scope['predicted_total'] == new_scope['predicted_total'] == 1
