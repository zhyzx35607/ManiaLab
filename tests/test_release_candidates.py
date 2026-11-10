from dataclasses import replace
import numpy as np
import pytest

from manialab.anchor_gestures import AnchorContext, labels, plan
from manialab.domain import Note
from manialab.release_candidates import ReleaseContext, AcousticReleaseContext
from scripts.evaluate_anchor_gestures import head_scoped_comparison, greedy_lanes

TIMING = ['0,500,4,2,1,60,1,0']


def test_extreme_red_effect_cannot_silently_destroy_duration_precision():
    with pytest.raises(ValueError, match='precision'):
        ReleaseContext([Note(1000, 1000, 0), Note(1500, 1500, 0)], 6,
                       ['0,1e-61,4,2,1,60,1,0', '1,500,4,2,1,60,1,0'])


def test_same_bpm_phase_reset_keeps_positive_durations():
    a = ReleaseContext([Note(t, t, 0) for t in (999, 1001, 1500)], 6,
                       TIMING+['1001,500,4,2,1,60,1,0'])
    assert 2001 in a.options(1) and 1500 in a.options(1)
    assert np.isfinite(a.features(1)).all()


def test_grid_release_without_new_attack_and_no_tail_leakage():
    notes = [Note(t, t, 0) for t in (0, 1000, 2000)]
    a = ReleaseContext(notes, 6, TIMING)
    b = ReleaseContext([replace(n, end=n.start+1234, lane=5) for n in notes], 6, TIMING)
    assert 500 in a.options(0) and 500 not in a.times
    assert any(0 < t < 100 for t in a.options(0))
    np.testing.assert_array_equal(a.options(0), b.options(0))
    np.testing.assert_array_equal(a.features(0), b.features(0))
    y, details = labels(a, 0, [Note(0, 500, 1)])
    assert y[list(a.options(0)).index(500)] == 1 and details['represented_holds'] == 1
    old_y, old_details = labels(AnchorContext(notes, 6, TIMING), 0, [Note(0, 500, 1)])
    assert old_y[-1] == 1 and old_details['unrepresented_holds'] == 1


def test_new_grid_uses_tempo_change_and_merges_rounding_duplicates():
    a = ReleaseContext([Note(1, 1, 0), Note(1001, 1001, 0), Note(2000, 2000, 0)], 6,
                       TIMING+['1000,250,4,2,1,60,1,0'])
    assert a.time_at(3) == 1250
    options = a.options(0)
    assert 1001 in options and 1000 not in options
    assert 1250 in options
    assert len(set(options)) == len(options)
    assert (np.diff(options[1:-1]) > 3).all()
    assert np.isfinite(a.features(0)).all()


def test_independent_endpoint_capacity_counts_all_intervening_attacks():
    notes = [Note(0, 0, k) for k in range(5)]+[Note(500, 500, k) for k in range(6)]+[Note(1000, 1000, 0)]
    a = ReleaseContext(notes, 6, TIMING)
    predictions = []
    for i in range(len(a.times)):
        options = a.options(i); p = np.zeros(len(options))
        p[list(options).index(750 if i == 0 else int(a.times[i]))] = 1
        predictions.append(dict(options=options, probabilities=p))
    output, report = plan(a, predictions)
    assert len(output) == len(notes)
    assert all(n.end <= 500 for n in output if n.start == 0)
    greedy_lanes(output, 6)
    assert report['optimal']


def test_scoring_does_not_hide_a_tail_outside_the_excerpt():
    reference = [Note(500, 700, 0)]
    predicted = [Note(500, 5000, 0)]
    score = head_scoped_comparison(reference, predicted, 0, 1000)
    assert score['complete_holds']['predicted'] == 1
    assert score['complete_holds']['matched'] == 0


def test_acoustic_release_features_at_arbitrary_time_keep_endpoint_provenance():
    notes = [Note(0, 0, 0), Note(1000, 1000, 1)]
    data = dict(notes=[dict(onset=.0225, offset=.5225, pitch=60, program=40)],
                forced=[False], energy=np.ones(200)*.2, hop_ms=10)
    a = AcousticReleaseContext(notes, 6, TIMING, .35, {'piano':data, 'other':data})
    at = list(a.options(0)).index(500)
    x = a.features(0)
    assert x[at, a.feature_names.index('other_explicit_near_tail')] == 1
    assert x[at, a.feature_names.index('release_at_attack')] == 0
    assert x.shape[1] == len(a.feature_names) and np.isfinite(x).all()
