from dataclasses import replace

import numpy as np
import pytest

from manialab.anchor_gestures import AnchorContext, FEATURES, labels, plan, arrange_gestures
from manialab.domain import Note

TIMING = ['0,500,4,2,1,60,1,0']


def context(notes, keys=6):
    return AnchorContext(notes, keys, TIMING)


def test_no_input_tail_or_lane_leakage():
    notes = [Note(i*250, i*250+1000, i % 6, voice='piano') for i in range(20)]
    a = context(notes)
    b = context([replace(n, end=n.start, lane=5-n.lane) for n in notes])
    for i in range(len(a.times)):
        np.testing.assert_array_equal(a.features(i), b.features(i))
        assert a.features(i).shape[1] == len(FEATURES)


def test_unrepresentable_tail_is_not_labelled_tap():
    a = context([Note(0, 0, 0), Note(500, 500, 1)])
    y, report = labels(a, 0, [Note(0, 300, 0)])
    assert y[0] == 0 and y[-1] == 1
    assert report['unrepresented_holds'] == 1


def test_candidate_distance_and_limit_without_target():
    a = context([Note(i*25, i*25, 0) for i in range(200)])
    options = a.options(0)
    assert options[0] == 0 and options[-1] == -1 and len(options) == 34
    assert all(100 <= a.times[i] <= 4000 for i in options[1:-1])
    assert len(options) == len(set(options))


def predictions(a, choose):
    result = []
    for i in range(len(a.times)):
        options = a.options(i)
        p = np.zeros(len(options)); p[list(options).index(choose(i))] = 1
        result.append(dict(options=options, probabilities=p))
    return result


def test_occupancy_preserves_attacks_and_endpoint_provenance():
    notes = [Note(0, 0, k, id=f'a{k}') for k in range(5)]
    notes += [Note(250, 250, k, id=f'b{k}') for k in range(6)]
    notes += [Note(500, 500, 0, id='c')]
    a = context(notes)
    generated, report = plan(a, predictions(a, lambda i: 2 if i == 0 else i))
    # No five-held + six-attacks overflow; shortened tails are real anchors.
    assert sorted((n.id, n.start) for n in notes) == sorted((n.id, n.start) for n in generated)
    assert all(n.end in a.times for n in generated)
    assert all(n.end <= 250 for n in generated if n.start == 0)
    assert report['optimal']


def test_release_and_new_attack_same_timestamp_are_feasible():
    notes = [Note(t, t, k, raw=f'{k},192,{t},1,0,0:0:0:0:') for t in (0, 500) for k in range(6)]
    a = context(notes)
    generated, _ = plan(a, predictions(a, lambda i: 1))
    assert sum(n.hold for n in generated) == 6
    assert all(not n.raw for n in generated)


def test_shared_release_objective_uses_real_selected_endpoints():
    a = context([Note(t, t, 0) for t in (0, 250, 500, 750)])
    pred = predictions(a, lambda i: 3 if i < 2 else i)
    generated, report = plan(a, pred, coordination=.25)
    assert [(n.start, n.end) for n in generated if n.hold] == [(0, 750), (250, 750)]
    assert report['shared_release_objects'] == 2


def test_ln_ceiling_does_not_drop_attacks_or_fill_a_quota():
    a = context([Note(t, t, 0) for t in (0, 250, 500, 750)])
    pred = predictions(a, lambda i: 3 if i < 3 else i)
    generated, report = plan(a, pred, max_ln_ratio=.25)
    assert len(generated) == 4 and sum(n.hold for n in generated) <= 1
    assert report['ln_limit'] == 1
    taps, _ = plan(a, predictions(a, lambda i: i), max_ln_ratio=1.)
    assert all(not n.hold for n in taps)


def test_invalid_predictions_and_overfull_input_rejected():
    with pytest.raises(ValueError, match='capacity'):
        context([Note(0, 0, 0) for _ in range(7)])
    a = context([Note(0, 0, 0)])
    with pytest.raises(ValueError, match='candidate'):
        plan(a, [dict(options=[0, -1], probabilities=[np.nan, 0.])])


def test_timing_change_is_integrated_and_green_lines_ignored():
    notes = [Note(t, t, 0) for t in (0, 500, 1000, 1250)]
    a = AnchorContext(notes, 6, TIMING+['1000,250,4,2,1,60,1,0'])
    np.testing.assert_allclose(a.beats, [0, 1, 2, 3])
    b = AnchorContext(notes, 6, TIMING+['750,-50,4,2,1,60,0,1', '1000,250,4,2,1,60,1,1'])
    np.testing.assert_array_equal(a.features(0), b.features(0))


@pytest.mark.parametrize('keys', [6, 7, 8, 9, 10])
def test_source_column_is_not_a_performance_lane_preference(keys):
    notes = [Note(i*500, i*500+(250 if i % 4 == 0 else 0), 0,
                  id=f'n{i}', voice='kick') for i in range(32)]
    output = arrange_gestures(notes, keys, TIMING, dict(pattern_prior=0.))
    assert sorted((n.id, n.start, n.end) for n in output) == sorted((n.id, n.start, n.end) for n in notes)
    assert sum(n.lane < keys//2 for n in output) >= 8
    assert sum(n.lane >= (keys+1)//2 for n in output) >= 8
