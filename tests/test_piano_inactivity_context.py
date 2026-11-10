from dataclasses import replace

import numpy as np
import pytest

from manialab.domain import Note
from manialab.expression_articulation import articulation_policy
from scripts.piano_inactivity_context import inactive_intervals, gap_support, activity_policy


def source():
    notes = [Note(t, t+300, pitch=60, source='pitch', voice='piano', id=str(t),
                  release_reason='release_regression') for t in (1000, 1500, 2000)]
    notes += [replace(notes[0], start=t, end=t+1200, id=str(t)) for t in (2500, 2750, 3000, 3250, 3500)]
    notes[0] = replace(notes[0], release_reason='reattack')
    return notes


def test_activity_must_span_eighty_ms_of_sampled_time_without_extrapolation():
    for count in (0, 8, 9, 10):
        zeros = np.zeros((count, 88), np.float32)
        assert inactive_intervals(zeros, zeros) == ([] if count < 9 else [(0, (count-1)*10)])


@pytest.mark.parametrize('head', ['frame', 'onset'])
def test_any_pitch_activity_breaks_low_activity_context(head):
    frame, onset = np.zeros((17, 88)), np.zeros((17, 88))
    (frame if head == 'frame' else onset)[8, 87] = .11 if head == 'frame' else .31
    assert inactive_intervals(frame, onset) == []


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1., 1.1])
def test_invalid_probabilities_are_not_silence(bad):
    a = np.zeros((20, 88));a[4, 4] = bad
    with pytest.raises(ValueError):
        inactive_intervals(a, np.zeros_like(a))


def test_context_outside_candidate_gap_does_not_count():
    assert gap_support([(100, 200)], 200, 500) is None
    assert gap_support([(100, 200)], 121, 500) is None
    assert gap_support([(100, 200)], 120, 500) == (120, 200)


def test_model_context_can_stop_unknown_history_from_flattening_later_sustains():
    notes = source();original = [n.data() for n in notes]
    result, gaps = activity_policy(notes, notes, [(1300, 1490), (1800, 1990), (2300, 2490)], observed_until_ms=6000)
    assert [result[n.id]['tap'] for n in notes] == [False]*3+[True]*4+[False]
    assert len(gaps) == 3 and [n.data() for n in notes] == original
    assert result['1000']['continuation'] == 'model_inactive_gap'
    assert not result['1000']['source_activity_context']['physical_silence_verified']


def test_no_activity_evidence_retains_baseline_decisions():
    notes = source();baseline = articulation_policy(notes, observed_until_ms=6000)
    result, gaps = activity_policy(notes, notes, [], observed_until_ms=6000)
    assert not gaps
    assert {k: v['tap'] for k, v in result.items()} == {k: v['tap'] for k, v in baseline.items()}


@pytest.mark.parametrize('strength', [.1, .9])
def test_a_raw_sustain_crossing_the_gap_vetoes_low_model_activity(strength):
    notes = source();blocker = replace(notes[0], start=900, end=6000, id='blocker', strength=strength)
    result, gaps = activity_policy(notes, notes+[blocker], [(0, 6000)], observed_until_ms=6000)
    assert not gaps and result['1500']['tap']


def test_source_context_does_not_become_separate_instrument_evidence():
    notes = [replace(n, instrument_id='model:strings') for n in source()]
    baseline = articulation_policy(notes, observed_until_ms=6000)
    result, gaps = activity_policy(notes, notes, [(0, 6000)], observed_until_ms=6000)
    assert result == baseline and not gaps


@pytest.mark.parametrize('mode', ['articulation', 'sustain', 'onsets', 'omit'])
def test_unknown_release_and_explicit_modes_survive_context(mode):
    from unittest.mock import patch
    import manialab.expression as expression
    from manialab.arrange import parameters
    from tests.test_structure_expression import analysis
    notes = source();a = analysis(notes, length=6000);p = parameters()
    plan = expression.propose_expression(a, p)
    plan['sections'][0]['voices']['piano']['mode'] = mode
    def contextual(local, *, observed_until_ms):
        return activity_policy(local, notes, [(1300, 1490), (1800, 1990), (2300, 2490)],
                               observed_until_ms=observed_until_ms)[0]
    with patch.object(expression, 'articulation_policy', contextual):
        selected, _, trace = expression.expression_evidence(a, plan, p)
    assert all(n.end in (n.start, next(x.end for x in notes if x.id == n.id)) for n in selected)
    assert all(not n.hold for n in selected if n.id == '1000')
    if mode == 'articulation':
        assert [n.hold for n in selected[:3]] == [False, True, True]
        assert trace['1000']['reason'] == 'unobserved_release'
    elif mode == 'omit':
        assert not selected
    elif mode == 'onsets':
        assert not any(n.hold for n in selected)
    else:
        assert all(n.hold for n in selected if n.id != '1000')
