import copy
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import select_expression
from manialab.expression_sustain import add_rewards, coverage_intervals, coverage_report


def note(identity, end=1000, strength=.8, **extra):
    return Note(0, end, id=identity, strength=strength, source='pitch', voice='other',
                evidence_start=0, release_reason='model_predicted_offset', instrument_id=identity, **extra)


def inputs(notes, mode='sustain'):
    return ({n.id:dict(section=0, role='support') for n in notes},
            {n.id:dict(id=n.id, start=n.start, mode=mode, reason='supported_sustain') for n in notes})


def choose(notes, weight=.25, mode='sustain'):
    meta, trace = inputs(notes, mode)
    return select_expression(notes, meta, trace, parameters(dict(chord_size=1, density=4)), 500,
                             sustain_coverage=weight)


def test_explicit_sustain_prefers_supported_duration_under_small_attack_difference():
    sustained, short = note('held'), note('short', end=0, strength=.85)
    notes = [sustained, short]
    before = copy.deepcopy(notes)
    assert choose(notes, 0)[0] == [short]
    selected, report = choose(notes)
    assert selected == [sustained] and notes == before
    assert report['sustain_coverage']['reward'] == pytest.approx(.525)


def test_stronger_attack_can_beat_sustain_and_long_tail_cannot_raise_cap():
    weak, strong = note('weak', strength=.35), note('strong', end=0, strength=1.)
    assert choose([weak, strong])[0] == [strong]
    long = replace(weak, end=60000)
    assert choose([long, strong])[0] == [strong]
    for held in (weak, long):
        _, report = choose([held])
        assert report['sustain_coverage']['reward'] == pytest.approx(.3)


@pytest.mark.parametrize('mode', ['onsets', 'articulation', 'phrase', 'omit'])
def test_other_expression_modes_do_not_gain_duration_reward(mode):
    held, short = note('held'), note('short', end=0, strength=.85)
    selected, report = choose([held, short], mode=mode)
    assert selected == [short] and report['sustain_coverage']['reward'] == 0


@pytest.mark.parametrize('changes', [dict(release_reason=''), dict(release_reason='unresolved_or_postprocessed'),
    dict(release_reason='reattack'), dict(source='transient'), dict(phrase=True), dict(support=('a','b'))])
def test_unsupported_or_synthetic_endpoints_never_gain_coverage(changes):
    held = replace(note('held'), **changes)
    meta, trace = inputs([held])
    assert coverage_intervals([[held]], meta, trace, 500, .25, lambda n:1.) == []


def test_overlap_uses_maximum_of_selected_options_not_unselected_chord_strength():
    a, b = note('a', end=500), note('b', end=500)
    units = [[a], [b]]
    meta, trace = inputs([a,b])
    intervals = coverage_intervals(units, meta, trace, 500, .25, lambda n: 1 if n.id=='a' else 3)
    assert len(intervals) == 1
    utility, constraints = [0.,0.], []
    add_rewards(intervals, utility, lambda weights, cap:constraints.append((weights,cap)))
    for primary in product((0,1), repeat=2):
        possible = []
        for aux in product((0,1), repeat=len(utility)-2):
            chosen = np.asarray(primary+aux)
            if all(sum(chosen[i]*v for i,v in weights.items()) <= cap for weights,cap in constraints):
                possible.append(np.asarray(utility)@chosen)
        expected = .25*max([score for flag,score in zip(primary,(1,3)) if flag],default=0)
        assert max(possible) == expected
        assert coverage_report(intervals, primary, .25)['reward'] == expected


def test_same_voice_chord_and_class_split_cannot_multiply_coverage():
    a, b = note('a'), note('b', pitch=72)
    meta, trace = inputs([a,b])
    for units, chosen in (([[a,b]],[1]), ([[a],[b]],[1,1])):
        rows = coverage_intervals(units, meta, trace, 500, .25, lambda n:1.)
        result = coverage_report(rows, chosen, .25)
        assert result['reward'] == .5 and result['covered_voice_ms'] == 1000


def test_nonoverlapping_pieces_and_independent_voices_are_counted_without_bridging():
    a = note('a', end=250)
    b = replace(note('b', end=1000), start=750)
    c = replace(note('c', end=250), voice='piano')
    meta, trace = inputs([a,b,c])
    rows = coverage_intervals([[a],[b],[c]], meta, trace, 500, .25, lambda n:1.)
    result = coverage_report(rows, [1,1,1], .25)
    assert result['covered_voice_ms'] == 750 and result['reward'] == .375
    assert all(not (r['start'] < 750 and r['end'] > 250) for r in rows)


@pytest.mark.parametrize('weight', [True,-1,1.01,float('nan'),float('inf'),'0.25'])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError, match='持续覆盖'):
        choose([], weight)


def test_disabled_path_and_solo_endpoints_preserved():
    held = note('held', end=4321)
    selected, default = choose([held], 0)
    assert selected == [held] and 'sustain_coverage' not in default
    assert choose([held])[0] == [held]


def test_empty_selection_reports_zero_sustain_coverage():
    selected, report = choose([], .25)
    assert selected == []
    assert report['sustain_coverage'] == coverage_report([], [], .25)
