import copy
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import select_expression
from manialab.expression_coherence import add_costs, group_links


def candidate(time, voice, strength, **extra):
    return Note(time, time, pitch=60, voice=voice, source='pitch', strength=strength,
                id=f'{voice}:{time}', evidence_start=time, **extra)


def choose(notes, coherence=0., sections=None, segments=(), **prefs):
    metadata = {n.id: dict(section=(sections or {}).get(n.id, 0), role='support') for n in notes}
    trace = {n.id: dict(id=n.id, start=n.start, reason='section_onsets') for n in notes}
    settings = parameters(dict(chord_size=1, density=4, **prefs))
    return select_expression(notes, metadata, trace, settings, 500, segments, coherence=coherence)


def test_small_quality_fluctuations_can_retain_one_continuous_source():
    notes = [candidate(t, voice, .85 if (t//1000)%2 == index else .8)
             for t in range(0, 4000, 500) for index, voice in enumerate(('piano', 'other'))]
    original = copy.deepcopy(notes)
    old, _ = choose(notes)
    selected, report = choose(notes, .25)
    assert [n.voice for n in old] == ['piano']*2+['other']*2+['piano']*2+['other']*2
    assert len({n.voice for n in selected}) == 1
    assert len(selected) == 8 and {n.start for n in selected} == {n.start for n in notes}
    assert report['group_coherence']['transitions'] == 0
    assert notes == original and all(n in notes for n in selected)


def test_clear_source_change_overcomes_continuity_preference():
    notes = [candidate(t, voice, 1. if (t < 2000) == (voice == 'piano') else .35)
             for t in range(0, 4000, 500) for voice in ('piano', 'other')]
    selected, report = choose(notes, .25)
    assert [n.voice for n in selected] == ['piano']*4+['other']*4
    assert report['group_coherence']['transitions'] == 2


def test_explicit_section_boundary_allows_an_unpenalized_change():
    notes = [candidate(t, voice, .85 if (t < 2000) == (voice == 'piano') else .8)
             for t in range(0, 4000, 500) for voice in ('piano', 'other')]
    selected, report = choose(notes, .25, sections={n.id: int(n.start >= 2000) for n in notes})
    assert [n.voice for n in selected] == ['piano']*4+['other']*4
    assert report['group_coherence']['transitions'] == 0


def test_rests_different_classes_and_manual_boundaries_are_not_linked():
    notes = [candidate(t, 'other', .8, instrument_id=identity)
             for t, identity in ((0, 'a'), (500, 'b'), (2000, 'a'), (2500, 'a'))]
    families = [([n], [i]) for i, n in enumerate(notes)]
    meta = {n.id: dict(section=0, role='support') for n in notes}
    links = group_links(families, meta, 500, .25, lambda n: 1.)
    assert [(r['left'], r['right']) for r in links] == [(2, 3)]
    assert group_links(families, meta, 500, .25, lambda n: 1., (2300,)) == []
    transients = [([replace(n, source='transient')], indices) for (n,), indices in families]
    assert group_links(transients, meta, 500, .25, lambda n: 1.) == []


def test_cost_does_not_reward_chord_depth_or_borrow_an_unselected_strong_pitch():
    a, b = candidate(0, 'piano', .4), candidate(500, 'piano', .8)
    strong = replace(a, id='strong', pitch=72, strength=1.)
    families = [([a, strong], [0, 1]), ([b], [2])]
    meta = {n.id: dict(section=0, role='support') for n in (a, b, strong)}
    links = group_links(families, meta, 500, .25, lambda n: n.strength)
    assert links[0]['penalty'] == pytest.approx(.1)
    constraints, utility = [], [0., 0., 0.]
    add_costs(links, families, utility, lambda weights, cap: constraints.append((weights, cap)))
    for left, right in product((None, 0, 1), (None, 2)):
        solutions = []
        for auxiliary in (0, 1):
            chosen = np.array([int(left == 0), int(left == 1), int(right == 2), auxiliary])
            if all(sum(chosen[i]*v for i, v in weights.items()) <= cap for weights, cap in constraints):
                solutions.append(-np.asarray(utility)@chosen)
        assert min(solutions) == pytest.approx(.1*int((left is None) != (right is None)))


def test_solo_available_attacks_and_known_ends_survive_without_capacity_pressure():
    notes = [replace(candidate(t, 'piano', .8), end=t+300, release_reason='model_predicted_offset')
             for t in (0, 500, 1000, 1500, 4000)]
    selected, report = choose(notes, 1., max_held=1)
    assert selected == notes
    assert report['group_coherence']['transitions'] == 0


@pytest.mark.parametrize('weight', [True, -1, 1.01, float('nan'), float('inf'), '0.25'])
def test_invalid_research_weight_rejected(weight):
    with pytest.raises(ValueError, match='连贯权重'):
        choose([], weight)


def test_disabled_research_path_preserves_decisions_and_does_not_add_report_claim():
    notes = [candidate(t, 'piano', .8) for t in (0, 500, 1000)]
    selected, report = choose(notes)
    assert selected == notes and 'group_coherence' not in report
