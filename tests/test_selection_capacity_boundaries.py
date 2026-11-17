"""Sustained occupancy and source-group viability at later section boundaries."""
from copy import deepcopy

import pytest

from manialab.arrange import assign_lanes, parameters, select_notes
from manialab.domain import Note, validate
from manialab.expression import expression_evidence, propose_expression, select_expression
from manialab.preferences import ChartPreferences
from manialab.structure import edit_boundaries
from manialab.timing import chart_clock_notes


def expression_selection(notes, prefs, segments, shift=0):
    analysis = dict(duration=8, tempo=dict(bpm=120,offset=0), candidates=[n.data() for n in notes])
    analysis['structure'] = edit_boundaries({}, [0,8000], analysis)
    original = deepcopy(analysis)
    evidence, metadata, trace = expression_evidence(analysis, propose_expression(analysis,prefs), prefs)
    chosen, report = select_expression(chart_clock_notes(evidence,shift), metadata,trace,prefs,500,
        ChartPreferences(segments,shift))
    assert analysis == original
    return chosen, report


@pytest.mark.parametrize('shift', [0,22.5,-50])
@pytest.mark.parametrize('keys', [6,10])
def test_group_keeps_feasible_full_ln_when_later_section_reduces_capacity(shift,keys):
    prefs = parameters(dict(keys=keys,density=8,chord_size=2,max_held=2,ln_ratio=1))
    notes = [Note(1000,4000,pitch=60,strength=.9,source='pitch',voice='piano',id='a'),
             Note(1500,4500,pitch=64,strength=.8,source='pitch',voice='piano',id='b')]
    segments = [dict(start_ms=2000,end_ms=5000,preferences=dict(max_held=1))]
    chosen, _ = expression_selection(notes,prefs,segments,shift)
    assert len(chosen) == 1 and chosen[0].hold
    expected = {n.id:(n.start,n.end) for n in chart_clock_notes(notes,shift)}
    assert (chosen[0].start,chosen[0].end) == expected[chosen[0].id]
    assert validate(assign_lanes(chosen,prefs),keys)['valid']


@pytest.mark.parametrize('shift', [0,22.5,-50])
def test_legacy_holds_obey_later_capacity_without_deleting_attacks(shift):
    prefs = parameters(dict(keys=6,density=8,chord_size=2,max_held=2,ln_ratio=1))
    notes = [Note(1000,4000,pitch=60,strength=.9,source='pitch',voice='piano',id='a'),
             Note(1500,4500,pitch=64,strength=.8,source='pitch',voice='piano',id='b')]
    shifted = chart_clock_notes(notes,shift)
    timeline = ChartPreferences([dict(start_ms=2000,end_ms=5000,preferences=dict(max_held=1))],shift)
    chosen, report = select_notes(shifted,prefs,timeline)
    assert {n.id:n.start for n in chosen} == {n.id:n.start for n in shifted}
    assert sum(n.hold for n in chosen) == 1
    assert report['feasible_ln_count'] == 1
    assert all(n.end == next(x.end for x in shifted if x.id == n.id) for n in chosen if n.hold)


def test_other_voice_checkpoint_does_not_erase_entire_feasible_rhythm():
    prefs = parameters(dict(keys=6,density=2,chord_size=2,ln_ratio=0))
    notes = [Note(600,600,strength=.9,source='pitch',voice='piano',id='a'),
             Note(800,800,strength=.8,source='pitch',voice='piano',id='b'),
             Note(1100,1100,strength=.4,source='pitch',voice='bass',id='quiet')]
    segments = [dict(start_ms=1100,end_ms=8000,preferences=dict(density=1))]
    chosen,_ = expression_selection(notes,prefs,segments)
    assert {n.id for n in chosen} == {'a'}


def test_unconstrained_group_is_not_fragmented():
    prefs = parameters(dict(keys=6,density=8,chord_size=2,max_held=2,ln_ratio=1))
    notes = [Note(1000,4000,pitch=60,source='pitch',voice='piano',id='a'),
             Note(1500,4500,pitch=64,source='pitch',voice='piano',id='b')]
    chosen,_ = expression_selection(notes,prefs,[])
    assert {(n.id,n.start,n.end) for n in chosen} == {(n.id,n.start,n.end) for n in notes}


@pytest.mark.parametrize('separate_instrument', [False,True])
def test_only_offered_pitches_tighten_other_groups(separate_instrument):
    prefs = parameters(dict(keys=6,density=3,chord_size=1,ln_ratio=0))
    notes = [Note(600,600,strength=.9,source='pitch',voice='piano',id='a'),
             Note(800,800,strength=.8,source='pitch',voice='piano',id='b'),
             Note(1000,1000,pitch=40,strength=.6,source='pitch',voice='bass',id='loose',audio_start=990),
             Note(1000,1000,pitch=44,strength=.5,source='pitch',voice='bass',id='tight',audio_start=1010,
                  instrument_id='second' if separate_instrument else '')]
    segments = [dict(start_ms=1000,end_ms=8000,preferences=dict(density=1))]
    chosen,_ = expression_selection(notes,prefs,segments)
    # A pitch outside every chord option does not own capacity. In a separate
    # instrument it is offered, so its original section tightens the shared row.
    assert {n.id for n in chosen} == ({'a'} if separate_instrument else {'a','b','loose'})


def test_legacy_boundary_end_restores_tighter_global_hold_limit():
    prefs = parameters(dict(keys=6,density=8,chord_size=2,max_held=1,ln_ratio=1))
    notes = [Note(1000,4000,pitch=60,source='pitch',id='a'),
             Note(1500,4500,pitch=64,source='pitch',id='b')]
    segments = [dict(start_ms=0,end_ms=2000,preferences=dict(max_held=2))]
    chosen,report = select_notes(notes,prefs,segments)
    assert len(chosen) == 2 and report['feasible_ln_count'] == 1
    assert sum(n.hold for n in chosen) == 1


def test_legacy_release_exactly_at_boundary_does_not_consume_hold_capacity():
    prefs = parameters(dict(keys=6,density=8,chord_size=2,max_held=2,ln_ratio=1))
    notes = [Note(1000,2000,pitch=60,source='pitch',id='a'),
             Note(1500,4500,pitch=64,source='pitch',id='b')]
    segments = [dict(start_ms=2000,end_ms=5000,preferences=dict(max_held=1))]
    chosen,report = select_notes(notes,prefs,segments)
    assert report['feasible_ln_count'] == 2
    assert [(n.start,n.end) for n in chosen] == [(n.start,n.end) for n in notes]
