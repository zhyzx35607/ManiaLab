from dataclasses import replace
from pathlib import Path
import random

import pytest

from manialab.arrange import parameters
from manialab.domain import Note, invariant, validate
from manialab.layout import fingers, search_layout


@pytest.mark.parametrize('keys',range(6,11))
def test_beam_preserves_lns_and_taps_deterministically(keys):
    rng=random.Random(17)
    notes=[]
    for lane in range(keys):
        start=1000+rng.randrange(100)
        for i in range(12):
            end=start+rng.choice([0,100,400,1500])
            notes.append(Note(start,end,lane,id=f'{lane}:{i}'))
            start=end+rng.randrange(70,600)
    prefs=parameters({'keys':keys})
    a=search_layout(notes,prefs)
    b=search_layout(notes,prefs)
    assert a==b
    assert invariant(a)==invariant(notes)
    assert validate(a,keys)['valid']
    assert len(fingers(keys))==keys


def test_beam_keeps_locked_notes_including_future_crossings():
    locked=[Note(0,5000,0,id='a'),Note(4000,7000,1,id='b')]
    notes=[Note(1000,6000,id='new'),Note(2000,2000,id='tap')]
    before=[n.data() for n in notes+locked]
    output=search_layout(notes,parameters({'keys':6}),locked)
    assert [n.data() for n in notes+locked]==before
    assert next(n for n in output if n.id=='new').lane not in (0,1)
    assert [n for n in output if n.id in {'a','b'}]==locked
    assert invariant(output)==invariant(notes+locked)


def test_beam_reports_over_capacity_without_dropping():
    notes=[Note(0,5000,i,id=str(i)) for i in range(6)]+[Note(1000,1000,id='tap')]
    with pytest.raises(ValueError,match='1000ms'):
        search_layout(notes,parameters({'keys':6}))
    with pytest.raises(InterruptedError):
        search_layout(notes,parameters({'keys':6}),cancelled=lambda:True)


def test_full_chord_does_not_fail_after_long_note_release():
    notes=[Note(0,1000,i,id=f'h{i}') for i in range(6)]
    notes+=[Note(1000,1000,i,id=f't{i}') for i in range(6)]
    output=search_layout(notes,parameters({'keys':6}))
    assert len(output)==12 and validate(output,6)['valid']


def test_segment_lane_preference_is_applied_with_the_rhythm_unchanged():
    notes=[Note(t,t,pitch=60,source='pitch',id=str(t)) for t in list(range(1000,2000,125))+list(range(3000,4000,125))]
    settings=parameters({'keys':10,'pitch_follow':1,'balance':0,'jack_avoid':1,'pattern_prior':0})
    output=search_layout(notes,settings,segment_preferences=[dict(start_ms=1000,end_ms=2000,preferences={'jack_avoid':0})])
    assert len({n.lane for n in output if n.start<2000})==1
    assert len({n.lane for n in output if n.start>=3000})>1
    assert invariant(output)==invariant(notes)


@pytest.mark.parametrize('engine', ['beam', 'classic'])
def test_variable_tempo_remap_retains_all_endpoints(engine):
    from manialab.arrange import assign_lanes
    notes = [Note(t,t+length,lane,id=str(i)) for i,(t,length,lane) in enumerate(
        [(0,2400,0),(500,0,1),(1000,2000,2),(2000,0,3),(2125,0,4),(2250,0,5),(2500,1000,0)])]
    points = ['0,500,4,2,0,70,1,0','2000,250,4,2,0,70,1,0']
    output = assign_lanes(notes,parameters({'keys':6,'layout_engine':engine}),timing_points=points)
    assert invariant(output) == invariant(notes)
    assert validate(output,6)['valid']
