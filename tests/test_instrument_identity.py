from dataclasses import replace
import json

import pytest

from manialab.arrange import assign_lanes, parameters, select_notes
from manialab.domain import Note, note_from, validate
from manialab.expression import propose_expression, expression_evidence, select_expression
from manialab.expression_articulation import articulation_policy
from manialab.layout import source_key
from scripts.instrument_candidate_adapter import accompaniment_candidates
from tests.test_structure_expression import analysis


def candidate(program, start, end):
    return dict(program=program,pitch=60,onset=start,offset=end,is_drum=False)


def bridge(events):
    provenance=[dict(note_index=i,endpoint_evidence=['explicit_model_note_off']) for i in range(len(events))]
    return accompaniment_candidates(events,provenance,10000)[0]


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('keys',range(6,11))
@pytest.mark.parametrize('engine',['beam','classic'])
def test_two_instrument_unison_keeps_both_tails_through_selection_and_layout(reverse,keys,engine):
    events=[candidate(0,1.,1.4),candidate(48,1.,3.)]
    notes=bridge(events[::-1] if reverse else events)
    a=analysis(notes)
    p=parameters(dict(keys=keys,density=24,chord_size=10,max_held=keys,ln_ratio=1,layout_engine=engine))
    evidence,meta,trace=expression_evidence(a,propose_expression(a,p),p)
    selected,report=select_expression(evidence,meta,trace,p,500)
    assert sorted((n.instrument_id,n.start,n.end) for n in selected)==[
        ('yourmt3:program:0',1000,1400),('yourmt3:program:48',1000,3000)]
    assert not report['omission_reasons']
    placed=assign_lanes(selected,p)
    assert len(placed)==2 and validate(placed,keys)['valid']
    fields=lambda n:(n.id,n.start,n.end,n.voice,n.pitch,n.instrument_id)
    assert {fields(n) for n in placed}=={fields(n) for n in selected}
    legacy,_=select_notes(notes,p)
    assert {n.instrument_id for n in legacy}=={n.instrument_id for n in notes}


def test_other_instrument_attacks_cannot_turn_a_sustained_string_into_tap():
    notes=bridge([candidate(48,1,3),candidate(24,1.25,1.35),candidate(24,1.5,1.6)])
    a=analysis(notes);p=parameters()
    plan=propose_expression(a,p)
    plan['sections'][0]['voices']['other']['mode']='articulation'
    evidence,_,_=expression_evidence(a,plan,p)
    assert evidence[0].end==3000
    assert articulation_policy(notes,observed_until_ms=10000)[notes[0].id]['articulation']=='independent_sustain'
    assert articulation_policy(iter(notes[:1]),observed_until_ms=10000)==articulation_policy(notes[:1],observed_until_ms=10000)


@pytest.mark.parametrize('split',[False,True])
@pytest.mark.parametrize('strong',['a','b'])
def test_instrument_labels_do_not_multiply_a_sources_chord_reward(split,strong):
    notes=[Note(1000,1000,pitch=60,voice='other',source='pitch',id='a',instrument_id='program:0',
                strength=1. if strong=='a' else .8),
           Note(1000,1000,pitch=64,voice='other',source='pitch',id='b',
                instrument_id='program:48' if split else 'program:0',strength=1. if strong=='b' else .8),
           Note(1000,1000,pitch=72,voice='piano',source='pitch',id='p',strength=.9)]
    a=analysis(notes);p=parameters(dict(chord_size=2,density=24,ln_ratio=0))
    plan=propose_expression(a,p)
    for policy in plan['sections'][0]['voices'].values():policy['role']='support'
    evidence,meta,trace=expression_evidence(a,plan,p)
    selected,_=select_expression(evidence,meta,trace,p,500)
    assert {n.id for n in selected}=={strong,'p'}


def choose_supported(notes,**settings):
    a=analysis(notes);p=parameters(dict(density=24,ln_ratio=1,**settings))
    plan=propose_expression(a,p)
    for voice,policy in plan['sections'][0]['voices'].items():
        policy.update(mode='sustain',role='primary' if voice=='bass' else 'support')
    evidence,meta,trace=expression_evidence(a,plan,p)
    return select_expression(evidence,meta,trace,p,500)[0]


def test_unselected_strong_hold_cannot_lend_its_reward_to_weak_tap():
    notes=[Note(0,2000,voice='bass',source='pitch',id='bass'),
           Note(1000,1500,voice='other',source='pitch',id='strong',instrument_id='a'),
           Note(1000,1000,voice='other',source='pitch',id='weak',instrument_id='b',strength=.35),
           Note(1000,1000,voice='piano',source='pitch',id='piano',strength=.7)]
    selected=choose_supported(notes,chord_size=1,max_held=1)
    assert {n.id for n in selected}=={'bass','piano'}


@pytest.mark.parametrize('room_for_layers',[False,True])
def test_chord_depth_only_receives_evidence_for_layers_actually_selected(room_for_layers):
    notes=[Note(1000,1000,pitch=60,voice='other',source='pitch',id='misaligned',instrument_id='a',evidence_start=1100),
           Note(1000,1000,pitch=64,voice='other',source='pitch',id='aligned',instrument_id='a',evidence_start=1000,strength=.9),
           Note(1000,1000,pitch=67,voice='other',source='pitch',id='weak',instrument_id='b',strength=.35)]
    if not room_for_layers:notes.append(Note(1000,1000,voice='bass',source='pitch',id='bass'))
    selected=choose_supported(notes,chord_size=3 if room_for_layers else 2)
    assert {n.id for n in selected}==({'misaligned','aligned','weak'} if room_for_layers else {'bass','weak'})


def test_same_program_duplicates_remain_duplicates_and_unknown_is_not_guessed():
    notes=bridge([candidate(48,1,3),candidate(48,1,3)])
    a=analysis(notes);p=parameters(dict(ln_ratio=1,density=24))
    evidence,meta,trace=expression_evidence(a,propose_expression(a,p),p)
    selected,report=select_expression(evidence,meta,trace,p,500)
    assert len(selected)==1 and report['omission_reasons']=={'duplicate_pitch':1}
    unknown=note_from(dict(start=1000,end=3000,voice='other',pitch=60))
    assert unknown.instrument_id==''
    assert source_key(unknown)!=source_key(notes[0])
    restored=note_from(json.loads(json.dumps(notes[0].data())))
    assert (restored.instrument_id,restored.start,restored.end)==(notes[0].instrument_id,1000,3000)


def test_phrase_cannot_claim_another_instruments_interior_attack():
    notes=bridge([candidate(48,1,2),candidate(24,1.5,1.8),candidate(48,1.7,1.9)])
    a=analysis(notes)
    a['phrase_candidates']=[replace(notes[0],end=3000,phrase=True,
        support=tuple(n.id for n in notes),support_coverage=1.).data()]
    p=parameters(dict(ln_ratio=1))
    plan=propose_expression(a,p);plan['sections'][0]['voices']['other']['mode']='phrase'
    evidence,_,trace=expression_evidence(a,plan,p)
    assert notes[1].id in {n.id for n in evidence}
    assert trace[notes[2].id]['reason']=='represented_by_phrase'


@pytest.mark.parametrize('value',[True,48.5])
def test_bridge_rejects_noninteger_program(value):
    with pytest.raises(ValueError):bridge([candidate(value,1,2)])


@pytest.mark.parametrize('value',[True,48,None,'bad\nidentifier','a'*129])
def test_project_rejects_invalid_instrument_identity(value):
    with pytest.raises(ValueError):note_from(dict(start=0,end=100,instrument_id=value))
