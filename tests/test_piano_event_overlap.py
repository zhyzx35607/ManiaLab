"""Input-tail perturbations preserve attacks, notation targets and open context."""
import copy
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest

from scripts.piano_event_overlap import augment_events, summarize_augmentation
from scripts.piano_event_training import _geometry, build_piece_items, read_meter_map
from scripts.piano_musicxml_events import parse_musicxml
from scripts.prepare_piano_event_overlap import assert_clean_cache, supervised_identity
from tests.test_piano_event_training import note, xml


def source_of(data):
    source=asdict(parse_musicxml(data))
    return source,read_meter_map(data,source)


def event(identifier,start,end,pitch=60,part='P1',eligible=True):
    return dict(id=identifier,part=part,pitch=pitch,start_tick=start,end_tick=end,
                written_ids=[identifier],supervision_eligible=eligible)


def raw_source(events,unit=1):
    return dict(ticks_per_quarter=unit,representation=dict(events=events))


@pytest.mark.parametrize('quarters',[1,2])
def test_exact_odd_tick_clock_extension_preserves_identity_and_does_not_mutate_source(quarters):
    source=raw_source([event('a',0,1),event('b',1,2,pitch=62)],unit=3)
    before=copy.deepcopy(source)
    changed=augment_events(source,quarters)
    assert isinstance(changed,list) and source==before
    assert [e['end_tick'] for e in changed]==[1+3*quarters,2+3*quarters]
    for a,b in zip(source['representation']['events'],changed):
        assert {k:v for k,v in a.items() if k!='end_tick'}=={k:v for k,v in b.items() if k!='end_tick'}
    stats=summarize_augmentation(source,changed)
    assert stats['events']==2 and stats['extended']==2 and stats['max_added_ticks']==3*quarters
    assert stats['onset_pitch_id_unchanged']


def test_next_same_pitch_any_part_caps_without_reading_part_labels():
    source=raw_source([event('a',0,2,part='upper'),event('b',3,4,part='lower')])
    changed=augment_events(source,2)
    assert [e['end_tick'] for e in changed]==[3,6]
    altered=copy.deepcopy(source)
    for row in altered['representation']['events']:
        row['part']='all-same';row['supervision_eligible']=False
    assert [(e['start_tick'],e['end_tick'],e['pitch']) for e in augment_events(altered,2)]==[
        (e['start_tick'],e['end_tick'],e['pitch']) for e in changed]


def test_simultaneous_unisons_remain_distinct_and_overlapping_true_intervals_never_shorten():
    source=raw_source([event('long',0,5),event('unison',0,1),event('reattack',2,3)])
    changed=augment_events(source,2)
    assert [(e['id'],e['start_tick'],e['end_tick']) for e in changed]==[
        ('long',0,5),('unison',0,2),('reattack',2,5)]
    assert len(changed)==3


def test_tie_continuation_is_not_an_augmentation_reattack():
    source,meter=source_of(xml(note(duration=4,ties=('start',)),
        note(duration=2,ties=('stop',),voice='7')+note('E',duration=2,voice='7')))
    changed=augment_events(source,2)
    assert len(changed)==2 and changed[0]['start_tick']==0 and changed[0]['end_tick']==8
    assert len(changed[0]['written_ids'])==2
    clean=build_piece_items(source,meter,piece_id='tie')
    overlap=build_piece_items(source,meter,piece_id='tie',observed_events=changed)
    assert supervised_identity(clean)==supervised_identity(overlap)
    assert any(supervised_identity(overlap)['voice'].values())


def test_source_successor_stays_positive_when_observed_end_overlaps_next_attack():
    source,meter=source_of(xml(note()+note('D')+note('E')+note('F')))
    clean=build_piece_items(source,meter,piece_id='run')
    overlap=build_piece_items(source,meter,piece_id='run',observed_events=augment_events(source,2))
    assert supervised_identity(clean)==supervised_identity(overlap)
    p=overlap[0]['payload']
    labels={tuple(edge):(bool(y),bool(m)) for edge,y,m in zip(p['potential_edges'].T,p['voice'],p['voice_mask'])}
    assert labels[(0,1)]==(True,True) and p['notes']['duration_div'][0]==3
    assert (0,1,2) in set(map(tuple,p['observed_edges'].T))
    assert (0,1,1) not in set(map(tuple,p['observed_edges'].T))


def test_new_augmented_carry_keeps_original_attack_id_and_never_owns_second_core():
    source,meter=source_of(xml(*[note('C' if i==2 else 'D' if i%2 else 'E',duration=4) for i in range(8)]))
    clean=build_piece_items(source,meter,piece_id='carry')
    changed=augment_events(source,2)
    overlap=build_piece_items(source,meter,piece_id='carry',observed_events=changed)
    before=clean[1]['payload'];after=overlap[1]['payload']
    extra=set(after['event_ids'])-set(before['event_ids'])
    assert len(extra)==1
    i=list(after['event_ids']).index(next(iter(extra)))
    assert after['notes']['onset_div'][i]==8 and after['source_end_tick'][i]==12
    assert after['notes']['duration_div'][i]==6
    assert after['carry_in'][i] and after['added_carry'][i] and not after['core_mask'][i]
    assert supervised_identity(clean)==supervised_identity(overlap)
    events=sorted(changed,key=lambda e:(e['start_tick'],e['pitch'],e['part'],e['id']))
    indices={e['id']:j for j,e in enumerate(events)}
    selected=[indices[str(identifier).split(':',1)[1]] for identifier in after['event_ids']]
    expected=_geometry(events,selected)
    assert np.array_equal(after['observed_edges'],expected)


def test_global_future_pitch_cap_applies_even_outside_first_window():
    events=[event('a',0,4),event('b',5,6)]
    observed=augment_events(raw_source(events),2)
    assert observed[0]['end_tick']==5
    # Constructing the cap from a clipped first-window list would incorrectly
    # produce six; the production API always receives the complete source.
    assert augment_events(raw_source(events[:1]),2)[0]['end_tick']==6


def test_unknown_events_and_grace_keep_masks_but_still_get_input_augmentation():
    source,meter=source_of(xml(note(ties=('start',))+note('D')+note('E',extra='<grace/>')+note('F')))
    clean=build_piece_items(source,meter,piece_id='unknown')
    changed=augment_events(source,2)
    overlap=build_piece_items(source,meter,piece_id='unknown',observed_events=changed)
    assert all(b['end_tick']>a['end_tick'] for a,b in zip(source['representation']['events'],changed))
    assert supervised_identity(clean)==supervised_identity(overlap)
    assert not overlap[0]['payload']['voice_mask'].any()
    assert len(overlap[0]['payload']['notes'])==len(clean[0]['payload']['notes'])


def test_crossbar_unproven_voice_edges_remain_unlabelled_under_overlap():
    source,meter=source_of(xml(note(duration=4),note('E',duration=4)))
    item=build_piece_items(source,meter,piece_id='bars',observed_events=augment_events(source,2))[0]
    assert item['payload']['potential_edges'].shape[1]==1
    assert not item['payload']['voice_mask'].any() and not item['payload']['voice'].any()


def test_nominal_meter_is_not_changed_by_extended_end_or_pickup():
    source,meter=source_of(xml(note(),note('D',duration=3),meter='<beats>6</beats><beat-type>8</beat-type>'))
    p=build_piece_items(source,meter,piece_id='meter',observed_events=augment_events(source,2))[0]['payload']
    assert list(p['notes']['ts_beats'])==[3.,3.]
    assert list(p['notes']['onset_div'])==[0,1]
    assert list(p['notes']['duration_div'])==[3,5]


@pytest.mark.parametrize('value',[0,-1,.5,1.,True,3,Fraction(1,2),'1'])
def test_nonprotocol_augmentation_rejects_instead_of_rounding(value):
    with pytest.raises(ValueError):
        augment_events(raw_source([event('a',0,1)]),value)


@pytest.mark.parametrize('field,value',[('start_tick',1),('pitch',61),('part','other'),
    ('supervision_eligible',False),('written_ids',['made-up']),('end_tick',0),('end_tick',1.5)])
def test_observation_override_cannot_move_attacks_change_labels_or_shorten(field,value):
    source,meter=source_of(xml(note()))
    observed=copy.deepcopy(list(source['representation']['events']))
    observed[0][field]=value
    with pytest.raises(ValueError,match='Observed'):
        build_piece_items(source,meter,piece_id='bad',observed_events=observed)


def test_observation_override_requires_complete_unique_identity_set():
    source,meter=source_of(xml(note()+note('D')))
    events=list(source['representation']['events'])
    for rows in (events[:1],[events[0],events[0]],events+[event('fake',8,9)]):
        with pytest.raises(ValueError,match='complete source identity'):
            build_piece_items(source,meter,piece_id='bad',observed_events=rows)


def test_clean_cache_checker_checks_every_field_and_exact_dtype(tmp_path):
    import json
    from scripts.piano_event_training import file_sha256
    source,meter=source_of(xml(note()+note('D')))
    items=build_piece_items(source,meter,piece_id='clean')
    item=items[0];path=tmp_path/'cache.npz'
    np.savez_compressed(path,**item['payload'],metadata_json=np.asarray(json.dumps(item['metadata'])))
    rows=[dict(chunk=0,cache=path.name,cache_sha256=file_sha256(path))]
    assert_clean_cache(items,rows,tmp_path)
    altered=copy.deepcopy(items);altered[0]['metadata']['piece_id']='changed'
    with pytest.raises(ValueError,match='Default clean metadata changed'):
        assert_clean_cache(altered,rows,tmp_path)
    altered=copy.deepcopy(items);altered[0]['payload']['notes']['duration_div'][0]+=1
    with pytest.raises(ValueError,match='Default clean payload changed'):
        assert_clean_cache(altered,rows,tmp_path)
    altered=copy.deepcopy(items);altered[0]['payload']['voice']=altered[0]['payload']['voice'].astype(np.float64)
    with pytest.raises(ValueError,match='Default clean payload changed'):
        assert_clean_cache(altered,rows,tmp_path)


def test_duplicate_supervision_cannot_hide_in_context_windows():
    source,meter=source_of(xml(note()+note('D')))
    items=build_piece_items(source,meter,piece_id='duplicate')
    with pytest.raises(ValueError,match='Duplicated successor'):
        supervised_identity(items+items)
