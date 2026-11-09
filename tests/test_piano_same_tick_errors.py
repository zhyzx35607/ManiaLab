"""Independent source-notation audit: known errors, unknowns and chord loss."""
import numpy as np
import pytest

from scripts.summarize_piano_same_tick_errors import (classify_same_tick_negatives,
    error_record, false_positive_transitions, source_context)


def example():
    notes = [dict(id='a', part='p', pitch=60, start_tick=0, end_tick=120, measure=1, staff=1, voice='1'),
             dict(id='b', part='p', pitch=62, start_tick=120, end_tick=240, measure=1, staff=1, voice='1'),
             dict(id='c', part='p', pitch=65, start_tick=120, end_tick=240, measure=1, staff=1, voice='1'),
             dict(id='d', part='p', pitch=70, start_tick=120, end_tick=240, measure=1, staff=1, voice='2'),
             dict(id='e', part='p', pitch=40, start_tick=120, end_tick=240, measure=1, staff=2, voice='1'),
             dict(id='f', part='q', pitch=40, start_tick=120, end_tick=240, measure=1, staff=1, voice='1')]
    source = dict(written=[dict(note=n) for n in notes], excluded=[],
        representation=dict(events=[dict(id=n['id'],part=n['part'],pitch=n['pitch'],start_tick=n['start_tick'],
            end_tick=n['end_tick'],written_ids=[n['id']],supervision_eligible=True) for n in notes]))
    data = dict(source_ids=np.asarray(['hash:a']*5),target_ids=np.asarray(['hash:'+x for x in 'bcdef']),
        target_ticks=np.asarray([120]*5),actual=np.asarray([True,True,False,False,False]),
        score_mask=np.asarray([True,True,True,True,False]))
    return source, data


def test_classifies_known_same_tick_negative_staff_voice_and_excludes_unknown_part():
    source, data = example()
    assert classify_same_tick_negatives(data,source_context(source),'hash').tolist()==[0,0,1,2,0]


def test_uses_final_written_fragment_after_tie_crosses_measure_and_staff():
    source,data = example()
    first=source['written'][0]['note'];first.update(end_tick=60,measure=0,staff=2,voice='9')
    last=dict(first,id='a-tail',start_tick=60,end_tick=120,measure=1,staff=1,voice='1')
    source['written'].append(dict(note=last));source['representation']['events'][0]['written_ids'].append('a-tail')
    assert classify_same_tick_negatives(data,source_context(source),'hash').tolist()==[0,0,1,2,0]


def test_rejects_missing_true_chord_target():
    source,data=example(); data={k:v[[0,2,3,4]] for k,v in data.items()}
    with pytest.raises(ValueError,match='omitted'):
        classify_same_tick_negatives(data,source_context(source),'hash')


@pytest.mark.parametrize('field', ['actual','score_mask','target_ticks','source_ids'])
def test_changed_identity_clock_or_label_is_not_silently_reclassified(field):
    source,data=example()
    if field in ('actual','score_mask'):data[field][2]=~data[field][2]
    elif field=='target_ticks':data[field][2]+=1
    else:data[field][2]='wrong:a'
    with pytest.raises(ValueError):
        classify_same_tick_negatives(data,source_context(source),'hash')


def test_counts_repaired_introduced_and_persistent_known_false_predictions():
    result=false_positive_transitions(np.asarray([True,True,True,False]),
        np.asarray([True,False,True,True]),np.asarray([False,True,True,False]))
    assert result==dict(before_fp=2,after_fp=2,repaired_fp=1,introduced_fp=1,persistent_fp=1)


def test_partial_chord_preservation_does_not_count_as_complete():
    source,data=example();codes=classify_same_tick_negatives(data,source_context(source),'hash')
    previous={**data,'logits':np.asarray([1.,1.,1.,-1.,100.]),'new_selected':np.asarray([True,True,True,False,True])}
    data.update(logits=np.asarray([1.,-1.,-1.,1.,100.]),new_selected=np.asarray([True,False,False,True,True]))
    result=error_record(data,previous,codes)
    assert result['negative_categories']['all_same_tick']['raw']==dict(before_fp=1,after_fp=1,repaired_fp=1,introduced_fp=1,persistent_fp=0)
    groups=result['true_successor_groups']['decoded']['multiple_true_targets']
    assert groups['before_all']==1 and groups['after_all']==0 and groups['all_lost']==1
    assert groups['before_at_least_one']==groups['after_at_least_one']==1
    assert groups['true_edges']==2
