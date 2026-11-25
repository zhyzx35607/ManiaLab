import copy
import json

import pytest

from scripts import recover_tsumugi_attacks as recovery
from scripts.research_tsumugi_transcription import convert_notes


CLASSES=['piano','drums','melody']


def note(start=100,end=300,*,pitch=60,slot=0,instrument=0,onset=True,offset=True):
    return dict(instrument_id=instrument,pitch=pitch,start_sample=start,end_sample=end,velocity=100,
        slot_index=slot,has_onset=onset,has_offset=offset,instrument_candidates=[instrument])


def fixture(baseline=(),candidates=(),*,sr=1000,total=5000):
    rows=[dict(constructed_note=n,v1_slot_index=n['slot_index'],
               arguments=dict(boundary_flag=[n['has_onset'],n['has_offset'],.5,.5],window_start_frame=0))
          for n in candidates]
    identity=dict(resampled_sample_rate=sr,resampled_samples=total,
        model_config=dict(semi_crf_version='v1',num_pitch_slots=3))
    audit=dict(identity=identity,notes=[dict(note=n) for n in baseline],
        filtered_groups=[dict(sources=[i]) for i in range(len(rows))],
        filtered_early_unclosed_onset_source_indices=list(range(len(rows))),
        official_finalize_equal=True,saved_raw_notes_equal=True,window_snapshots_equal=True)
    events=convert_notes(baseline,sample_rate=sr,total_samples=total,classes=CLASSES)
    return audit,rows,events,CLASSES


def test_recovered_attack_has_unknown_tail_and_preserves_original_evidence():
    args=fixture([note()],[note(600,850,offset=False,instrument=2)])
    before=copy.deepcopy(args)
    result=recovery.recover_attacks(*args)
    attack=result['recovered_attacks'][0]
    assert attack['onset']==attack['offset']==.6
    assert attack['release_reason']=='unresolved_or_postprocessed'
    assert attack['event_kind']=='onset_only' and attack['program']==-1
    assert attack['instrument_class']=='melody'
    assert attack['application_ln_eligible'] is False
    assert result['decisions'][0]['original_model_interval']['end_sample']==850
    assert args==before and result['baseline_notes']==before[2]


def test_inside_existing_long_note_is_retained_without_changing_the_note():
    args=fixture([note(100,1000)],[note(600,900,offset=False)])
    result=recovery.recover_attacks(*args)
    assert result['counts']['recovered_attacks']==1
    assert result['notes'][0]==args[2][0]
    assert result['notes'][0]['offset']==1.


@pytest.mark.parametrize('delta,accepted',[(49,False),(50,False),(51,True),(-50,False),(-51,True)])
def test_existing_onset_deduplication_is_inclusive_at_50ms(delta,accepted):
    result=recovery.recover_attacks(*fixture([note(1000,1500)],[note(1000+delta,1600,offset=False)]))
    assert result['counts']['recovered_attacks']==int(accepted)


@pytest.mark.parametrize('samples,accepted',[(1102,False),(1103,True)])
def test_22050hz_deduplication_does_not_round_50ms_to_a_false_boundary(samples,accepted):
    result=recovery.recover_attacks(*fixture([note(1000,3000)],
        [note(1000+samples,4000,offset=False)],sr=22050))
    assert result['counts']['recovered_attacks']==int(accepted)


def test_pitch_and_slot_are_both_required_but_class_is_not_a_dedup_key():
    candidates=[note(100,300,offset=False,pitch=61),note(100,300,offset=False,slot=1),
                note(100,300,offset=False,instrument=2)]
    result=recovery.recover_attacks(*fixture([note()],candidates))
    assert result['counts']['recovered_attacks']==2
    assert result['counts']['duplicate_existing_onsets']==1


def test_candidates_use_earliest_accepted_greedy_rule_without_transitive_closure():
    candidates=[note(180,300,offset=False),note(140,300,offset=False),note(100,300,offset=False)]
    result=recovery.recover_attacks(*fixture([],candidates))
    assert [n['onset'] for n in result['recovered_attacks']]==[.1,.18]
    assert [n['recovery_source_index'] for n in result['recovered_attacks']]==[2,0]
    assert result['decisions'][1]['recovered_source_indices']==[2]


def test_existing_duplicates_take_priority_over_recovered_duplicates():
    result=recovery.recover_attacks(*fixture([note(100,300)],
        [note(140,300,offset=False),note(180,300,offset=False)]))
    assert [d['decision'] for d in result['decisions']]==['duplicate_existing_onset','recovered_onset_only']


def test_errors_in_class_prediction_and_fallback_onsets_are_preserved():
    args=fixture([],[note(offset=False,instrument=1)])
    args[1][0]['arguments']['boundary_flag']=None
    result=recovery.recover_attacks(*args)
    assert result['recovered_attacks'][0]['is_drum']
    assert result['decisions'][0]['arguments']['boundary_flag'] is None


@pytest.mark.parametrize('field',['official_finalize_equal','saved_raw_notes_equal','window_snapshots_equal'])
def test_unverified_replay_cannot_recover(field):
    args=fixture([],[note(offset=False)])
    args[0][field]=False
    with pytest.raises(ValueError,match='verified'):
        recovery.recover_attacks(*args)


@pytest.mark.parametrize('change',['missing_group','duplicate_index','closed','no_onset','bad_slot','changed_baseline'])
def test_invalid_sources_and_altered_baseline_are_rejected(change):
    args=fixture([note()],[note(600,800,offset=False)])
    if change=='missing_group': args[0]['filtered_groups']=[]
    elif change=='duplicate_index': args[0]['filtered_early_unclosed_onset_source_indices']=[0,0]
    elif change=='closed': args[1][0]['constructed_note']['has_offset']=True
    elif change=='no_onset': args[1][0]['constructed_note']['has_onset']=False
    elif change=='bad_slot': args[1][0]['v1_slot_index']=3
    else: args[2][0]['offset']+=.01
    with pytest.raises(ValueError): recovery.recover_attacks(*args)


def event(onset=.2,offset=.6,*,pitch=60,drum=False,unknown=False):
    result=dict(onset=onset,offset=offset,pitch=pitch,program=-1,is_drum=drum)
    if unknown:
        result.update(offset=onset,event_kind='onset_only',release_reason='unresolved_or_postprocessed',
                      application_ln_eligible=False)
    return result


def score(reference,predicted):
    return recovery.onset_metrics(reference,predicted,begin=0,end=1,context_start=0,context_end=2)


def test_unknown_release_matches_onset_without_exposing_interval_or_release_scores():
    metrics,_=score([event()],[event(unknown=True)])
    assert metrics['standard']['matched']==1
    assert all('release' not in k and 'interval' not in k for v in metrics.values() for k in v)


def test_drums_and_duplicates_stay_in_denominator_and_cannot_match_twice():
    metrics,scope=score([event()],[event(unknown=True),event(unknown=True),event(drum=True,unknown=True)])
    value=metrics['standard']
    assert (value['matched'],value['predicted'],value['false_positives'])==(1,3,2)
    assert scope['predicted_drums']==1 and scope['onset_only_predictions']==3
    metrics,_=score([event()],[event(drum=True,unknown=True)])
    assert metrics['standard']['matched']==0


def test_onset_selection_is_half_open_and_ignores_crossing_or_unknown_tails():
    metrics,scope=score([event(0,.5),event(.999,1.5),event(1,1.8)],
        [event(0,unknown=True),event(.999,3),event(1,unknown=True)])
    assert scope['reference_indices']==[0,1] and scope['predicted_indices']==[0,1]
    assert metrics['standard']['matched']==2


def test_matching_is_maximum_one_to_one_and_tolerances_remain_distinct():
    metrics,_=score([event(0,.3),event(.08,.3)],[event(.04,unknown=True),event(0,unknown=True)])
    assert metrics['standard']['matched']==2
    metrics,_=score([event()],[event(.245,unknown=True)])
    assert metrics['standard']['matched']==1 and metrics['fixed_40_80']['matched']==0


def test_bare_zero_length_event_and_nonfinite_values_are_rejected():
    with pytest.raises(ValueError,match='Zero duration'): score([event()],[event(.2,.2)])
    with pytest.raises(ValueError,match='Nonfinite'): score([event()],[event(float('nan'),unknown=True)])


def test_new_output_does_not_overwrite_earlier_evidence(tmp_path):
    path=tmp_path/'report.json'
    recovery.write_new(path,{'before':1})
    with pytest.raises(ValueError,match='Preserve'): recovery.write_new(path,{'after':2})
    assert json.loads(path.read_text())=={'before':1}
