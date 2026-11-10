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


def test_incomplete_cohort_fails_before_reading_manifest_or_any_labels(tmp_path,monkeypatch):
    def unexpected(*args,**kwargs): raise AssertionError('Read attempted before completion')
    monkeypatch.setattr(recovery,'read_json',unexpected)
    with pytest.raises(ValueError,match='100 Tsumugi'):
        recovery.evaluate_cohort(tmp_path,tmp_path/'unfinished',tmp_path/'source')


def test_new_output_does_not_overwrite_earlier_evidence(tmp_path):
    path=tmp_path/'report.json'
    recovery.write_new(path,{'before':1})
    with pytest.raises(ValueError,match='Preserve'): recovery.write_new(path,{'after':2})
    assert json.loads(path.read_text())=={'before':1}


def test_protocol_is_the_fixed_published_prelabel_document():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'docs/tsumugi-attack-recovery-protocol.md'
    assert recovery.sha(path)==recovery.PROTOCOL_SHA256
    assert recovery.DEDUPLICATION_MS==50


def synthetic_completed_cohort(tmp_path):
    """Public-format synthetic data; no audio, checkpoint, or benchmark labels."""
    root,predictions=tmp_path/'root',tmp_path/'predictions'
    root.mkdir()
    predictions.mkdir()
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8')
    cases,rows=[],[]
    references={}
    for name,pitch in (('strings',60),('A1',60),('A2',62)):
        label=root/(name+'.json')
        write(label,[event(pitch=pitch)])
        references[name]=dict(path=str(label),sha256=recovery.sha(label),count=1,scored_count=1)
    for index in range(100):
        dataset='GuitarSet' if index<60 else 'vocadito'
        annotations=['strings'] if index<60 else ['A1','A2']
        name=f'case-{index:03}'
        cases.append(dict(name=name,dataset=dataset,references={a:references[a] for a in annotations},
            audio=str(root/(name+'.wav')),audio_sha256='a'*64,sample_rate=1000,samples=3000,
            duration_seconds=3.,seconds=1.,start_ms=0))
        folder=predictions/name
        write(folder/'report.json',{})
        write(folder/'notes.json',[event()])
        refs={}
        for ann in annotations:
            metrics,_=score([event(pitch=62 if ann=='A2' else 60)],[event()])
            refs[ann]=dict(tsumugi=dict(metrics={k:dict(pitched_onsets=v) for k,v in metrics.items()}))
        rows.append(dict(name=name,report_sha256=recovery.sha(folder/'report.json'),
            notes_sha256=recovery.sha(folder/'notes.json'),references=refs))
    write(root/'manifest.json',dict(revision=1,cases=cases))
    plan=dict(manifest_sha256=recovery.sha(root/'manifest.json'),
        runner_sha256=recovery.sha('scripts/run_tsumugi_guitar_vocal_comparison.py'),
        adapter_sha256=recovery.sha('scripts/research_tsumugi_transcription.py'),
        evaluator_sha256=recovery.sha(recovery.cohort_evaluation.__file__),
        matcher_sha256=recovery.sha(recovery.onset_evaluation.__file__),
        stem_filter='all',normalization=False,inference_scope='full_original_audio',
        source_commit='synthetic',source_files={},model_revision='synthetic',
        selection={'GuitarSet':'guitar','vocadito':'vocal'},models={'guitar':'g'*64,'vocal':'v'*64})
    write(predictions/'plan.json',plan)
    write(predictions/'comparison.json',dict(identity=plan,cases=100,annotations=140,rows=rows))
    write(predictions/'completed.json',dict(identity=plan,cases=100,
        plan_sha256=recovery.sha(predictions/'plan.json'),comparison_sha256=recovery.sha(predictions/'comparison.json')))
    def recover_case(path,source):
        case=next(c for c in cases if c['name']==path.name)
        model=plan['selection'][case['dataset']]
        identity=dict(source_commit=plan['source_commit'],source_files=plan['source_files'],
            runner_sha256=plan['adapter_sha256'],model_revision=plan['model_revision'],model=model,
            model_sha256=plan['models'][model],audio=str(__import__('pathlib').Path(case['audio']).resolve()),
            audio_sha256=case['audio_sha256'],original_sample_rate=1000,original_samples=3000,
            original_channels=1,resampled_sample_rate=22050,resampled_samples=66150,
            normalization=False,inference_scope='full_original_audio',stem_filter='all',evidence_revision=2)
        return dict(identity=identity,baseline_notes=[event()],
            notes=[event(),event(pitch=62,unknown=True),event(drum=True,unknown=True)])
    return root,predictions,recover_case


def test_full_synthetic_100_case_entry_keeps_a1_a2_independent_and_wrong_classes(tmp_path,monkeypatch):
    root,predictions,recover_case=synthetic_completed_cohort(tmp_path)
    monkeypatch.setattr(recovery,'recover_case',recover_case)
    result=recovery.evaluate_cohort(root,predictions,tmp_path/'source')
    assert result['cases']==100 and result['annotations']==140 and result['complete']
    assert result['interval_scoring'] is False
    before,after=result['groups']['baseline'],result['groups']['recovered']
    assert before['GuitarSet/strings']['standard']['total_cases']==60
    assert before['vocadito/A1']['standard']['micro']['matched']==40
    assert before['vocadito/A2']['standard']['micro']['matched']==0
    for ann in ('A1','A2'):
        assert after['vocadito/'+ann]['standard']['micro']['matched']==40
        assert after['vocadito/'+ann]['standard']['micro']['predicted']==120
        assert after['vocadito/'+ann]['standard']['total_cases']==40
    assert all(r['references'][ann]['recovered']['scope']['predicted_drums']==1
        for r in result['rows'][60:] for ann in ('A1','A2'))


@pytest.mark.parametrize('change',['row_removed','missing_a2','changed_report','changed_manifest','changed_notes'])
def test_completion_preflight_rejects_missing_or_changed_evidence(tmp_path,change):
    root,predictions,_=synthetic_completed_cohort(tmp_path)
    if change in ('row_removed','missing_a2'):
        path=predictions/'comparison.json'
        value=json.loads(path.read_text())
        if change=='row_removed': value['rows'].pop()
        else: value['rows'][-1]['references'].pop('A2')
        path.write_text(json.dumps(value),encoding='utf-8')
        completed=json.loads((predictions/'completed.json').read_text())
        completed['comparison_sha256']=recovery.sha(path)
        (predictions/'completed.json').write_text(json.dumps(completed),encoding='utf-8')
    elif change=='changed_report': (predictions/'case-000'/'report.json').write_text('{"changed":true}')
    elif change=='changed_notes': (predictions/'case-000'/'notes.json').write_text('[]')
    else: (root/'manifest.json').write_text((root/'manifest.json').read_text()+' ')
    with pytest.raises(ValueError): recovery.completed_inputs(root,predictions)
