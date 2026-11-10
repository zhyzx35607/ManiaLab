"""Synthetic scoring, provenance and scheduling controls; never run a model."""
from dataclasses import asdict,dataclass,replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_bass_model_comparison as runner
from scripts.evaluate_instrument_benchmark import match


def note(onset=.2,offset=.8,*,pitch=40,drum=False,pluck='FS',expression='NO'):
    return dict(onset=onset,offset=offset,pitch=pitch,program=-1,is_drum=drum,
        plucking_style=pluck,expression_style=expression)


def test_one_global_assignment_cannot_reuse_prediction_for_two_techniques():
    refs=[note(.1,.5,pluck='FS'),note(.14,.6,pluck='PK')]
    predicted=[note(.12,.6)]
    pairs=runner.global_matches(refs,predicted)
    recall=runner.technique_recall(refs,pairs)
    assert len(pairs)==1
    assert sum(v['matched'] for v in recall['plucking_style'].values())==1
    assert sum(v['reference'] for v in recall['plucking_style'].values())==2
    assert all('f1' not in v and 'precision' not in v for v in recall['plucking_style'].values())


def test_maximum_cardinality_matching_beats_greedy_nearest():
    refs=[note(0,.5),note(.08,.5)]
    predicted=[note(.04,.5),note(0,.5)]
    assert runner.global_matches(refs,predicted)==[[0,1],[1,0]]


def test_pitch_agnostic_matching_accepts_wrong_pitch_and_drum_without_hiding_denominator():
    refs=[note()]
    predicted=[note(pitch=90,drum=True),note(.9,1.2,pitch=100)]
    result=runner.score_case(refs,predicted,end=1,context_end=2)
    assert result['metrics']['standard']['pitched_onsets']['matched']==0
    assert result['pitch_agnostic']['standard']['matched']==1
    assert result['pitch_agnostic']['standard']['predicted']==2
    assert result['interval_scope']['predicted_drums']==1


def test_zero_length_attack_matches_onset_never_interval_and_adds_interval_fp():
    refs=[note()]
    first=runner.score_case(refs,[note()],end=1,context_end=2)
    result=runner.score_case(refs,[note(),note(.2,.2)],end=1,context_end=2)
    old=first['metrics']['standard']['complete_intervals']
    new=result['metrics']['standard']['complete_intervals']
    assert old['matched']==new['matched']==1
    assert new['available'] and new['predicted']==2 and new['false_positives']==1
    assert new['precision']==.5 and new['f1']==pytest.approx(2/3)
    zero=runner.score_case(refs,[note(.2,.2)],end=1,context_end=2)
    assert zero['metrics']['standard']['pitched_onsets']['matched']==1
    assert zero['metrics']['standard']['complete_intervals']['matched']==0
    assert zero['matched_indices']['standard']['nominal_complete_intervals']==[]


def test_only_unobserved_reference_release_invalidates_interval_metric():
    result=runner.score_case([note(.2,2.)],[note(.2,.2)],end=1,context_end=2)
    assert result['metrics']['standard']['complete_intervals']['available'] is False
    assert result['metrics']['standard']['complete_intervals']['predicted']==1
    assert result['metrics']['standard']['pitched_onsets']['matched']==1


def test_unknown_crossing_prediction_tails_are_preserved_in_full_denominator():
    predicted=[note(.2,5.),note(.4,.4),note(1.,2.)]
    predicted[0]['release_reason']='input_boundary'
    result=runner.score_case([note()],predicted,end=1,context_end=2)
    scope=result['interval_scope']
    assert scope['predicted_indices']==[0,1] and scope['predicted_total']==3
    assert scope['predicted_crosses_context_end']==1
    assert result['metrics']['standard']['complete_intervals']['predicted']==2
    assert predicted[0]['offset']==5.


def test_matching_indices_refer_to_original_arrays_after_window_selection():
    refs=[note(1.2,1.8),note(.2,.8),note(.5,.9)]
    predicted=[note(1.1,1.4),note(.5,.9),note(.2,.8)]
    result=runner.score_case(refs,predicted,end=1,context_end=2)
    assert result['matched_indices']['standard']['nominal_pitched_onsets']==[[1,2],[2,1]]


def test_nominal_global_matching_agrees_with_frozen_matcher_in_both_tolerances():
    import random
    random.seed(13)
    for _ in range(20):
        refs=[note(random.randrange(30)/100,1.,pitch=random.randrange(39,42)) for _ in range(7)]
        predicted=[note(random.randrange(30)/100,1.,pitch=random.randrange(39,42),drum=i%5==0) for i in range(9)]
        for strict in (False,True):
            for offsets in (False,True):
                pairs=runner.global_matches(refs,predicted,strict=strict,offset=offsets)
                assert len(pairs)==match(refs,predicted,strict=strict,offset=offsets)['matched']


def test_technique_aggregate_does_not_invent_fp_attribution_or_subgroup_f1():
    result=runner.score_case([note(expression='DN')],[note(),note(.7,.9)],end=1,context_end=2)
    aggregate=runner.aggregate_extra([result,result])['standard']
    assert aggregate['pitch_agnostic']['false_positives']==2
    value=aggregate['technique_recall']['nominal_pitched_onsets']['expression_style']['DN']
    assert value==dict(reference=2,matched=2,false_negatives=0,recall=1.)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def saved_output(tmp_path,identity,artifacts=runner.APP_ARTIFACTS):
    tmp_path.mkdir(parents=True,exist_ok=True)
    for filename in artifacts:
        write(tmp_path/filename,[note()] if filename=='notes.json' else [])
    report=dict(identity=identity,notes=1,context_start_seconds=0,
        context_end_seconds=identity['original_samples']/identity['original_sample_rate'],
        **{key:runner.sha(tmp_path/name) for name,key in artifacts.items()})
    write(tmp_path/'report.json',report)
    return report


@pytest.mark.parametrize('change',['identity','notes','activations','suppressed','count','clock'])
def test_reused_output_must_match_identity_all_evidence_counts_and_clock(tmp_path,change):
    expected=dict(original_samples=3000,original_sample_rate=1000,threshold=.45)
    report=saved_output(tmp_path,expected)
    if change=='identity': report['identity']=dict(expected,threshold=.5)
    elif change=='count': report['notes']=2
    elif change=='clock': report['context_end_seconds']=2.
    else:
        file={'notes':'notes.json','activations':'activations.npz','suppressed':'articulation-suppressed.json'}[change]
        (tmp_path/file).write_text('["changed"]')
    write(tmp_path/'report.json',report)
    with pytest.raises(ValueError): runner.verify_output(tmp_path,expected,runner.APP_ARTIFACTS)


def test_valid_output_is_reused_without_model_or_source_audio_access(tmp_path):
    expected=dict(original_samples=3000,original_sample_rate=1000)
    saved_output(tmp_path,expected)
    _,notes=runner.verify_output(tmp_path,expected,runner.APP_ARTIFACTS)
    assert notes==[note()]


@pytest.mark.parametrize('change',['source','runtime','threshold','pcm','resampled_clock'])
def test_application_reuse_is_bound_to_current_frozen_bass_pipeline(tmp_path,change):
    application=dict(source_files={'audio.py':'a'*64},versions={'numpy':'1'},settings={'onset_threshold':.45})
    plan=dict(application=application,manifest_sha256='b'*64,runner_sha256='c'*64)
    case=dict(audio=str(tmp_path/'original.wav'),audio_sha256='d'*64,sample_rate=44100,samples=132300)
    expected=dict(application,manifest_sha256=plan['manifest_sha256'],runner_sha256=plan['runner_sha256'],
        audio=str(Path(case['audio']).resolve()),audio_sha256=case['audio_sha256'],original_sample_rate=44100,
        original_samples=132300,original_channels=1,resampled_samples=66150,pcm_sha256='e'*64,resampled_pcm_sha256='f'*64)
    report=saved_output(tmp_path,expected)
    report['resampled_duration_seconds']=3.
    write(tmp_path/'report.json',report)
    assert runner.verify_application(tmp_path,plan,case)[1]==[note()]
    report=json.loads(json.dumps(report))
    if change=='source': report['identity']['source_files']['audio.py']='0'*64
    elif change=='runtime': report['identity']['versions']['numpy']='2'
    elif change=='threshold': report['identity']['settings']['onset_threshold']=.5
    elif change=='pcm': report['identity']['pcm_sha256']='unknown'
    else: report['resampled_duration_seconds']=3.1
    write(tmp_path/'report.json',report)
    with pytest.raises(ValueError): runner.verify_application(tmp_path,plan,case)


def test_tsumugi_reuse_rejects_changed_window_merge_runtime_or_class_filter(tmp_path):
    plan={'tsumugi':dict(source_files={},adapter_sha256='a'*64,versions={'test':'1'})}
    case=dict(audio=str(tmp_path/'audio.wav'),audio_sha256='b'*64,sample_rate=44100,samples=132300)
    expected=runner.expected_tsumugi(plan,case)
    expected['model_config']={'semi_crf_version':'v1'}
    report=saved_output(tmp_path/'out',expected,runner.TSUMUGI_ARTIFACTS)
    write(tmp_path/'out'/'identity.json',expected)
    assert runner.verify_tsumugi(tmp_path/'out',plan,case)[1]==[note()]
    for key,value in (('merge_onset_ms',55.),('window_ms',7000),('allowed_instrument_ids',[0])):
        changed=json.loads(json.dumps(report))
        changed['identity']['settings'][key]=value
        write(tmp_path/'out'/'report.json',changed)
        write(tmp_path/'out'/'identity.json',changed['identity'])
        with pytest.raises(ValueError): runner.verify_tsumugi(tmp_path/'out',plan,case)


def test_partial_cohort_fails_before_reading_labels(tmp_path,monkeypatch):
    def no_read(*args): raise AssertionError('Partial experiment attempted to read labels')
    monkeypatch.setattr(runner,'read_json',no_read)
    with pytest.raises(ValueError,match='partial'): runner.collect_scores(tmp_path,tmp_path,{},[{'name':'001'}])


@dataclass
class Candidate:
    start:int
    end:int
    pitch:int=40
    strength:float=1.
    release_reason:str='input_boundary'
    voice:str='mix'
    id:str='test'
    def data(self): return asdict(self)


def test_application_sequence_uses_bass_suppression_filter_refine_and_fuse_without_dropping_zero():
    calls=[]
    attack=Candidate(100,100)
    class Backend:
        SR=1000
        def rhythmic_evidence(self,signal,estimate):
            assert estimate is False
            calls.append('rhythm')
            return [attack],None
        def source_evidence(self,notes,stem,mix):
            assert stem is mix
            calls.append('source')
            return notes,0
        def pitched_notes(self,*args,suppressed):
            calls.append('pitch-suppressed' if suppressed else 'pitch-all')
            return [Candidate(100,300)] if suppressed else [Candidate(100,300),Candidate(200,400)]
        def refine_releases(self,notes,signal):
            calls.append('refine')
            return [replace(notes[0],end=100)],{'shortened':1}
        def fuse_attacks(self,notes,attacks):
            calls.append('fuse')
            return notes,attacks,0
    def steady(*args):
        calls.append('steady')
        assert args[-2:]==(.45,.3)
        return [{'pitch':40,'time_ms':200}]
    events,evidence,suppressed=runner.application_pipeline(Backend(),steady,[0]*1000,[],[],[],
        dict(onset_threshold=.45,frame_threshold=.3))
    assert calls==['rhythm','source','steady','pitch-all','pitch-suppressed','source','refine','fuse']
    assert len(events)==1 and events[0]['onset']==events[0]['offset']==.1
    assert len(evidence['pitched_without_articulation_suppression'])==2
    assert evidence['final_candidates'][0]['voice']=='bass'
    assert suppressed==[{'pitch':40,'time_ms':200}]


@pytest.mark.parametrize('incomplete',[False,True])
def test_runner_finishes_all_application_cases_before_any_tsumugi_case(tmp_path,monkeypatch,incomplete):
    root=tmp_path/'root'
    output=tmp_path/'output'
    cases=[dict(name=f'{i:03}',sample_rate=44100,samples=132300,duration_seconds=3.,seconds=1.,
        audio=str(root/f'{i:03}.wav'),audio_sha256='a'*64) for i in range(1,18)]
    write(root/'manifest.json',dict(revision=1,cases=cases))
    calls=[]
    actual_sha=runner.sha
    monkeypatch.setattr(runner,'sha',lambda path:actual_sha(path) if Path(path).is_file() else 'a'*64)
    monkeypatch.setattr(runner,'verified_hash',lambda *args:None)
    monkeypatch.setattr(runner,'verify_source',lambda *args:{})
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *args,**kwargs:'{}')
    monkeypatch.setattr(runner,'verify_tsumugi',lambda *args:None)
    monkeypatch.setattr(runner,'collect_scores',lambda *args:dict(groups={},cases=17))
    def fake_run(argv,**kwargs):
        if 'application-stage' in argv:
            plan=runner.read_json(output/'plan.json')
            selected=cases[:-1] if incomplete else cases
            for case in selected:
                calls.append(('application',case['name']))
                write(output/'application'/case['name']/'report.json',{})
            write(output/'application'/'completed.json',dict(identity=plan['application'],cases=17,
                plan_sha256=runner.sha(output/'plan.json'),case_reports={c['name']:dict(
                    report_sha256=runner.sha(output/'application'/c['name']/'report.json')) for c in selected}))
        else:
            folder=Path(argv[argv.index('--output')+1])
            calls.append(('tsumugi',folder.name))
            write(folder/'report.json',{})
    monkeypatch.setattr(runner.subprocess,'run',fake_run)
    args=SimpleNamespace(root=root,output=output,source=tmp_path/'source',model=tmp_path/'model',
        application_python=tmp_path/'app-python',tsumugi_python=tmp_path/'model-python',dependencies=tmp_path/'deps')
    if incomplete:
        with pytest.raises(ValueError,match='incomplete'): runner.run(args)
        assert all(engine=='application' for engine,_ in calls)
        assert not (output/'completed.json').exists()
    else:
        runner.run(args)
        assert calls==[('application',f'{i:03}') for i in range(1,18)]+[('tsumugi',f'{i:03}') for i in range(1,18)]
        assert (output/'completed.json').exists()
