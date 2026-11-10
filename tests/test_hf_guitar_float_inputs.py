"""Float-chain and production-replay controls; no model/ORT session is loaded."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.signal import resample_poly

from scripts import check_hf_guitar_float_inputs as probe
from manialab import guitar
from manialab.piano import decode


def gate():
    names=[f'case-{i:02d}' for i in range(60)]
    rows=[dict(name=name,physical_comparisons={key:dict(preserved=True) for key in probe.adapter.OUTPUTS},
               window_comparisons={key:dict(preserved=True) for key in probe.adapter.OUTPUTS},
               events_preserved=True,interval_scope_unchanged=True,scoring_count_changes=[]) for name in names]
    rows[0]['window_comparisons']['velocity_output']['preserved']=False
    result=dict(plan=dict(model='guitar_fl',graph_sha256=probe.GRAPH_SHA,source_report_sha256=probe.SOURCE_SHA,
        cases=names,activation_rtol=.001,activation_atol=.0001,endpoint_tolerance_seconds=.001,
        device='CPUExecutionProvider',intra_threads=2,inter_threads=1),completed_cases=60,rows=rows,
        all_physical_activations_preserved=True,all_physical_events_preserved=True,
        all_scoring_counts_and_scopes_preserved=True,all_window_activations_preserved=False,
        original_all_window_protocol_passed=False)
    return result,names


def test_gate_allows_new_float_probe_but_preserves_original_window_failure():
    report,names=gate()
    result=probe.validate_gate(report,names)
    assert result['physical_events_scores_passed'] is True
    assert result['original_all_window_protocol_passed'] is False
    assert result['all_window_activations_preserved'] is False


@pytest.mark.parametrize('damage', ['incomplete','duplicate','order','graph','physical','event','score','scope',
                                  'tolerance','hidden_window_failure','missing_window_head','nonboolean'])
def test_gate_rejects_missing_or_failed_prerequisites(damage):
    report,names=gate()
    if damage=='incomplete':report['rows'].pop()
    elif damage=='duplicate':report['rows'][1]=deepcopy(report['rows'][0])
    elif damage=='order':report['rows'].reverse()
    elif damage=='graph':report['plan']['graph_sha256']='0'*64
    elif damage=='physical':report['rows'][1]['physical_comparisons']['frame_output']['preserved']=False
    elif damage=='event':report['rows'][1]['events_preserved']=False
    elif damage=='score':report['rows'][1]['scoring_count_changes']=[dict(field='matched',before=10,after=9)]
    elif damage=='scope':report['rows'][1]['interval_scope_unchanged']=False
    elif damage=='tolerance':report['plan']['activation_atol']=.01
    elif damage=='hidden_window_failure':report['original_all_window_protocol_passed']=True
    elif damage=='missing_window_head':report['rows'][1]['window_comparisons'].pop('velocity_output')
    elif damage=='nonboolean':report['all_physical_events_preserved']=1
    with pytest.raises(ValueError):probe.validate_gate(report,names)


def test_two_stage_chain_is_exact_and_retains_original_physical_clock():
    integer=np.array([-32768,32767,128,-7,0]*8909,dtype=np.int16)
    floating,model,identity=probe.derived_input(integer,44100)
    expected_float=resample_poly(integer.astype(np.float32)/32768,1,2).astype(np.float32)
    expected_model=resample_poly(expected_float,320,441).astype(np.float32)
    np.testing.assert_array_equal(floating,expected_float)
    np.testing.assert_array_equal(model,expected_model)
    assert identity['physical_samples']==len(integer) and identity['physical_rate']==44100
    assert len(floating)==(len(integer)+1)//2
    assert identity['float_pcm_sha256']==probe.pcm_hash(expected_float)
    assert identity['normalization'] is False and identity['clipping'] is False


def test_float_resampling_preserves_overshoot_and_low_level_detail():
    samples=np.array([1.35,-.17,.00000417]*1000,dtype=np.float32)
    actual=probe.resample_float(samples,22050,16000)
    expected=resample_poly(samples,320,441).astype(np.float32)
    np.testing.assert_array_equal(actual,expected)
    quantized=resample_poly((np.clip(samples,-1,1)*32767).astype(np.int16).astype(np.float32)/32768,320,441)
    assert not np.array_equal(actual,quantized)
    np.testing.assert_array_equal(probe.resample_float(samples,22050,22050),samples)


@pytest.mark.parametrize('array,rate', [(np.zeros(100,dtype=np.float32),44100),
    (np.zeros(100,dtype=np.int16),22050),(np.zeros((2,2),dtype=np.int16),44100)])
def test_derived_inputs_cannot_silently_change_source_encoding(array,rate):
    with pytest.raises(ValueError):probe.derived_input(array,rate)


def recording():
    integer=np.zeros(441*101,dtype=np.int16)
    floating,pcm,identity=probe.derived_input(integer,44100)
    heads={key:np.zeros((1,1001,88),dtype=np.float32) for key in probe.adapter.OUTPUTS}
    # A short observed decay and a sustained unknown boundary tail.
    for pitch,start,stop in [(39,5,20),(40,90,1001)]:
        heads['reg_onset_output'][0,start,pitch]=.9
        heads['frame_output'][0,start:stop,pitch]=.8
        heads['velocity_output'][0,:,pitch]=.8
    physical={key:value[0,:101] for key,value in heads.items()}
    return floating,pcm,np.array([0],dtype=np.int64),heads,physical,identity


def test_recorded_replay_keeps_extra_application_frame_and_unknown_tail():
    floating,pcm,starts,windows,physical,_=recording()
    heads,events,modes,report=probe.replay_runtime(floating,pcm,starts,windows,physical)
    original=decode(physical,**probe.adapter.THRESHOLDS)
    assert report['recorded_windows_replayed']==1 and report['new_neural_inference'] is False
    assert report['original_clock_frames']==101 and report['application_float_clock_frames']==102
    assert report['application_extra_physical_frames']==1 and len(heads['frame_output'])==102
    assert len(events)==len(original)==2
    assert events[-1]['offset']-original[-1]['offset']==pytest.approx(.01,abs=1e-6)
    assert events[-1]['release_reason']=='input_boundary' and events[-1]['release_observed'] is False
    assert all(n['release_reason'] in guitar.UNOBSERVED_RELEASES for n in modes['default_onsets']['candidates'])
    assert modes['predicted_releases']['trace'][0]['model_predicted_release_hypothesis'] is True
    assert modes['predicted_releases']['trace'][-1]['model_predicted_release_hypothesis'] is False
    assert modes['predicted_releases']['trace'][-1]['raw_event']==events[-1]


def test_recorded_replay_rejects_modified_float_samples_without_running_nn():
    floating,pcm,starts,windows,physical,_=recording()
    floating[50]=.1
    with pytest.raises(ValueError,match='float input sample'):
        probe.replay_runtime(floating,pcm,starts,windows,physical)


def test_recorded_replay_rejects_missing_or_extra_windows():
    floating,pcm,starts,windows,physical,_=recording()
    with pytest.raises(ValueError,match='extra inference windows'):
        probe.replay_runtime(floating,pcm,starts[:0],{k:v[:0] for k,v in windows.items()},physical)
    extra={k:np.repeat(v,2,axis=0) for k,v in windows.items()}
    with pytest.raises(ValueError,match='omitted recorded'):
        probe.replay_runtime(floating,pcm,np.array([0,80000]),extra,physical)


def test_inference_capture_preserves_padding_and_full_unconsumed_heads():
    floating,pcm,_,windows,_,identity=recording()
    seen=[]
    class Symbolic:
        def run(self,names,inputs):
            seen.append(inputs['audio'].copy())
            return [windows[k] for k in names]
    starts,captured,heads,physical,times,valid=probe.infer_windows(Symbolic(),pcm,
        identity['physical_samples'],identity['physical_rate'])
    assert starts.tolist()==[0] and len(heads['frame_output'])==1001 and valid.sum()==101
    np.testing.assert_array_equal(seen[0][0,:len(pcm)],pcm)
    assert not seen[0][0,len(pcm):].any()
    np.testing.assert_array_equal(captured['frame_output'],windows['frame_output'])
    assert len(physical['frame_output'])==101 and len(times)==1001


def test_saved_evidence_keeps_both_clocks_and_candidate_modes(tmp_path):
    floating,pcm,_,windows,_,identity=recording()
    class Symbolic:
        def run(self,names,inputs):return [windows[k] for k in names]
    folder=tmp_path/'recording'
    events,row=probe.save_prediction(folder,Symbolic(),floating,pcm,identity)
    assert row['events']==row['runtime_events']==len(events)==2
    assert row['physical_frames']==101 and row['runtime_replay']['application_float_clock_frames']==102
    assert set(row['artifacts'])=={p.name for p in folder.iterdir()}
    for name,digest in row['artifacts'].items():assert probe.sha(folder/name)==digest
    with np.load(folder/'activations.npz',allow_pickle=False) as physical:
        assert physical['valid_frame'].sum()==101
    with np.load(folder/'runtime-physical-mask.npz',allow_pickle=False) as application:
        assert application['valid_frame'].sum()==102
    with np.load(folder/'runtime-activations.npz',allow_pickle=False) as application:
        assert len(application['times'])==102
    modes=probe.read(folder/'runtime-candidates.json')
    assert set(modes)=={'default_onsets','predicted_releases'}
    assert modes['predicted_releases']['trace'][-1]['model_predicted_release_hypothesis'] is False


def test_scoring_difference_is_reported_without_requiring_same_predictions():
    before={tol:{metric:dict(available=True,reference=10,predicted=12,matched=8,precision=2/3,recall=.8,f1=16/22)
            for metric in probe.evaluation.METRICS} for tol in probe.evaluation.TOLERANCES}
    after=deepcopy(before)
    after['standard']['pitched_onsets'].update(predicted=13,matched=9,precision=9/13,recall=.9,f1=18/23)
    changes=probe.metric_changes(before,after)
    assert {r['field'] for r in changes}=={'predicted','matched','precision','recall','f1'}
    assert all(r['tolerance']=='standard' and r['metric']=='pitched_onsets' for r in changes)
