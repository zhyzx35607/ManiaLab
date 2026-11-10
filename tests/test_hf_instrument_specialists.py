import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.research_hf_instrument import OUTPUTS, physical_heads, stitch_windows, write_new


def test_hf_stitching_retains_raw_padding_on_the_physical_clock():
    windows = {key:np.stack([np.broadcast_to((np.arange(1001)+start)[:,None],(1001,88))
                            for start in range(0,2500,500)]) for key in OUTPUTS}
    stitched = stitch_windows(windows)
    for values in stitched.values():
        np.testing.assert_array_equal(values[:,0],np.arange(3000))
    heads,times,valid = physical_heads(stitched,23*44100+1,44100)
    assert len(times)==3000 and valid.sum()==2301
    np.testing.assert_array_equal(heads[OUTPUTS[0]][:,0],np.arange(2301))


def test_hf_partial_last_frame_uses_original_sample_clock():
    heads = {key:np.zeros((3000,88),dtype=np.float32) for key in OUTPUTS}
    actual,times,valid = physical_heads(heads,23*44100+1,44100)
    assert len(times)==3000 and valid.sum()==2301 and len(actual[OUTPUTS[0]])==2301
    assert times[2300]==23. and not valid[2301]
    exact,_,valid = physical_heads(heads,23*44100,44100)
    assert len(exact[OUTPUTS[0]])==2300 and not valid[2300]
    with pytest.raises(ValueError,match='cover'):
        physical_heads(heads,31*44100,44100)


def test_hf_single_window_keeps_center_stft_extra_frame():
    raw = {key:np.zeros((1,1001,88),dtype=np.float32) for key in OUTPUTS}
    stitched = stitch_windows(raw)
    _,times,valid = physical_heads(stitched,160000,16000)
    assert len(times)==1001 and valid.sum()==1000 and not valid[-1]
    assert all(value.shape==(1,1001,88) for value in raw.values())


def test_hf_unknown_tail_events_are_retained_in_bass_denominators():
    from manialab.piano import decode
    from scripts.run_bass_model_comparison import score_case
    raw = {key:np.zeros((1000,88),dtype=np.float32) for key in OUTPUTS}
    raw['reg_onset_output'][99:102,0]=[.2,.8,.2]
    raw['frame_output'][100:,0]=.9
    note = dict(decode(raw)[0],program=-1,is_drum=False)
    assert note['release_reason']=='duration_limit' and not note['release_observed']
    ref = [dict(onset=1.,offset=2.,pitch=21,program=-1,plucking_style='FS',expression_style='NO')]
    result = score_case(ref,[note],end=8.,context_end=10.)
    scores = result['metrics']['standard']
    assert scores['pitched_onsets']['matched']==1
    assert scores['complete_intervals']['predicted']==1 and scores['complete_intervals']['matched']==0


def test_hf_rejects_bad_arrays_and_existing_evidence(tmp_path):
    raw = {key:np.zeros((1,1001,88),dtype=np.float32) for key in OUTPUTS}
    raw[OUTPUTS[0]][0,0,0]=np.nan
    with pytest.raises(ValueError,match='activations'):
        stitch_windows(raw)
    target=tmp_path/'evidence.json'
    write_new(target,{'unchanged':True})
    with pytest.raises(FileExistsError):
        write_new(target,{'unchanged':False})
    assert 'true' in target.read_text()


def test_hf_import_and_help_do_not_load_torch():
    root=Path(__file__).resolve().parents[1]
    code="import sys;import scripts.research_hf_instrument;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],cwd=root,check=True,capture_output=True)
    result=subprocess.run([sys.executable,'scripts/research_hf_instrument.py','--help'],cwd=root,
                          check=True,capture_output=True,text=True)
    assert '--model' in result.stdout


def test_hf_comparison_keeps_both_tolerances_and_ignores_program_metric():
    from scripts.check_hf_instrument_specialists import normalized_metrics, CASE_NAMES
    cell=dict(available=True,reference=2,predicted=3,matched=1)
    values={'pitched_onsets':cell,'complete_intervals':cell,'instrument_onsets':{'matched':0}}
    result=normalized_metrics({'standard':values,'strict':values})
    assert list(result)==['standard','fixed_40_80']
    assert result['fixed_40_80']['complete_intervals'] is cell
    assert set(result['standard'])=={'pitched_onsets','complete_intervals'}
    assert len(CASE_NAMES)==19 and CASE_NAMES[-2:]==['sax-25-4-sax','sax-39-4-sax']


def test_hf_comparison_import_does_not_load_a_network():
    root=Path(__file__).resolve().parents[1]
    code="import sys;import scripts.check_hf_instrument_specialists;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],cwd=root,check=True,capture_output=True)


def test_hf_legacy_aggregation_requires_observable_context():
    from scripts.check_hf_instrument_specialists import normalized_metrics
    from scripts.evaluate_guitar_vocal_benchmark import aggregate
    cell=dict(reference=2,predicted=3,matched=1,precision=1/3,recall=.5,f1=.4)
    values={'pitched_onsets':cell,'complete_intervals':cell}
    legacy={'standard':values,'strict':values}
    scope=dict(context_start_seconds=0,begin_seconds=0,end_seconds=30,
               context_end_seconds=40,unobservable_reference_offsets=0)
    with pytest.raises(ValueError,match='observable'):
        normalized_metrics(legacy)
    with pytest.raises(ValueError,match='observable'):
        normalized_metrics(legacy,dict(scope,unobservable_reference_offsets=1))
    metrics=normalized_metrics(legacy,scope)
    score=aggregate([dict(name='sax-a',metrics=metrics),dict(name='sax-b',metrics=metrics)])
    micro=score['standard']['complete_intervals']['micro']
    assert (micro['reference'],micro['predicted'],micro['matched'],micro['f1'])==(4,6,2,.4)
    assert 'available' not in cell  # Preserve original evidence.


def test_hf_unavailable_tail_score_is_not_promoted():
    from scripts.check_hf_instrument_specialists import normalized_metrics
    cell=dict(available=False,reason='reference_release_reaches_input_boundary',
              reference=2,predicted=3,matched=None)
    assert normalized_metrics({'strict':{'complete_intervals':cell}})['fixed_40_80']['complete_intervals'] is cell


def test_hf_summary_rejects_partial_cohort():
    from scripts.check_hf_instrument_specialists import summarize
    with pytest.raises(ValueError,match='partial'):
        summarize({},[])


def test_hf_reuse_requires_complete_matching_evidence(tmp_path):
    from scripts.check_hf_instrument_specialists import verified_reuse
    from scripts.research_hf_instrument import sha
    identity={'model':'bass','strict_weights':True}
    case=dict(audio=str(tmp_path/'audio.wav'),audio_sha256='a'*64,original_rate=16000,original_samples=16000)
    note=dict(onset=.1,offset=.8,pitch=40,program=-1,is_drum=False,
              release_reason='input_boundary',release_observed=False)
    write_new(tmp_path/'notes.json',[note])
    write_new(tmp_path/'raw-notes.json',[])
    for filename in ('activations.npz','windows.npz'):
        (tmp_path/filename).write_bytes(b'fixed-evidence')
    report=dict(identity=identity,input=dict(audio=str(Path(case['audio']).resolve()),
                audio_sha256=case['audio_sha256'],original_sample_rate=16000,original_samples=16000),
                raw_decoder_parity=True,unknown_tails_retained=True,notes=1)
    for filename,key in (('notes.json','notes_sha256'),('raw-notes.json','raw_notes_sha256'),
                         ('activations.npz','activations_sha256'),('windows.npz','windows_sha256')):
        report[key]=sha(tmp_path/filename)
    write_new(tmp_path/'report.json',report)
    assert verified_reuse(tmp_path,identity,case)[0]==[note]
    with pytest.raises(ValueError,match='identity'):
        verified_reuse(tmp_path,dict(identity,model='sax'),case)
    (tmp_path/'activations.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Hash mismatch'):
        verified_reuse(tmp_path,identity,case)
