import copy

import numpy as np
from scipy.io import wavfile

from scripts.research_acoustic_releases import AcousticEvidence, FEATURES, apply, score
from scripts.research_instrument_overlap import refine


def note(onset=.3,offset=1.2):
    return dict(onset=onset,offset=offset,pitch=69,program=73,is_drum=False)


def test_acoustic_release_features_follow_physical_audio_not_window_clock(tmp_path):
    rate=16000;time=np.arange(rate*3)/rate
    # Known abrupt release with a brief fade to avoid treating a sample click
    # as the only release cue. The original model endpoint is 200ms too late.
    envelope=np.clip((time-.3)/.02,0,1)*np.clip((1.-time)/.02,0,1)
    audio=tmp_path/'tone.wav'
    wavfile.write(audio,rate,(np.sin(2*np.pi*440*time)*envelope*12000).astype(np.int16))
    evidence=AcousticEvidence(audio,3.)
    base=note();alternate=note(offset=1.)
    report=dict(segment_samples=32767,sample_rate=16000,context_start_seconds=0.)
    a,b=evidence.feature_pair(base,alternate,['forced_segment_boundary'],['explicit_model_note_off'],
        report,dict(report,context_start_seconds=1.0239375),[base])
    assert a.shape==b.shape==(len(FEATURES),)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert b[FEATURES.index('harmonic_drop')]>a[FEATURES.index('harmonic_drop')]+3


def test_ranking_retains_all_attack_fields_and_does_not_mutate_evidence():
    item=dict(notes=[note()],examples=[dict(index=0,alternate_index=2,
        x=np.zeros(len(FEATURES)),new_offset=1.)])
    original=copy.deepcopy(item['notes'])
    model=dict(scale=np.ones(len(FEATURES)),weights=np.zeros(len(FEATURES)+1))
    kept,decisions=apply(item,model,.5)
    assert kept==original and not decisions[0]['selected']
    changed,_=apply(item,model,.4)
    assert changed==[note(offset=1.)] and item['notes']==original
    assert apply(item,model,1.)[0]==original


def test_explicit_endpoint_candidates_need_opt_in_and_still_require_same_pitch():
    base=[note()];alternative=[note(offset=1.)]
    flags={0:['explicit_model_note_off']}
    args=dict(shifted_start=.5,segment_seconds=2.)
    assert refine(base,alternative,flags,flags,**args)[0]==base
    assert refine(base,alternative,flags,flags,only_forced=False,**args)[0]==alternative
    alternative[0]['pitch']=70
    assert refine(base,alternative,flags,flags,only_forced=False,**args)[0]==base


def test_endpoint_training_scores_do_not_hide_overlong_prediction():
    item=dict(reference=[note(onset=29.,offset=30.5)],context_start=0.,context_end=32.1)
    result=score(item,[note(onset=29.,offset=40.)])
    assert result['standard']['complete_intervals']['predicted']==1
    assert result['standard']['complete_intervals']['reference']==1
    assert result['standard']['complete_intervals']['matched']==0
