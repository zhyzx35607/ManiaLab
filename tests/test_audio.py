from pathlib import Path
import numpy as np
from scipy.io import wavfile

from manialab.audio import SR, analyze, rhythmic_evidence
from manialab.arrange import parameters


def test_stem_eof_rounding_never_changes_the_attack_clock():
    import pytest
    from manialab.audio import align_stem_samples
    samples=np.zeros(1000,dtype=np.float32);samples[91]=.8
    longer,delta=align_stem_samples(samples,1001)
    assert delta==1 and len(longer)==1001 and np.argmax(longer)==91
    assert np.array_equal(longer[:-1],samples) and longer[-1]==0
    shorter,delta=align_stem_samples(samples,999)
    assert delta==-1 and np.array_equal(shorter,samples[:-1])
    with pytest.raises(ValueError):align_stem_samples(samples,997)
    invalid=samples.copy();invalid[30]=np.nan
    with pytest.raises(ValueError):align_stem_samples(invalid,1000)


def test_silence():
    notes,_=rhythmic_evidence(np.zeros(SR*3))
    assert notes==[]


def test_residual_stem_cannot_be_amplified_into_musical_evidence():
    from manialab.audio import source_evidence
    from manialab.domain import Note
    mix=np.full(SR*3,.2,dtype=np.float32);mix[SR*2:]=0
    residual=np.full_like(mix,.00006)
    notes=[Note(500,500,source='transient'),Note(2500,2500,source='transient')]
    kept,rejected=source_evidence(notes,residual,mix)
    assert kept==[] and rejected==2
    # A quiet but real source remains eligible; this is a conservative noise
    # floor, not a rule that only the dominant instrument may be mapped.
    quiet=residual.copy();quiet[:SR]=.008
    kept,rejected=source_evidence(notes,quiet,mix)
    assert kept==[notes[0]] and rejected==1


def test_release_refinement_recovers_silence_without_extending_notes():
    from manialab.audio import refine_releases
    from manialab.domain import Note
    wave=np.zeros(SR*4,dtype=np.float32);wave[SR:SR*3]=.1
    notes=[Note(1000,3050,source='pitch'),Note(1000,2500,source='pitch')]
    output,report=refine_releases(notes,wave)
    assert abs(output[0].end-3000)<=5 and output[1].end==2500
    assert report['shortened']==1
    assert all(new.start==old.start and new.end<=old.end for new,old in zip(output,notes))


def test_real_model_timing_and_sustains(tmp_path):
    y=np.zeros(SR*30,dtype=np.float32)
    expected=[(1,2,60),(5,4,64),(12,1,67),(20,3,60),(26,2,72)]
    for start,duration,midi in expected:
        t=np.arange(duration*SR)/SR
        freq=440*2**((midi-69)/12)
        tone=.3*(np.sin(2*np.pi*freq*t)+.3*np.sin(4*np.pi*freq*t)+.1*np.sin(6*np.pi*freq*t))
        tone*=np.minimum(1,t/.015)*np.minimum(1,(duration-t)/.04)
        y[start*SR:(start+duration)*SR]+=tone.astype(np.float32)
    wavfile.write(tmp_path/'synth.wav',SR,y)
    result=analyze(tmp_path/'synth.wav',tmp_path,Path(__file__).parents[1]/'assets/basic-pitch.onnx',parameters())
    pitched=[n for n in result['candidates'] if n['source']=='pitch']
    assert len(pitched) == len(expected), pitched
    assert all(n['release_reason'] == 'frame_decay' for n in pitched)
    for start,duration,midi in expected:
        matches=[n for n in pitched if n['pitch']==midi and abs(n['start']-start*1000)<120]
        assert matches, (start,midi,pitched)
        assert abs(matches[0]['end']-(start+duration)*1000)<180
    assert result['duration']==30
    attacks=[n['start'] for n in result['candidates'] if n['source']=='transient']
    assert not any(abs(attack-(start+length)*1000)<100 for attack in attacks for start,length,_ in expected)
def test_short_cross_source_leakage_is_not_an_extra_instrument():
    import numpy as np
    from manialab.audio import reject_source_bleed
    from manialab.domain import Note
    rate=22050;t=np.arange(rate*4)/rate
    other=np.sin(2*np.pi*261.63*t).astype(np.float32)
    stems={'other':other,'bass':other*.05,'vocals':other*.5}
    notes=[Note(1000,3000,pitch=60,strength=.8,source='pitch',voice='other',id='real'),
           Note(1020,1120,pitch=61,strength=.4,source='pitch',voice='bass',id='duplicate'),
           Note(2900,3000,pitch=60,strength=.4,source='pitch',voice='bass',id='tail'),
           Note(1500,1600,pitch=67,strength=.4,source='pitch',voice='bass',id='other-pitch'),
           Note(1020,1120,pitch=60,strength=.4,source='pitch',voice='vocals',id='unison')]
    result,rejected=reject_source_bleed(notes,stems,[])
    assert {n.id for n in result}=={'real','other-pitch','unison'}
    assert {r['reason'] for r in rejected}=={'weak_duplicate_attack','release_edge_bleed'}
    result,_=reject_source_bleed(notes,stems,[Note(2900,2900)])
    assert 'tail' in {n.id for n in result}


def test_steady_bass_retrigger_can_be_removed_without_smoothing_a_real_rest():
    from manialab.articulation import steady_bass_onsets
    from manialab.audio import pitched_notes
    times=np.arange(300)*256/SR
    frames=np.zeros((300,88));onsets=np.zeros_like(frames)
    frames[40:180,12]=.7;onsets[40,12]=.8;onsets[100,12]=.5
    t=np.arange(SR*4)/SR;frequency=440*2**((33-69)/12)
    wave=.2*np.sin(2*np.pi*frequency*t)
    original=onsets.copy()
    suppressed=steady_bass_onsets(frames,onsets,times,wave)
    assert len(suppressed)==1 and suppressed[0]['time_ms']==round(times[100]*1000)
    notes=pitched_notes(frames,onsets,times,4,suppressed=suppressed)
    assert len(notes)==1 and notes[0].end>2000
    assert np.array_equal(onsets,original)
    # Even a short true rest around the same suspected reattack prevents
    # suppression; the test leaves model activation constant intentionally.
    at=times[100];wave[(t>=at-.015)&(t<at)]=0
    assert steady_bass_onsets(frames,onsets,times,wave)==[]
