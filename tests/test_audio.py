from pathlib import Path
import numpy as np
from scipy.io import wavfile

from manialab.audio import SR, analyze, rhythmic_evidence
from manialab.arrange import parameters


def test_silence():
    notes,_=rhythmic_evidence(np.zeros(SR*3))
    assert notes==[]


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
    for start,duration,midi in expected:
        matches=[n for n in pitched if n['pitch']==midi and abs(n['start']-start*1000)<120]
        assert matches, (start,midi,pitched)
        assert abs(matches[0]['end']-(start+duration)*1000)<180
    assert result['duration']==30
    attacks=[n['start'] for n in result['candidates'] if n['source']=='transient']
    assert not any(abs(attack-(start+length)*1000)<100 for attack in attacks for start,length,_ in expected)
