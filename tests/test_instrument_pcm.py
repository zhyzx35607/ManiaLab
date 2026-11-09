import numpy as np
import pytest
from scipy.io import wavfile

from scripts.research_instrument_transcription import load_sources


def test_float_sources_keep_overshoot_low_level_detail_and_mix_gain(tmp_path):
    paths=[tmp_path/'a.wav',tmp_path/'b.wav']
    a=np.array([1.4,-1.2,1e-8,.000015],dtype=np.float32)
    b=np.array([.2,-.1,0.,.00002],dtype=np.float32)
    for path,signal in zip(paths,[a,b]):wavfile.write(path,22050,signal)
    rate,mix,evidence=load_sources(paths)
    assert rate==22050
    np.testing.assert_array_equal(mix,a+b)
    assert all(not row['normalized'] and not row['clipped'] for row in evidence)


def test_int16_pcm_remains_compatible_with_historical_probes(tmp_path):
    path=tmp_path/'audio.wav'
    signal=np.array([-32768,-1,0,1,32767],dtype=np.int16)
    wavfile.write(path,16000,signal)
    _,result,_=load_sources([path])
    np.testing.assert_array_equal(result,signal.astype(np.float32)/32768)


def test_clock_mismatch_and_nonfinite_audio_are_rejected(tmp_path):
    paths=[tmp_path/'a.wav',tmp_path/'b.wav']
    wavfile.write(paths[0],16000,np.zeros(4,dtype=np.float32))
    wavfile.write(paths[1],22050,np.zeros(4,dtype=np.float32))
    with pytest.raises(ValueError,match='clocks differ'):load_sources(paths)
    wavfile.write(paths[0],16000,np.array([float('nan')],dtype=np.float32))
    with pytest.raises(ValueError,match='Nonfinite'):load_sources(paths[:1])
