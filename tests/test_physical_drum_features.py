import numpy as np
from scripts.physical_drum_features import RATE,HOP,features,target_frames,peaks


def test_frame_clock_does_not_drift_from_rounding_to_100fps():
    audio=np.zeros(RATE*12,np.float32)
    audio[10*RATE]=1
    x,times=features(audio)
    assert x.shape==(len(times),80)
    np.testing.assert_allclose(times,np.arange(len(times))*HOP/RATE*1000,atol=1e-9)
    assert abs(times[int(np.argmax(x.sum(axis=1)))]-10000)<HOP/RATE*1000
    assert np.isfinite(x).all()


def test_simultaneous_and_boundary_targets_retain_each_class():
    times=np.arange(101)*HOP/RATE*1000
    events=[dict(time_ms=-.204,kind='hi-hat'),dict(time_ms=500,kind='kick'),dict(time_ms=500,kind='snare')]
    targets=target_frames(times,events)
    at=int(np.argmin(abs(times-500)))
    assert targets[0,2]==1 and targets[1,2]==.25
    assert targets[at,:2].tolist()==[1,1]
    detected=peaks(targets,times,[.5]*3)
    assert len(detected)==3
    assert [e['kind'] for e in detected if e['time_ms']==times[at]]==['kick','snare']
