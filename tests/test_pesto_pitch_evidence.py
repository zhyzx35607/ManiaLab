import numpy as np
import pytest

from scripts.research_pesto_pitch import frame_clock, note_support


def test_actual_integer_sample_hop_prevents_fractional_rate_clock_drift():
    samples, rate = 60*22050, 22050
    times, valid, hop = frame_clock(samples//221+1, rate, samples)
    assert hop == 221
    assert times[-1] == ((samples//hop)*hop)/rate
    assert times[-1] - (len(times)-1)*.01 > .13
    assert valid.all()


def test_padding_frame_is_retained_but_not_audio_evidence():
    times, valid, hop = frame_clock(401, 16000, 64000)
    assert times[-1] == 4 and not valid[-1] and valid[:-1].all()
    frames = dict(times=times, valid=valid, pitch_midi=np.full(401,69.), voicing=np.ones(401))
    result = note_support([dict(onset=3.99,offset=4.01,pitch=69)], frames)[0]
    assert result['frames'] == result['supported_frames'] == 1


def test_voiced_wrong_octave_and_unvoiced_correct_pitch_do_not_support_a_note():
    frames = dict(times=np.array([0.,.01,.02,.03]), valid=np.ones(4,dtype=bool),
                  pitch_midi=np.array([60.,72.,60.,60.]), voicing=np.array([.9,.9,.1,.9]))
    notes = [dict(onset=0.,offset=.04,pitch=60),dict(onset=1.,offset=2.,pitch=60)]
    result = note_support(notes, frames)
    assert result[0]['support_fraction'] == .5 and result[0]['voiced_frames'] == 3
    assert result[1]['support_fraction'] is None


def test_unexpected_cqt_frame_count_rejected():
    with pytest.raises(ValueError):frame_clock(400,16000,64000)
