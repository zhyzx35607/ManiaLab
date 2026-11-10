import numpy as np
import pytest

from scripts.oaf_drums_runtime import decode, HOP, SAMPLE_RATE, prepare_audio


def test_frame_threshold_retains_adjacent_hits_and_unknown_classes():
    p = np.zeros((3, 88), np.float32)
    v = np.ones_like(p) * .5
    p[0, 36-21] = .5
    p[1:, 36-21] = .7
    p[2, 46-21] = .8
    p[2, 90-21] = .9
    events = decode(p, v)
    assert [e['kind'] for e in events] == ['kick', 'kick', 'hi-hat', 'unknown']
    assert [e['frame'] for e in events] == [1, 2, 2, 2]
    assert events[0]['onset'] == HOP/SAMPLE_RATE
    assert events[0]['velocity'] == 63
    assert all('end' not in e and 'duration' not in e for e in events)


def test_invalid_outputs_fail_instead_of_silent_filtering():
    with pytest.raises(ValueError):
        decode(np.zeros((4, 8)), np.zeros((4, 8)))
    p = np.zeros((2, 88))
    p[0, 0] = np.nan
    with pytest.raises(ValueError):
        decode(p, np.zeros_like(p))


def test_float_peak_precedes_pcm_quantization(tmp_path):
    pytest.importorskip('librosa')
    from scipy.io import wavfile
    source = np.array([0, 2., -1., .51], np.float32)
    path = tmp_path/'unclipped.wav'
    wavfile.write(path, SAMPLE_RATE, source)
    y, info = prepare_audio(path)
    np.testing.assert_array_equal(y, ((source/2*32767).astype(np.int16).astype(np.float32)/32767))
    assert info['source_mono_peak'] == 2
    assert y[2] > -.501


def test_pcm16_and_stereo_are_decoded_before_normalization(tmp_path):
    pytest.importorskip('librosa')
    from scipy.io import wavfile
    pcm = np.array([[-32768, 0], [16384, 16384], [100, -100]], np.int16)
    path = tmp_path/'pcm.wav'
    wavfile.write(path, SAMPLE_RATE, pcm)
    y, info = prepare_audio(path)
    np.testing.assert_array_equal(y, [-1, 1, 0])
    assert info['source_shape'] == [3, 2]
