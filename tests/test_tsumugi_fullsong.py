import numpy as np
import pytest

from scripts.research_tsumugi_fullsong import validate_pcm


def test_full_song_clock_and_unclipped_float_gain_are_preserved():
    pcm = np.full((2838, 2), 1.6, dtype=np.float32)
    before = pcm.copy()
    assert validate_pcm(pcm, 10) == 283.8
    np.testing.assert_array_equal(pcm, before)


@pytest.mark.parametrize('samples,rate', [(1, 10), (12000, 10)])
def test_duration_limits_accept_endpoints(samples, rate):
    assert validate_pcm(np.zeros((samples, 1), dtype=np.float32), rate) == samples / rate


@pytest.mark.parametrize('pcm,rate', [
    (np.zeros((0, 1)), 10), (np.zeros((12001, 1)), 10),
    (np.zeros((1, 1)), 11), (np.zeros((10, 3)), 10),
    (np.zeros(10), 10), (np.array([[np.nan]]), 10),
    (np.array([[np.inf]]), 10), (np.zeros((10, 1)), 0),
    (np.zeros((10, 1)), True),
])
def test_invalid_audio_is_rejected_without_normalization_or_channel_dropping(pcm, rate):
    with pytest.raises(ValueError, match='PCM'):
        validate_pcm(pcm, rate)
