import numpy as np
import pytest
from scipy.io import wavfile

from manialab.audition import grid_events, render_audition, waveform_window


def test_audition_keeps_music_and_clicks_on_one_sample_clock(tmp_path):
    rate = 22050
    music = np.zeros(rate * 4, dtype=np.int16)
    music[round(1.091 * rate)] = 12000
    path = tmp_path / "preview.wav"
    wavfile.write(path, rate, music)
    events = [(1091, "head"), (1091, "head"), (2091, "tail")]
    content, report = render_audition(path, 1000, 2000, events)
    sr, result = wavfile.read(content)
    assert sr == rate and result.shape == (2 * rate, 2)
    assert len(report["events"]) == 2
    assert report["events"][0]["sample"] == round(1.091 * rate) - rate
    assert np.all(result[:report["events"][0]["sample"]] == 0)
    assert np.array_equal(result[:, 0], result[:, 1])


def test_audition_slices_original_time_without_rebasing_offset(tmp_path):
    path = tmp_path / "preview.wav"
    wavfile.write(path, 8000, np.zeros(8000 * 60, dtype=np.int16))
    events = grid_events(210, 91, 50000, 5000)
    _, a = render_audition(path, 50000, 5000, events)
    _, b = render_audition(path, 50000, 5000, [(t + 24, k) for t, k in events])
    assert b["events"][0]["sample"] - a["events"][0]["sample"] == round(.024 * 8000)
    assert len(a["events"]) > 10
    with pytest.raises(ValueError, match="结尾"):
        render_audition(path, 60000, 1000, [])
    with pytest.raises(ValueError, match="1–30"):
        render_audition(path, 0, 31000, [])


def test_waveform_envelope_preserves_an_impulse_on_original_sample_clock(tmp_path):
    path = tmp_path / 'preview.wav'
    data = np.zeros(44100 * 3, np.int16)
    at = round(1.091 * 44100)
    data[at] = 16384
    data[at + 1] = -16384
    wavfile.write(path, 44100, data)
    result = waveform_window(path, 1000, 500)
    index = int(np.argmax(result['high']))
    first = result['first_sample'] + index * result['step_samples']
    assert first <= at < first + result['step_samples']
    assert max(result['high']) == .5 and min(result['low']) == -.5
    assert result['start_ms'] == 1000 and result['end_ms'] == 1500
    assert len(result['high']) <= 1600
    with pytest.raises(ValueError):
        waveform_window(path, float('nan'), 1000)
    with pytest.raises(ValueError, match='结尾'):
        waveform_window(path, 3000, 1000)
