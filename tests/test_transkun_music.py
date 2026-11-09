import numpy as np
import pytest
from scipy.io import wavfile

from scripts.check_transkun_music import nova_comparison, onset_comparison, prepare_input, summarize_music


def event(start, pitch=60, end=None):
    return dict(onset=start, offset=start+.2 if end is None else end, pitch=pitch)


def test_equal_counts_can_hide_a_new_miss_and_recovery():
    ref = [event(1, 36), event(2, 60), event(3, 96)]
    old = ref[:2]
    new = [ref[0], ref[2], event(5, 108)]
    result = onset_comparison(ref, old, new)
    assert result['new_missing_ref_indices'] == [1]
    assert result['recovered_ref_indices'] == [2]
    assert result['transkun']['predicted'] == 3
    assert result['bands']['middle']['new_missing'] == 1
    assert result['bands']['high']['recovered'] == 1
    assert result['bands']['high']['transkun']['precision'] == .5


def test_predictions_with_long_tails_and_duplicates_stay_in_denominator():
    ref = [event(1), event(1.4)]
    result = onset_comparison(ref, [event(1, end=100)], [event(1, end=100), event(1.01)])
    assert result['transkun']['matched'] == 1
    assert result['transkun']['predicted'] == 2
    assert result['kong']['recall'] == .5


def test_summary_preserves_per_recording_tradeoffs():
    first = onset_comparison([event(1)], [], [event(1)])
    second = onset_comparison([event(1, 96)], [event(1, 96)], [])
    rows = [dict(name=n, kind='smd', comparison={'50ms':c, '40ms':c})
            for n,c in [('a',first), ('b',second)]]
    result = summarize_music(rows)
    assert result['models']['50ms']['kong']['pooled'] == result['models']['50ms']['transkun']['pooled']
    assert result['new_missing']['50ms'] == result['recovered']['50ms'] == 1
    assert result['models']['50ms']['transkun']['per_recording_f1'] == {'a':1., 'b':0.}


@pytest.mark.parametrize('integer', [True, False])
def test_input_preserves_gain_and_clock(tmp_path, integer):
    value = np.full(22050, 8192 if integer else .25, dtype=np.int16 if integer else np.float32)
    path = tmp_path/'piano.wav'
    wavfile.write(path, 22050, value)
    shared, actual = prepare_input(path, 1.)
    assert shared.dtype == actual.dtype == np.float32
    assert shared.shape == (16000,) and actual.shape == (44100,)
    assert np.median(shared) == pytest.approx(.25, abs=.0001)
    assert np.median(actual) == pytest.approx(.25, abs=.0001)
    with pytest.raises(ValueError, match='shorter'):
        prepare_input(path, 1.1)


def test_author_grouping_does_not_chain_and_shift_is_explicit():
    text = ('osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:10\n'
            '[HitObjects]\n230,192,1000,1,0,0:0:0:0:\n281,192,1050,1,0,0:0:0:0:\n')
    notes = [event(1.0225), event(1.0475, 64), event(1.0725, 67)]
    result = nova_comparison(notes, notes, text)
    assert result['same_pitch_50ms']['matched'] == 3
    view = result['author_onset_diagnostic']['transkun']['22.5']
    assert view['onset_groups']['predicted'] == 2
    assert view['onset_groups']['matched'] == 2
    assert view['note_multiplicity']['predicted'] == 3
    assert result['acoustic_offset_truth_available'] is False
