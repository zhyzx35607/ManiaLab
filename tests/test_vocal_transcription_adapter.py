import pytest

from scripts.research_vocal_transcription import decode_intervals


def test_vocal_decode_preserves_repeated_same_pitch_and_labels_artificial_final_tail():
    events = decode_intervals([0, 0, 1, 0, 1, 0], [60, 60, 60],
                              sample_rate=100, hop_size=10, duration=.6)
    assert [(n['onset'], n['offset'], n['pitch']) for n in events] == [(0, .2, 60), (.2, .4, 60), (.4, .5, 60)]
    assert [n['release_reason'] for n in events] == ['predicted_joint_note_word_boundary']*2+['input_boundary']
    assert all(n['strength_semantics'] == 'unit_weight_not_confidence' for n in events)


def test_vocal_decode_preserves_rest_intervals_in_raw_evidence():
    events = decode_intervals([0, 1, 0, 1, 0], [0, 64, 0],
                              sample_rate=100, hop_size=10, duration=.5)
    assert [n['rest'] for n in events] == [True, False, True]
    assert events[1]['onset'] == .1 and events[1]['offset'] == .3


@pytest.mark.parametrize('bd,pitches,duration', [([], [], 1), ([1, 0], [60, 60], 1),
    ([0, 1], [60, 60], 1), ([0, 0, 0], [60, 64], 1), ([0, 0, 0], [60], .1),
    ([0, 2, 0], [60], 1), ([0, 0, 0], [130], 1)])
def test_vocal_decoder_rejects_misaligned_outputs_instead_of_dropping_intervals(bd, pitches, duration):
    with pytest.raises(ValueError):
        decode_intervals(bd, pitches, sample_rate=100, hop_size=10, duration=duration)
