import copy

import numpy as np
import pytest

from scripts.research_instrument_refinement import observation_counts, query_arrays


def event(start, end, pitch=60, drum=False):
    return dict(onset=start, offset=end, pitch=pitch, is_drum=drum)


def test_queries_preserve_chords_overlap_duplicates_and_sub_millisecond_notes():
    notes = [event(.2, .6), event(.1, .5, 64), event(.1, .4),
             event(.1, .4), event(.1, .1001, 67), event(.3, .4, drum=True)]
    frozen = copy.deepcopy(notes)
    result = query_arrays(notes, 1)
    assert result['original_indices'].tolist() == [2, 3, 1, 4, 0]
    assert len(result['pitch']) == 5
    assert notes == frozen
    assert result['end_seconds'][3] > result['start_seconds'][3]


def test_empty_and_drum_only_input_remains_explicitly_empty():
    for notes in ([], [event(.1, .2, drum=True)]):
        assert all(len(v) == 0 for v in query_arrays(notes, 1).values())


@pytest.mark.parametrize('note', [event(-.1, .2), event(.1, .1), event(.2, .1),
    event(.1, 1.1), event(float('nan'), .2), event(.1, float('inf')),
    event(.1, .2, -1), event(.1, .2, True), event(.1, .2, drum=1)])
def test_invalid_raw_events_are_not_repaired_or_dropped(note):
    with pytest.raises(ValueError, match='original note'):
        query_arrays([note], 1)


def test_float32_collapsed_interval_is_rejected():
    with pytest.raises(ValueError, match='collapsed'):
        query_arrays([event(1000., 1000.000001)], 1200.)


def test_observation_counts_follow_half_open_actual_windows_including_final_partial():
    starts = np.asarray([0., 4., 7.999, 8., 10., 12.], dtype=np.float32)
    counts = observation_counts(starts, [0, 400, 450], 100)
    assert counts.tolist() == [1, 2, 3, 2, 2, 1]


def test_missing_query_observation_is_visible():
    assert observation_counts(np.asarray([12.5]), [0, 400, 450], 100).tolist() == [0]
