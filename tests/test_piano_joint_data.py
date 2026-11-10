from copy import deepcopy

import numpy as np
import pytest

from scripts.piano_joint_data import frame_targets, owned_mask


def note(onset, offset, pitch=60, **kwargs):
    return dict(onset=onset, offset=offset, pitch=pitch, **kwargs)


def test_frame_targets_use_half_open_physical_intervals_without_rounding():
    result = frame_targets([note(.01, .03, 21), note(.011, .029, 108)], 0, frames=5)
    assert result.dtype == np.float32 and result.shape == (5, 88)
    np.testing.assert_array_equal(result[:, 0], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(result[:, 87], [0, 0, 1, 0, 0])
    assert not result[:, 1:87].any()


def test_overlaps_and_retriggered_sustain_take_union_without_false_rest():
    notes = [note(0, .02), note(.02, .04), note(.01, .03),
             note(.01, .025, 64), note(.04, .05, 64)]
    original = deepcopy(notes)
    result = frame_targets(notes, 0, frames=6)
    np.testing.assert_array_equal(result[:, 39], [1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(result[:, 43], [0, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(result, frame_targets(reversed(notes), 0, frames=6))
    assert notes == original


def test_window_external_sustain_is_kept_without_local_or_global_fake_release():
    notes = [note(.5, 12, 21, key_offset=.7), note(179, 190, 108),
             note(20, 21, 30)]
    local = frame_targets(notes, 1)
    assert local[:, 0].all() and local[-1, 0] == 1
    assert local.sum() == 1001
    global_end = frame_targets(notes, 170)
    assert global_end[900:, 87].all()  # Includes t=180, masked only in loss.
    assert not global_end[:900, 87].any()
    assert notes[1]['offset'] == 190


def test_empty_labels_and_numpy_numeric_scalars_are_valid():
    assert frame_targets([], 0).shape == (1001, 88)
    assert not frame_targets([], 2, frames=1).any()
    result = frame_targets([note(np.float64(1), np.float64(2), np.int64(60))],
                           np.float64(1), frames=np.int64(2))
    assert result[:, 39].all()


@pytest.mark.parametrize('invalid', [
    note(0, 1, 20), note(0, 1, 109), note(0, 1, 60.), note(0, 1, True),
    note(-.1, 1), note(1, 1), note(2, 1), note(float('nan'), 1),
    note(0, float('inf')), note(False, 1), note(0, True), note('0', 1),
    note(300, 301, 200), {'pitch': 60, 'onset': 0}, None,
])
def test_invalid_notes_are_rejected_even_outside_window(invalid):
    with pytest.raises(ValueError):
        frame_targets([invalid], 10)


@pytest.mark.parametrize('start,frames', [
    (-1, 1001), (float('nan'), 1001), (float('inf'), 1001), (True, 1001),
    ('0', 1001), (0, 0), (0, -1), (0, 1.), (0, True),
])
def test_invalid_physical_clock_is_rejected(start, frames):
    with pytest.raises(ValueError):
        frame_targets([], start, frames)


def test_global_mask_owns_each_guarded_physical_frame_exactly_once():
    owned = []
    for index in range(35):
        mask = owned_mask(index)
        assert mask.dtype == np.bool_ and mask.shape == (1001,)
        owned.extend((500 * index + np.flatnonzero(mask)).tolist())
    np.testing.assert_array_equal(owned, np.arange(5, 17995))
    assert len(owned) == 17990
    assert owned_mask(0).sum() == owned_mask(34).sum() == 745
    assert all(owned_mask(index).sum() == 500 for index in range(1, 34))
    assert not owned_mask(34)[995:].any()
    assert owned_mask(1)[250]  # No extra mask at internal window boundaries.


@pytest.mark.parametrize('index', [-1, 35, 1., True, '1', None])
def test_invalid_window_index_is_rejected(index):
    with pytest.raises(ValueError):
        owned_mask(index)
