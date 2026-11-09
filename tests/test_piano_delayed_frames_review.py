"""Independent activation-only checks; no model, score labels or audio corpus."""
import numpy as np
import pytest

from manialab.piano import FPS, OUTPUTS
from scripts.piano_delayed_frames import diagnose


def blank(frames=100):
    return {name: np.zeros((frames, 88), dtype=np.float32) for name in OUTPUTS}


def attack(values, frame, pitch=60, shape=(.1, .5, .9, .5, .1)):
    key = pitch - 21
    values['reg_onset_output'][frame-2:frame+3, key] = shape
    values['velocity_output'][frame, key] = .7


def active(values, begin, end, pitch=60, level=.8):
    values['frame_output'][begin:end, pitch-21] = level


def release(values, frame, pitch=60):
    values['reg_offset_output'][frame-4:frame+5, pitch-21] = [.01, .1, .2, .5, .9, .5, .2, .1, .01]


def test_isolated_first_activation_does_not_search_a_later_favorable_interval():
    values = blank()
    attack(values, 10)
    active(values, 12, 13)
    active(values, 14, 40)
    old, new, rows = diagnose(values)
    assert old == new
    assert rows[0]['status'] == 'isolated_activation'
    assert rows[0]['first_active_frame'] == 12


def test_release_peak_without_frame_decay_does_not_claim_observed_release():
    values = blank()
    attack(values, 10)
    active(values, 12, 100)
    release(values, 70)
    old, new, rows = diagnose(values)
    assert old == new
    assert rows[0]['status'] == 'active_without_observed_decay'
    assert not new[0]['release_observed']


def test_different_pitch_reattack_cannot_truncate_an_observed_tail():
    values = blank()
    attack(values, 10, 60)
    attack(values, 20, 64)
    active(values, 12, 40, 60)
    active(values, 20, 35, 64)
    old, new, rows = diagnose(values)
    assert [event['pitch'] for event in old] == [60, 64]
    assert new[0]['offset'] == float(np.float32(.4))
    assert new[1] == old[1]
    assert rows[1]['status'] == 'existing_active_tail'


@pytest.mark.parametrize('first,changed', [(15, True), (16, False)])
def test_exact_waiting_boundary_is_finite_and_inclusive(first, changed):
    values = blank()
    attack(values, 10)
    active(values, first, 40)
    old, new, rows = diagnose(values)
    assert (old != new) == changed
    if changed:
        assert rows[0]['delay_ms'] == 50
        assert new[0]['offset'] == float(np.float32(.4))


@pytest.mark.parametrize('next_attack,changed', [(15, False), (19, True)])
def test_waiting_boundary_cannot_borrow_next_attack_evidence(next_attack, changed):
    values = blank()
    attack(values, 10)
    attack(values, next_attack)
    active(values, 15, 18)
    old, new, rows = diagnose(values)
    assert (old[0] != new[0]) == changed
    if changed:
        assert new[0]['offset'] == float(np.float32(.18))
        assert new[0]['offset'] < new[1]['onset']
    else:
        assert rows[0]['status'] == 'reattack_before_activation'


@pytest.mark.parametrize('end,changed', [(610, True), (611, False)])
def test_six_second_limit_requires_actual_decay_evidence(end, changed):
    values = blank(700)
    attack(values, 10)
    active(values, 12, end)
    old, new, rows = diagnose(values)
    assert (old != new) == changed
    assert rows[0]['search_boundary'] == 610
    if changed:
        assert new[0]['offset'] == float(np.float32(610/FPS))
        assert new[0]['release_observed']
    else:
        assert rows[0]['status'] == 'active_without_observed_decay'


def test_last_input_frame_is_evidence_only_when_it_contains_actual_decay():
    values = blank(15)
    attack(values, 10)
    active(values, 12, 14)
    old, new, rows = diagnose(values)
    assert new[0]['offset'] == float(np.float32(.14))
    assert rows[0]['active_frames'] == 2
    active(values, 12, 15)
    old, new, rows = diagnose(values)
    assert old == new
    assert rows[0]['status'] == 'active_without_observed_decay'


def test_threshold_equality_is_inactive_and_does_not_count_as_two_frames():
    values = blank()
    attack(values, 10)
    active(values, 12, 13)
    active(values, 13, 14, level=.1)
    old, new, rows = diagnose(values)
    assert old == new
    assert rows[0]['active_frames'] == 1
    assert rows[0]['status'] == 'isolated_activation'


def test_attack_subframe_shift_and_every_nonrelease_field_are_preserved():
    values = blank()
    attack(values, 10, shape=(.1, .3, .9, .6, .1))
    active(values, 12, 40)
    old, new, _ = diagnose(values)
    assert old[0]['onset'] != float(np.float32(.1))
    for name in old[0]:
        if name not in ('offset', 'release_reason', 'release_observed'):
            assert new[0][name] == old[0][name]


def test_diagnostic_never_mutates_activation_arrays_or_key_order():
    values = blank()
    attack(values, 10)
    active(values, 12, 40)
    copied = {name: value.copy() for name, value in values.items()}
    expected = diagnose(values)
    assert diagnose(dict(reversed(list(values.items())))) == expected
    for name in values:
        assert np.array_equal(values[name], copied[name])


def test_plateau_onsets_skipped_by_decoder_do_not_desynchronize_event_identity():
    values = blank()
    values['reg_onset_output'][8:15, 39] = [.1, .5, .9, .9, .9, .5, .1]
    active(values, 14, 40)
    attack(values, 50, 72)
    active(values, 52, 70, 72)
    old, new, rows = diagnose(values)
    assert len(old) == len(new) == len(rows) == 2
    assert [event['pitch'] for event in old] == [60, 72]
    assert [row['pitch'] for row in rows] == [60, 72]
    assert all(row['index'] == index for index, row in enumerate(rows))


def test_empty_valid_activations_have_no_fabricated_events():
    assert diagnose(blank(0)) == ([], [], [])
