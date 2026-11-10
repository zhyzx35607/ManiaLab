import numpy as np
import pytest
from manialab.piano import OUTPUTS
from scripts.piano_delayed_frames import diagnose


def activation(*, start=10, end=100, first_active=12, active_end=30, next_attack=None, release=None):
    values={key:np.zeros((end,88),dtype=np.float32) for key in OUTPUTS}
    for attack in (start,next_attack):
        if attack is not None:
            values['reg_onset_output'][attack-2:attack+3,39]=[.1,.5,.9,.5,.1]
            values['velocity_output'][attack,39]=.7
    if first_active is not None:
        values['frame_output'][first_active:active_end,39]=.8
    if release is not None:
        values['reg_offset_output'][release-4:release+5,39]=[.01,.1,.2,.5,.9,.5,.2,.1,.01]
    return values


@pytest.mark.parametrize('delay',[2,5])
def test_delayed_frame_requires_real_decay_and_preserves_attack(delay):
    old,new,rows=diagnose(activation(first_active=10+delay))
    assert old[0]['release_reason']=='frame_inactive'
    assert rows[0]['delay_ms']==delay*10
    assert new[0]['offset']==float(np.float32(.3))
    assert new[0]['onset']==old[0]['onset']
    assert new[0]['pitch']==old[0]['pitch']
    assert new[0]['release_reason']=='delayed_frame_decay'


def test_existing_active_output_is_unchanged():
    old,new,rows=diagnose(activation(first_active=10))
    assert old==new and rows[0]['status']=='existing_active_tail'


@pytest.mark.parametrize('kwargs,status',[
    ({'first_active':None},'no_activation_in_window'),
    ({'first_active':16},'no_activation_in_window'),
    ({'first_active':12,'active_end':13},'isolated_activation'),
    ({'first_active':12,'active_end':100},'active_without_observed_decay'),
    ({'first_active':14,'next_attack':14},'reattack_before_activation'),
    ({'first_active':None,'end':14},'input_boundary_before_activation'),
])
def test_uncertain_observation_does_not_invent_a_tail(kwargs,status):
    old,new,rows=diagnose(activation(**kwargs))
    assert rows[0]['status']==status
    assert new[0]==old[0]


def test_offset_peak_before_activation_cannot_end_the_delayed_note():
    old,new,rows=diagnose(activation(first_active=15,release=12))
    assert rows[0]['first_release_frame'] is None
    assert new[0]['offset']==float(np.float32(.3))


def test_observed_release_peak_after_activation_may_select_tail():
    old,new,rows=diagnose(activation(first_active=12,release=28))
    assert rows[0]['status']=='delayed_release_regression'
    assert new[0]['offset']==float(np.float32(.28))


def test_same_pitch_reattack_cannot_be_used_as_a_decay():
    old,new,rows=diagnose(activation(first_active=12,active_end=30,next_attack=25))
    assert rows[0]['status']=='active_without_observed_decay'
    assert new[0]==old[0]
