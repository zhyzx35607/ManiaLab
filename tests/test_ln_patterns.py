from dataclasses import replace

import numpy as np
import pytest

from manialab.arrange import parameters
from manialab.domain import Note, invariant, validate
from manialab.layout import release_choices, search_layout
from manialab.ln_patterns import analyze_ln_patterns, hand_matrix, ln_row_costs


def test_release_only_rows_and_occupancy_are_counted_without_heads():
    notes = [Note(0, 1000, 0), Note(0, 2000, 1), Note(500, 500, 2), Note(1500, 1500, 5)]
    report = analyze_ln_patterns(notes, 6)
    assert report['counts']['releases'] == 2
    assert report['counts']['release_only_rows'] == 2
    assert report['counts']['split_release_hands'] == 1
    assert report['counts']['hybrid_taps'] == 2
    assert report['counts']['same_hand_hybrid_taps'] == 1
    assert report['occupied_key_fraction'] == .25
    assert report['held_seconds_by_lane'] == [1., 2., 0., 0., 0., 0.]
    assert report['peak_held_by_hand'] == [2., 0.]
    assert report['peak_releases_per_second'] == 1


def test_density_integrates_holds_across_empty_sections_and_locations_cover_ending():
    notes=[Note(0,40000,0)]+[Note(t,t,1) for t in range(1000,40000,500)]
    r=analyze_ln_patterns(notes,10)
    assert [s['occupied_key_fraction'] for s in r['sections']]==[.1,.1,.1]
    assert [s['peak_held'] for s in r['sections']]==[1,1,1]
    assert r['sections'][-1]['end_ms']==40000
    hybrid=[e for e in r['locations'] if e['kind']=='same_hand_hybrid_taps']
    assert len(hybrid)<=20 and hybrid[-1]['start_ms']>=38000


def test_inverse_is_positive_gap_in_local_beats_not_every_long_note():
    notes = [Note(0, 500, 0), Note(600, 800, 0), Note(800, 1000, 0),
             Note(2000, 2500, 0), Note(2600, 2600, 0)]
    result = analyze_ln_patterns(notes, 6, timing_points=['0,500,4,2,0,70,1,0','2000,250,4,2,0,70,1,0'])
    assert result['counts']['inverse_heads'] == 1  # 100ms fits at 120 but not 240 BPM.
    assert result['counts']['tight_repress_heads'] == 1  # exact release/head chain
    assert result['counts']['release_press_hands'] == 1


def test_hybrid_does_not_count_simultaneous_heads_or_a_finished_hold():
    notes = [Note(0,1000,0), Note(0,0,1), Note(1000,1000,2)]
    assert analyze_ln_patterns(notes,6)['hybrid_tap_fraction'] == 0


def test_shields_are_distinct_from_inverse_lns():
    notes=[Note(0,0,0),Note(100,500,0),Note(600,600,0),Note(1000,1400,1),Note(1500,1900,1)]
    c=analyze_ln_patterns(notes,6)['counts']
    assert c['shield_heads']==1 and c['reverse_shield_heads']==1 and c['inverse_heads']==1


def test_nearby_asynchronous_releases_include_past_and_future_but_not_own_tail():
    notes=[Note(0,1000,0),Note(950,950,1),Note(1050,1050,2),Note(2000,2050,1)]
    r=analyze_ln_patterns(notes,6)
    assert r['counts']['near_release_press_hands']==2


def test_release_order_choices_can_place_low_voice_on_higher_free_lane():
    choices = release_choices(6, np.array([2000,1000]))
    assert [5,0] in choices.tolist()
    assert all(len(set(c)) == len(c) for c in choices)
    assert len(release_choices(6,np.array([2000,2000]))) == 15


def test_release_control_looks_forward_to_independent_tails():
    choices = np.array([[1], [5]])
    ends, last = np.array([1000,-9999,-9999,-9999,-9999,-9999]), np.array([0,-9999,-9999,-9999,-9999,-9999])
    p = parameters({'keys':6,'ln_release':1,'ln_coordination':.5})
    costs = ln_row_costs(choices,500,np.array([1050]),ends,last,hand_matrix(6),500,p)
    assert costs[0] > costs[1]  # near asynchronous releases in same hand
    assert np.array_equal(ln_row_costs(choices,500,np.array([1000]),ends,last,hand_matrix(6),500,p), [0,0])


def test_simultaneous_same_hand_releases_are_one_timing_action():
    p=parameters({'keys':10,'ln_release':1,'ln_coordination':.5})
    hands=hand_matrix(10)
    previous=np.array([1000]+[-9999]*9);last=np.array([0]+[-9999]*9)
    one=ln_row_costs(np.array([[4],[9]]),950,np.array([1500]),previous,last,hands,500,p)
    previous[1:3]=1000;last[1:3]=0
    chord=ln_row_costs(np.array([[4],[9]]),950,np.array([1500]),previous,last,hands,500,p)
    assert np.array_equal(one,chord)
    # Likewise two new notes ending together should not double the temporal
    # penalty. The layout engine handles their separate occupied fingers.
    two=ln_row_costs(np.array([[3,4],[8,9]]),950,np.array([1500,1500]),previous,last,hands,500,p)
    assert np.array_equal(two,chord)
    previous[1]=1020
    staggered=ln_row_costs(np.array([[4],[9]]),950,np.array([1500]),previous,last,hands,500,p)
    assert staggered[0]>chord[0]


@pytest.mark.parametrize('keys', range(6,11))
def test_coordination_and_inverse_controls_change_lanes_but_not_audio(keys):
    notes = [Note(0,2000,0,source='pitch',id='hold'),
             *[Note(t,t,source='pitch',id=f't{t}') for t in range(250,1751,250)]]
    p = parameters({'keys':keys,'pattern_prior':0,'balance':0,'pitch_follow':0,'ln_release':0})
    separate = search_layout(notes,p|{'ln_coordination':0})
    hybrid = search_layout(notes,p|{'ln_coordination':1})
    a,b = [analyze_ln_patterns(n,keys) for n in (separate,hybrid)]
    assert b['same_hand_hybrid_fraction'] > a['same_hand_hybrid_fraction']
    assert invariant(separate) == invariant(hybrid) == invariant(notes)
    notes = [Note(i*500, i*500+400, pitch=60, source='pitch', id=str(i)) for i in range(12)]
    base = search_layout(notes,p|{'balance':.8,'ln_inverse':0})
    inverse = search_layout(notes,p|{'balance':.8,'ln_inverse':1})
    assert analyze_ln_patterns(inverse,keys)['inverse_head_fraction'] > analyze_ln_patterns(base,keys)['inverse_head_fraction']
    assert invariant(inverse) == invariant(notes)
    assert validate(inverse,keys)['valid']


def test_future_locked_note_can_require_reversing_release_order():
    # Only lanes 0 and 1 free now; lane 0 becomes occupied at 2000.
    locked = [Note(0,5000,i,id=f'held{i}') for i in range(2,6)] + [Note(2000,5000,0,id='future')]
    notes = [Note(1000,4000,pitch=40,id='low'),Note(1000,1500,pitch=80,id='high')]
    result = search_layout(notes,parameters({'keys':6}),locked=locked)
    assert next(n for n in result if n.id=='low').lane == 1
    assert next(n for n in result if n.id=='high').lane == 0
    assert invariant(result)==invariant(notes+locked)
