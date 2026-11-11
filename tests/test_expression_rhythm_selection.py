import copy

import pytest

from manialab.domain import Note
from manialab.expression import select_expression


def choose(notes, policies=None, *, density=2, coherence=0., sustain=.25, beat_ms=500):
    metadata = {n.id: dict(section=0, role='support') for n in notes}
    trace = {n.id: dict(id=n.id, start=n.start, mode='sustain' if n.hold else 'onsets',
        reason='supported_sustain' if n.hold else 'section_onsets') for n in notes}
    rhythm = None if policies is None else dict(clock=dict(bpm=120, offset=0, division=4),
        confirmed=True, policies=policies)
    return select_expression(notes, metadata, trace, dict(keys=6, chord_size=2, density=density),
        beat_ms, coherence=coherence, sustain_coverage=sustain, rhythm_policy=rhythm)


def piano(time, strength=.5, **kwargs):
    return Note(time, time, voice='piano', source='pitch', id=f'p{time}', strength=strength, **kwargs)


POLICY = [dict(section=0, voice='piano', instrument_id='', stride=2, phase=0)]


def test_explicit_rhythm_subset_can_share_capacity_with_an_existing_sustain():
    bass = Note(0, 1000, voice='bass', source='pitch', id='b', strength=1.,
                release_reason='model_predicted_offset')
    notes = [piano(0), piano(125), bass]
    original = copy.deepcopy(notes)
    full, _ = choose(notes)
    selected, report = choose(notes, POLICY)
    assert full == [bass]
    assert {n.id for n in selected} == {'b', 'p0'}
    assert notes == original and all(n in notes for n in selected)
    trace = {d['id']: d for d in report['decisions']}
    assert trace['p125']['reason'] == 'rhythm_policy_omitted'
    assert trace['p125']['rhythm_policy']['tick'] == 1
    assert trace['p125']['rhythm_source_group'] == trace['p0']['rhythm_source_group']
    assert 'represented_by' not in trace['p125']


def test_omitted_strong_attack_cannot_lend_reward_to_a_retained_weak_attack():
    notes = [piano(0, .4), piano(125, 1.),
             Note(0, 0, voice='other', source='pitch', id='o', strength=.9)]
    selected, report = choose(notes, POLICY, density=1, sustain=0.)
    assert [n.id for n in selected] == ['o']
    trace = {d['id']: d for d in report['decisions']}
    assert trace['p0']['reason'] == 'group_capacity_or_density'
    assert trace['p125']['reason'] == 'rhythm_policy_omitted'


def test_mask_does_not_join_original_groups_or_fill_an_empty_group():
    notes = [piano(125), piano(250), piano(1125), piano(2125), piano(2250)]
    selected, report = choose(notes, POLICY, density=8, sustain=0.)
    assert {n.id for n in selected} == {'p250', 'p2250'}
    trace = {d['id']: d for d in report['decisions']}
    assert trace['p125']['rhythm_source_group'] == trace['p250']['rhythm_source_group']
    assert trace['p1125']['rhythm_source_group'] != trace['p250']['rhythm_source_group']
    assert trace['p1125']['rhythm_source_group'] != trace['p2250']['rhythm_source_group']
    assert trace['p1125']['decision'] == 'omitted'


def test_all_masked_attacks_remain_explicit_when_selection_is_empty():
    selected, report = choose([piano(125)], POLICY)
    assert not selected
    assert report['decisions'][0]['reason'] == 'rhythm_policy_omitted'
    assert 'rhythm_policy' in report


def test_same_attack_chord_options_keep_original_durations():
    notes = [Note(0, 80, voice='piano', source='pitch', id='lo', pitch=48, strength=.6),
             Note(0, 90, voice='piano', source='pitch', id='hi', pitch=72, strength=.6), piano(125)]
    selected, report = choose(notes, POLICY, density=8, sustain=0.)
    assert selected == notes[:2]
    assert report['selected'] == 2


def test_disabled_policy_preserves_full_selection_report():
    notes = [piano(0), piano(125)]
    default, report = choose(notes)
    empty, empty_report = choose(notes, [])
    assert empty == default and empty_report == report
    assert 'rhythm_policy' not in report


def test_masked_only_attack_still_checks_local_density_and_drum_fallback():
    notes = [Note(0, 0, voice='drums', source='transient', pitch=36, id='kick', instrument_id='kick'),
             Note(0, 0, voice='drums', source='transient', pitch=38, id='snare', instrument_id='snare'), piano(125)]
    metadata = {n.id: dict(section=0, role='support') for n in notes}
    trace = {n.id: dict(id=n.id, start=n.start, mode='onsets', reason='section_onsets') for n in notes}
    selected, report = select_expression(notes, metadata, trace,
        dict(keys=6, chord_size=2, density=2), 500,
        [dict(start_ms=125, end_ms=500, preferences=dict(density=1))],
        rhythm_policy=dict(clock=dict(bpm=120, offset=0, division=4), confirmed=True, policies=POLICY))
    assert len(selected) == 1 and selected[0].voice == 'drums'
    assert report['drum_fallback_solver'] is None


@pytest.mark.parametrize('extra,message', [
    (dict(coherence=.25), '连贯'), (dict(beat_ms=600), '拍长')])
def test_unvalidated_objective_combination_or_mismatched_clock_is_rejected(extra, message):
    with pytest.raises(ValueError, match=message):
        choose([piano(0)], POLICY, **extra)
