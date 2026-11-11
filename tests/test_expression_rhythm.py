import copy
from dataclasses import replace

import pytest

from manialab.domain import Note
from manialab.expression_rhythm import plan_rhythm
from manialab.timing import DIVISIONS


CLOCK = dict(bpm=120, offset=91, division=4)


def note(identity, start, **kwargs):
    return Note(start, kwargs.pop('end', start), id=identity, voice=kwargs.pop('voice', 'piano'),
                source='pitch', instrument_id=kwargs.pop('instrument_id', 'piano'), **kwargs)


def policy(**kwargs):
    return dict(section=0, voice='piano', instrument_id='piano', stride=2, phase=0) | kwargs


def metadata(notes):
    return {n.id: dict(section=0, role='support') for n in notes}


def plan(notes, *, clock=CLOCK, policies=None, **kwargs):
    return plan_rhythm(notes, metadata(notes), clock=clock, confirmed=True,
                       policies=[policy()] if policies is None else policies, **kwargs)


def test_disabled_path_does_not_inspect_candidates_metadata_or_clock():
    for empty in ((), []):
        assert plan_rhythm(None, None, clock=False, confirmed=False, policies=empty) == ({}, None)


def test_nonzero_offset_uses_original_grid_and_reports_actual_omissions():
    notes = [note(str(i), t) for i, t in enumerate((91, 216, 341, 466, 591))]
    decisions, report = plan(notes)
    assert [decisions[n.id]['tick'] for n in notes] == [0, 1, 2, 3, 4]
    assert [decisions[n.id]['keep'] for n in notes] == [True, False, True, False, True]
    assert report['kept_candidates'] == 3 and report['omitted_candidates'] == 2
    row = report['policies'][0]
    assert row['attack_groups'] == 5 and row['kept_attack_groups'] == 3
    assert row['omitted_attack_groups'] == 2
    assert report['semantics'] == 'explicit_chart_grid_subset'
    assert report['automatic_role_inference'] is False


def test_section_boundary_does_not_reset_phase_to_its_first_attack():
    notes = [note('before', 91), note('first_after', 216), note('next', 341)]
    meta = metadata(notes)
    meta['first_after']['section'] = meta['next']['section'] = 1
    decisions, _ = plan_rhythm(notes, meta, clock=CLOCK, confirmed=True,
                               policies=[policy(), policy(section=1)])
    assert [decisions[n.id]['keep'] for n in notes] == [True, False, True]
    assert decisions['first_after']['tick'] == 1
    assert decisions['first_after']['scope']['section'] == 1


def test_triplet_grid_retains_explicit_phase_without_assuming_half_beats():
    notes = [note(str(i), round(91 + i * 500 / 3)) for i in range(7)]
    decisions, report = plan(notes, clock=dict(CLOCK, division=3),
                             policies=[policy(stride=3, phase=1)])
    assert {identity for identity, d in decisions.items() if d['keep']} == {'1', '4'}
    assert report['clock']['division'] == 3
    with pytest.raises(ValueError, match='stride'):
        plan(notes, clock=dict(CLOCK, division=3))


@pytest.mark.parametrize('division', DIVISIONS)
def test_supported_subdivisions_accept_integer_millisecond_serialization(division):
    clock = dict(bpm=74, offset=178, division=division)
    notes = [note(str(i), round(clock['offset'] + i * 60000 / clock['bpm'] / division))
             for i in range(division * 2)]
    decisions, _ = plan(notes, clock=clock, policies=[policy(stride=division, phase=0)])
    assert {identity for identity, d in decisions.items() if d['keep']} == {'0', str(division)}


def test_all_chord_pitches_share_decision_and_ln_endpoints_are_unchanged():
    notes = [note('low', 91, end=1800, pitch=48, strength=.4),
             note('high', 91, end=800, pitch=72, strength=1.),
             note('dropped_hold', 216, end=950, pitch=60),
             note('dropped_tap', 216, pitch=67)]
    meta, clock, rules = metadata(notes), dict(CLOCK), [policy()]
    original = copy.deepcopy((notes, meta, clock, rules))
    decisions, report = plan_rhythm(notes, meta, clock=clock, confirmed=True, policies=rules)
    assert decisions['low'] == decisions['high']
    assert decisions['dropped_hold'] == decisions['dropped_tap']
    assert decisions['low']['keep'] and not decisions['dropped_hold']['keep']
    assert (notes, meta, clock, rules) == original
    assert report['kept_candidates'] == 2 and report['omitted_candidates'] == 2
    assert report['policies'][0]['kept_attack_groups'] == 1
    assert report['policies'][0]['omitted_attack_groups'] == 1


def test_rule_is_local_to_exact_section_source_and_instrument_even_if_others_are_off_grid():
    notes = [note('target', 216), note('class', 217, instrument_id='other_piano'),
             note('source', 217, voice='other'), note('section', 217)]
    meta = metadata(notes)
    meta['section']['section'] = 1
    decisions, _ = plan_rhythm(notes, meta, clock=CLOCK, confirmed=True, policies=[policy()])
    assert set(decisions) == {'target'} and not decisions['target']['keep']


def test_empty_instrument_identity_is_an_explicit_scope_not_a_wildcard():
    notes = [note('unknown', 216, instrument_id=''), note('known', 216)]
    decisions, _ = plan(notes, policies=[policy(instrument_id='')])
    assert set(decisions) == {'unknown'}


def test_empty_subset_never_adds_a_strongest_note_or_changes_phase():
    notes = [note('weak', 216, strength=.4), note('strong', 466, strength=1.)]
    decisions, report = plan(notes)
    assert not any(d['keep'] for d in decisions.values())
    assert report['kept_candidates'] == 0
    assert report['policies'][0]['kept_attack_groups'] == 0


def test_negative_ticks_and_audio_shift_metadata_do_not_reset_or_shift_chart_clock():
    clock = dict(bpm=120, offset=1000, division=4, audio_shift_ms=100)
    notes = [note('even', 0), note('odd', 125)]
    decisions, report = plan(notes, clock=clock)
    assert decisions['even']['tick'] == -8 and decisions['even']['keep']
    assert decisions['odd']['tick'] == -7 and not decisions['odd']['keep']
    assert report['clock'] == dict(bpm=120, offset=1000, division=4)


def test_stride_one_retains_every_attack_without_regrouping():
    notes = [note('a', 216), note('b', 591)]
    decisions, _ = plan(notes, policies=[policy(stride=1, phase=0)])
    assert all(d['keep'] for d in decisions.values())
    assert [d['tick'] for d in decisions.values()] == [1, 4]


@pytest.mark.parametrize('confirmed', [False, None, 1, 'true'])
def test_clock_requires_explicit_boolean_confirmation(confirmed):
    notes = [note('a', 91)]
    with pytest.raises(ValueError, match='确认'):
        plan_rhythm(notes, metadata(notes), clock=CLOCK, confirmed=confirmed, policies=[policy()])


@pytest.mark.parametrize('clock', [None, {}, False, '120',
    dict(CLOCK, bpm=True), dict(CLOCK, bpm=0), dict(CLOCK, bpm=1001),
    dict(CLOCK, bpm=float('nan')), dict(CLOCK, bpm=float('inf')), dict(CLOCK, bpm='120'),
    dict(CLOCK, offset=True), dict(CLOCK, offset=float('nan')), dict(CLOCK, offset=float('inf')),
    dict(CLOCK, offset=600001), dict(CLOCK, offset='91'),
    dict(CLOCK, division=True), dict(CLOCK, division=4.), dict(CLOCK, division=5)])
def test_invalid_clock_rejected(clock):
    with pytest.raises(ValueError, match='节奏子集'):
        plan([note('a', 91)], clock=clock)


@pytest.mark.parametrize('points', [
    ['0,500,4,2,1,70,1,0', '1000,400,4,2,1,70,1,0'],
    ['0,500,4,2,1,70,1,0'], ['500,-100,4,2,1,70,0,0'], 'variable'])
def test_timing_points_are_not_silently_ignored(points):
    with pytest.raises(ValueError, match='timing_points'):
        plan([note('a', 91)], clock=dict(CLOCK, timing_points=points))


@pytest.mark.parametrize('points', [None, [], ()])
def test_empty_timing_points_do_not_prevent_constant_clock(points):
    decisions, _ = plan([note('a', 91)], clock=dict(CLOCK, timing_points=points))
    assert decisions['a']['keep']


@pytest.mark.parametrize('changes', [dict(stride=True), dict(stride=0), dict(stride=-1),
    dict(stride=3), dict(stride=2.), dict(stride='2'), dict(phase=True), dict(phase=-1),
    dict(phase=2), dict(phase=.5), dict(phase='0'), dict(section=True), dict(section=-1),
    dict(section=0.), dict(voice=''), dict(voice=None), dict(instrument_id=None)])
def test_invalid_policy_fields_rejected(changes):
    with pytest.raises(ValueError, match='节奏子集'):
        plan([note('a', 91)], policies=[policy(**changes)])


@pytest.mark.parametrize('policies', [None, False, {}, 'full', [None], [{}],
    [policy(extra=True)], [{k:v for k,v in policy().items() if k != 'phase'}]])
def test_malformed_policy_collection_rejected(policies):
    notes = [note('a', 91)]
    with pytest.raises(ValueError, match='节奏子集'):
        plan_rhythm(notes, metadata(notes), clock=CLOCK, confirmed=True, policies=policies)


def test_duplicate_scope_is_rejected_even_when_rules_agree():
    with pytest.raises(ValueError, match='一个节奏子集'):
        plan([note('a', 91)], policies=[policy(), policy()])


def test_scope_without_candidates_is_not_silently_accepted():
    with pytest.raises(ValueError, match='没有对应候选'):
        plan([note('a', 91)], policies=[policy(section=1)])


@pytest.mark.parametrize('start', [92, 92., True, -1])
def test_scoped_non_grid_or_non_integer_attack_is_rejected(start):
    with pytest.raises(ValueError, match='节奏子集候选'):
        plan([note('a', start)])


def test_millisecond_tolerance_is_validation_not_resnapping():
    notes = [note('a', 91)]
    decisions, _ = plan(notes, clock=dict(CLOCK, offset=91.5))
    assert decisions['a']['tick'] == 0 and notes[0].start == 91
    with pytest.raises(ValueError, match='没有落在'):
        plan(notes, clock=dict(CLOCK, offset=91.502))


def test_missing_or_duplicate_identity_and_metadata_are_rejected():
    original = note('a', 91)
    for notes, meta in (([replace(original, id='')], {'':dict(section=0)}),
                        ([original, original], metadata([original])),
                        ([original], {}), ([original], {'a':dict(section=True)})):
        with pytest.raises(ValueError, match='节奏子集'):
            plan_rhythm(notes, meta, clock=CLOCK, confirmed=True, policies=[policy()])
