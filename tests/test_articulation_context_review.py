"""Independent controls for source context across chart section boundaries."""
import copy

import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import expression_evidence, propose_expression
from manialab.expression_articulation import articulation_policy


def n(start, *, pitch=60, end=None, strength=1., voice='piano', identifier=None):
    return Note(start, start+1200 if end is None else end, pitch=pitch, strength=strength,
                source='pitch', voice=voice, id=identifier or f'{voice}:{start}:{pitch}',
                release_reason='release_regression')


def apply(notes, cuts, modes, *, duration_ms=6000):
    source = dict(duration=duration_ms/1000, tempo=dict(bpm=120, offset=0),
        candidates=[note.data() for note in notes], structure=dict(segments=[
            dict(start_ms=left, end_ms=right) for left, right in zip(cuts, cuts[1:])]))
    before = copy.deepcopy(source)
    settings = parameters()
    plan = propose_expression(source, settings)
    for section, mode in zip(plan['sections'], modes):
        for policy in section['voices'].values():
            policy['mode'] = mode
    output, metadata, trace = expression_evidence(source, plan, settings)
    assert source == before
    # Expression decisions may retain or collapse a tail, never move an attack.
    raw = {note.id: note for note in notes}
    assert all(note.start == raw[note.id].start and note.end in (note.start, raw[note.id].end)
               for note in output)
    return output, metadata, trace


def test_section_boundary_inside_thirty_ms_chord_keeps_full_source_expression():
    notes = [n(start, pitch=pitch) for start, pitch in
             ((1000,60),(1029,64),(1250,62),(1279,65),(1500,64),(1529,67))]
    full, _, whole_trace = apply(notes, [0,6000], ['articulation'])
    split, metadata, trace = apply(notes, [0,1015,1265,1515,6000], ['articulation']*4)
    assert [note.data() for note in split] == [note.data() for note in full]
    assert [note.hold for note in split] == [False,False,False,False,True,True]
    assert metadata[notes[0].id]['section'] != metadata[notes[1].id]['section']
    for note in notes:
        assert trace[note.id]['articulation'] == whole_trace[note.id]['articulation']
    assert trace[notes[0].id]['articulation']['run_start'] == 1000


@pytest.mark.parametrize('other_mode,expected', [
    ('sustain', [False,False,True,True]),
    ('onsets', [False,False,False,False]),
    ('omit', [False,False]),
])
def test_explicit_later_section_mode_keeps_its_intent_without_fake_earlier_tail(other_mode, expected):
    notes = [n(t) for t in (1000,1250,1500,1750)]
    output, _, trace = apply(notes, [0,1400,6000], ['articulation',other_mode])
    assert [note.hold for note in output] == expected
    assert trace[notes[1].id]['articulation']['continuation'] == 'next_attack'
    assert all('articulation' not in trace[note.id] for note in notes[2:])
    if other_mode == 'omit':
        assert all(trace[note.id]['reason'] == 'voice_not_selected' for note in notes[2:])


def test_mode_switch_inside_one_chord_is_explicit_not_an_artificial_source_split():
    notes = [n(t, pitch=p) for t,p in ((1000,60),(1250,62),(1279,65),(1500,64))]
    output, _, trace = apply(notes, [0,1265,6000], ['articulation','sustain'])
    assert [note.hold for note in output] == [False,False,True,True]
    assert trace[notes[1].id]['articulation']['attack_groups'] == 3
    assert 'articulation' not in trace[notes[2].id]


def test_insufficient_audio_right_context_does_not_turn_open_run_into_tail():
    notes = [n(t) for t in (1000,1250,1500)]
    output, _, trace = apply(notes, [0,1600], ['articulation'], duration_ms=1600)
    assert not any(note.hold for note in output)
    assert trace[notes[-1].id]['articulation']['continuation'] == 'unknown'
    assert trace[notes[-1].id]['articulation']['articulation'] == 'attack_run_open'
    # Enough observed time changes the context classification, not raw ends.
    complete, _, closed_trace = apply(notes, [0,6000], ['articulation'])
    assert complete[-1].end == notes[-1].end
    assert closed_trace[notes[-1].id]['articulation']['continuation'] == 'observed_gap'


def test_two_attacks_do_not_become_a_run_merely_from_a_short_right_context():
    notes = [n(1000), n(1250)]
    rules = articulation_policy(notes, observed_until_ms=1300)
    assert not any(value['tap'] for value in rules.values())
    assert rules[notes[-1].id]['continuation'] == 'unknown'
    assert rules[notes[-1].id]['articulation'] == 'independent_sustain'


def test_filtered_weak_attack_cannot_extend_a_run_or_enter_expression():
    notes = [n(t) for t in (1000,1250,1500)]
    weak = n(1750, strength=.349, identifier='weak')
    baseline, _, baseline_trace = apply(notes, [0,6000], ['articulation'])
    output, _, trace = apply(notes+[weak], [0,1400,6000], ['articulation']*2)
    assert [note.data() for note in output] == [note.data() for note in baseline]
    assert trace['weak']['reason'] == 'weak_detection'
    assert trace[notes[-1].id]['articulation'] == baseline_trace[notes[-1].id]['articulation']
    accepted, _, accepted_trace = apply(notes+[n(1750,strength=.35,identifier='threshold')],
        [0,1400,6000], ['articulation']*2)
    assert len(accepted) == 4 and not accepted[-2].hold and accepted[-1].hold
    assert accepted_trace[notes[-1].id]['articulation']['continuation'] == 'next_attack'


def test_interleaved_different_source_does_not_supply_missing_attack_context():
    piano = [n(1000),n(1500)]
    bass = [n(1250,voice='bass'),n(1750,voice='bass')]
    output, _, trace = apply(piano+bass, [0,1400,6000], ['articulation']*2)
    assert all(note.hold for note in output)
    assert all(value['articulation']['attack_groups'] == 2 for value in trace.values())


def test_input_order_does_not_change_expression_identity_or_decisions():
    notes = [n(t,pitch=p) for t,p in ((1000,60),(1020,64),(1250,62),(1500,64))]
    forward, _, first = apply(notes, [0,1010,1400,6000], ['articulation']*3)
    reverse, _, second = apply(list(reversed(notes)), [0,1010,1400,6000], ['articulation']*3)
    assert {note.id:note.data() for note in forward} == {note.id:note.data() for note in reverse}
    assert first == second


@pytest.mark.parametrize('mode', ['sustain','articulation','onsets','phrase','omit'])
@pytest.mark.parametrize('identity', ['',None,7,['unhashable']])
def test_invalid_candidate_identity_rejects_before_any_mode_can_overwrite_trace(mode, identity):
    note = n(1000)
    note.id = identity
    with pytest.raises(ValueError, match='标识'):
        apply([note], [0,6000], [mode])


@pytest.mark.parametrize('mode', ['sustain','articulation','onsets','phrase','omit'])
def test_duplicate_identity_across_sections_rejects_instead_of_copying_later_tail(mode):
    notes = [n(1000,identifier='duplicate'),n(1250),n(1500,identifier='duplicate')]
    with pytest.raises(ValueError, match='重复'):
        apply(notes, [0,1400,6000], [mode,mode])


@pytest.mark.parametrize('duration', [None,False,True,0,-1,float('nan'),float('inf'),'6'])
def test_articulation_requires_valid_audio_duration_instead_of_using_legacy_unknown_tail(duration):
    notes = [n(t) for t in (1000,1250,1500)]
    source = dict(duration=duration,tempo=dict(bpm=120,offset=0),candidates=[note.data() for note in notes],
                  structure=dict(segments=[dict(start_ms=0,end_ms=6000)]))
    settings = parameters()
    plan = propose_expression(source,settings)
    plan['sections'][0]['voices']['piano']['mode']='articulation'
    with pytest.raises(ValueError,match='音频时长'):
        expression_evidence(source,plan,settings)


def test_old_sustain_plan_without_duration_remains_independent_of_new_context_policy():
    notes = [n(t) for t in (1000,1250,1500)]
    source = dict(tempo=dict(bpm=120,offset=0),candidates=[note.data() for note in notes],
                  structure=dict(segments=[dict(start_ms=0,end_ms=6000)]))
    settings=parameters();plan=propose_expression(source,settings)
    plan['sections'][0]['voices']['piano']['mode']='sustain'
    output,_,trace=expression_evidence(source,plan,settings)
    assert [(note.start,note.end) for note in output]==[(note.start,note.end) for note in notes]
    assert all('articulation' not in row for row in trace.values())
