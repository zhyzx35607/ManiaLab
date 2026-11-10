"""Notation continuations must not become additional score attacks."""
from dataclasses import replace

import pytest

from scripts.piano_score_events import WrittenNote, score_events


def note(identifier, start, end, **kwargs):
    return WrittenNote(identifier, kwargs.get('part', 'P1'), kwargs.get('pitch', 60),
        start, end, kwargs.get('measure', 0), kwargs.get('staff', 1), kwargs.get('voice', '1'))


def known(notes, ties=()):
    notes = tuple(notes)
    return score_events(notes, ties, verified_note_ids=[n.id for n in notes])


def test_written_tie_chain_retains_one_attack_and_all_segments():
    notes = [note('a', 0, 120), note('b', 120, 240), note('c', 240, 480)]
    result = known(notes, [('a', 'b'), ('b', 'c')])
    assert len(result.events) == 1
    assert result.events[0].written_ids == ('a', 'b', 'c')
    assert (result.events[0].start_tick, result.events[0].end_tick) == (0, 480)
    assert [v.role for v in result.notation] == ['attack', 'continuation', 'continuation']
    assert [v.note for v in result.notation] == notes


def test_verified_tie_crosses_bar_staff_and_voice_without_inventing_role():
    notes = [note('a', 1680, 1920), note('b', 1920, 2400, measure=1, staff=2, voice='7')]
    result = known(notes, [('a', 'b')])
    assert len(result.events) == 1
    assert result.notation[1].note.staff == 2
    assert result.notation[1].note.voice == '7'
    assert result.notation[1].note.measure == 1
    assert result.notation[1].role == 'continuation'


def test_adjacent_same_pitch_rearticulations_are_never_inferred_as_ties():
    notes = [note('a', 0, 240), note('b', 240, 480)]
    result = known(notes)
    assert len(result.events) == 2
    assert [v.role for v in result.notation] == ['attack', 'attack']


def test_parallel_unison_voices_follow_explicit_edges_not_a_pitch_dictionary():
    notes = [note('a', 0, 240), note('b', 0, 120, voice='2'),
             note('c', 240, 480), note('d', 120, 480, voice='2')]
    result = known(notes, [('a', 'c'), ('b', 'd')])
    assert [event.written_ids for event in result.events] == [('a', 'c'), ('b', 'd')]
    assert sum(view.role == 'attack' for view in result.notation) == 2
    assert {view.note.id for view in result.notation} == {'a', 'b', 'c', 'd'}


def test_unverified_import_does_not_generate_positive_or_negative_attack_labels():
    notes = [note('a', 0, 240), note('b', 240, 480), note('safe', 600, 720)]
    result = score_events(notes, [], verified_note_ids=['safe'])
    assert [event.supervision_eligible for event in result.events] == [False, False, True]
    assert [view.role for view in result.notation] == ['unknown', 'unknown', 'attack']
    chained = score_events(notes, [('a', 'b')], verified_note_ids=['a', 'safe'])
    assert not chained.events[0].supervision_eligible
    assert [v.role for v in chained.notation[:2]] == ['unknown', 'unknown']


@pytest.mark.parametrize('change', [dict(pitch=61), dict(part='P2'), dict(start_tick=241),
                                  dict(start_tick=239)])
def test_rejects_unjustified_tie_connections(change):
    a, b = note('a', 0, 240), replace(note('b', 240, 480), **change)
    with pytest.raises(ValueError):
        known([a, b], [('a', 'b')])


@pytest.mark.parametrize('edges', [[('a', 'missing')], [('a', 'a')],
    [('a', 'b'), ('a', 'b')], [('a', 'b'), ('a', 'c')], [('a', 'c'), ('b', 'c')]])
def test_rejects_dangling_duplicate_or_branching_edges(edges):
    with pytest.raises(ValueError):
        known([note('a', 0, 240), note('b', 240, 480), note('c', 240, 480)], edges)


def test_rejects_duplicate_ids_and_unrelated_verification():
    a = note('a', 0, 240)
    with pytest.raises(ValueError):
        known([a, a])
    with pytest.raises(ValueError):
        score_events([a], [], verified_note_ids=['missing'])


def test_two_tie_predecessors_are_ambiguous_even_when_times_and_pitch_agree():
    notes = [note('a', 0, 240), note('b', 0, 240, voice='2'), note('c', 240, 480)]
    with pytest.raises(ValueError, match='branching'):
        known(notes, [('a', 'c'), ('b', 'c')])


@pytest.mark.parametrize('field,value', [
    ('end_tick', float('nan')), ('end_tick', float('inf')), ('end_tick', 1.5),
    ('start_tick', float('-inf')), ('start_tick', 0.5), ('start_tick', True),
    ('pitch', 60.5), ('pitch', float('nan')), ('pitch', False),
])
def test_invalid_numeric_fields_cannot_become_verified_supervision(field, value):
    with pytest.raises(ValueError, match='finite integers'):
        known([replace(note('a', 0, 240), **{field: value})])


def test_output_is_stable_under_input_order_and_preserves_input():
    notes = [note('a', -120, 0), note('b', 0, 240), note('c', 480, 600)]
    original = tuple(notes)
    expected = known(notes, [('a', 'b')])
    assert known(reversed(notes), [('a', 'b')]) == expected
    assert tuple(notes) == original
    assert known([], []).events == ()
