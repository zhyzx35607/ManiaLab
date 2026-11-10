"""Equal source weights must not hide better aligned chord attacks."""
from dataclasses import replace
from itertools import permutations

import pytest

from manialab.arrange import parameters, assign_lanes
from manialab.domain import Note, validate
from manialab.expression import chord_layer_options, select_expression


def note(identifier, pitch, shift=0, **values):
    return Note(start=1000, end=1500, pitch=pitch, id=identifier, voice='other',
                source='pitch', evidence_start=1000-shift, evidence_end=1500,
                release_reason='model_predicted_offset', **values)


def test_equal_weight_chord_prefers_aligned_middle_pitch_and_keeps_whole_note():
    notes = [note('low', 48, 75), note('middle', 60, 5), note('high', 72, 35)]
    original = [n.data() for n in notes]
    settings = parameters(dict(chord_size=3))
    for order in permutations(notes):
        options = chord_layer_options(order, settings, ())
        assert [[n.id for n in row] for row in options] == [
            ['middle'], ['middle', 'high'], ['middle', 'high', 'low']]
        assert all(n is notes[1] for n in (row[0] for row in options))
    assert [n.data() for n in notes] == original


def test_strength_priority_is_unchanged_for_unequal_source_weights():
    strong = note('strong', 72, 80, strength=.9)
    weak = note('weak', 48, 0, strength=.8)
    assert chord_layer_options([weak, strong], parameters(), ())[0] == [strong]


def test_missing_evidence_is_not_treated_as_a_perfectly_aligned_alternative():
    known = note('known', 48, 50)
    unknown = replace(note('unknown', 60), evidence_start=None)
    assert chord_layer_options([unknown, known], parameters(), ())[0] == [known]


@pytest.mark.parametrize('shift', [None, 0, 20])
def test_equal_alignment_retains_stable_pitch_then_id_order(shift):
    notes = [replace(note(identifier, pitch), evidence_start=None if shift is None else 1000-shift)
             for identifier, pitch in [('c', 72), ('b', 48), ('a', 48)]]
    assert [n.id for n in chord_layer_options(notes, parameters(), ())[-1]] == ['a', 'b', 'c']


def test_alignment_tie_break_respects_local_layer_caps_and_transients():
    notes = [note('off', 48, 50), note('on', 60),
             replace(note('later', 48), start=2000, end=2500, evidence_start=2000)]
    local = [dict(start_ms=500, end_ms=1500, preferences=dict(chord_size=1))]
    assert [n.id for n in chord_layer_options(notes, parameters(), local)[0]] == ['on', 'later']
    transients = [replace(n, source='transient', voice='drums', end=n.start) for n in notes[:2]]
    assert {n.id for n in chord_layer_options(transients, parameters(dict(chord_size=1)), ())[0]} == {'off', 'on'}


@pytest.mark.parametrize('keys', range(6, 11))
def test_reduction_cannot_drop_a_clear_chord_in_favor_of_a_weaker_competitor(keys):
    notes = [note('off', 48, 80), note('on', 60),
             replace(note('competing', 65, strength=.8), voice='piano')]
    settings = parameters(dict(keys=keys, chord_size=1, density=1, max_held=1))
    metadata = {n.id: dict(section=0, role='support') for n in notes}
    trace = {n.id: dict(id=n.id, start=n.start, reason='supported_sustain') for n in notes}
    chosen, report = select_expression(notes, metadata, trace, settings, 500)
    assert chosen == [notes[1]]
    rejected = {row['id']: row for row in report['decisions']}
    assert rejected['off']['reason'] == 'represented_by_chord'
    assert rejected['off']['represented_by'] == ['on']
    assert rejected['competing']['reason'] == 'group_capacity_or_density'
    assert validate(assign_lanes(chosen, settings), keys)['valid']
