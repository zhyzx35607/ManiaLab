from copy import deepcopy

import pytest

from manialab.arrange import assign_lanes, parameters
from manialab.domain import Note, validate
from manialab.expression import expression_evidence, propose_expression, select_expression
from manialab.structure import edit_boundaries


def select(notes, *, segments=(), **settings):
    analysis = dict(duration=8, tempo=dict(bpm=120, offset=0),
                    candidates=[note.data() for note in notes])
    analysis['structure'] = edit_boundaries({}, [0, 8000], analysis)
    original = deepcopy(analysis)
    config = parameters(settings)
    evidence, metadata, trace = expression_evidence(
        analysis, propose_expression(analysis, config), config)
    chosen, report = select_expression(evidence, metadata, trace, config, 500, segments)
    assert analysis == original
    return chosen, {row['id']: row for row in report['decisions']}, config


@pytest.mark.parametrize('keys', range(6, 11))
def test_blocked_drum_group_can_retain_one_attack_without_cutting_the_melody(keys):
    notes = [Note(1000, 1000, strength=.9, source='transient', voice='drums', id='free'),
             Note(1500, 1500, strength=.8, source='transient', voice='drums', id='blocked'),
             Note(1500, 2200, strength=1, source='pitch', voice='piano', id='melody')]
    chosen, trace, config = select(notes, keys=keys, density=3, chord_size=1, ln_ratio=1)
    assert {note.id for note in chosen} == {'free', 'melody'}
    assert next(note for note in chosen if note.id == 'melody').end == 2200
    assert trace['blocked']['reason'] == 'group_capacity_or_density'
    assert 'represented_by' not in trace['blocked']
    assert trace['free']['group_simplification'] == 'strongest_drum_attack'
    assert validate(assign_lanes(chosen, config), keys)['valid']


def test_full_drum_rhythm_is_retained_when_it_fits():
    notes = [Note(t, t, strength=.9, source='transient', voice='drums', id=str(t))
             for t in (1000, 1500, 2000, 2500)]
    chosen, trace, _ = select(notes, density=4, chord_size=2)
    assert {note.id for note in chosen} == {note.id for note in notes}
    assert all('group_simplification' not in row for row in trace.values())


def test_simultaneous_drum_reduction_does_not_claim_to_represent_other_timbres():
    notes = [Note(1000, 1000, pitch=pitch, strength=strength, source='transient',
                  voice='drums', id=name, evidence_start=evidence)
             for name, pitch, strength, evidence in
             [('loud-off-grid', 36, .95, 910), ('aligned', 42, .7, 1000)]]
    chosen, trace, _ = select(notes, density=2, chord_size=1)
    assert [note.id for note in chosen] == ['aligned']
    assert trace['loud-off-grid']['reason'] == 'group_capacity_or_density'
    assert 'represented_by' not in trace['loud-off-grid']


def test_fallback_still_obeys_rolling_density_and_explicit_drum_mute():
    notes = [Note(1000, 1000, strength=.9, source='transient', voice='drums', id='drum'),
             Note(1500, 1500, strength=.8, source='transient', voice='drums', id='drum2'),
             Note(1600, 2200, strength=1, source='pitch', voice='piano', id='melody')]
    for overrides in (dict(density=1), dict(density=3, drums_weight=0)):
        chosen, trace, _ = select(notes, chord_size=2, ln_ratio=1, **overrides)
        assert [note.id for note in chosen] == ['melody']
        assert all('represented_by' not in trace[key] for key in ('drum', 'drum2'))


def test_individually_possible_anchors_compete_for_shared_remaining_capacity():
    notes = [Note(t, t, pitch=pitch, strength=strength, source='transient', voice='drums',
                  id=name, instrument_id=instrument)
             for name, t, pitch, strength, instrument in
             [('a', 1000, 36, .9, 'kick'), ('a-blocked', 1500, 36, .35, 'kick'),
              ('b', 1250, 42, .8, 'hat'), ('b-blocked', 1500, 42, .35, 'hat')]]
    notes.append(Note(1500, 2200, strength=1, source='pitch', voice='piano', id='melody'))
    chosen, _, _ = select(notes, density=2, chord_size=1, ln_ratio=1)
    assert {note.id for note in chosen} == {'a', 'melody'}


def test_lower_section_density_still_applies_at_an_omitted_candidate_time():
    notes = [Note(600, 600, strength=.9, source='transient', voice='drums', id='drum'),
             Note(800, 800, strength=.8, source='transient', voice='drums', id='drum2'),
             Note(1000, 1000, strength=1, source='pitch', voice='piano', id='melody'),
             Note(1100, 1100, strength=.4, source='pitch', voice='piano', id='quiet')]
    chosen, _, _ = select(notes, density=2, chord_size=2, ln_ratio=0,
        segments=[dict(start_ms=1100, end_ms=8000, preferences=dict(density=1))])
    assert [note.id for note in chosen] == ['melody']
