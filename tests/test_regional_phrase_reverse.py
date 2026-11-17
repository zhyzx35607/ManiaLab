"""A newly generated phrase must not also retain its locked internal attack."""
from copy import deepcopy

import pytest

from manialab.domain import Note
from manialab.editing import edit_notes
from tests.test_regional_phrase_boundary import generate, project


def edit_active(state, op, **values):
    old = state['variants'][-1]
    notes, edit = edit_notes(old, dict(op=op, note_id='b', values=values), 3000)
    old.update(notes=[n.data() for n in notes], region=None, selection=dict(message='已手动修改'))
    old.setdefault('edits', []).append(edit)


@pytest.mark.parametrize('shift,offset,bpm,division,expected', [
    (0, 0, 120, 4, (1000, 1500)),
    (22.5, 103, 120, 4, (978, 1478)),
    (-50, 50, 120, 4, (1050, 1550)),
    (22.5, 17.3, 137, 3, (1039, 1477)),
])
def test_new_phrase_cannot_duplicate_an_internal_attack_locked_at_upper_bound(
        shift, offset, bpm, division, expected):
    state, tempo = project(shift, offset)
    tempo.update(bpm=bpm, division=division)
    before = deepcopy(generate(state, tempo, amount=0))
    old = {n['id']: n for n in before['notes']}
    assert (old['a']['start'], old['b']['start']) == expected
    proposal = generate(deepcopy(state), tempo)
    assert next(n for n in proposal['notes'] if n['id'] == 'a')['phrase']
    assert 'b' not in {n['id'] for n in proposal['notes']}
    with pytest.raises(ValueError, match='乐句.*范围'):
        generate(state, tempo, region=list(expected))
    assert state['variants'] == [before]


def test_internal_attack_moved_outside_region_is_checked_at_its_saved_position():
    state, tempo = project()
    generate(state, tempo, amount=0)
    edit_active(state, 'update', start=2250, end=2250)
    before = deepcopy(state['variants'])
    assert before[-1]['source_evidence']['b']['detected']['start'] == 1500
    with pytest.raises(ValueError, match='乐句.*范围'):
        generate(state, tempo, region=[1000, 1750])
    assert state['variants'] == before


def test_reanalysed_internal_id_does_not_bypass_locked_old_attack_protection():
    state, tempo = project()
    before = deepcopy(generate(state, tempo, amount=0))
    state['analysis']['candidates'][1].update(start=1250, end=1350, pitch=67)
    assert before['source_evidence']['b']['detected']['pitch'] == 60
    with pytest.raises(ValueError, match='乐句.*范围'):
        generate(state, tempo, region=[1000, 1500])
    assert state['variants'] == [before]


def test_rejected_phrase_does_not_conflict_with_a_locked_internal_attack():
    state, tempo = project()
    state['analysis']['candidates'][0]['strength'] = .4
    state['analysis']['phrase_candidates'][0]['strength'] = .4
    state['analysis']['candidates'].append(Note(750, 2500, id='piano', source='pitch',
        voice='piano', release_reason='frame_decay').data())
    before = deepcopy(generate(state, tempo, amount=0, max_held=1))
    actual = generate(state, tempo, region=[1000, 1500], max_held=1)
    decisions = {d['id']: d for d in actual['selection']['proposal']['decisions']}
    assert decisions['b']['reason'] == 'phrase_not_selected'
    assert not any(n['id'] == 'a' for n in actual['notes'])
    assert next(n for n in actual['notes'] if n['id'] == 'b') == next(
        n for n in before['notes'] if n['id'] == 'b')


@pytest.mark.parametrize('previous', ['already_replaced', 'deleted', 'moved_inside'])
def test_new_phrase_is_allowed_when_no_internal_attack_will_stay_locked(previous):
    state, tempo = project()
    generate(state, tempo, amount=1 if previous == 'already_replaced' else 0)
    if previous == 'deleted':
        edit_active(state, 'delete')
    elif previous == 'moved_inside':
        edit_active(state, 'update', start=1250, end=1250)
    before = deepcopy(state['variants'])
    actual = generate(state, tempo, region=[1000, 1500])
    assert next(n for n in actual['notes'] if n['id'] == 'a')['phrase']
    assert 'b' not in {n['id'] for n in actual['notes']}
    assert state['variants'][:-1] == before


def test_ordinary_cross_boundary_sustain_can_keep_its_separate_internal_attack():
    state, tempo = project()
    state['analysis']['phrase_candidates'] = []
    state['analysis']['candidates'][0]['end'] = 2100
    before = deepcopy(generate(state, tempo, amount=0))
    actual = generate(state, tempo, region=[1000, 1500])
    notes = {n['id']: n for n in actual['notes']}
    assert (notes['a']['start'], notes['a']['end'], notes['a']['phrase']) == (1000, 2125, False)
    assert notes['b'] == next(n for n in before['notes'] if n['id'] == 'b')
    decisions = {d['id']: d for d in actual['selection']['proposal']['decisions']}
    assert decisions['b']['decision'] == 'selected' and 'represented_by' not in decisions['b']
