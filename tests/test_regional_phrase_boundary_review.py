"""Independent checks of phrase scope using actual generation and saved edits."""
from copy import deepcopy

import pytest

from manialab.domain import Note
from manialab.editing import edit_notes
from tests.test_regional_phrase_boundary import generate, project


def test_capacity_rejected_phrase_still_cannot_remove_internal_regional_attack():
    state, tempo = project()
    state['analysis']['candidates'][0]['strength'] = .4
    state['analysis']['phrase_candidates'][0]['strength'] = .4
    state['analysis']['candidates'].append(Note(750, 2500, id='piano', source='pitch',
        voice='piano', release_reason='frame_decay').data())
    before = deepcopy(generate(state, tempo, amount=0, max_held=1))
    assert 'b' in {n['id'] for n in before['notes']}

    proposal = generate(deepcopy(state), tempo, max_held=1)
    decisions = {d['id']: d for d in proposal['selection']['decisions']}
    assert decisions['a']['decision'] == 'omitted'
    assert decisions['b']['reason'] == 'phrase_not_selected'
    with pytest.raises(ValueError, match='乐句.*范围'):
        generate(state, tempo, region=[1250, 2500], max_held=1)
    assert state['variants'] == [before]


@pytest.mark.parametrize('region,rejected', [([1477, 1490], True), ([1490, 1550], False)])
def test_fractional_grid_uses_quantized_attack_instead_of_raw_audio(region, rejected):
    state, tempo = project(shift=22.5, offset=17.3)
    tempo.update(bpm=137, division=3)
    before = deepcopy(generate(state, tempo, amount=0))
    assert next(n['start'] for n in before['notes'] if n['id'] == 'b') == 1477
    if rejected:
        with pytest.raises(ValueError, match='乐句.*范围'):
            generate(state, tempo, region=region)
        assert state['variants'] == [before]
    else:
        actual = generate(state, tempo, region=region)
        assert actual['notes'] == before['notes']


@pytest.mark.parametrize('history', ['missing_snapshot', 'shortened_phrase', 'reanalysed_support'])
def test_old_phrase_identity_does_not_bypass_cross_boundary_protection(history):
    state, tempo = project()
    old = generate(state, tempo)
    if history == 'missing_snapshot':
        old.pop('source_evidence')
    elif history == 'shortened_phrase':
        notes, edit = edit_notes(old, dict(op='update', note_id='a', values=dict(end=1250)), 3000)
        old.update(notes=[n.data() for n in notes], edits=[edit], region=None,
                   selection=dict(message='已手动修改'))
        assert next(n for n in old['notes'] if n['id'] == 'a')['phrase']
    else:
        state['analysis']['candidates'][1]['pitch'] = 67
    before = deepcopy(state['variants'])
    with pytest.raises(ValueError, match='乐句.*范围'):
        generate(state, tempo, region=[1250, 2500])
    assert state['variants'] == before


def test_collapsed_phrase_with_same_quantized_attack_does_not_create_false_boundary():
    state, tempo = project()
    tempo.update(bpm=10, division=3)
    generate(state, tempo, amount=0)
    actual = generate(state, tempo, region=[1900, 2100])
    decisions = {d['id']: d for d in actual['selection']['proposal']['decisions']}
    assert decisions['b']['reason'] == 'phrase_not_selected'
    anchor = next(n for n in actual['notes'] if n['id'] == 'a')
    assert (anchor['start'], anchor['end'], anchor['phrase']) == (2000, 2000, False)


def test_legacy_phrase_tail_replacement_keeps_internal_attacks_and_remains_allowed():
    state, tempo = project()
    state['analysis'].pop('structure')
    before = deepcopy(generate(state, tempo, amount=0))
    actual = generate(state, tempo, region=[1250, 2500])
    geometry = lambda v: [(n['id'], n['start'], n['end']) for n in v['notes']]
    assert geometry(actual) == geometry(before)
    assert 'b' in {n['id'] for n in actual['notes']}
