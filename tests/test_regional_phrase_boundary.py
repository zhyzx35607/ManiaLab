"""Regional rewrites must not rely on a new phrase outside their scope."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.generation import generate_variant


def project(shift=0, offset=0):
    notes = [Note(1000, 1100, id='a', source='pitch', voice='vocals', release_reason='frame_decay'),
             Note(1500, 1600, id='b', source='pitch', voice='vocals', release_reason='frame_decay'),
             Note(2700, 2700, id='d', source='transient', voice='drums')]
    phrase = replace(notes[0], end=2100, phrase=True, support=('a', 'b'),
        support_coverage=1, release_reason='vocal_activity_decay')
    analysis = dict(revision=13, duration=3, candidates=[n.data() for n in notes],
        phrase_candidates=[phrase.data()], timing_attacks=[],
        structure=dict(segments=[dict(start_ms=0, end_ms=3000)]))
    state = dict(analysis=analysis, variants=[], active=-1)
    tempo = dict(bpm=120, offset=offset, division=4, audio_shift_ms=shift)
    return state, tempo


def generate(state, tempo, *, amount=1, region=None, **settings):
    payload = dict(mode='audio', timing_confirmed=True)
    if region is not None:
        payload['region'] = region
    generate_variant(state, payload, parameters(dict(phrase_ln=amount, **settings)),
        tempo, lambda *_: None, lambda: False)
    return state['variants'][-1]


@pytest.mark.parametrize('shift,offset,region', [
    (0, 0, [1250, 2500]),
    (22.5, 103, [1478, 2500]),  # Internal raw 1500 becomes chart 1478, included.
    (-50, 50, [1550, 2500]),
])
def test_new_phrase_cannot_remove_internal_attack_using_locked_old_tap(shift, offset, region):
    state, tempo = project(shift, offset)
    before = deepcopy(generate(state, tempo, amount=0))
    with pytest.raises(ValueError, match='乐句.*范围'):
        generate(state, tempo, region=region)
    assert state['variants'] == [before]


def test_anchor_at_lower_bound_allows_complete_phrase_rewrite():
    state, tempo = project(22.5, 103)
    old = deepcopy(generate(state, tempo, amount=0))
    actual = generate(state, tempo, region=[978, 2500])
    assert [(n['id'], n['start'], n['end']) for n in actual['notes'] if n['voice']=='vocals'] == [('a', 978, 2103)]
    assert actual['selection']['generated'] == 1
    assert state['variants'][0] == old


def test_internal_attack_at_upper_bound_does_not_trigger_false_rejection():
    state, tempo = project(22.5, 103)
    before = generate(state, tempo, amount=0)
    actual = generate(state, tempo, region=[1100, 1478])
    assert actual['notes'] == before['notes']


def test_existing_phrase_also_requires_rewriting_its_replacement_scope():
    state, tempo = project()
    before = deepcopy(generate(state, tempo))
    with pytest.raises(ValueError, match='乐句.*范围'):
        generate(state, tempo, region=[1250, 2500])
    assert state['variants'] == [before]


def test_ordinary_sustain_without_covered_attacks_can_cross_region_boundary():
    state, tempo = project()
    state['analysis']['phrase_candidates'] = []
    state['analysis']['candidates'][0]['end'] = 2100
    before = deepcopy(generate(state, tempo))
    actual = generate(state, tempo, region=[1250, 2500])
    geometry = lambda v: [(n['id'], n['start'], n['end']) for n in v['notes']]
    assert geometry(actual) == geometry(before)
    assert actual['notes'][0] == before['notes'][0]


@pytest.mark.parametrize('region', [[1250, 2500], [1000, 1500]])
def test_api_rejection_preserves_saved_project_and_expansion_exports_complete_phrase(tmp_path, region):
    import io
    from manialab.domain import parse_osu
    from manialab.server import create_app
    from tests.test_server import wait_job

    root = Path(__file__).parents[1]
    app = create_app(root, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    try:
        state = client.post('/api/import', data={'files': (io.BytesIO(b'placeholder'), 'song.wav')}, headers=headers).json
        evidence, tempo = project()
        state['analysis'] = evidence['analysis']
        state['analysis']['tempo'] = dict(bpm=120, offset=0, division=4, stable=True)
        app.extensions['projects'].save(state)
        base = f"/api/projects/{state['id']}"
        payload = dict(mode='audio', timing_confirmed=True, settings=dict(phrase_ln=0), **tempo)
        assert wait_job(client, client.post(base+'/generate', json=payload, headers=headers).json)['status'] == 'done'
        before = client.get(base).json
        payload.update(settings=dict(phrase_ln=1), region=region)
        rejected = wait_job(client, client.post(base+'/generate', json=payload, headers=headers).json)
        assert rejected['status'] == 'error' and '乐句' in rejected['message']
        assert client.get(base).json == before
        payload['region'] = [1000,2500]
        assert wait_job(client, client.post(base+'/generate', json=payload, headers=headers).json)['status'] == 'done'
        after = client.get(base).json
        current = after['variants'][-1]
        assert [(n['id'],n['start'],n['end']) for n in current['notes'] if n['voice']=='vocals'] == [('a',1000,2125)]
        assert after['variants'][0] == before['variants'][0]
        assert current['source_evidence']['a']['expressed']['end'] == 2100
        exported = parse_osu(client.get(base+'/export/osu').data.decode())
        assert [(n.start,n.end) for n in exported['notes']] == [(1000,2125),(2750,2750)]
    finally:
        app.extensions['executor'].shutdown()

