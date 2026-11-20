from copy import deepcopy
import io
import json
from pathlib import Path

from manialab.domain import Note, export_osu
from manialab.selection_report import resolve_generation_origin


def test_duplicate_created_parent_keys_resolve_to_unknown_independent_of_order():
    full_song = dict(created=1.0)
    regional = dict(created=1.0, region=[3000, 5000])
    child = dict(created=2.0, region=None, edits=[dict(parent_created=1.0)])

    assert resolve_generation_origin(child, [full_song, regional, child]) == dict(kind='unknown')
    assert resolve_generation_origin(child, [regional, full_song, child]) == dict(kind='unknown')


def test_unique_legacy_parent_still_resolves_its_origin():
    parent = dict(created=1.0, region=[3000, 5000])
    child = dict(created=2.0, region=None, edits=[dict(parent_created=1.0)])

    assert resolve_generation_origin(child, [parent, child]) == dict(
        kind='regional_generation', region=[3000, 5000]
    )


def test_cyclic_legacy_parent_history_resolves_to_unknown():
    first = dict(created=1.0, edits=[dict(parent_created=2.0)])
    second = dict(created=2.0, edits=[dict(parent_created=1.0)])

    assert resolve_generation_origin(first, [first, second]) == dict(kind='unknown')
    assert resolve_generation_origin(second, [first, second]) == dict(kind='unknown')


def test_unhashable_legacy_candidate_created_value_does_not_break_resolution():
    parent = dict(created=1.0, region=[3000, 5000])
    child = dict(created=2.0, region=None, edits=[dict(parent_created=1.0)])
    malformed = dict(created=[1.0])

    assert resolve_generation_origin(child, [parent, malformed, child]) == dict(
        kind='regional_generation', region=[3000, 5000]
    )


def test_non_finite_created_values_from_json_resolve_to_unknown():
    for value in (float('nan'), float('inf'), float('-inf')):
        malformed_parent = json.loads(json.dumps(dict(
            created=value, region=[3000, 5000]
        )))
        child = dict(created=2.0, region=None, edits=[dict(parent_created=1.0)])
        assert resolve_generation_origin(child, [malformed_parent, child]) == dict(kind='unknown')

        parent = dict(created=1.0, region=[3000, 5000])
        malformed_child = json.loads(json.dumps(dict(
            created=2.0, region=None, edits=[dict(parent_created=value)]
        )))
        assert resolve_generation_origin(malformed_child, [parent, malformed_child]) == dict(
            kind='unknown'
        )


def test_only_missing_null_or_empty_list_edit_history_defaults_to_full_song():
    for candidate in (
        dict(created=1.0),
        dict(created=1.0, edits=None),
        dict(created=1.0, edits=[]),
    ):
        assert resolve_generation_origin(candidate, [candidate]) == dict(
            kind='full_song_generation'
        )

    for candidate in (
        dict(created=1.0, edits={}),
        dict(created=1.0, edits=''),
        dict(created=1.0, edits=0),
    ):
        assert resolve_generation_origin(candidate, [candidate]) == dict(kind='unknown')


def test_persisted_unknown_is_terminal_after_direct_origin_evidence():
    parent = dict(created=1.0, region=[3000, 5000])
    candidate = dict(created=2.0, generation_origin=dict(kind='unknown'),
                     edits=[dict(parent_created=1.0)])

    assert resolve_generation_origin(candidate, [parent, candidate]) == dict(kind='unknown')
    candidate['region'] = [6000, 7000]
    assert resolve_generation_origin(candidate, [parent, candidate]) == dict(
        kind='regional_generation', region=[6000, 7000]
    )
    candidate['generation_origin'] = dict(kind='full_song_generation')
    assert resolve_generation_origin(candidate, [parent, candidate]) == dict(
        kind='full_song_generation'
    )


def test_ambiguous_origin_survives_edit_history_pruning_restart_and_json_export(tmp_path):
    from manialab.server import create_app
    from tests.test_server import wait_job

    root = Path(__file__).parents[1]
    chart = export_osu([Note(1000, 1000, 0)], 6, root / 'vendor/mapperatorinator/template.osu',
                       audio='song.wav')
    app = create_app(root, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    try:
        imported = client.post('/api/import', data={'files': [
            (io.BytesIO(chart.encode()), 'chart.osu'),
            (io.BytesIO(b'synthetic audio'), 'song.wav'),
        ]}, headers=headers)
        assert imported.status_code == 200, imported.json
        project_id = imported.json['id']
        base = f'/api/projects/{project_id}'
        projects = app.extensions['projects']

        generated = client.post(base + '/generate', json={
            'mode': 'remap', 'settings': {'keys': 6},
        }, headers=headers)
        assert generated.status_code == 200, generated.json
        assert wait_job(client, generated.json)['status'] == 'done'

        seed = projects.load(project_id)['variants'][0]
        full_parent = deepcopy(seed)
        full_parent.update(created=1.0, region=None,
                           generation_origin=dict(kind='full_song_generation'))
        regional_parent = deepcopy(seed)
        regional_parent.update(created=1.0, region=[3000, 5000],
                               generation_origin=dict(kind='regional_generation', region=[3000, 5000]))
        legacy_child = deepcopy(seed)
        legacy_child.update(created=2.0, region=None, edits=[dict(parent_created=1.0)])
        legacy_child.pop('generation_origin', None)

        state = projects.load(project_id)
        state['variants'] = [full_parent, regional_parent, legacy_child]
        state['active'] = 2
        projects.save(state)

        def edit_active():
            current = projects.load(project_id)
            variant = current['variants'][current['active']]
            note = variant['notes'][0]
            response = client.post(base + '/edit', json={
                'op': 'update', 'base_created': variant['created'], 'note_id': note['id'],
                'values': {'lane': (note['lane'] + 1) % 6},
            }, headers=headers)
            assert response.status_code == 200, response.json
            result = wait_job(client, response.json)
            assert result['status'] == 'done', result
            return projects.load(project_id)

        state = edit_active()
        assert state['variants'][-1]['generation_origin'] == dict(kind='unknown')
        for _ in range(5):
            state = edit_active()

        matching_parents = [item for item in state['variants'] if item.get('created') == 1.0]
        assert len(matching_parents) == 1
        assert matching_parents[0].get('region') == [3000, 5000]

        state = edit_active()
        assert state['variants'][-1]['generation_origin'] == dict(kind='unknown')
    finally:
        app.extensions['executor'].shutdown()

    reopened = create_app(root, tmp_path)
    try:
        reopened_client = reopened.test_client()
        exported = reopened_client.get(base + '/export/json')
        assert exported.status_code == 200
        assert exported.json['variant']['generation_origin'] == dict(kind='unknown')
    finally:
        reopened.extensions['executor'].shutdown()
