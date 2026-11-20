import json

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
