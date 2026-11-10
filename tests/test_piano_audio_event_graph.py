"""Unlabelled audio windows preserve identity and never manufacture truth."""
import copy

import numpy as np
import pytest

from scripts.piano_audio_event_graph import (BAR_TICKS, build_audio_event_items,
                                            load_frozen_sakura_rows)
from scripts.piano_event_training import DTYPE


def row(identifier, tick, duration=120, pitch=60):
    return dict(id=identifier, tick=tick, duration=duration, pitch=pitch)


def owned_ids(items):
    return [str(identifier) for item in items for identifier, own in
            zip(item['payload']['event_ids'], item['payload']['node_output_owner']) if own]


def owned_edges(items, kind):
    result = []
    for item in items:
        p = item['payload']
        key = 'potential_edges' if kind == 'voice' else 'chord_edges'
        result.extend(tuple(str(p['event_ids'][i]) for i in pair) for pair in
                      p[key][:, p[kind+'_output_owner']].T)
    return result


def test_rows_and_audio_ids_remain_unchanged_with_zero_supervision():
    rows = [row('piano:b', 240, pitch=64), row('piano:a', 120), row('piano:c', 240)]
    before = copy.deepcopy(rows)
    items = build_audio_event_items(rows, piece_id='frozen-audio')
    assert rows == before and len(items) == 1
    p = items[0]['payload']
    assert p['notes'].dtype == np.dtype(DTYPE)
    assert list(p['event_ids']) == ['piano:a', 'piano:c', 'piano:b']
    assert list(p['source_row_indices']) == [1, 2, 0]
    for index, source_index in enumerate(p['source_row_indices']):
        source = rows[source_index]
        assert p['notes']['onset_div'][index] == source['tick']
        assert p['notes']['duration_div'][index] == source['duration']
        assert p['notes']['pitch'][index] == source['pitch']
    for kind in ('voice', 'chord', 'staff'):
        assert not p[kind+'_mask'].any() and not p[kind].any()
    assert set(p['notes']['staff']) == set(p['notes']['voice']) == {1}
    assert not items[0]['metadata']['targets_available']


def test_carry_retains_complete_interval_and_ownership_follows_destination_attack():
    rows = [row('carry', 0, duration=8000), row('b', 7800, pitch=62), row('c', 7800, pitch=65)]
    items = build_audio_event_items(rows, piece_id='carry')
    assert len(items) == 2
    assert sorted(owned_ids(items)) == ['b', 'c', 'carry']
    assert owned_edges(items, 'voice') == [('carry', 'b'), ('carry', 'c')]
    assert sorted(owned_edges(items, 'chord')) == [('b', 'c'), ('c', 'b')]
    first, second = (item['payload'] for item in items)
    assert not first['voice_output_owner'].any() and second['voice_output_owner'].all()
    assert not first['chord_output_owner'].any() and second['chord_output_owner'].all()
    assert second['carry_in'][0] and not second['node_output_owner'][0]
    assert second['notes']['onset_div'][0] == 0 and second['notes']['duration_div'][0] == 8000
    assert not second['voice_same_bar'].any()
    assert not second['voice_mask'].any()


def test_empty_bars_do_not_change_fixed_grid_or_chunk_identity():
    rows = [row('early', 0), row('late', 12*BAR_TICKS)]
    items = build_audio_event_items(rows, piece_id='rest')
    assert [item['chunk'] for item in items] == [0, 2, 3]
    assert items[1]['metadata']['core_ticks'] == [8*BAR_TICKS, 12*BAR_TICKS]
    assert not items[1]['payload']['node_output_owner'].any()
    assert owned_ids(items) == ['early', 'late']


def test_intervals_touching_context_boundary_are_not_carry_nodes():
    left = 3*BAR_TICKS
    rows = [row('ends_at_left', 0, duration=left), row('crosses_left', 120, duration=left),
            row('new', 4*BAR_TICKS)]
    second = build_audio_event_items(rows, piece_id='boundary')[1]
    assert second['metadata']['context_ticks'][0] == left
    assert 'ends_at_left' not in second['payload']['event_ids']
    assert 'crosses_left' in second['payload']['event_ids']


def test_graph_is_induced_from_one_complete_observed_graph(monkeypatch):
    import scripts.piano_audio_event_graph as module
    original = module._geometry
    calls = []
    def recorded(events, selected):
        calls.append((len(events), tuple(selected)))
        return original(events, selected)
    monkeypatch.setattr(module, '_geometry', recorded)
    rows = [row('a', 0, duration=4*BAR_TICKS), row('b', 3*BAR_TICKS),
            row('c', 4*BAR_TICKS), row('d', 9*BAR_TICKS)]
    items = build_audio_event_items(rows, piece_id='induced')
    assert calls == [(4, (0, 1, 2, 3))]
    # a contains b, ends exactly at c, while b rests until c; d has no future.
    global_edges = {('a', 'b', 2), ('a', 'c', 1), ('b', 'c', 3), ('c', 'd', 3)}
    for item in items:
        p = item['payload']; ids = list(map(str, p['event_ids']))
        actual = {(ids[a], ids[b], int(kind)) for a, b, kind in p['observed_edges'].T}
        assert actual == {edge for edge in global_edges if edge[0] in ids and edge[1] in ids}


def test_parallel_unisons_are_distinct_and_keep_bidirectional_chord_ownership():
    rows = [row('a', 0), row('b', 0), row('c', 120)]
    items = build_audio_event_items(rows, piece_id='unisons')
    assert owned_ids(items) == ['a', 'b', 'c']
    assert sorted(owned_edges(items, 'chord')) == [('a', 'b'), ('b', 'a')]
    assert sorted(owned_edges(items, 'voice')) == [('a', 'c'), ('b', 'c')]
    assert items[0]['payload']['voice_same_bar'].all()


def test_terminal_partial_bar_is_explicit_without_shortening_last_audio_interval():
    rows = [row('a', 0), row('tail', 4*BAR_TICKS, duration=BAR_TICKS//2)]
    last = build_audio_event_items(rows, piece_id='last')[-1]
    assert last['metadata']['core_ticks'] == [4*BAR_TICKS, 5*BAR_TICKS]
    assert last['metadata']['core_bars'] == [4]
    assert last['metadata']['bar_index_base'] == 0
    assert last['payload']['notes']['duration_div'][-1] == BAR_TICKS//2


def test_core_boundary_attack_belongs_only_to_right_core():
    rows = [row('before', 4*BAR_TICKS-1), row('boundary', 4*BAR_TICKS)]
    items = build_audio_event_items(rows, piece_id='exact')
    assert owned_ids(items) == ['before', 'boundary']
    assert list(items[0]['payload']['node_output_owner']) == [True, False]
    assert list(items[1]['payload']['node_output_owner']) == [False, True]


def test_int64_overflow_is_rejected_instead_of_wrapping():
    with pytest.raises(ValueError, match='int64'):
        build_audio_event_items([row('overflow', np.iinfo(np.int64).max, duration=1)], piece_id='bad')


@pytest.mark.parametrize('field,value', [('tick', -1), ('tick', 1.5), ('duration', 0),
    ('duration', True), ('pitch', 120), ('pitch', False), ('id', ''), ('id', ' ')])
def test_invalid_audio_geometry_or_identity_is_rejected(field, value):
    data = row('a', 0); data[field] = value
    with pytest.raises(ValueError):
        build_audio_event_items([data], piece_id='invalid')


@pytest.mark.parametrize('extra', ['part', 'voice', 'staff', 'written', 'supervision_eligible'])
def test_adapter_rejects_fabricated_notation_fields(extra):
    data = row('a', 0); data[extra] = 'invented'
    with pytest.raises(ValueError, match='no notation labels'):
        build_audio_event_items([data], piece_id='invalid')


def test_duplicate_audio_ids_are_not_merged():
    with pytest.raises(ValueError, match='unique'):
        build_audio_event_items([row('same', 0), row('same', 120)], piece_id='invalid')


def test_empty_input_produces_no_fabricated_windows():
    assert build_audio_event_items([], piece_id='empty') == []


def test_frozen_loader_rejects_modified_rows_before_parsing(tmp_path):
    path = tmp_path/'changed.json'; path.write_text('[]')
    with pytest.raises(ValueError, match='Frozen Sakura graph rows changed'):
        load_frozen_sakura_rows(path)
