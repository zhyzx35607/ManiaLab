"""Independent boundary checks for the frozen audio comparison's small helpers."""
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def helpers():
    # Avoid importing the inference runner and its optional torch stack.
    path = Path(__file__).resolve().parents[1] / 'scripts/probe_piano_audio_events.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    wanted = {'audio_ms', 'select_clips', 'describe_edges'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in wanted]
    assert {node.name for node in selected} == wanted
    namespace = {'defaultdict': defaultdict}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.fixture
def select(helpers):
    # Selection boundaries are tested in an exact millisecond coordinate system;
    # the actual diagnostic-clock conversion has its own separate check below.
    helpers['audio_ms'] = float
    return helpers['select_clips']


def item(chunk, begin, end, owns=True):
    return {'chunk': chunk, 'metadata': {'core_ticks': [begin, end]},
            'payload': {'node_output_owner': np.asarray([owns], dtype=bool)}}


def test_diagnostic_clock_retains_offset_and_advance_separately(helpers):
    clock = helpers['audio_ms']
    assert clock(0) == 200.5
    assert clock(480) == pytest.approx(60000 / 74 + 200.5)
    assert clock(7680) == pytest.approx(16 * 60000 / 74 + 200.5)


def test_targets_follow_pcm_duration_instead_of_available_window_indices(select):
    items = [item(0, 0, 200), item(5, 200, 400), item(30, 400, 600), item(90, 600, 800)]
    result = select(items, 800)
    assert [row['chunk'] for row in result] == [0, 5, 30, 90]
    assert [row['target_ms'] for row in result] == [100, 300, 500, 700]
    assert [row['fraction'] for row in result] == [.125, .375, .625, .875]


def test_context_only_window_cannot_be_selected(select):
    result = select([item(0, 0, 200, owns=False), item(1, 200, 400)], 400)
    assert [row['chunk'] for row in result] == [1]
    assert result[0]['target_ms'] == 50


def test_empty_core_chooses_nearest_owned_core(select):
    result = select([item(0, 0, 100), item(2, 200, 300)], 1440)
    assert result[0]['target_ms'] == 180
    assert result[0]['chunk'] == 2


def test_equal_distance_prefers_earlier_core_even_when_input_is_reversed(select):
    result = select([item(2, 200, 300), item(0, 0, 100)], 1200)
    assert result[0]['target_ms'] == 150
    assert result[0]['chunk'] == 0


def test_touching_core_boundary_belongs_to_right_half_open_interval(select):
    result = select([item(0, 0, 100), item(1, 100, 200)], 800)
    assert result[0]['target_ms'] == 100
    assert result[0]['chunk'] == 1


def test_duplicate_fraction_core_is_kept_once_without_filling_distant_cores(select):
    result = select([item(0, 0, 1000), item(9, 1200, 1400)], 1000)
    assert len(result) == 1
    assert result[0]['chunk'] == 0
    assert result[0]['fraction'] == .125


def test_selected_audio_interval_is_cropped_to_pcm_end(select):
    result = select([item(0, 0, 800)], 600)
    assert len(result) == 1
    assert result[0]['begin_ms'] == 0
    assert result[0]['end_ms'] == 600


def test_no_owned_attacks_produces_no_review_clips(select):
    assert select([], 1000) == []
    assert select([item(1, 200, 400, owns=False)], 1000) == []


def test_chord_notes_at_same_attack_do_not_create_multiple_attack_groups(helpers):
    rows = [{'id': 'a', 'tick': 0}, {'id': 'b', 'tick': 0},
            {'id': 'c', 'tick': 120}, {'id': 'd', 'tick': 120}]
    result = helpers['describe_edges'](rows, {('a', 'c'), ('a', 'd'), ('b', 'c')})
    assert result == {'edges': 3, 'sources_with_multiple_target_attack_groups': 0,
                      'targets_with_multiple_source_attack_groups': 0}


def test_distinct_attack_times_count_as_multiple_groups(helpers):
    rows = [{'id': 'a', 'tick': 0}, {'id': 'b', 'tick': 120},
            {'id': 'c', 'tick': 240}, {'id': 'd', 'tick': 360}]
    result = helpers['describe_edges'](rows, {('a', 'b'), ('a', 'c'), ('b', 'c'), ('c', 'd')})
    assert result == {'edges': 4, 'sources_with_multiple_target_attack_groups': 1,
                      'targets_with_multiple_source_attack_groups': 1}


def test_no_predictions_has_zero_structure_counts(helpers):
    assert helpers['describe_edges']([], set()) == {
        'edges': 0, 'sources_with_multiple_target_attack_groups': 0,
        'targets_with_multiple_source_attack_groups': 0}


@pytest.fixture
def edge_visibility():
    path = Path(__file__).resolve().parents[1] / 'scripts/render_piano_audio_events.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'edge_visibility']
    assert len(selected) == 1
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['edge_visibility']


@pytest.mark.parametrize(('first', 'last', 'expected'), [
    (1000, 1500, 'fully_visible'),
    (1800, 1200, 'fully_visible'),
    (999, 1500, 'boundary_clipped'),
    (1500, 2000, 'boundary_clipped'),
    (500, 2500, 'boundary_clipped'),
    (500, 999, 'outside_clip'),
    (2000, 2200, 'outside_clip'),
])
def test_edge_visibility_uses_raw_milliseconds_and_half_open_audio_clip(
        edge_visibility, first, last, expected):
    physical = {'a': {'start': first}, 'b': {'start': last}}
    result = edge_visibility({('a', 'b')}, physical, 1., 2.)
    assert sum(result.values()) == 1
    assert result[expected] == 1
