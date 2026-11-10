from copy import deepcopy

import numpy as np
import pytest

from scripts.tsumugi_instrument_scores import (
    ambiguity_bins, describe_logits, note_decisions, rebuild_intervals, score_bin,
)


def construction(pitch=60, slot=0, begin=1, end=4):
    return dict(arguments=dict(pair_id=27*88+pitch-21, begin_frame=begin, end_frame=end,
                               valid_model_frames=345), v1_slot_index=slot, constructed_note={})


def test_saved_pitch_slots_do_not_collapse_same_pitch_intervals():
    rows = [construction(), construction(slot=2), construction(pitch=61)]
    before = deepcopy(rows)
    decoded, entries, keys = rebuild_intervals(rows, [0,1,2], slots=3, classes=36, valid_frames=345)
    assert entries == [(0,117,0,1,4), (0,119,0,1,4), (0,120,0,1,4)]
    assert keys == {(0,117,0):0, (0,119,0):1, (0,120,0):2}
    assert decoded[0][118] == [] and len(decoded[0]) == 264
    assert rows == before


@pytest.mark.parametrize('field,value', [('v1_slot_index', None), ('v1_slot_index', 3),
                                         ('constructed_note', None)])
def test_unknown_slot_and_rejected_intervals_are_not_guessed(field, value):
    row = construction()
    row[field] = value
    with pytest.raises(ValueError):
        rebuild_intervals([row], [0], slots=3, classes=36, valid_frames=345)


def test_duplicate_out_of_order_and_out_of_window_intervals_fail():
    rows = [construction(), construction(pitch=61)]
    for indices in ([0,0], [1,0]):
        with pytest.raises(ValueError):
            rebuild_intervals(rows, indices, slots=3, classes=36, valid_frames=345)
    rows[0]['arguments']['end_frame'] = 345
    with pytest.raises(ValueError):
        rebuild_intervals(rows, [0], slots=3, classes=36, valid_frames=345)


def test_repeated_intervals_preserve_distinct_indices_and_empty_windows():
    rows = [construction(), construction(begin=6, end=8)]
    _, entries, _ = rebuild_intervals(rows, [0,1], slots=3, classes=36, valid_frames=345)
    assert entries == [(0,117,0,1,4), (0,117,1,6,8)]
    decoded, entries, keys = rebuild_intervals([], [], slots=3, classes=36, valid_frames=345)
    assert decoded == [[[] for _ in range(264)]] and entries == [] and keys == {}


def test_extreme_logits_and_ties_do_not_fabricate_confidence():
    first = describe_logits([1000, 1000, -1000], preferred_class=1)
    assert first == dict(class_id=1, top_sigmoid=1., logit_margin=0., positive_classes=2, tied_top=2)
    assert describe_logits([-1000, -1001], preferred_class=1)['top_sigmoid'] == 0
    assert describe_logits([2,2,1], preferred_class=2)['class_id'] == 0
    with pytest.raises(ValueError):
        describe_logits([float('nan'), 1], preferred_class=0)


def test_lineage_averages_only_contributors_without_changing_notes():
    values = np.array([[4,0], [100,-100], [-1,6]], dtype=np.float32)
    ranks = np.array([[0,1], [0,1], [1,0]])
    lineage = [dict(sources=[0,2], note=dict(instrument_id=0, instrument_candidates=[0,1]))]
    before = deepcopy(lineage)
    decision = note_decisions(lineage, values, ranks)[0]
    assert decision['first']['class_id'] == 0 and decision['mean']['class_id'] == 1
    assert decision['mean']['logit_margin'] == 1.5
    assert decision['sources'] == [0,2] and not decision['sources_agree']
    assert lineage == before


def test_inconsistent_source_or_ranking_cannot_be_silently_replaced():
    lineage = [dict(sources=[0], note=dict(instrument_id=1, instrument_candidates=[1,0]))]
    with pytest.raises(ValueError):
        note_decisions(lineage, [[2,1]], [[0,1]])
    with pytest.raises(ValueError):
        note_decisions(lineage, [[2,1]], [[1,1]])
    lineage[0]['sources'] = [0,0]
    with pytest.raises(ValueError):
        note_decisions(lineage, [[1,2]], [[1,0]])


@pytest.mark.parametrize('value,expected', [(0,0), (.499,0), (.5,1), (.8,2), (.95,3), (1.,3)])
def test_score_bins_are_fixed_and_include_boundaries(value, expected):
    assert score_bin(value, (.5,.8,.95)) == expected


def test_ambiguity_report_counts_unmatched_drums_and_both_directions_of_unison():
    def note(time, pitch=60, drum=False):
        return dict(onset=time, pitch=pitch, is_drum=drum, program=40)
    refs = [note(1), note(2), note(2), note(3), note(4)]
    preds = [note(1), note(2), note(3), note(3), note(4), note(5), note(6, drum=True)]
    decisions = [dict(first=dict(class_id=(1 if i==4 else 0), top_sigmoid=.96, logit_margin=4.),
                      mean=dict(class_id=0, top_sigmoid=.6, logit_margin=1.)) for i in range(7)]
    result = ambiguity_bins(refs, preds, decisions, {40:0})
    assert result['first']['categories'] == dict(unique_correct_family=1, unique_wrong_family=1,
        ambiguous_correspondence=3, no_compatible_reference=1, original_drum_prediction=1)
    assert result['mean']['categories']['unique_correct_family'] == 2
    for mode in result.values():
        assert sum(mode['categories'].values()) == 7
        for bins in mode['bins'].values():
            assert sum(sum(b.values()) for b in bins) == 7
