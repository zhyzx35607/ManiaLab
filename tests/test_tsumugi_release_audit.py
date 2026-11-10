"""Offline provenance controls; no model, Torch, audio, or vendor checkout needed."""
import copy
import json

import pytest

from scripts.audit_tsumugi_releases import (
    audit, classify_endpoints, loss_diagnostics, read_json, replay_lineage,
    sha, source_endpoint,
)


def note(start=100, end=200, *, onset=True, offset=True, slot=0, pitch=60, candidates=(2,)):
    return dict(instrument_id=0, pitch=pitch, start_sample=start, end_sample=end,
                velocity=100, slot_index=slot, has_onset=onset, has_offset=offset,
                instrument_candidates=list(candidates))


def construction(value, *, window=0, boundary=True, hop=10, total=1000, valid=None):
    start, end = value['start_sample']-window, value['end_sample']-window
    return dict(arguments=dict(pair_id=value['pitch']-21, begin_frame=start//hop, end_frame=end//hop,
        boundary_flag=([value['has_onset'],value['has_offset'],(start%hop)/hop,(end%hop)/hop]
                       if boundary else None), window_start_frame=window,
        valid_audio_frames=total-window if valid is None else valid),
        constructed_note=copy.deepcopy(value), v1_slot_index=value['slot_index'],
        hop_length=hop, total_audio_samples=total)


def identity(*, original_samples=1000, original_rate=1000, model_rate=1000):
    return dict(original_samples=original_samples,original_sample_rate=original_rate,
                resampled_sample_rate=model_rate)


def replay(values, *, rows=None, gap=10, onset=50):
    rows = rows if rows is not None else [construction(n) for n in values]
    tracks = {}
    for index,n in enumerate(values):
        tracks.setdefault((n['pitch'],n['slot_index']),[]).append(dict(source=index,note=n))
    result = replay_lineage(tracks,merge_gap_samples=gap,merge_onset_samples=onset)
    classify_endpoints(result[0],rows,identity(),result[2])
    return result,loss_diagnostics(result[0],result[1],rows)


def test_finalizer_closes_last_segment_before_input_end_without_prediction():
    (kept,dropped,forced),_ = replay([note(offset=False)])
    assert dropped == [] and forced == [0]
    assert kept[0]['note']['has_offset'] is True
    assert kept[0]['note']['end_sample'] == 200
    assert kept[0]['classification'] == 'forced_finalization_endpoint'
    assert not kept[0]['unmodified_boundary_head_endpoint']
    assert not kept[0]['application_ln_eligible']


def test_earlier_positive_offset_flag_does_not_validate_later_unknown_endpoint():
    values = [note(),note(120,300,offset=False),note(700,800)]
    (kept,_,_),_ = replay(values)
    first = kept[0]
    assert first['note']['has_offset'] is True
    assert first['endpoint_sources'] == [1]
    assert first['model_supported_endpoint_sources'] == []
    assert first['classification'] == 'unverified_stitched_endpoint'
    assert first['merge_extended_endpoint']


def test_continuation_with_predicted_offset_keeps_support_and_merge_history_separate():
    values = [note(offset=False),note(190,300,onset=False),note(700,800)]
    (kept,_,_),_ = replay(values)
    assert kept[0]['endpoint_sources'] == [1]
    assert kept[0]['model_supported_endpoint_sources'] == [1]
    assert kept[0]['classification'] == 'merge_extended_predicted_offset'
    assert not kept[0]['unmodified_boundary_head_endpoint']
    assert kept[0]['merges'][0]['kind'] == 'continuation'


def test_same_pitch_different_slots_remain_independent():
    (kept,_,_),diagnostics = replay([note(slot=0),note(120,300,slot=1)])
    assert len(kept) == 2
    assert diagnostics['merged_repeated_onset_events'] == []


def test_two_distinct_same_window_boundary_onsets_are_counted_when_merged():
    (kept,_,_),diagnostics = replay([note(100,130),note(140,170),note(700,800)])
    assert len(kept) == 2
    events = diagnostics['same_window_distinct_boundary_onset_merge_events']
    assert len(events) == 1
    assert events[0]['distinct_predicted_onset_source_pairs'] == [[0,1]]


def test_cross_window_duplicate_is_not_counted_as_same_window_retrigger():
    values = [note(100,200),note(120,220),note(700,800)]
    rows = [construction(values[0]),construction(values[1],window=100),construction(values[2])]
    _,diagnostics = replay(values,rows=rows)
    assert len(diagnostics['merged_repeated_onset_events']) == 1
    assert diagnostics['same_window_distinct_boundary_onset_merge_events'] == []


@pytest.mark.parametrize('change',['fallback','same_begin_frame','no_head_onset'])
def test_same_window_retrigger_requires_two_distinct_explicit_head_onsets(change):
    values = [note(100,130),note(140,170),note(700,800)]
    rows = [construction(n) for n in values]
    if change == 'fallback':
        rows[0]['arguments']['boundary_flag'] = None
    elif change == 'same_begin_frame':
        rows[1]['arguments']['begin_frame'] = rows[0]['arguments']['begin_frame']
    else:
        rows[0]['arguments']['boundary_flag'][0] = False
    _,diagnostics = replay(values,rows=rows)
    assert diagnostics['same_window_distinct_boundary_onset_merge_events'] == []


def test_finalizer_filters_early_unclosed_onset_and_counts_candidate():
    (kept,dropped,forced),diagnostics = replay([note(offset=False),note(700,800)])
    assert len(kept) == 1 and len(dropped) == 1 and forced == []
    assert diagnostics['filtered_onset_source_indices'] == [0]
    assert diagnostics['filtered_early_unclosed_onset_source_indices'] == [0]


def test_offset_overwrite_can_filter_previously_closed_onset_but_not_count_as_unclosed_input():
    values = [note(),note(190,300,onset=False,offset=False),note(700,800)]
    (kept,dropped,_),diagnostics = replay(values)
    assert len(dropped) == 1 and len(kept) == 1
    assert diagnostics['filtered_onset_source_indices'] == [0]
    assert diagnostics['filtered_early_unclosed_onset_source_indices'] == []


def test_equal_endpoint_coordinates_do_not_erase_mixed_evidence():
    values = [note(),note(120,200,offset=False),note(700,800)]
    (kept,_,_),_ = replay(values)
    assert kept[0]['endpoint_sources'] == [0,1]
    assert kept[0]['model_supported_endpoint_sources'] == [0]
    assert kept[0]['classification'] == 'mixed_endpoint_evidence'


def test_same_time_other_track_positive_endpoint_cannot_support_this_track():
    (kept,_,_),_ = replay([note(offset=False),note(pitch=61)])
    assert kept[0]['classification'] == 'forced_finalization_endpoint'
    assert kept[0]['model_supported_endpoint_sources'] == []
    assert kept[1]['classification'] == 'boundary_head_predicted_offset'


def test_output_order_is_before_v1_restores_instrument_and_input_is_unchanged():
    values = [note(pitch=60,candidates=(8,)),note(pitch=61,candidates=(1,))]
    original = copy.deepcopy(values)
    (kept,_,_),_ = replay(values)
    assert [n['note']['instrument_id'] for n in kept] == [8,1]
    assert values == original


def test_gap_merge_overwrites_offset_and_preserves_candidate_order():
    values = [note(candidates=(2,3)),note(190,300,onset=False,offset=False,candidates=(3,1)),note(700,800)]
    (_,dropped,_),_ = replay(values)
    assert dropped[0]['note']['instrument_candidates'] == [2,3,1]
    assert dropped[0]['note']['has_offset'] is False


def test_boundary_head_support_never_grants_ln_eligibility():
    (kept,_,_),_ = replay([note()])
    assert kept[0]['unmodified_boundary_head_endpoint'] is True
    assert kept[0]['application_ln_eligible'] is False


@pytest.mark.parametrize('kind,reason',[
    ('fallback','no_boundary_head_fallback'),
    ('absent_offset','boundary_head_has_no_offset'),
    ('clipped','constructor_adjusted_endpoint'),
    ('window_end','at_window_observation_boundary'),
    ('input_end','at_original_input_boundary'),
])
def test_endpoint_limitations_are_not_accepted_as_predicted_release(kind,reason):
    row = construction(note())
    if kind == 'fallback':
        row['arguments']['boundary_flag'] = None
    elif kind == 'absent_offset':
        row['arguments']['boundary_flag'][1] = False
    elif kind == 'clipped':
        row['arguments']['end_frame'] = 25
    elif kind == 'window_end':
        row['arguments']['valid_audio_frames'] = 200
    else:
        row = construction(note(100,1000))
    result = source_endpoint(row,identity())
    assert reason in result['unsupported_reasons']
    assert not result['model_predicted_offset_support']


def test_original_duration_excludes_ceil_resampling_tail_from_observed_release():
    row = construction(note(1,5),hop=1,total=5)
    result = source_endpoint(row,identity(original_samples=3,original_rate=2,model_rate=3))
    assert result['at_original_input_boundary']
    row = construction(note(1,4),hop=1,total=5)
    assert source_endpoint(row,identity(original_samples=3,original_rate=2,model_rate=3))['model_predicted_offset_support']


@pytest.mark.parametrize('text',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'])
def test_evidence_json_rejects_ambiguous_or_nonfinite_data(tmp_path,text):
    path = tmp_path/'bad.json'
    path.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError):
        read_json(path)


@pytest.mark.parametrize('changed',['notes.json','raw-notes.json','window-notes.json','interval-constructions.json'])
def test_hash_mismatch_fails_before_loading_official_source(tmp_path,changed):
    evidence = [('notes.json','notes_sha256'),('raw-notes.json','raw_notes_sha256'),
                ('window-notes.json','window_notes_sha256'),('interval-constructions.json','interval_constructions_sha256')]
    report = {'identity':{}}
    for filename,key in evidence:
        (tmp_path/filename).write_text('[]',encoding='utf-8')
        report[key] = sha(tmp_path/filename)
    (tmp_path/'identity.json').write_text('{}',encoding='utf-8')
    (tmp_path/'report.json').write_text(json.dumps(report),encoding='utf-8')
    (tmp_path/changed).write_text('[1]',encoding='utf-8')
    with pytest.raises(ValueError,match='Evidence hash mismatch'):
        audit(tmp_path,tmp_path/'does-not-exist')
