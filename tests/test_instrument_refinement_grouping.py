from scripts.audit_instrument_refinement import part_grouping


def note(time, part=1):
    return dict(onset=time, offset=time+.5, pitch=60, is_drum=False, part=part)


def test_merged_players_and_fragmented_single_player_are_reported_separately():
    notes = [note(1), note(2), note(3, 2), note(4, 2)]
    result = part_grouping(notes, notes, {'merged':[0,0,0,0], 'correct':[0,0,1,1], 'split':[0,1,2,3]})
    assert result['unambiguous_notes'] == 4
    assert result['grouping_metrics']['merged']['precision'] == 2/6
    assert result['grouping_metrics']['merged']['recall'] == 1
    assert result['grouping_metrics']['correct']['f1'] == 1
    assert result['grouping_metrics']['split']['precision'] is None
    assert result['grouping_metrics']['split']['recall'] == 0


def test_unisons_are_not_resolved_using_part_or_predicted_group_labels():
    ref = [note(1,1),note(1,2),note(2,1)]
    pred = [note(1),note(1),note(2),note(3)]
    result = part_grouping(ref,pred,{'classes':[1,2,1,2]})
    assert result['matched_pairs'] == [(2,2)]
    assert result['unambiguous_notes'] == 1
    assert result['ambiguous_reference'] == 2
    assert result['no_compatible_reference'] == 1
    assert result['grouping_metrics']['classes']['f1'] is None


def test_duplicate_detections_and_absent_notes_reduce_explicit_coverage():
    ref = [note(1),note(2)]
    pred = [note(1),note(1)]
    result = part_grouping(ref,pred,{'classes':[0,0]})
    assert result['unambiguous_notes'] == 0
    assert result['no_compatible_prediction'] == 1
    assert result['ambiguous_reference'] == 1
    assert result['matched_prediction_fraction'] == 0
