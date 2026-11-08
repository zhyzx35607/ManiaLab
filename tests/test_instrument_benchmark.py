from scripts.evaluate_instrument_benchmark import match


def note(onset=1.,offset=3.,pitch=60,program=40,is_drum=False):
    return dict(onset=onset,offset=offset,pitch=pitch,program=program,is_drum=is_drum)


def test_wrong_instrument_and_percussion_cannot_hide_as_correct_transcription():
    truth=[note()];prediction=[note(program=73),note(is_drum=True)]
    assert match(truth,prediction)['matched']==1
    assert match(truth,prediction)['precision']==.5
    assert match(truth,prediction,instrument=True)['matched']==0


def test_duration_relative_offset_tolerance_is_distinct_from_strict_clock():
    truth=[note()];prediction=[note(offset=3.3)]
    assert match(truth,prediction,offset=True)['matched']==1
    assert match(truth,prediction,offset=True,strict=True)['matched']==0


def test_duplicate_predictions_are_penalized_and_matching_maximizes_coverage():
    truth=[note(onset=1.),note(onset=1.08)]
    prediction=[note(onset=1.04),note(onset=.98),note(onset=1.04)]
    result=match(truth,prediction)
    assert result['matched']==2 and result['precision']==2/3 and result['recall']==1
    assert match(truth,[])['f1']==0
