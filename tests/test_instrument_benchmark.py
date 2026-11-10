import pytest

from scripts.evaluate_instrument_benchmark import match,window_metrics


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


def test_wrong_long_tails_cannot_disappear_from_interval_denominators():
    truth=[note(onset=29.,offset=29.5),note(onset=29.8,offset=30.5)]
    predictions=[note(onset=29.,offset=100.),note(onset=29.8,offset=30.5)]
    values,scope=window_metrics(truth,predictions,begin=0,end=30,context_start=0,context_end=32.1)
    result=values['standard']['complete_intervals']
    assert (result['reference'],result['predicted'],result['matched'])==(2,2,1)
    assert result['f1']==.5
    assert scope['predicted_crosses_context_end']==1 and scope['reference_crosses_end']==1


def test_unobservable_reference_tail_requires_more_context():
    with pytest.raises(ValueError,match='reference tails exceed'):
        window_metrics([note(onset=29.,offset=33.)],[],begin=0,end=30,context_start=0,context_end=32.1)


def test_context_events_are_not_scored_but_exact_end_release_is_retained():
    rows=[note(onset=.9),note(onset=1.,offset=30.),note(onset=30.,offset=30.5)]
    values,scope=window_metrics(rows,rows,begin=1,end=30,context_start=0,context_end=32.1)
    assert values['standard']['complete_intervals']['matched']==1
    assert values['standard']['complete_intervals']['predicted']==1
    assert scope['reference_crosses_end']==1
