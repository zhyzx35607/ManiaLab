from copy import deepcopy

from scripts.evaluate_piano_acoustics import metrics
from scripts.train_piano_onset_head import eligible


def note(start, end, pitch=60):
    return dict(onset=start, offset=end, pitch=pitch)


def scored(name, reference, predicted):
    return dict(id=name, metrics={semantics:{label:metrics(reference,predicted,strict=strict)
        for label,strict in [('50ms',False),('40ms',True)]} for semantics in ('key','sustain')})


def test_baseline_is_always_eligible():
    reference=[note(1.,1.2),note(1.25,2.),note(3.,4.)]
    baseline=[scored('a',reference,reference[:2])]
    assert eligible(baseline,baseline)['eligible']


def test_recovering_missing_note_without_tradeoff_is_eligible():
    reference=[note(1.,1.2),note(1.25,2.),note(3.,4.)]
    baseline=[scored('a',reference,reference[:2])]
    current=[scored('a',reference,reference)]
    assert eligible(baseline,current)['eligible']


def test_better_onset_recall_does_not_excuse_extra_prediction():
    reference=[note(1.,2.),note(3.,4.),note(5.,6.)]
    baseline=[scored('a',reference,reference[:1])]
    current=[scored('a',reference,reference+[note(7.,8.)])]
    result=eligible(baseline,current)
    assert not result['eligible']
    assert not result['checks']['extra_not_increased']


def test_improvement_on_one_recording_does_not_hide_another_regression():
    reference=[note(1.,2.),note(3.,4.),note(5.,6.)]
    baseline=[scored('a',reference,reference[:1]),scored('b',reference,reference)]
    current=[scored('a',reference,reference),scored('b',reference,reference[:2])]
    result=eligible(baseline,current)
    assert result['pooled_onsets']['recall'] > eligible(baseline,baseline)['pooled_onsets']['recall']
    assert not result['checks']['all_recording_f1_not_lower']
    assert not result['eligible']


def test_equal_onset_counts_do_not_hide_reattack_loss():
    reference=[note(1.,1.2),note(1.25,2.),note(3.,4.)]
    baseline=[scored('a',reference,reference[:2])]
    current=[scored('a',reference,[reference[0],reference[2]])]
    result=eligible(baseline,current)
    assert result['checks']['all_recording_f1_not_lower']
    assert not result['checks']['all_reattack_coverage_not_lower']
    assert not result['eligible']


def test_unchanged_onsets_do_not_hide_sustain_endpoint_regression():
    reference=[note(1.,2.),note(3.,4.)]
    baseline=[scored('a',reference,reference)]
    current=deepcopy(baseline)
    for label,strict in [('50ms',False),('40ms',True)]:
        current[0]['metrics']['sustain'][label]=metrics(reference,[note(1.,1.1),note(3.,3.1)],strict=strict)
    result=eligible(baseline,current)
    assert result['checks']['all_recording_f1_not_lower']
    assert not result['checks']['sustain_interval_f1_not_lower']
    assert not result['eligible']
