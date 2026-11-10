from scripts.evaluate_piano_acoustics import metrics


def test_interval_metric_counts_duplicate_and_wrong_long_release():
    reference=[dict(onset=1.,offset=3.,pitch=60),dict(onset=4.,offset=4.5,pitch=64)]
    predicted=[dict(onset=1.01,offset=3.02,pitch=60),dict(onset=1.02,offset=3.03,pitch=60),
               dict(onset=4.01,offset=80.,pitch=64)]
    report=metrics(reference,predicted,strict=True)
    assert report['onsets']['matched']==2
    assert report['intervals']['matched']==1
    assert report['intervals']['predicted']==3
    assert report['long_reference']==1 and report['long_matched']==1


def test_joint_matching_can_choose_another_valid_release_without_reusing_prediction():
    reference=[dict(onset=1.,offset=2.,pitch=60),dict(onset=1.03,offset=3.,pitch=60)]
    predicted=[dict(onset=1.005,offset=3.01,pitch=60),dict(onset=1.025,offset=2.01,pitch=60)]
    report=metrics(reference,predicted,strict=True)
    assert report['intervals']['matched']==2


def test_repeated_attack_covered_by_a_single_sustain_is_not_two_hits():
    reference=[dict(onset=1.,offset=2.,pitch=60),dict(onset=2.03,offset=3.,pitch=60)]
    predicted=[dict(onset=1.,offset=3.,pitch=60)]
    report=metrics(reference,predicted,strict=True)
    assert report['onsets']['matched']==1
    assert report['close_repeats']==dict(reference=1,both_onsets_matched=0,bridged_missing_reattack=1)
