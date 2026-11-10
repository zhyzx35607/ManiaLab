from scripts.audit_piano_window_continuity import cardinality


def note(start, end=1., pitch=60):
    return dict(onset=start, offset=end, pitch=pitch)


def test_independent_score_handles_one_to_one_greedy_trap():
    reference = [note(.1), note(.14)]
    predicted = [note(.11), note(.08)]
    assert cardinality(reference, predicted, False, False) == 2
    assert cardinality(reference, predicted[:1], False, False) == 1


def test_independent_score_separates_pitch_onset_and_release_requirements():
    reference = [note(.1, 3.)]
    assert cardinality(reference, [note(.1, 3., 61)], False, False) == 0
    assert cardinality(reference, [note(.145, 3.)], False, False) == 1
    assert cardinality(reference, [note(.145, 3.)], True, False) == 0
    assert cardinality(reference, [note(.1, 3.1)], False, True) == 1
    assert cardinality(reference, [note(.1, 3.1)], True, True) == 0
    assert cardinality(reference, [note(.1, 7.)], False, False) == 1
    assert cardinality(reference, [note(.1, 7.)], False, True) == 0
