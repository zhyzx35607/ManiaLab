"""Keep removed predictions and release regressions visible in review clips."""
from scripts.review_piano_head_fit import choose_examples


def note(onset, offset, pitch=60):
    return dict(onset=onset, offset=offset, pitch=pitch)


def test_removed_extra_is_shown_even_when_no_reference_onset_changes():
    reference = [note(0., .2)]
    before = reference + [note(1., 1.3, 62)]
    after = [note(.001, .2)]
    assert ('removed-unmatched-prediction', 1., 62, None) in choose_examples(reference, before, after)


def test_small_extra_timing_shift_is_not_presented_as_a_removed_prediction():
    reference = [note(0., .2)]
    before = reference + [note(1., 1.3, 62)]
    after = reference + [note(1.001, 1.3, 62)]
    assert all(row[0] != 'removed-unmatched-prediction' for row in choose_examples(reference, before, after))


def test_sustain_loss_anchors_the_reference_tail_not_the_attack():
    reference = [note(1., 1.1)]
    sustain = [note(1., 1.9)]
    before, after = [note(1., 1.91)], [note(1., 2.6)]
    assert ('lost-sustain-interval', 1.9, 60, None) in choose_examples(reference, before, after, sustain)
