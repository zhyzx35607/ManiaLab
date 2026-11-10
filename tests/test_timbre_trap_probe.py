import numpy as np

from scripts.research_timbre_trap import match_pitches, peak_mask, score_frames


def test_frequency_peaks_preserve_distinct_chord_pitches_and_reject_plateaus():
    a = np.asarray([[.6, .8], [.1, .8], [.7, .1], [.1, .5]])
    assert peak_mask(a).tolist() == [[True, False], [False, False], [True, False], [False, True]]


def test_duplicate_predictions_cannot_double_match_a_unison_pitch():
    assert match_pitches([60, 60, 64], [59.7, 60.2, 64.4]) == (2, 3, 2)
    assert match_pitches([60, 61], [60.5, 61.4]) == (2, 2, 2)


def test_padding_and_scoring_endpoint_do_not_inflate_activity_scores():
    reference = [dict(onset=0., offset=.2, pitch=60), dict(onset=0., offset=.1, pitch=64)]
    counts = score_frames(reference, [0., .1, .2, .3, 30.],
        np.asarray([True, True, True, False, True]), [[60, 64], [60], [60], [60], [60]])
    assert counts == dict(frames=3, reference=3, predicted=4, matched=3,
        rest_frames=1, rest_false_active=1, polyphonic_frames=1)
