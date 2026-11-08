from dataclasses import replace
import numpy as np

from manialab.acoustic_anchors import AcousticAnchorContext, ACOUSTIC_FEATURES
from manialab.domain import Note


def evidence(forced=False):
    value = dict(notes=[dict(onset=.0225, offset=1.5225, pitch=60, program=40)],
                 forced=[forced], energy=np.ones(400)*.2, hop_ms=10.)
    return {'piano': value, 'other': value}


def context(notes, observed=None):
    return AcousticAnchorContext(notes, 10, ['0,500,4,2,1,60,1,0'], .35, observed or evidence())


def test_acoustic_features_never_read_input_chart_tails():
    notes = [Note(t, t, 0) for t in (0, 500, 1000, 1500, 2000)]
    a = context(notes); b = context([replace(n, end=n.start+1234, lane=7) for n in notes])
    np.testing.assert_array_equal(a.features(0), b.features(0))
    assert a.features(0).shape[1] == len(ACOUSTIC_FEATURES)


def test_forced_offset_is_distinguished_from_explicit_observation():
    notes = [Note(t, t, 0) for t in (0, 500, 1000, 1500)]
    explicit = context(notes).features(0)
    forced = context(notes, evidence(True)).features(0)
    at = ACOUSTIC_FEATURES.index('other_explicit_near_tail')
    forced_at = ACOUSTIC_FEATURES.index('other_forced_near_tail')
    assert explicit[3, at] == 1 and forced[3, at] == 0
    assert forced[3, forced_at] == 1


def test_no_observed_source_head_produces_finite_masked_features():
    a = context([Note(t, t, 0) for t in (2000, 2500)])
    x = a.features(0)
    assert np.isfinite(x).all()
    assert (x[:, ACOUSTIC_FEATURES.index('other_present')] == 0).all()


def test_acoustic_evidence_does_not_create_or_replace_release_candidates():
    a = context([Note(t, t, 0) for t in (0, 500, 1000, 2000)])
    assert 1500 not in a.times[a.options(0)[1:-1]]
    assert a.options(0).tolist() == [0, 1, 2, 3, -1]
