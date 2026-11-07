"""Metric regressions prevent flattering scores from duplicate or reused spans."""
from manialab.domain import Note
from scripts.evaluate_ln import available_support, compare


def test_ln_evaluation_deduplicates_chords_and_matches_once():
    predicted=[Note(100,1000,lane=0),Note(100,1000,lane=1),Note(110,1010,lane=2)]
    reference=[Note(100,1000)]
    result=compare(predicted,reference)
    assert result['endpoints']==dict(predicted=2,reference=1,matched=1,precision=.5,recall=1.,f1=.6667)


def test_ln_evaluation_uses_maximum_matching_instead_of_greedy():
    # First prediction can use either reference; the second can only use the
    # first. A nearest/first greedy match would incorrectly report one match.
    result=compare([Note(100,1000),Note(80,1000)],[Note(90,1000),Note(120,1000)])
    assert result['endpoints']['matched']==2


def test_ln_evaluation_separates_endpoint_accuracy_from_duration_overlap():
    result=compare([Note(100,900)],[Note(100,1200)])
    assert result['endpoints']['matched']==0
    assert result['overlap']['matched']==1
    empty=compare([],[])
    assert empty['endpoints']['f1'] is None
    assert compare([], [Note(100,1200)])['endpoints']['f1']==0


def test_support_bound_distinguishes_missing_release_options_from_short_evidence():
    candidates=[Note(100,2000),Note(4000,4200),Note(6000,6500)]
    human=[Note(100,1000),Note(4000,5000),Note(6000,6500),Note(8000,9000)]
    result=available_support(candidates,human)
    assert result==dict(reference_intervals=4,no_attack=1,insufficient_duration=1,
                        sufficient_duration=2,existing_endpoint_nearby=1)
