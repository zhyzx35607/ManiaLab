"""Protect event identities, tails, and matching ambiguity in joint selection."""
import random

import pytest

from scripts.evaluate_piano_acoustics import metrics
from scripts.piano_joint_selection import matching, compare_recording, eligible, select_epoch


def note(onset, offset, pitch=60):
    return dict(onset=onset, offset=offset, pitch=pitch)


def decision(reference, before, after):
    def scored(notes):
        return [dict(id='a', metrics={kind: {label: metrics(reference, notes, strict=strict)
                      for label, strict in [('50ms', False), ('40ms', True)]}
                      for kind in ('key', 'sustain')})]
    return eligible(scored(before), scored(after),
                    {'a': compare_recording(reference, reference, before, after)})


def exhaustive(reference, predicted, strict, offsets):
    best, solutions = -1, []

    def visit(i, used, edges):
        nonlocal best, solutions
        if i == len(reference):
            if len(edges) > best:
                best, solutions = len(edges), [edges]
            elif len(edges) == best:
                solutions.append(edges)
            return
        visit(i+1, used, edges)
        ref = reference[i]
        tail = .08 if strict else max(.05, .2*(ref['offset']-ref['onset']))
        for j, pred in enumerate(predicted):
            if (j not in used and ref['pitch'] == pred['pitch']
                    and abs(ref['onset']-pred['onset']) <= (.04 if strict else .05)
                    and (not offsets or abs(ref['offset']-pred['offset']) <= tail)):
                visit(i+1, used | {j}, edges | {(i, j)})

    visit(0, set(), set())
    sets = [{i for i, _ in solution} for solution in solutions]
    return best, set.union(*sets), set.intersection(*sets), set.intersection(*solutions)


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('offsets', [False, True])
def test_coverage_and_fixed_edges_match_exhaustive_assignments(strict, offsets):
    rng = random.Random(2701)
    for _ in range(70):
        samples = lambda: [note(rng.randrange(8)/64, 1+rng.randrange(4)/8, rng.randrange(60, 62))
                           for _ in range(rng.randrange(6))]
        reference, predicted = samples(), samples()
        size, possible, necessary, edges = exhaustive(reference, predicted, strict, offsets)
        result = matching(reference, predicted, strict=strict, offsets=offsets, fixed_pairs=True)
        assert len(result['selected']) == size
        assert set(result['possible']) == possible
        assert set(result['necessary']) == necessary
        assert {tuple(pair) for pair in result['fixed_pairs']} == edges


def test_interval_swap_cannot_hide_in_equal_total():
    reference = [note(1, 2), note(3, 4)]
    result = decision(reference, [note(1, 2), note(3, 3.2)], [note(1, 1.2), note(3, 4)])
    assert not result['eligible']
    assert not result['checks']['50ms']['baseline_sustain_intervals_guaranteed']


def test_removing_extra_does_not_excuse_known_tail_regression():
    reference = [note(51.68359375, 52.60546875)]
    result = decision(reference, [note(51.68, 52.59), note(52.597, 53.44)], [note(51.68, 53.44)])
    assert result['strict_improvement']
    assert not result['eligible']
    assert not result['checks']['50ms']['no_tail_error_increase_over_80ms']


def test_tail_regression_is_rejected_even_inside_standard_interval_tolerance():
    reference = [note(1, 5)]
    result = decision(reference, [note(1, 5), note(7, 8)], [note(1, 5.125)])
    assert result['checks']['50ms']['baseline_sustain_intervals_guaranteed']
    assert not result['checks']['50ms']['no_tail_error_increase_over_80ms']


def test_ambiguous_coverage_does_not_count_as_preserved_identity():
    reference = [note(0, 1), note(.03125, 1.2)]
    result = decision(reference, [note(0, 1), note(3, 4)], [note(.015625, 1)])
    assert not result['checks']['50ms']['baseline_onsets_guaranteed']


def test_ambiguous_pair_tails_are_recorded_as_incomparable():
    reference = [note(0, 1), note(.03125, 2)]
    predicted = [note(.015625, 1), note(.015625, 2)]
    comparison = compare_recording(reference, reference, predicted, predicted)['50ms']
    assert comparison['onset_after']['necessary'] == [0, 1]
    assert comparison['tail_changes'] == []
    assert comparison['incomparable_onset_references'] == [0, 1]


def test_clean_recovery_passes_and_identical_candidate_does_not():
    reference = [note(1, 2), note(3, 4)]
    assert decision(reference, reference[:1], reference)['eligible']
    assert not decision(reference, reference, reference)['eligible']


def test_recording_roster_cannot_be_silently_shortened():
    with pytest.raises(ValueError, match='same unique'):
        eligible([{'id': 'a'}], [], {})


def test_selection_retains_baseline_if_no_candidate_passes():
    reference = [note(1, 2)]
    result = decision(reference, reference, reference)
    history = [dict(epoch=e, selection=result) for e in range(3)]
    assert select_epoch(history) == 0
    with pytest.raises(ValueError, match='all three'):
        select_epoch(history[:2])
