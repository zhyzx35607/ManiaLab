"""Check ambiguity claims against exhaustive small onset assignments."""
import random

from scripts.piano_match_coverage import coverage, changes, summarize_changes


def notes(times, pitch=60):
    return [dict(onset=t, offset=t+.1, pitch=pitch) for t in times]


def exhaustive(reference, predicted, tolerance):
    best, masks = -1, []

    def visit(i, used, selected):
        nonlocal best, masks
        if i == len(reference):
            if len(selected) > best:
                best, masks = len(selected), [selected]
            elif len(selected) == best:
                masks.append(selected)
            return
        visit(i+1, used, selected)
        for j, note in enumerate(predicted):
            if (j not in used and reference[i]['pitch'] == note['pitch']
                    and abs(reference[i]['onset']-note['onset']) <= tolerance):
                visit(i+1, used | {j}, selected | {i})

    visit(0, set(), set())
    possible = set().union(*masks)
    necessary = set.intersection(*masks)
    return best, possible, necessary


def test_microsecond_assignment_change_is_not_proven_loss_and_recovery():
    reference = notes([0., .03125])
    result = changes(reference, notes([.015625]), notes([.015624]))
    assert result['before']['possible'] == {0, 1}
    assert result['after']['possible'] == {0, 1}
    assert result['before']['necessary'] == set()
    assert result['after']['necessary'] == set()
    assert result['unavoidable'] == dict(recovered=set(), new_missing=set())


def test_unambiguous_loss_and_recovery():
    reference = notes([0., 1., 2.])
    result = changes(reference, notes([0., 1.]), notes([1., 2.]))
    assert result['selected'] == result['unavoidable']
    assert summarize_changes(result, 2) == dict(
        recovered=1, new_missing=1, unmatched_predictions=0,
        unavoidable_recovered=1, unavoidable_new_missing=1,
        before_ambiguous_coverage=0, after_ambiguous_coverage=0)


def test_alternating_chain_can_change_distant_reference_coverage():
    result = coverage(notes([0., .0625, .125]), notes([.03125, .09375]))
    assert len(result['selected']) == 2
    assert result['possible'] == {0, 1, 2}
    assert result['necessary'] == set()


def test_complete_matching_has_necessary_coverage_even_if_pairs_vary():
    result = coverage(notes([0., .03125]), notes([.015625, .015625]))
    assert result['necessary'] == {0, 1}


def test_empty_and_different_pitch_inputs():
    for reference, predicted in [(notes([0.]), []), ([], notes([0.])),
                                 (notes([0.]), notes([0.], 61))]:
        result = coverage(reference, predicted)
        assert result['possible'] == result['necessary'] == set()


def test_coverage_matches_exhaustive_maximum_assignments():
    rng = random.Random(314159)
    for strict, tolerance in [(False, .05), (True, .04)]:
        for _ in range(100):
            reference = [dict(onset=rng.randrange(8)/64, offset=1., pitch=rng.randrange(60, 62))
                         for _ in range(rng.randrange(6))]
            predicted = [dict(onset=rng.randrange(8)/64, offset=1., pitch=rng.randrange(60, 62))
                         for _ in range(rng.randrange(6))]
            size, possible, necessary = exhaustive(reference, predicted, tolerance)
            result = coverage(reference, predicted, strict=strict)
            assert len(result['selected']) == size
            assert result['possible'] == possible
            assert result['necessary'] == necessary
