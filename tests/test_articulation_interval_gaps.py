"""A distant decay sequence must not flatten separate observed intervals."""
from dataclasses import replace

import pytest

from manialab.domain import Note, UNOBSERVED_RELEASES
from manialab.expression_articulation import articulation_policy, attack_runs


def note(start, end, pitch=60, **kwargs):
    return Note(start, end, pitch=pitch, voice='piano', source='pitch',
                release_reason='release_regression', id=f'{start}:{pitch}', **kwargs)


def separated_then_overlapping():
    return ([note(t, t+300) for t in (1000, 1500, 2000)]
            + [note(t, t+1200) for t in (2500, 2750, 3000, 3250, 3500)])


def test_later_overlapping_attacks_do_not_flatten_separate_sustains():
    notes = separated_then_overlapping()
    before = [n.data() for n in notes]
    result = articulation_policy(notes, observed_until_ms=6000)
    assert [result[n.id]['tap'] for n in notes] == [False]*3 + [True]*4 + [False]
    assert [n.data() for n in notes] == before


def test_gap_protection_does_not_create_new_taps_from_a_previously_sustained_run():
    notes = separated_then_overlapping()[:-2]
    rules = articulation_policy(notes, observed_until_ms=6000)
    assert all(not rules[n.id]['tap'] for n in notes)


def test_gap_keeps_preceding_run_tail_and_following_run_separate():
    left = [note(t, 1900) for t in (1000, 1250, 1500)]
    right = [note(t, t+1200) for t in (2000, 2250, 2500)]
    rules = articulation_policy(left+right, observed_until_ms=5000)
    assert [rules[n.id]['tap'] for n in left+right] == [True, True, False]*2
    assert rules[left[-1].id]['run'] != rules[right[0].id]['run']
    assert rules[left[-1].id]['continuation'] == 'observed_interval_gap'


def test_an_earlier_sustain_crossing_the_gap_prevents_an_interval_split():
    notes = separated_then_overlapping()
    notes[0] = replace(notes[0], end=5000)
    runs, _ = attack_runs(notes)
    assert len(runs) == 1


def test_one_chord_layer_crossing_the_gap_prevents_an_interval_split():
    notes = separated_then_overlapping()
    notes += [note(1000, 5000, pitch=48)]
    runs, _ = attack_runs(notes)
    assert len(runs) == 1


def test_tail_from_an_earlier_onset_run_still_prevents_a_false_gap():
    notes = [note(1000, 5000)] + [note(t, t+300) for t in (2500, 3000, 3500)]
    notes += [note(t, t+1200) for t in (3750, 4000, 4250, 4500)]
    runs, _ = attack_runs(notes)
    assert [[group[0].start for group in run] for run in runs] == [
        [1000], [2500, 3000, 3500, 3750, 4000, 4250, 4500]]


def test_eighty_ms_interval_gap_is_sufficient_without_extending_a_tail():
    left = [note(t, 1920) for t in (1000, 1250, 1500)]
    right = [note(t, t+1200) for t in (2000, 2250, 2500)]
    rules = articulation_policy(left+right, observed_until_ms=5000)
    assert not rules[left[-1].id]['tap']
    assert rules[left[-1].id]['continuation'] == 'observed_interval_gap'
    assert left[-1].end == 1920


@pytest.mark.parametrize('reason', sorted(UNOBSERVED_RELEASES))
def test_unobserved_release_cannot_establish_an_interval_gap(reason):
    notes = separated_then_overlapping()
    notes[0] = replace(notes[0], release_reason=reason)
    runs, _ = attack_runs(notes)
    assert len(runs) == 1


@pytest.mark.parametrize('gap', [0, 1, 79])
def test_short_interval_gap_does_not_split_a_decay_sequence(gap):
    left = [note(t, 2000-gap) for t in (1000, 1250, 1500)]
    right = [note(t, t+1200) for t in (2000, 2250, 2500)]
    runs, _ = attack_runs(left+right)
    assert len(runs) == 1


def test_transient_timestamps_do_not_imply_observed_note_releases():
    notes = [replace(n, source='transient', end=n.start) for n in separated_then_overlapping()]
    runs, _ = attack_runs(notes)
    assert len(runs) == 1


def test_order_and_duplicate_observations_do_not_change_interval_gap():
    notes = separated_then_overlapping()
    expected = articulation_policy(notes, observed_until_ms=6000)
    extra = replace(notes[1], id='duplicate')
    actual = articulation_policy(list(reversed(notes+[extra])), observed_until_ms=6000)
    assert {n.id: actual[n.id] for n in notes} == expected


def test_other_instrument_sustain_does_not_close_a_source_local_gap():
    notes = separated_then_overlapping()
    expected = articulation_policy(notes, observed_until_ms=6000)
    other = note(0, 10000, instrument_id='other-piano')
    actual = articulation_policy(notes+[other], observed_until_ms=6000)
    assert {n.id: actual[n.id] for n in notes} == expected


def test_section_cuts_and_manual_sustain_keep_their_separate_meanings():
    from manialab.arrange import parameters
    from manialab.expression import expression_evidence, propose_expression
    from tests.test_structure_expression import analysis

    notes = separated_then_overlapping()
    source = analysis(notes, length=6000)
    settings = parameters()
    full, _, _ = expression_evidence(source, propose_expression(source, settings), settings)
    source['structure']['segments'] = [dict(start_ms=a, end_ms=b)
                                      for a, b in ((0, 1800), (1800, 2900), (2900, 6000))]
    plan = propose_expression(source, settings)
    split, _, _ = expression_evidence(source, plan, settings)
    assert split == full
    for section in plan['sections']:
        section['voices']['piano']['mode'] = 'sustain'
    manual, _, _ = expression_evidence(source, plan, settings)
    assert manual == notes
