from copy import deepcopy

from scripts.research_guitar_release_transfer import transfer_releases


def note(onset=1., offset=3., *, pitch=60, reason='reattack', observed=False):
    return dict(onset=onset, offset=offset, pitch=pitch, program=-1, is_drum=False,
                release_reason=reason, release_observed=observed, strength=.31)


def test_decay_transfer_preserves_attack_count_strength_and_both_original_inputs():
    targets = [note(), note(4., 5., pitch=64, reason='release_regression', observed=True)]
    donors = [note(1.02, 2., reason='frame_decay', observed=True)]
    original = deepcopy((targets, donors))
    result, trace = transfer_releases(targets, donors, context_end=6.)
    assert len(result) == 2 and result[0]['onset'] == 1. and result[0]['strength'] == .31
    assert result[0]['offset'] == 2. and result[0]['release_evidence']['original_offset'] == 3.
    assert result[1] == targets[1] and (targets, donors) == original
    assert trace[0]['decision'] == 'transferred_model_decay'


def test_unison_and_fast_repeats_do_not_borrow_a_single_donor_tail_twice():
    donor = [note(1.02, 2., reason='frame_decay', observed=True)]
    for second_onset in (1., 1.04):
        targets = [note(), note(second_onset, 3.1, reason='release_regression', observed=True)]
        result, trace = transfer_releases(targets, donor, context_end=6.)
        assert result == targets
        assert trace[0]['decision'] == 'no_unique_bidirectional_attack_match'


def test_ineligible_second_donor_must_still_make_the_attack_ambiguous():
    targets = [note()]
    donors = [note(1.01, 2., reason='frame_decay', observed=True), note(1.03, 2.5)]
    result, trace = transfer_releases(targets, donors, context_end=6.)
    assert result == targets and trace[0]['matching_donor_indices'] == [0, 1]


def test_artificial_cuts_and_context_boundary_are_never_promoted():
    targets = [note()]
    for reason, observed, offset in (('same_pitch_reattack', True, 2.), ('frame_decay', False, 2.),
                                     ('frame_decay', True, 6.), ('input_boundary', False, 6.)):
        donors = [note(1.01, offset, reason=reason, observed=observed)]
        assert transfer_releases(targets, donors, context_end=6.)[0] == targets


def test_other_pitch_or_drum_cannot_supply_the_release():
    targets = [note()]
    donor = note(1.01, 2., reason='frame_decay', observed=True)
    for changed in (dict(donor, pitch=61), dict(donor, is_drum=True)):
        assert transfer_releases(targets, [changed], context_end=6.)[0] == targets


def test_unique_attack_match_cannot_extend_through_a_later_same_pitch_attack():
    targets = [note(1., 1.19), note(1.2, 3.)]
    donors = [note(1.01, 2., reason='frame_decay', observed=True)]
    result, trace = transfer_releases(targets, donors, context_end=6.)
    assert result == targets
    assert trace[0]['matching_donor_indices'] == [0]
    assert trace[0]['crossed_later_target_indices'] == [1]
    assert trace[0]['decision'] == 'donor_tail_crosses_later_target_attack'
