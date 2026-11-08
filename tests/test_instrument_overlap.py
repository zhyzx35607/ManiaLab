from scripts.research_instrument_overlap import refine


def note(onset=1., offset=2., pitch=60, program=40, is_drum=False):
    return dict(onset=onset, offset=offset, pitch=pitch, program=program, is_drum=is_drum)


def run(base, alt, base_flags=None, alt_flags=None):
    return refine(base, alt,
        base_flags or {i:['forced_segment_boundary'] for i in range(len(base))},
        alt_flags or {i:['explicit_model_note_off'] for i in range(len(alt))},
        shifted_start=.5, segment_seconds=2.)


def test_only_truncated_tail_changes_and_explicit_note_is_not_truth():
    base = [note(), note(onset=4., offset=4.8), note(onset=7., is_drum=True)]
    alt = [note(onset=1.01, offset=2.8), note(onset=4., offset=5.8)]
    result, decisions = run(base, alt, {0:['forced_segment_boundary'],
        1:['explicit_model_note_off'], 2:['drum_fixed_10ms']})
    assert result[0] == note(offset=2.8) and base[0]['offset'] == 2.
    assert result[1:] == base[1:]
    assert len(decisions) == 1 and decisions[0]['alternate_note_index'] == 0


def test_ambiguous_and_boundary_endpoints_remain_unmodified():
    for flags, end in ((['forced_segment_boundary', 'explicit_model_note_off'], 2.8),
                       (['unresolved_or_postprocessed'], 2.8),
                       (['explicit_model_note_off'], 2.51)):
        result, _ = run([note()], [note(offset=end)], alt_flags={0:flags})
        assert result == [note()]
    assert run([note()], [note(program=42, offset=2.8)])[0] == [note()]


def test_distinct_reattacks_prevent_merging_from_either_pass():
    for base, alt in (([note(), note(onset=2.6, offset=3.)], [note(offset=2.8)]),
                      ([note()], [note(offset=2.8), note(onset=2.6, offset=3.)])):
        result, decisions = run(base, alt)
        assert result[0] == base[0]
        assert decisions[0]['status'] == 'intervening_reattack'


def test_matches_are_one_to_one_and_closer_nontruncated_note_owns_evidence():
    base = [note(onset=1.), note(onset=1.06, offset=1.2)]
    result, decisions = run(base, [note(onset=1.04, offset=2.8)],
        {0:['forced_segment_boundary'], 1:['explicit_model_note_off']})
    assert result == base and decisions[0]['status'] == 'no_matching_attack'


def test_explicit_second_pass_can_shorten_forced_tail_without_moving_attack():
    result, decisions = run([note()], [note(onset=.98, offset=1.8)])
    assert result == [note(offset=1.8)]
    assert decisions[0]['predicted_offset'] < decisions[0]['original_offset']
