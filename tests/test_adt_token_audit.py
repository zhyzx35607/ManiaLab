import pytest

from scripts.adt_token_audit import audit_tokens, GROUPS, NAMES, group


def test_custom_pitch_semantics_are_not_general_midi():
    assert group(43) == group(44) == 'hi-hat'
    assert group(46) == group(48) == 'cymbal'
    assert group(47) == 'tom'
    assert len(NAMES) == 26
    assert set.union(*GROUPS.values()) == set(range(35, 61))
    assert sum(map(len, GROUPS.values())) == 26


@pytest.mark.parametrize('tokens', [[2, 3], [2, 0, 3], [2, 0, 3, 3, 3]])
def test_empty_and_batch_end_padding(tokens):
    result = audit_tokens(tokens)
    assert result['terminated'] and result['empty_stream']
    assert result['events'] == result['issues'] == []


def test_valid_triplets_preserve_ten_millisecond_clock_and_simultaneous_hits():
    result = audit_tokens([2, 4, 335, 500, 4, 338, 490, 259, 344, 527, 3])
    assert [e['tick'] for e in result['events']] == [0, 0, 255]
    assert [e['kind'] for e in result['events']] == ['kick', 'snare', 'hi-hat']
    assert result['issues'] == []


def test_missing_pitch_cannot_pair_old_time_with_next_note():
    result = audit_tokens([2, 4, 500, 104, 338, 490, 3])
    assert len(result['events']) == 1
    assert result['events'][0]['tick'] == 100
    assert result['events'][0]['token_index'] == 3
    assert result['issues']


def test_missing_velocity_does_not_take_later_velocity():
    result = audit_tokens([2, 4, 335, 104, 338, 490, 3])
    assert [(e['tick'], e['velocity']) for e in result['events']] == [(100, 90)]


def test_invalid_range_and_eos_are_never_silently_repaired():
    result = audit_tokens([2, 4, 399, 500, 104, 335, 900, 3, 204, 335, 500])
    assert result['events'] == []
    assert 'non_eos_after_end' in {i['kind'] for i in result['issues']}


def test_nonterminated_prefix_is_explicit():
    result = audit_tokens([2, 4, 335, 500], max_length=4)
    assert not result['terminated'] and result['hit_max_length']
    assert len(result['events']) == 1
    assert result['issues'][0]['kind'] == 'missing_eos'


def test_window_and_actual_input_end_are_distinguished():
    result = audit_tokens([2, 103, 335, 500, 104, 338, 500, 260, 342, 500, 3], valid_samples=24000)
    assert [e['in_audio'] for e in result['events']] == [True, False]
    assert {i['kind'] for i in result['issues']} == {'onset_in_padding', 'onset_outside_chunk'}


def test_duplicate_and_reversed_hits_are_reported_without_reordering():
    result = audit_tokens([2, 104, 335, 500, 104, 335, 500, 4, 335, 400, 3])
    assert [e['tick'] for e in result['events']] == [100, 100, 0]
    assert {i['kind'] for i in result['issues']} == {'duplicate_onset_class', 'nonmonotonic_onset', 'zero_velocity'}


@pytest.mark.parametrize('tokens', [[2., 3], [True], ['2', 3]])
def test_noninteger_tokens_rejected(tokens):
    with pytest.raises(ValueError):
        audit_tokens(tokens)
