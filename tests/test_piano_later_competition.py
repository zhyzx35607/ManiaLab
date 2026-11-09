import numpy as np
import pytest

from scripts.piano_successor_decode import select_successor_groups
from scripts.summarize_piano_later_competition import competition_diagnostics


def example(logits, *, same_winning_tick=False):
    data = dict(source_ids=np.asarray(['s']*4), target_ticks=np.asarray([120,120,240,240 if same_winning_tick else 360]),
                actual=np.asarray([True,True,False,False]), score_mask=np.asarray([True,True,False,False]))
    logits = np.asarray(logits, dtype=float)
    selected = select_successor_groups(data['source_ids'], data['target_ticks'], logits)
    proved = np.asarray([False,False,True,False]); auxiliary_eligible = proved.copy()
    return data, logits, selected, proved, auxiliary_eligible


def test_counts_each_lost_chord_member_for_proved_later_winner():
    result = competition_diagnostics(*example([1,1,3,-1]))
    assert result['removed_true_with_proved_maximum_only'] == 2
    assert result['removed_true_with_same_window_proved_maximum_only'] == 2
    assert result['true_sources_with_proved_maximum_only'] == 1


def test_selected_proved_edge_below_unproved_maximum_is_not_blame_for_maximum():
    result = competition_diagnostics(*example([1,1,2,3], same_winning_tick=True))
    assert result['decoded_proved_positive'] == 1
    assert result['removed_true_with_proved_maximum_any'] == 0


def test_tied_mixed_maxima_are_reported_as_any_but_not_only():
    result = competition_diagnostics(*example([1,1,3,3], same_winning_tick=True))
    assert result['removed_true_with_proved_maximum_any'] == 2
    assert result['removed_true_with_proved_maximum_only'] == 0


def test_maximum_only_is_within_winning_tick_and_does_not_claim_recovery():
    data, logits, selected, proved, eligible = example([1,1,3,3])
    result = competition_diagnostics(data, logits, selected, proved, eligible)
    assert result['removed_true_with_proved_maximum_only'] == 2
    remaining = np.asarray([True,True,False,True])
    after_removal = select_successor_groups(data['source_ids'][remaining], data['target_ticks'][remaining], logits[remaining])
    assert not after_removal[:2].any()  # The other later tick still wins.


def test_already_negative_true_edge_is_not_counted_as_decoder_loss():
    result = competition_diagnostics(*example([1,-1,3,-1]))
    assert result['removed_true_with_proved_maximum_only'] == 1


def test_proved_edge_outside_positive_owner_window_is_distinguished():
    values = list(example([1,1,3,-1])); values[-1][:] = False
    result = competition_diagnostics(*values)
    assert result['removed_true_with_proved_maximum_only'] == 2
    assert result['removed_true_with_same_window_proved_maximum_only'] == 0


@pytest.mark.parametrize('field', ['proof', 'auxiliary_eligible', 'selected', 'shape', 'nan'])
def test_rejects_invalid_masks_or_changed_decoder(field):
    values = list(example([1,1,3,-1]))
    if field == 'proof': values[3][0] = True
    elif field == 'auxiliary_eligible': values[4][3] = True
    elif field == 'selected': values[2][:] = False
    elif field == 'shape': values[3] = values[3][:-1]
    else: values[1][0] = np.nan
    with pytest.raises(ValueError):
        competition_diagnostics(*values)
