import numpy as np
import pytest

from scripts.evaluate_sequence_prior_groups import (
    pair_accuracy_counts,
    summarize_group_counts,
)


def test_pair_accuracy_splits_ln_and_taps_and_counts_ties_as_half():
    counts = pair_accuracy_counts(
        positive_scores=np.array([1.0, 0.0, 2.0]),
        negative_scores=np.array([
            [1.0, 0.0],
            [1.0, -1.0],
            [1.0, 2.0],
        ]),
        ln_context=np.array([True, False, True]),
    )

    assert counts["overall"] == [4.0, 6]
    assert counts["ln"] == [3.0, 4]
    assert counts["tap"] == [1.0, 2]


def test_song_group_macro_weights_groups_equally_and_skips_empty_subsets():
    groups = {
        "large-group": {
            "overall": [1.0, 2],
            "ln": [1.0, 2],
            "tap": [0.0, 0],
        },
        "small-group": {
            "overall": [1.0, 1],
            "ln": [0.0, 0],
            "tap": [1.0, 1],
        },
    }

    result = summarize_group_counts(groups, bootstrap_samples=500, seed=17)

    assert result["overall"]["groups"] == 2
    assert result["overall"]["pairs"] == 3
    assert result["overall"]["macro_accuracy"] == pytest.approx(.75)
    assert result["ln"]["groups"] == 1
    assert result["ln"]["macro_accuracy"] == pytest.approx(.5)
    assert result["tap"]["groups"] == 1
    assert result["tap"]["macro_accuracy"] == pytest.approx(1.0)


def test_song_group_bootstrap_interval_is_reproducible():
    groups = {
        "a": {"overall": [1., 2], "ln": [0., 0], "tap": [0., 0]},
        "b": {"overall": [1., 1], "ln": [0., 0], "tap": [0., 0]},
        "c": {"overall": [0., 1], "ln": [0., 0], "tap": [0., 0]},
    }

    first = summarize_group_counts(groups, bootstrap_samples=1000, seed=29)
    second = summarize_group_counts(groups, bootstrap_samples=1000, seed=29)

    assert first == second
    assert first["overall"]["ci_95"] == [0.0, 1.0]


@pytest.mark.parametrize("sample_count", [0, -1, True, 2.5, "5"])
def test_song_group_bootstrap_rejects_invalid_sample_count(sample_count):
    with pytest.raises(ValueError, match="bootstrap_samples"):
        summarize_group_counts({}, bootstrap_samples=sample_count)
