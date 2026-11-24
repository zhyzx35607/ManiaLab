import pytest

from manialab.arrange import hand_weights, layout_metrics, parameters
from manialab.domain import Note
from manialab.layout import fingers
from manialab.review import review_chart


@pytest.mark.parametrize("keys,expected", [
    (6, ((0, 4), (0, 3), (0, 2), (1, 2), (1, 3), (1, 4))),
    (7, ((0, 4), (0, 3), (0, 2), (2, 1), (1, 2), (1, 3), (1, 4))),
    (8, ((0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4))),
    (9, ((0, 5), (0, 4), (0, 3), (0, 2), (2, 1), (1, 2), (1, 3), (1, 4), (1, 5))),
    (10, ((0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5))),
])
def test_standard_key_layout_profiles_match_expected_lane_hands_and_fingers(keys, expected):
    assert fingers(keys) == expected
    assert fingers(keys, parameters({"keys": keys})["key_layout"]) == expected


def test_8k_4_4_is_a_supported_finger_profile():
    assert fingers(8, "8k_4_4") == (
        (0, 5), (0, 4), (0, 3), (0, 2), (1, 2), (1, 3), (1, 4), (1, 5),
    )


@pytest.mark.parametrize("keys,key_layout", [(8, "10k_5_5"), (7, "8k_4_4"), (6, "unknown")])
def test_metrics_and_review_reject_invalid_layout_combinations(keys, key_layout):
    with pytest.raises(ValueError, match="键型"):
        hand_weights(keys, 0, key_layout)
    with pytest.raises(ValueError, match="键型"):
        layout_metrics([], keys, key_layout=key_layout)
    with pytest.raises(ValueError, match="键型"):
        review_chart([], keys, key_layout=key_layout)


def test_assignment_and_generation_reports_use_the_same_explicit_key_layout(monkeypatch):
    from manialab import generation
    from manialab.arrange import assign_lanes as real_assign_lanes
    from manialab.arrange import layout_metrics as real_layout_metrics
    from manialab.review import review_chart as real_review_chart
    from manialab.ln_patterns import analyze_ln_patterns as real_analyze_ln_patterns

    seen = {}

    def assign(*args, **kwargs):
        seen["assign"] = kwargs["key_layout"]
        return real_assign_lanes(*args, **kwargs)

    def metrics(*args, **kwargs):
        seen["metrics"] = kwargs["key_layout"]
        return real_layout_metrics(*args, **kwargs)

    def review(*args, **kwargs):
        seen["review"] = kwargs["key_layout"]
        return real_review_chart(*args, **kwargs)

    def ln_patterns(*args, **kwargs):
        seen["ln_patterns"] = kwargs["key_layout"]
        return real_analyze_ln_patterns(*args, **kwargs)

    monkeypatch.setattr(generation, "assign_lanes", assign)
    monkeypatch.setattr(generation, "layout_metrics", metrics)
    monkeypatch.setattr(generation, "review_chart", review)
    monkeypatch.setattr(generation, "analyze_ln_patterns", ln_patterns)

    note = Note(1000, 1000, lane=2, id="tap")
    state = dict(input_notes=[note.data()], input_keys=8, input_timing=[], variants=[], active=-1)
    prefs = parameters({"keys": 8, "key_layout": "8k_4_4"})
    generation.generate_variant(
        state,
        {"mode": "remap"},
        prefs,
        {"bpm": 120, "offset": 0, "division": 4, "audio_shift_ms": 0},
        lambda *_: None,
        lambda: False,
    )

    assert seen == {
        "assign": "8k_4_4", "metrics": "8k_4_4", "review": "8k_4_4",
        "ln_patterns": "8k_4_4",
    }
    assert state["variants"][-1]["settings"]["key_layout"] == "8k_4_4"
