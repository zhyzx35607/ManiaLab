from manialab.domain import Note
from scripts.analyze_sakura_expression_profile import duration_beats, profile_segment


def test_duration_beats_integrates_redline_changes():
    timing = ["0,500,4,2,0,70,1,0", "2000,1000,4,2,0,70,1,0"]

    assert duration_beats(0, 4000, timing) == 6.0


def test_profile_uses_half_open_sections_and_counts_ln_interactions():
    notes = [
        Note(100, 2100, lane=0),
        Note(100, 1100, lane=1),
        Note(600, 600, lane=2),
        Note(2000, 2000, lane=3),
    ]

    profile = profile_segment(
        notes, 10, ["0,500,4,2,0,70,1,0"], 0, 2000,
        lane_sources={"drums": [1, 2], "piano": [3], "other": [7, 8, 9, 10]},
    )

    assert profile["objects"] == 3
    assert profile["ln"] == 2
    assert profile["ln_fraction"] == 2 / 3
    assert profile["ln_median_beats"] == 3.0
    assert profile["ln_p90_beats"] == 4.0
    assert profile["ln_at_least_2_beats"] == 2
    assert profile["ln_crossing_segment_end"] == 1
    assert profile["peak_held"] == 2
    assert profile["chord_groups"] == 1
    assert profile["ln_chord_groups"] == 1
    assert profile["staggered_ln_chord_groups"] == 1
    assert profile["taps_during_other_lane_ln"] == 1
    assert profile["source_groups"]["drums"] == {"objects": 2, "ln": 2}
    assert profile["source_groups"]["piano"] == {"objects": 1, "ln": 0}


def test_ln_ending_exactly_at_boundary_does_not_cross_section():
    profile = profile_segment(
        [Note(100, 2000, lane=0)], 10, ["0,500,4,2,0,70,1,0"], 0, 2000,
    )

    assert profile["ln_crossing_segment_end"] == 0
