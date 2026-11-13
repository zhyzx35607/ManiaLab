import pytest

from manialab.domain import UNOBSERVED_RELEASES
from scripts.summarize_expression_tradeoffs import build_expression_tradeoff_audit


def _fixture():
    analysis = {
        "candidates": [
            {
                "id": "other:selected",
                "start": 100,
                "voice": "other",
                "source": "pitch",
                "release_reason": "model_predicted_offset",
                "strength": 0.9,
            },
            {
                "id": "other:long",
                "start": 500,
                "voice": "other",
                "source": "pitch",
                "release_reason": "model_predicted_offset",
                "strength": 0.8,
            },
            {
                "id": "other:short",
                "start": 500,
                "voice": "other",
                "source": "pitch",
                "release_reason": "model_predicted_offset",
                "strength": 0.7,
            },
            {
                "id": "other:represented",
                "start": 100,
                "voice": "other",
                "source": "pitch",
                "release_reason": "model_predicted_offset",
                "strength": 0.6,
            },
            {
                "id": "other:tap",
                "start": 1300,
                "voice": "other",
                "source": "pitch",
                "release_reason": "model_predicted_offset",
                "strength": 0.5,
            },
            {
                "id": "other:brief",
                "start": 1500,
                "voice": "other",
                "source": "pitch",
                "release_reason": "model_predicted_offset",
                "strength": 0.4,
            },
            {
                "id": "other:unknown",
                "start": 1700,
                "voice": "other",
                "source": "pitch",
                "release_reason": next(iter(UNOBSERVED_RELEASES)),
                "strength": 0.3,
            },
        ],
        "structure": {
            "segments": [
                {"start_ms": 0, "end_ms": 1000, "label": "A"},
                {"start_ms": 1000, "end_ms": 2000, "label": "B"},
            ],
        },
    }
    decisions = [
        {
            "id": "other:selected",
            "voice": "other",
            "start": 100,
            "section": 0,
            "decision": "selected",
            "reason": "supported_sustain",
            "mode": "sustain",
            "expression_end": 1000,
            "chart_start": 100,
            "chart_end": 1000,
        },
        {
            "id": "other:long",
            "voice": "other",
            "start": 500,
            "section": 0,
            "decision": "omitted",
            "reason": "group_capacity_or_density",
            "mode": "sustain",
            "expression_end": 1200,
            "group": 1,
        },
        {
            "id": "other:short",
            "voice": "other",
            "start": 500,
            "section": 0,
            "decision": "omitted",
            "reason": "represented_by_chord",
            "mode": "sustain",
            "expression_end": 800,
            "group": 1,
        },
        {
            "id": "other:represented",
            "voice": "other",
            "start": 100,
            "section": 0,
            "decision": "omitted",
            "reason": "represented_by_chord",
            "mode": "sustain",
            "expression_end": 600,
            "group": 0,
        },
        {
            "id": "other:tap",
            "voice": "other",
            "start": 1300,
            "section": 1,
            "decision": "omitted",
            "reason": "group_capacity_or_density",
            "mode": "articulation",
            "expression_end": 1800,
            "articulation": {"tap": True},
            "group": 2,
        },
        {
            "id": "other:brief",
            "voice": "other",
            "start": 1500,
            "section": 1,
            "decision": "omitted",
            "reason": "group_capacity_or_density",
            "mode": "sustain",
            "expression_end": 1600,
            "group": 3,
        },
        {
            "id": "other:unknown",
            "voice": "other",
            "start": 1700,
            "section": 1,
            "decision": "omitted",
            "reason": "group_capacity_or_density",
            "mode": "sustain",
            "expression_end": 2000,
            "group": 4,
        },
    ]
    selection = {
        "selected": 1,
        "omitted": 6,
        "omission_reasons": {
            "group_capacity_or_density": 4,
            "represented_by_chord": 2,
        },
        "decisions": decisions,
    }
    return analysis, selection


def test_groups_alternatives_and_measures_uncovered_interval_individually():
    analysis, selection = _fixture()
    report = build_expression_tradeoff_audit(analysis, selection, min_ln_ms=240)

    assert report["counts"] == {
        "analysis_candidate_count": 7,
        "selected_candidate_count": 1,
        "omitted_candidate_count": 6,
        "selected_ln_candidate_count": 1,
        "selected_ln_onset_group_count": 1,
        "omitted_hold_eligible_candidate_count": 3,
        "omitted_hold_eligible_onset_group_count": 2,
        "omitted_hold_groups_without_same_onset_selected_ln": 1,
    }
    groups = report["omitted_hold_onset_groups"]
    same_onset = next(row for row in groups if row["start_analysis_ms"] == 100)
    later = next(row for row in groups if row["start_analysis_ms"] == 500)
    assert same_onset["has_selected_ln_at_same_voice_onset"] is True
    assert same_onset["individually_uncovered_ms"] == 0
    assert later["candidate_count"] == 2
    assert later["individually_uncovered_intervals_analysis_ms"] == [[1000, 1200]]
    assert later["individually_uncovered_ms"] == 200
    assert report["interpretation_limits"]["omitted_intervals_are_not_additive_gains"] is True
    assert report["interpretation_limits"]["musical_value_scored"] is False


def test_selection_ids_and_reported_totals_must_match():
    analysis, selection = _fixture()
    selection["decisions"][0]["id"] = "unknown"
    with pytest.raises(ValueError, match="IDs differ"):
        build_expression_tradeoff_audit(analysis, selection, min_ln_ms=240)

    analysis, selection = _fixture()
    selection["omission_reasons"]["group_capacity_or_density"] = 3
    with pytest.raises(ValueError, match="omission reasons"):
        build_expression_tradeoff_audit(analysis, selection, min_ln_ms=240)


def test_unknown_release_taps_and_short_durations_are_not_ln_markers():
    analysis, selection = _fixture()
    report = build_expression_tradeoff_audit(analysis, selection, min_ln_ms=240)
    ids = {
        identity
        for row in report["omitted_hold_onset_groups"]
        for identity in row["candidate_ids"]
    }
    assert ids == {"other:long", "other:short", "other:represented"}


def test_candidate_section_must_reference_frozen_structure():
    analysis, selection = _fixture()
    selection["decisions"][1]["section"] = 9
    with pytest.raises(ValueError, match="Invalid section"):
        build_expression_tradeoff_audit(analysis, selection, min_ln_ms=240)
