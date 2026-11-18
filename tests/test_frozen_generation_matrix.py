from __future__ import annotations

import hashlib
import json

import pytest

from scripts.summarize_frozen_generation_matrix import build_matrix_report


def _voice(*, candidates, selected, ln=0, unknown=0, omitted=None):
    release_sources = {}
    if ln > unknown:
        release_sources["known_test_fixture"] = ln - unknown
    if unknown:
        release_sources["unknown"] = unknown
    return {
        "candidates": candidates,
        "selected": selected,
        "retained_raw_ids": selected,
        "ln": ln,
        "union_ln_ms": ln * 300,
        "unknown_release_ln": unknown,
        "large_attack_shifts": 1 if selected else 0,
        "max_attack_shift_ms": 31 if selected else 0,
        "large_release_shifts": 0,
        "max_release_shift_ms": 0,
        "later_release_shifts": 0,
        "earlier_release_shifts": 0,
        "release_sources": release_sources,
        "omitted_reasons": omitted or {},
    }


def _case(case_id, analysis_sha, keys, voice, *, accepted=True, plan=None):
    total_notes = sum(value["selected"] for value in voice.values())
    total_ln = sum(value["ln"] for value in voice.values())
    return {
        "id": case_id,
        "analysis": f"test-data/audio-choice-pairs/{analysis_sha[:6]}/deep/analysis.json",
        "analysis_sha256": analysis_sha,
        "settings": {"keys": keys, "density": 7},
        "tempo": {
            "bpm": 120,
            "offset": 10,
            "division": 4,
            "audio_shift_ms": 22.5,
            "stable": accepted,
        },
        "plan": plan,
    }


def _summary(case_id, keys, voice, *, accepted=True):
    return {
        "id": case_id,
        "keys": keys,
        "passed": True,
        "notes": sum(value["selected"] for value in voice.values()),
        "ln": sum(value["ln"] for value in voice.values()),
        "voices": voice,
        "source_snapshot_notes": sum(value["selected"] for value in voice.values()),
        "timing_suspect_windows": 0,
        "automatic_timing_gate": {"accepted": accepted},
    }


def _build(plan_cases, summaries, variant_notes=None):
    plan = {"schema": "manialab-frozen-generation-v1", "cases": plan_cases}
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    plan_sha = hashlib.sha256(payload).hexdigest()
    replay = {
        "schema": "manialab-frozen-generation-results-v1",
        "plan_sha256": plan_sha,
        "passed": True,
        "source_files": {"manialab/generation.py": "a" * 64},
        "cases": summaries,
    }
    return build_matrix_report(
        plan,
        replay,
        plan_sha256=plan_sha,
        replay_sha256="b" * 64,
        tool_sha256="c" * 64,
        variant_notes_by_case=variant_notes,
    )


def test_matrix_deduplicates_timing_by_analysis_and_keeps_unknown_ln_distinct():
    analysis_a, analysis_b = "1" * 64, "2" * 64
    a_voice = {"piano": _voice(candidates=10, selected=5, ln=2, unknown=2)}
    b_voice = {
        "piano": _voice(
            candidates=8,
            selected=2,
            ln=1,
            unknown=0,
            omitted={"group_capacity_or_density": 4},
        )
    }
    plan_cases = [
        _case("a-6k", analysis_a, 6, a_voice),
        _case("a-10k", analysis_a, 10, a_voice),
        _case("b-6k", analysis_b, 6, b_voice, accepted=False),
    ]
    summaries = [
        _summary("a-6k", 6, a_voice),
        _summary("a-10k", 10, a_voice),
        _summary("b-6k", 6, b_voice, accepted=False),
    ]

    report = _build(plan_cases, summaries)

    assert report["summary"] == {
        "case_count": 3,
        "analysis_count": 2,
        "key_counts": [6, 10],
        "automatic_timing_accepted_analyses": 1,
        "automatic_timing_rejected_analyses": 1,
    }
    key6 = next(row for row in report["by_key_profile"] if row["keys"] == 6)
    piano = key6["voices"]["piano"]
    assert key6["analysis_count"] == 2
    assert piano["candidate_count"] == 18
    assert piano["selected_count"] == 7
    assert piano["median_per_analysis_selection_rate"] == 0.375
    assert piano["ln_count"] == 3
    assert piano["unknown_release_ln"] == 2
    assert piano["known_release_ln"] == 1
    assert piano["unknown_release_fraction"] == round(2 / 3, 6)
    assert piano["release_sources"] == {"known_test_fixture": 1, "unknown": 2}
    assert report["limitations"]["musical_value_scored"] is False


def test_matrix_separates_recommendation_profiles_and_rejects_duplicate_units():
    analysis_sha = "3" * 64
    voice = {"vocals": _voice(candidates=4, selected=2)}
    recommendation = {
        "source": "local-audio-recommendation",
        "choice": "melody",
        "segments": [],
    }
    report = _build(
        [_case("melody-7k", analysis_sha, 7, voice, plan=recommendation)],
        [_summary("melody-7k", 7, voice)],
    )
    assert report["by_key_profile"][0]["profile"] == "recommendation:melody"

    with pytest.raises(ValueError, match="Multiple cases share"):
        _build(
            [
                _case("default-6k-a", analysis_sha, 6, voice),
                _case("default-6k-b", analysis_sha, 6, voice),
            ],
            [
                _summary("default-6k-a", 6, voice),
                _summary("default-6k-b", 6, voice),
            ],
        )


def test_matrix_rejects_inconsistent_timing_for_the_same_analysis():
    analysis_sha = "4" * 64
    voice = {"drums": _voice(candidates=3, selected=2)}
    inconsistent = _case("second-10k", analysis_sha, 10, voice, accepted=False)

    with pytest.raises(ValueError, match="Timing result changes"):
        _build(
            [
                _case("first-6k", analysis_sha, 6, voice),
                inconsistent,
            ],
            [
                _summary("first-6k", 6, voice),
                _summary("second-10k", 10, voice, accepted=False),
            ],
        )


def test_matrix_rejects_case_or_release_accounting_mismatch():
    analysis_sha = "5" * 64
    voice = {"bass": _voice(candidates=5, selected=2, ln=1, unknown=1)}
    bad = _summary("bass-6k", 6, voice)
    bad["voices"]["bass"]["release_sources"] = {}
    with pytest.raises(ValueError, match="Release sources"):
        _build([_case("bass-6k", analysis_sha, 6, voice)], [bad])

    valid_voice = {"bass": _voice(candidates=5, selected=2, ln=1, unknown=1)}
    with pytest.raises(ValueError, match="IDs differ"):
        _build(
            [_case("bass-6k", analysis_sha, 6, valid_voice)],
            [_summary("different-id", 6, valid_voice)],
        )


def test_matrix_summarizes_lane_load_and_cross_key_mapping_changes():
    analysis_sha = "6" * 64
    voice = {"piano": _voice(candidates=2, selected=2, ln=1)}
    plan_cases = [
        _case("song-6k", analysis_sha, 6, voice),
        _case("song-10k", analysis_sha, 10, voice),
    ]
    summaries = [
        _summary("song-6k", 6, voice),
        _summary("song-10k", 10, voice),
    ]
    variant_notes = {
        "song-6k": [
            {"id": "tap", "start": 100, "end": 100, "lane": 0, "voice": "piano"},
            {"id": "hold", "start": 200, "end": 500, "lane": 5, "voice": "piano"},
        ],
        "song-10k": [
            {"id": "tap", "start": 100, "end": 100, "lane": 1, "voice": "piano"},
            {"id": "hold", "start": 200, "end": 500, "lane": 9, "voice": "piano"},
        ],
    }

    report = _build(plan_cases, summaries, variant_notes)

    six_key = next(case for case in report["cases"] if case["keys"] == 6)
    assert six_key["key_mapping"]["lane_counts"] == [1, 0, 0, 0, 0, 1]
    comparison = report["key_mapping_comparisons"][0]
    assert comparison["analysis_count"] == 1
    assert comparison["common_note_ids"] == 2
    assert comparison["lane_changed_note_ids"] == 2
    assert comparison["lane_changed_fraction"] == 1
    assert comparison["lane_changed_by_voice"] == {"piano": 2}
    assert comparison["timing_changed_note_ids"] == 0
    assert comparison["added_note_ids"] == comparison["removed_note_ids"] == 0
    assert report["limitations"]["lane_changes_are_not_player_validated"] is True


def test_matrix_rejects_invalid_key_mapping_variant_data():
    analysis_sha = "7" * 64
    voice = {"piano": _voice(candidates=1, selected=1)}
    with pytest.raises(ValueError, match="lane is outside"):
        _build(
            [_case("song-6k", analysis_sha, 6, voice)],
            [_summary("song-6k", 6, voice)],
            {
                "song-6k": [
                    {"id": "out-of-range", "start": 100, "end": 100, "lane": 6, "voice": "piano"}
                ]
            },
        )
