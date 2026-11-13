import pytest

from scripts.summarize_onset_retention import build_onset_retention_ledger


def _fixture():
    analysis = {
        "candidates": [
            {
                "id": "other:a",
                "start": 100,
                "voice": "other",
                "source": "pitch",
                "strength": 1.0,
                "evidence_start": 105,
            },
            {
                "id": "other:b",
                "start": 100,
                "voice": "other",
                "source": "pitch",
                "strength": 0.7,
                "evidence_start": 98,
            },
            {
                "id": "piano:c",
                "start": 100,
                "voice": "piano",
                "source": "pitch",
                "strength": 0.8,
                "evidence_start": 101,
            },
            {
                "id": "piano:cross",
                "start": 100,
                "voice": "piano",
                "source": "pitch",
                "strength": 0.6,
                "evidence_start": 100,
            },
            {
                "id": "bass:d",
                "start": 200,
                "voice": "bass",
                "source": "pitch",
                "strength": 0.5,
                "evidence_start": None,
            },
        ]
    }
    chart_candidates = [
        {"id": "other:a", "start": 104},
        {"id": "other:b", "start": 104},
        {"id": "piano:c", "start": 102},
    ]
    selection = {
        "selected": 1,
        "omitted": 4,
        "omission_reasons": {
            "group_capacity_or_density": 2,
            "represented_by_chord": 1,
            "weak_detection": 1,
        },
        "decisions": [
            {
                "id": "other:a",
                "voice": "other",
                "start": 100,
                "section": 0,
                "decision": "selected",
                "reason": "supported_sustain",
                "group": 4,
                "instrument_id": "class:choir",
                "chart_start": 104,
            },
            {
                "id": "other:b",
                "voice": "other",
                "start": 100,
                "section": 0,
                "decision": "omitted",
                "reason": "represented_by_chord",
                "group": 4,
                "instrument_id": "class:strings",
                "represented_by": ["other:a"],
                "represented_at": 100,
            },
            {
                "id": "piano:c",
                "voice": "piano",
                "start": 100,
                "section": 0,
                "decision": "omitted",
                "reason": "group_capacity_or_density",
                "group": 5,
                "instrument_id": "",
            },
            {
                "id": "piano:cross",
                "voice": "piano",
                "start": 100,
                "section": 1,
                "decision": "omitted",
                "reason": "group_capacity_or_density",
                "group": 6,
                "instrument_id": "",
            },
            {
                "id": "bass:d",
                "voice": "bass",
                "start": 200,
                "section": 1,
                "decision": "omitted",
                "reason": "weak_detection",
                "instrument_id": "",
            },
        ],
    }
    scopes = [
        {"rows": [{"candidate_ids": ["other:a", "other:b", "piano:c"]}]},
    ]
    return analysis, chart_candidates, selection, scopes


def test_same_voice_classes_share_one_onset_and_ids_keep_group_and_chart_time():
    report = build_onset_retention_ledger(*_fixture())

    assert report["counts"] == {
        "analysis_candidate_count": 5,
        "chart_candidate_count": 3,
        "selected_candidate_count": 1,
        "omitted_candidate_count": 4,
        "unique_raw_onset_times": 2,
        "unique_section_voice_onsets": 4,
        "same_time_multi_voice_onset_count": 1,
        "same_time_multi_voice_candidate_count": 4,
        "outside_chart_candidate_count": 2,
    }
    onset = report["onsets"][0]
    assert onset["start_ms"] == 100 and onset["voice_count"] == 2
    assert onset["section_voice_onset_count"] == 3
    other, piano, piano_next_section = onset["voice_onsets"]
    assert other["voice"] == "other"
    assert other["section"] == 0
    assert other["candidate_count"] == 2
    assert other["selection_group_indices"] == [4]
    assert other["model_class_labels"] == ["class:choir", "class:strings"]
    assert other["candidate_ids"] == ["other:a", "other:b"]
    assert other["candidates"][0]["chart_start_ms"] == 104
    assert other["candidates"][1]["represented_by_candidate_ids"] == ["other:a"]
    assert piano["voice"] == "piano" and piano["candidate_count"] == 1
    assert piano_next_section["voice"] == "piano"
    assert piano_next_section["section"] == 1
    assert report["by_voice"]["other"]["section_voice_onset_count"] == 1
    assert report["by_voice"]["other"]["candidate_count"] == 2
    assert report["limitations"]["musical_acceptance"] is False
    assert "music_value" not in report


def test_candidate_and_legacy_scope_id_mismatches_are_rejected():
    analysis, chart, selection, scopes = _fixture()
    selection["decisions"][0]["id"] = "unknown"
    with pytest.raises(ValueError, match="IDs differ"):
        build_onset_retention_ledger(analysis, chart, selection, scopes)

    analysis, chart, selection, scopes = _fixture()
    scopes[0]["rows"][0]["candidate_ids"].append("other:a")
    with pytest.raises(ValueError, match="cover chart candidates exactly once"):
        build_onset_retention_ledger(analysis, chart, selection, scopes)


def test_selected_candidate_must_reach_the_chart_candidate_set():
    analysis, chart, selection, scopes = _fixture()
    chart[:] = [row for row in chart if row["id"] != "other:a"]
    scopes[0]["rows"][0]["candidate_ids"].remove("other:a")

    with pytest.raises(ValueError, match="Selected candidate IDs"):
        build_onset_retention_ledger(analysis, chart, selection, scopes)
