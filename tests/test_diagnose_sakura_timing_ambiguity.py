from copy import deepcopy

import pytest

from manialab.domain import Note
from scripts.diagnose_sakura_timing_ambiguity import (
    EXPECTED_INPUT_SHA256,
    build_report,
)
from manialab.timing import grid_diagnostics


def _candidate(bpm, offset=0):
    return {
        "bpm": bpm,
        "offset": offset,
        "division": 4,
        "score": 0.8,
        "coverage": 1.0,
        "median_error_ms": 0.0,
    }


def _fixture():
    beat = 60_000 / 148
    attacks = [
        Note(round(204 + index * beat), round(204 + index * beat), strength=1.0, id=str(index))
        for index in range(120)
    ]
    candidates = [
        _candidate(74.0, 204),
        _candidate(148.0, 204),
        _candidate(296.0, 2),
    ]
    sections = grid_diagnostics(attacks, 148, 204, 4)
    source = {
        "bpm": 148.0,
        "offset": 204,
        "division": 4,
        "score": 0.8,
        "coverage": 1.0,
        "median_error_ms": 0.0,
        "stable": False,
        "candidates": candidates,
        "sections": sections,
    }
    data = {
        "duration": 49.0,
        "timing_attacks": [
            {
                "start": attack.start,
                "end": attack.end,
                "strength": attack.strength,
                "id": attack.id,
            }
            for attack in attacks
        ],
        "tempo": {
            "bpm": 148.0,
            "offset": 204,
            "division": 4,
            "score": 0.8,
            "coverage": 1.0,
            "median_error_ms": 0.0,
            "stable": False,
            "evidence_source": "drums",
            "comparisons": {
                "drums": source,
                "mix": deepcopy(source),
            },
        },
    }
    data["tempo"]["comparisons"]["mix"]["offset"] = 206
    for candidate in data["tempo"]["comparisons"]["mix"]["candidates"]:
        candidate["offset"] = candidate["offset"] + 2
    return data


def test_report_describes_half_double_phase_and_separates_beat_support():
    report = build_report(_fixture(), EXPECTED_INPUT_SHA256)

    hypotheses = {
        row["bpm"]: row for row in report["selected_attack_grid_hypotheses"]
    }
    assert hypotheses[74.0]["half_beat_coverage"] == hypotheses[148.0]["beat_coverage"]
    assert abs(
        hypotheses[148.0]["half_beat_coverage"]
        - hypotheses[296.0]["beat_coverage"]
    ) < 0.01
    assert hypotheses[74.0]["beat_coverage"] < hypotheses[148.0]["beat_coverage"]
    assert all(row["windows"]["window_count"] == 4 for row in hypotheses.values())
    assert [row["wrapped_phase_difference_ms"] for row in report["cross_source_phase"]] == [2, 2, 2]
    assert report["interpretation_limits"]["metrical_beat_identified"] is False


def test_report_rejects_wrong_fingerprint_and_inconsistent_stored_coverage():
    with pytest.raises(ValueError, match="SHA-256"):
        build_report(_fixture(), "0" * 64)

    data = _fixture()
    data["tempo"]["comparisons"]["drums"]["candidates"][1]["coverage"] = 0.2
    with pytest.raises(ValueError, match="coverage differs"):
        build_report(data, EXPECTED_INPUT_SHA256)


def test_report_requires_all_three_selected_source_aliases():
    data = _fixture()
    data["tempo"]["comparisons"]["drums"]["candidates"].pop()
    with pytest.raises(ValueError, match="missing 296 BPM"):
        build_report(data, EXPECTED_INPUT_SHA256)
