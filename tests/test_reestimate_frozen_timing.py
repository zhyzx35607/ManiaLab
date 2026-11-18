import hashlib
import json
from pathlib import Path

import pytest

from scripts import reestimate_frozen_timing as reestimate


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    root = tmp_path
    analysis_path = root / "test-data" / "audio-choice-pairs" / "clip" / "analysis.json"
    analysis = {
        "duration": 12,
        "timing_attacks": [{"id": "private-note-id", "start": 100, "end": 100, "strength": 1}],
        "tempo": {
            "bpm": 120,
            "offset": 10,
            "division": 4,
            "stable": False,
            "evidence_source": "mix",
        },
        "separation": {
            "beat_evidence": {
                "model_sha256": "b" * 64,
                "beats_ms": [110 + index * 500 for index in range(24)],
            },
        },
    }
    analysis_hash = _write_json(analysis_path, analysis)
    case_clock = dict(bpm=120, offset=-12, division=4, audio_shift_ms=22.5, stable=False)
    cases = [
        dict(id="clip-6k", analysis=str(analysis_path.relative_to(root)),
             analysis_sha256=analysis_hash, tempo=case_clock),
        dict(id="clip-10k", analysis=str(analysis_path.relative_to(root)),
             analysis_sha256=analysis_hash, tempo=case_clock),
    ]
    plan_path = root / "test-data" / "plan.json"
    plan = dict(schema="manialab-frozen-generation-v1", cases=cases)
    plan_hash = _write_json(plan_path, plan)
    for name in reestimate.SOURCE_FILES:
        _write_json(root / name, {"source": name})
    monkeypatch.setattr(reestimate, "ROOT", root)
    monkeypatch.setattr(reestimate, "FROZEN_PLAN_SHA256", plan_hash)
    return root, analysis_path, plan_path


def test_run_reestimates_unique_analysis_and_uses_frozen_beats(tmp_path, monkeypatch):
    root, analysis_path, plan_path = _fixture(tmp_path, monkeypatch)
    calls = []

    def estimate(attacks, duration, beats_ms):
        calls.append((len(attacks), duration, beats_ms))
        return {
            "bpm": 120,
            "offset": 10,
            "division": 4,
            "stable": True,
            "coverage": 0.91,
            "median_error_ms": 3.4,
            "confidence": 0.91,
            "tempo_ambiguity": None,
            "sparse_windows": [],
            "candidates": [{"bpm": 120, "offset": 10, "division": 4, "coverage": 0.91}],
            "precise_candidates": [],
        }

    monkeypatch.setattr(reestimate, "estimate_timing", estimate)
    output = root / "test-data" / "frozen-timing-reestimate-test"
    report = reestimate.run(plan_path, output)

    assert len(calls) == 1
    assert calls[0][0:2] == (1, 12.0)
    assert calls[0][2] == [110 + index * 500 for index in range(24)]
    assert report["plan_cases"] == 2
    assert report["unique_analyses"] == 1
    row = report["cases"][0]
    assert row["case_ids"] == ["clip-6k", "clip-10k"]
    assert row["reestimated"]["chart_offset"] == -12
    assert row["delta"]["chart_offset_ms_from_plan"] == 0
    assert row["frozen"]["beat_evidence"]["used"] is True
    saved_report = (output / "report.json").read_text(encoding="utf-8")
    saved_data = json.loads(saved_report)
    assert "private-note-id" not in saved_report
    assert "timing_attacks" not in saved_data["cases"][0]
    assert (output / "report.json").is_file()
    assert analysis_path.is_file()


def test_run_rejects_existing_output_before_writing(tmp_path, monkeypatch):
    root, _, plan_path = _fixture(tmp_path, monkeypatch)
    output = root / "test-data" / "frozen-timing-reestimate-existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        reestimate.run(plan_path, output)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_run_rejects_changed_frozen_analysis(tmp_path, monkeypatch):
    root, analysis_path, plan_path = _fixture(tmp_path, monkeypatch)
    analysis_path.write_text('{"duration":12}', encoding="utf-8")
    output = root / "test-data" / "frozen-timing-reestimate-changed"

    with pytest.raises(ValueError, match="Frozen analysis changed"):
        reestimate.run(plan_path, output)

    assert not output.exists()


def test_group_cases_rejects_conflicting_clocks_for_same_analysis(tmp_path, monkeypatch):
    root, _, plan_path = _fixture(tmp_path, monkeypatch)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["cases"][1]["tempo"]["bpm"] = 121
    _write_json(plan_path, plan)
    monkeypatch.setattr(reestimate, "FROZEN_PLAN_SHA256", hashlib.sha256(plan_path.read_bytes()).hexdigest())

    with pytest.raises(ValueError, match="same frozen clock"):
        reestimate._group_cases(reestimate.read(plan_path))
