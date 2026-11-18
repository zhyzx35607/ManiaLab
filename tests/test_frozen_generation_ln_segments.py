import hashlib
import json

import pytest

from scripts import summarize_frozen_generation_ln_segments as summarize


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _note(note_id, *, audio_start=None, start=100, end=200, evidence_end=None, voice="piano"):
    row = {
        "id": note_id,
        "voice": voice,
        "pitch": 60,
        "start": start,
        "end": end,
        "release_reason": "model_predicted_offset",
    }
    if audio_start is not None:
        row["audio_start"] = audio_start
    if evidence_end is not None:
        row["evidence_end"] = evidence_end
    return row


def _fixture(tmp_path, monkeypatch):
    root = tmp_path
    plan_path = root / "test-data" / "plan.json"
    replay_path = root / "test-data" / "replay" / "report.json"
    variant_path = root / "test-data" / "replay" / "sample-6k" / "variant.json"
    plan = {
        "schema": "manialab-frozen-generation-v1",
        "cases": [{"id": "sample-6k", "settings": {"keys": 6}}],
    }
    plan_hash = _write_json(plan_path, plan)
    notes = [
        _note("source-large", audio_start=999, end=700, evidence_end=670),
        _note("quant-large", audio_start=1000, end=1301, evidence_end=1270),
        _note("missing-snapshot", audio_start=1500, end=1700),
        _note("outside-segments", audio_start=2500, end=2100, evidence_end=2100),
        _note("missing-audio-start", end=1840, evidence_end=1800),
        _note("tap", audio_start=400, start=900, end=900, voice="drums"),
        _note("outside-tap", audio_start=2500, start=2100, end=2100, evidence_end=500),
    ]
    variant = {
        "notes": notes,
        "source_evidence": {
            "source-large": {
                "detected": {"start": 998, "end": 600, "release_reason": "model_predicted_offset"},
                "expressed": {"start": 998, "end": 650, "release_reason": "model_predicted_offset"},
            },
            "quant-large": {
                "detected": {"start": 1000, "end": 1200, "release_reason": "frame_decay"},
                "expressed": {"start": 1000, "end": 1170, "release_reason": "frame_decay"},
            },
        },
    }
    variant_hash = _write_json(variant_path, variant)
    replay = {
        "schema": "manialab-frozen-generation-results-v1",
        "plan_sha256": plan_hash,
        "passed": True,
        "cases": [{
            "id": "sample-6k",
            "keys": 6,
            "passed": True,
            "variant_sha256": variant_hash,
            "sections": [
                {"index": 0, "start_ms": 0, "end_ms": 1000},
                {"index": 1, "start_ms": 1000, "end_ms": 2000},
            ],
        }],
    }
    _write_json(replay_path, replay)
    monkeypatch.setattr(summarize, "ROOT", root)
    return root, plan_path, replay_path, variant_path


def _voice(report, section_index, voice):
    section = next(row for row in report["cases"][0]["sections"] if row["index"] == section_index)
    return next(row for row in section["voices"] if row["voice"] == voice)


def test_run_keeps_audio_tail_and_chart_quantization_clocks_separate(tmp_path, monkeypatch):
    root, plan_path, replay_path, _ = _fixture(tmp_path, monkeypatch)
    output = root / "test-data" / "ln-segment-report"

    report = summarize.run(plan_path, replay_path, output)

    assert report["summary"] == {
        "case_count": 1,
        "ln_count": 5,
        "assigned_ln_count": 3,
        "unassigned_ln_count": 2,
    }
    section_zero = _voice(report, 0, "piano")
    assert section_zero["ln_count"] == 1
    assert section_zero["source_tail_movement_audio_clock"] == {
        "measured_count": 1,
        "missing_count": 0,
        "later_count": 1,
        "earlier_count": 0,
        "unchanged_count": 0,
        "abs_gt_30ms_count": 1,
        "max_abs_ms": 50.0,
    }
    assert section_zero["quantization_movement_chart_clock"]["later_count"] == 1
    assert section_zero["quantization_movement_chart_clock"]["abs_gt_30ms_count"] == 0
    assert section_zero["large_movement_candidates"][0]["id"] == "source-large"
    assert set(section_zero["large_movement_candidates"][0]["movements"]) == {"source_tail"}

    section_one = _voice(report, 1, "piano")
    assert section_one["ln_count"] == 2
    assert section_one["source_tail_movement_audio_clock"]["measured_count"] == 1
    assert section_one["source_tail_movement_audio_clock"]["earlier_count"] == 1
    assert section_one["source_tail_movement_audio_clock"]["abs_gt_30ms_count"] == 0
    assert section_one["source_tail_movement_audio_clock"]["missing_count"] == 1
    assert section_one["quantization_movement_chart_clock"]["later_count"] == 1
    assert section_one["quantization_movement_chart_clock"]["abs_gt_30ms_count"] == 1
    assert section_one["quantization_movement_chart_clock"]["missing_count"] == 1
    assert section_one["missing_source_snapshot_count"] == 1
    assert section_one["missing_detected_end_count"] == 1
    assert section_one["missing_expressed_end_count"] == 1
    assert section_one["release_reasons"]["detected"] == {
        "frame_decay": 1,
        "unknown": 1,
    }
    assert [row["id"] for row in section_one["large_movement_candidates"]] == ["quant-large"]
    assert section_one["large_movement_candidates"][0]["movements"] == {
        "quantization": {"direction": "later", "delta_ms": 31.0},
    }
    assert report["cases"][0]["unassigned_ln"] == [
        {
            "id": "missing-audio-start",
            "voice": "piano",
            "pitch": 60,
            "audio_start_ms": None,
            "reason": "missing_audio_start",
        },
        {
            "id": "outside-segments",
            "voice": "piano",
            "pitch": 60,
            "audio_start_ms": 2500.0,
            "reason": "outside_structural_segments",
        },
    ]
    assert [
        row["id"] for row in report["cases"][0]["unassigned_large_movement_candidates"]
    ] == ["missing-audio-start"]
    assert report["limitations"]["physical_key_release_verified"] is False
    assert (output / "report.json").is_file()


def test_exact_segment_boundary_belongs_to_following_half_open_section(tmp_path, monkeypatch):
    root, plan_path, replay_path, _ = _fixture(tmp_path, monkeypatch)
    report = summarize.run(plan_path, replay_path, root / "test-data" / "boundary-report")

    assert _voice(report, 0, "piano")["ln_count"] == 1
    assert _voice(report, 1, "piano")["ln_count"] == 2


def test_plan_and_variant_hash_mismatches_are_rejected_before_output(tmp_path, monkeypatch):
    root, plan_path, replay_path, variant_path = _fixture(tmp_path, monkeypatch)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["plan_sha256"] = "0" * 64
    _write_json(replay_path, replay)
    output = root / "test-data" / "bad-plan-report"

    with pytest.raises(ValueError, match="does not match the frozen plan hash"):
        summarize.run(plan_path, replay_path, output)
    assert not output.exists()

    _, plan_path, replay_path, variant_path = _fixture(tmp_path / "variant", monkeypatch)
    with variant_path.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    output = tmp_path / "variant" / "test-data" / "bad-variant-report"

    with pytest.raises(ValueError, match="variant hash mismatch"):
        summarize.run(plan_path, replay_path, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    root, plan_path, replay_path, _ = _fixture(tmp_path, monkeypatch)
    output = root / "test-data" / "existing-report"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Preserve previous runs"):
        summarize.run(plan_path, replay_path, output)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_overlapping_structural_sections_are_rejected(tmp_path, monkeypatch):
    root, plan_path, replay_path, _ = _fixture(tmp_path, monkeypatch)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["cases"][0]["sections"][1]["start_ms"] = 999
    _write_json(replay_path, replay)
    plan = summarize.read_json(plan_path)
    variants = {
        "sample-6k": summarize.read_json(
            root / "test-data" / "replay" / "sample-6k" / "variant.json"
        )
    }

    with pytest.raises(ValueError, match="Overlapping or unordered structural sections"):
        summarize.build_report(
            plan,
            replay,
            variants,
            summarize.sha256_file(plan_path),
            summarize.sha256_file(replay_path),
        )


def test_duplicate_plan_case_ids_are_rejected(tmp_path, monkeypatch):
    root, plan_path, replay_path, _ = _fixture(tmp_path, monkeypatch)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["cases"].append(dict(plan["cases"][0]))
    plan_hash = _write_json(plan_path, plan)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["plan_sha256"] = plan_hash
    _write_json(replay_path, replay)

    with pytest.raises(ValueError, match="duplicate case ID"):
        summarize.run(
            plan_path,
            replay_path,
            root / "test-data" / "duplicate-case-report",
        )


def test_duplicate_variant_note_ids_are_rejected(tmp_path, monkeypatch):
    root, plan_path, replay_path, variant_path = _fixture(tmp_path, monkeypatch)
    variant = json.loads(variant_path.read_text(encoding="utf-8"))
    variant["notes"].append(dict(variant["notes"][0]))
    variant_hash = _write_json(variant_path, variant)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["cases"][0]["variant_sha256"] = variant_hash
    _write_json(replay_path, replay)

    with pytest.raises(ValueError, match="duplicate note ID"):
        summarize.run(
            plan_path,
            replay_path,
            root / "test-data" / "duplicate-note-report",
        )
