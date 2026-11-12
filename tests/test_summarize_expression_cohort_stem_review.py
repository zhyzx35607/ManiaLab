import hashlib
import json
from pathlib import Path
import wave

import pytest

from scripts.summarize_expression_cohort_stem_review import (
    LABEL_SCHEMA,
    SUMMARY_SCHEMA,
    summarize_labels,
    validate_bundle,
)


TRACK_ORDER = ("mix", "drums", "bass", "other", "vocals")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"\x00\x00" * 8)


def _write_bundle(root: Path, case_ids: tuple[str, ...] = ("case-a", "case-b")):
    bundle = root / "bundle"
    (bundle / "clips").mkdir(parents=True)
    cases = []
    group_ids = []
    for case_id in case_ids:
        clips = {}
        for track in TRACK_ORDER:
            relative = f"clips/{case_id}-{track}.wav"
            _write_wav(bundle / relative)
            clips[track] = relative
        cases.append({
            "case_id": case_id,
            "group": f"group-{case_id}",
            "clips": clips,
        })
        group_ids.append({"group": f"group-{case_id}"})

    selection_sha = "a" * 64
    run_sha = "b" * 64
    builder_sha = "c" * 64
    protocol_sha = "d" * 64
    review_key = hashlib.sha256(
        f"{selection_sha}:{run_sha}:{builder_sha}:{protocol_sha}".encode("ascii")
    ).hexdigest()[:24]
    manifest = {
        "schema": "manialab-expression-stem-review-manifest-v1",
        "review_key": review_key,
        "cohort": {
            "selection_sha256": selection_sha,
            "split": "train",
            "group_count": len(cases),
            "case_count": len(cases),
            "groups": group_ids,
        },
        "separation": {"run_identity_sha256": run_sha},
        "selection": {"cases": cases},
        "playback": {
            "sample_rate": 22050,
            "channels": 1,
            "format": "PCM16",
            "track_order": list(TRACK_ORDER),
        },
        "review": {
            "human_listening_completed": False,
            "musical_acceptance": False,
            "instrument_identity_established": False,
            "ln_tail_ground_truth_established": False,
        },
        "builder_sha256": builder_sha,
        "protocol_sha256": protocol_sha,
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (bundle / "index.html").write_text("review page", encoding="utf-8")
    (bundle / "README.md").write_text("review notes", encoding="utf-8")
    (bundle / "protocol-snapshot.md").write_text("protocol", encoding="utf-8")
    (bundle / "builder-snapshot.py").write_text("# builder", encoding="utf-8")
    artifact_paths = [
        path for path in bundle.rglob("*")
        if path.is_file() and path.name != "report.json"
    ]
    report = {
        "schema": "manialab-expression-stem-review-report-v1",
        "selection_sha256": selection_sha,
        "run_identity_sha256": run_sha,
        "case_count": len(cases),
        "source_track_count": len(TRACK_ORDER),
        "clip_count": len(cases) * len(TRACK_ORDER),
        "validation_test_audio_read": False,
        "model_inference_rerun": False,
        "human_listening_completed": False,
        "musical_acceptance": False,
        "artifact_sha256": {
            path.relative_to(bundle).as_posix(): _sha256(path)
            for path in artifact_paths
        },
    }
    (bundle / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bundle, review_key


def _labels(review_key: str, rows: list[dict]) -> dict:
    return {
        "schema": LABEL_SCHEMA,
        "review_key": review_key,
        "exported_at": "2026-11-13T12:00:00.000Z",
        "review": rows,
    }


def _row(
    case_id: str,
    attack: str = "",
    track_evidence: list[str] | None = None,
    chart_timing: str = "",
    tail_evidence: str = "",
    note: str = "",
) -> dict:
    return {
        "case_id": case_id,
        "attack": attack,
        "track_evidence": [] if track_evidence is None else track_evidence,
        "chart_timing": chart_timing,
        "tail_evidence": tail_evidence,
        "note": note,
    }


def test_summary_counts_complete_labels_without_returning_note_text(tmp_path):
    bundle, review_key = _write_bundle(tmp_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_labels(review_key, [
        _row("case-a", "clear", ["drums", "other"], "rough", "past_chart_end"),
        _row("case-b", "possible", ["unclear"], "late", "unclear", "private note"),
    ])), encoding="utf-8")

    summary = summarize_labels(labels_path, bundle)

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["case_count"] == 2
    assert summary["attack_label_counts"] == {
        "clear": 1, "possible": 1, "not_distinct": 0, "unclear": 0,
    }
    assert summary["track_evidence_counts"]["drums"] == 1
    assert summary["track_evidence_counts"]["other"] == 1
    assert summary["track_evidence_counts"]["unclear"] == 1
    assert summary["chart_timing_counts"]["late"] == 1
    assert summary["tail_evidence_counts"]["past_chart_end"] == 1
    assert summary["fully_labeled_count"] == 2
    assert summary["complete_label_coverage"] is True
    assert summary["attack_follow_up_case_ids"] == ["case-b"]
    assert summary["chart_timing_follow_up_case_ids"] == ["case-b"]
    assert summary["tail_follow_up_case_ids"] == ["case-b"]
    assert summary["notes_present"] == 1
    assert "private note" not in json.dumps(summary)
    assert summary["bundle_artifacts_verified"] is True
    assert summary["musical_acceptance"] is False


def test_summary_marks_missing_fields_as_incomplete(tmp_path):
    bundle, review_key = _write_bundle(tmp_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_labels(review_key, [
        _row("case-a", "clear", ["vocals"], "rough", "not_applicable"),
        _row("case-b", note="not reviewed"),
    ])), encoding="utf-8")

    summary = summarize_labels(labels_path, bundle)

    assert summary["fully_labeled_count"] == 1
    assert summary["complete_label_coverage"] is False
    assert summary["incomplete_case_ids"] == ["case-b"]
    assert summary["incomplete_fields"] == {
        "case-b": ["attack", "track_evidence", "chart_timing", "tail_evidence"]
    }
    assert summary["notes_present"] == 1


@pytest.mark.parametrize(
    "rows",
    [
        [_row("case-a"), _row("case-a")],
        [_row("case-a"), _row("unknown")],
        [_row("case-a")],
    ],
)
def test_summary_rejects_duplicate_unknown_or_missing_rows(tmp_path, rows):
    bundle, review_key = _write_bundle(tmp_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_labels(review_key, rows)), encoding="utf-8")

    with pytest.raises(ValueError):
        summarize_labels(labels_path, bundle)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema": "other"}, "schema"),
        ({"review_key": "wrong"}, "different frozen review"),
        ({"exported_at": "2026-11-13T12:00:00"}, "timezone"),
    ],
)
def test_summary_rejects_wrong_schema_key_or_timestamp(tmp_path, change, message):
    bundle, review_key = _write_bundle(tmp_path)
    payload = _labels(review_key, [_row("case-a"), _row("case-b")])
    payload.update(change)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        summarize_labels(labels_path, bundle)


@pytest.mark.parametrize(
    "row",
    [
        {**_row("case-a"), "attack": "probably"},
        {**_row("case-a"), "chart_timing": "maybe"},
        {**_row("case-a"), "tail_evidence": "likely"},
        {**_row("case-a"), "track_evidence": ["drums", "drums"]},
        {**_row("case-a"), "track_evidence": ["mix"]},
        {key: value for key, value in _row("case-a").items() if key != "note"},
    ],
)
def test_summary_rejects_invalid_labels_or_row_shape(tmp_path, row):
    bundle, review_key = _write_bundle(tmp_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_labels(review_key, [
        row,
        _row("case-b"),
    ])), encoding="utf-8")

    with pytest.raises(ValueError):
        summarize_labels(labels_path, bundle)


def test_validate_bundle_rejects_changed_artifacts_and_path_traversal(tmp_path):
    bundle, _ = _write_bundle(tmp_path)
    manifest, cases = validate_bundle(bundle)
    assert manifest["review_key"]
    assert len(cases) == 2

    (bundle / "index.html").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bundle(bundle)

    bundle, _ = _write_bundle(tmp_path / "traversal")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"]["cases"][0]["clips"]["mix"] = "clips\\..\\outside.wav"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid artifact path"):
        validate_bundle(bundle)
