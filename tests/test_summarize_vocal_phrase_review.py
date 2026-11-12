import hashlib
import json
from pathlib import Path

import pytest

from scripts.summarize_vocal_phrase_review import (
    LABEL_SCHEMA,
    summarize_labels,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path, candidate_ids: tuple[str, ...] = ("candidate-a", "candidate-b")):
    bundle = root / "bundle"
    bundle.mkdir()
    identity = {
        "analysis_sha256": "a" * 64,
        "vocal_pcm_sha256": "b" * 64,
    }
    review_key = hashlib.sha256(
        (identity["analysis_sha256"] + identity["vocal_pcm_sha256"]).encode("ascii")
    ).hexdigest()[:20]
    (bundle / "index.html").write_text("fixed review page", encoding="utf-8")
    manifest = {
        "schema": "manialab-vocal-phrase-audio-review-v1",
        "review_key": review_key,
        "sources": identity,
        "phrases": [{"id": candidate_id} for candidate_id in candidate_ids],
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "schema": "manialab-vocal-phrase-audio-review-report-v1",
        "input_identity": identity,
        "candidate_count": len(candidate_ids),
        "clip_count": len(candidate_ids) * 2,
        "artifact_sha256": {
            "manifest.json": _sha256(bundle / "manifest.json"),
            "index.html": _sha256(bundle / "index.html"),
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
        "exported_at": "2026-11-12T20:00:00.000Z",
        "review": rows,
    }


def test_summary_counts_labels_and_preserves_unresolved_work(tmp_path):
    bundle, review_key = _write_bundle(tmp_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_labels(review_key, [
        {"id": "candidate-a", "source": "clear_voice", "boundary": "reasonable", "note": ""},
        {"id": "candidate-b", "source": "unclear", "boundary": "", "note": "试听环境有噪声"},
    ])), encoding="utf-8")

    summary = summarize_labels(labels_path, bundle)

    assert summary["candidate_count"] == 2
    assert summary["source_label_counts"] == {
        "clear_voice": 1,
        "possible_voice": 0,
        "other_or_bleed": 0,
        "unclear": 1,
    }
    assert summary["boundary_label_counts"]["reasonable"] == 1
    assert summary["fully_labeled_count"] == 1
    assert summary["complete_label_coverage"] is False
    assert summary["incomplete_ids"] == ["candidate-b"]
    assert summary["notes_present"] == 1
    assert summary["bundle_artifacts_verified"] is True
    assert summary["musical_acceptance"] is False


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"id": "candidate-a", "source": "", "boundary": "", "note": ""},
            {"id": "candidate-a", "source": "", "boundary": "", "note": ""},
        ],
        [
            {"id": "candidate-a", "source": "", "boundary": "", "note": ""},
            {"id": "unknown", "source": "", "boundary": "", "note": ""},
        ],
        [
            {"id": "candidate-a", "source": "", "boundary": "", "note": ""},
        ],
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
    ],
)
def test_summary_rejects_wrong_schema_or_review_key(tmp_path, change, message):
    bundle, review_key = _write_bundle(tmp_path)
    payload = _labels(review_key, [
        {"id": "candidate-a", "source": "", "boundary": "", "note": ""},
        {"id": "candidate-b", "source": "", "boundary": "", "note": ""},
    ])
    payload.update(change)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        summarize_labels(labels_path, bundle)


def test_summary_rejects_invalid_label_values_and_changed_bundle_artifacts(tmp_path):
    bundle, review_key = _write_bundle(tmp_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_labels(review_key, [
        {"id": "candidate-a", "source": "probably", "boundary": "", "note": ""},
        {"id": "candidate-b", "source": "", "boundary": "", "note": ""},
    ])), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid source label"):
        summarize_labels(labels_path, bundle)

    (bundle / "index.html").write_text("changed page", encoding="utf-8")
    labels_path.write_text(json.dumps(_labels(review_key, [
        {"id": "candidate-a", "source": "", "boundary": "", "note": ""},
        {"id": "candidate-b", "source": "", "boundary": "", "note": ""},
    ])), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        summarize_labels(labels_path, bundle)
