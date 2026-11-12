"""Validate exported vocal-review labels and report coverage without scoring quality."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "test-data" / "sakura-benchmark" / "vocal-phrase-audio-review-v1"
MANIFEST_SCHEMA = "manialab-vocal-phrase-audio-review-v1"
REPORT_SCHEMA = "manialab-vocal-phrase-audio-review-report-v1"
LABEL_SCHEMA = "manialab-vocal-phrase-review-v1"
SUMMARY_SCHEMA = "manialab-vocal-phrase-review-summary-v1"
SOURCE_LABELS = (
    "clear_voice",
    "possible_voice",
    "other_or_bleed",
    "unclear",
)
BOUNDARY_LABELS = (
    "reasonable",
    "start_review",
    "end_review",
    "both_review",
    "unclear",
)
BOUNDARY_FOLLOW_UP = frozenset({"start_review", "end_review", "both_review", "unclear"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(bundle_path: Path = DEFAULT_BUNDLE) -> tuple[dict, list[dict]]:
    bundle = Path(bundle_path).resolve()
    manifest_path = bundle / "manifest.json"
    report_path = bundle / "report.json"
    if not bundle.is_dir():
        raise ValueError(f"Review bundle does not exist: {bundle}")

    manifest = _read_json(manifest_path)
    report = _read_json(report_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Unexpected review manifest schema")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("Unexpected review report schema")

    phrases = manifest.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise ValueError("Review manifest has no candidates")
    candidate_ids = []
    for phrase in phrases:
        if not isinstance(phrase, dict) or not isinstance(phrase.get("id"), str) or not phrase["id"]:
            raise ValueError("Review manifest contains a candidate without an ID")
        candidate_ids.append(phrase["id"])
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Review manifest contains duplicate candidate IDs")
    if report.get("candidate_count") != len(phrases):
        raise ValueError("Review report candidate count does not match the manifest")
    if report.get("clip_count") != len(phrases) * 2:
        raise ValueError("Review report clip count does not match the manifest")

    identity = report.get("input_identity")
    if not isinstance(identity, dict) or manifest.get("sources") != identity:
        raise ValueError("Review manifest source identity does not match the report")
    analysis_hash = identity.get("analysis_sha256")
    vocal_hash = identity.get("vocal_pcm_sha256")
    if not all(
        isinstance(value, str) and SHA256_RE.fullmatch(value)
        for value in (analysis_hash, vocal_hash)
    ):
        raise ValueError("Review report has invalid source identity hashes")
    expected_key = hashlib.sha256((analysis_hash + vocal_hash).encode("ascii")).hexdigest()[:20]
    if manifest.get("review_key") != expected_key:
        raise ValueError("Review key does not match the frozen source identity")

    artifact_hashes = report.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or "manifest.json" not in artifact_hashes:
        raise ValueError("Review report has no manifest artifact hash")
    for relative_name, expected_hash in artifact_hashes.items():
        relative = PurePosixPath(relative_name) if isinstance(relative_name, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or not isinstance(expected_hash, str)
            or not SHA256_RE.fullmatch(expected_hash)
        ):
            raise ValueError("Review report contains an invalid artifact path or hash")
        artifact_path = bundle.joinpath(*relative.parts).resolve()
        if not artifact_path.is_relative_to(bundle) or not artifact_path.is_file():
            raise ValueError(f"Review artifact is missing or outside the bundle: {relative_name}")
        if _sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"Review artifact hash mismatch: {relative_name}")

    return manifest, phrases


def summarize_labels(labels_path: Path, bundle_path: Path = DEFAULT_BUNDLE) -> dict:
    manifest, phrases = validate_bundle(bundle_path)
    labels = _read_json(labels_path)
    if labels.get("schema") != LABEL_SCHEMA:
        raise ValueError("Unexpected exported label schema")
    if labels.get("review_key") != manifest["review_key"]:
        raise ValueError("Exported labels belong to a different frozen review")

    exported_at = labels.get("exported_at")
    if not isinstance(exported_at, str) or not exported_at:
        raise ValueError("Exported labels have no export timestamp")
    try:
        timestamp = datetime.fromisoformat(exported_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Exported labels have an invalid export timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("Export timestamp must include a timezone")

    rows = labels.get("review")
    if not isinstance(rows, list):
        raise ValueError("Exported labels have no review list")
    expected_ids = [phrase["id"] for phrase in phrases]
    expected_set = set(expected_ids)
    by_id = {}
    required_fields = {"id", "source", "boundary", "note"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required_fields:
            raise ValueError("Each review row must contain exactly id, source, boundary, and note")
        candidate_id = row["id"]
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Exported review contains an invalid candidate ID")
        if candidate_id not in expected_set:
            raise ValueError(f"Exported review contains an unknown candidate ID: {candidate_id}")
        if candidate_id in by_id:
            raise ValueError(f"Exported review contains a duplicate candidate ID: {candidate_id}")
        if row["source"] not in ("", *SOURCE_LABELS):
            raise ValueError(f"Invalid source label for {candidate_id}: {row['source']!r}")
        if row["boundary"] not in ("", *BOUNDARY_LABELS):
            raise ValueError(f"Invalid boundary label for {candidate_id}: {row['boundary']!r}")
        if not isinstance(row["note"], str):
            raise ValueError(f"Invalid note for {candidate_id}")
        by_id[candidate_id] = row

    missing_ids = [candidate_id for candidate_id in expected_ids if candidate_id not in by_id]
    if missing_ids:
        raise ValueError(f"Exported review is missing {len(missing_ids)} candidate rows")

    source_counts = Counter(row["source"] for row in by_id.values() if row["source"])
    boundary_counts = Counter(row["boundary"] for row in by_id.values() if row["boundary"])
    incomplete_ids = [
        candidate_id for candidate_id in expected_ids
        if not by_id[candidate_id]["source"] or not by_id[candidate_id]["boundary"]
    ]
    follow_up_ids = [
        candidate_id for candidate_id in expected_ids
        if by_id[candidate_id]["boundary"] in BOUNDARY_FOLLOW_UP
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "review_key": manifest["review_key"],
        "exported_at": timestamp.isoformat(),
        "candidate_count": len(expected_ids),
        "source_label_counts": {label: source_counts[label] for label in SOURCE_LABELS},
        "boundary_label_counts": {label: boundary_counts[label] for label in BOUNDARY_LABELS},
        "source_labeled_count": sum(bool(row["source"]) for row in by_id.values()),
        "boundary_labeled_count": sum(bool(row["boundary"]) for row in by_id.values()),
        "fully_labeled_count": len(expected_ids) - len(incomplete_ids),
        "complete_label_coverage": not incomplete_ids,
        "incomplete_ids": incomplete_ids,
        "boundary_follow_up_ids": follow_up_ids,
        "notes_present": sum(bool(row["note"].strip()) for row in by_id.values()),
        "bundle_artifacts_verified": True,
        "musical_acceptance": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Preserve previous label summaries: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Summary output directory does not exist: {output.parent}")
    summary = summarize_labels(args.labels, args.bundle)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "candidate_count": summary["candidate_count"],
        "fully_labeled_count": summary["fully_labeled_count"],
        "complete_label_coverage": summary["complete_label_coverage"],
        "musical_acceptance": summary["musical_acceptance"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
