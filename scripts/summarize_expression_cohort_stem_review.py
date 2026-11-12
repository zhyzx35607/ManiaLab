"""Validate exported stem-review labels and summarize coverage without scoring quality."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import wave


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "test-data" / "expression-audio-cohort-v3-stem-review-v1"
MANIFEST_SCHEMA = "manialab-expression-stem-review-manifest-v1"
REPORT_SCHEMA = "manialab-expression-stem-review-report-v1"
LABEL_SCHEMA = "manialab-expression-stem-review-v1"
SUMMARY_SCHEMA = "manialab-expression-stem-review-summary-v1"
TRACK_ORDER = ("mix", "drums", "bass", "other", "vocals")
ATTACK_LABELS = ("clear", "possible", "not_distinct", "unclear")
CHART_TIMING_LABELS = ("rough", "early", "late", "no_attack", "unclear")
TAIL_EVIDENCE_LABELS = (
    "past_chart_end",
    "before_chart_end",
    "unclear",
    "not_applicable",
)
TRACK_EVIDENCE_LABELS = ("drums", "bass", "other", "vocals", "unclear")
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
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_bundle_path(bundle: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Review bundle contains an invalid artifact path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("Review bundle contains an invalid artifact path")
    path = bundle.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(bundle) or not path.is_file():
        raise ValueError(f"Review artifact is missing or outside the bundle: {value}")
    return value, path


def validate_bundle(bundle_path: Path = DEFAULT_BUNDLE) -> tuple[dict, list[dict]]:
    bundle = Path(bundle_path).resolve()
    if not bundle.is_dir():
        raise ValueError(f"Review bundle does not exist: {bundle}")

    manifest = _read_json(bundle / "manifest.json")
    report = _read_json(bundle / "report.json")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Unexpected review manifest schema")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("Unexpected review report schema")

    cohort = manifest.get("cohort")
    separation = manifest.get("separation")
    selection = manifest.get("selection")
    playback = manifest.get("playback")
    review_state = manifest.get("review")
    if not all(isinstance(value, dict) for value in (
        cohort, separation, selection, playback, review_state
    )):
        raise ValueError("Review manifest is missing a required section")

    identity_parts = (
        cohort.get("selection_sha256"),
        separation.get("run_identity_sha256"),
        manifest.get("builder_sha256"),
        manifest.get("protocol_sha256"),
    )
    if not all(
        isinstance(value, str) and SHA256_RE.fullmatch(value)
        for value in identity_parts
    ):
        raise ValueError("Review manifest has invalid identity hashes")
    selection_sha, run_sha, builder_sha, protocol_sha = identity_parts
    expected_key = hashlib.sha256(
        f"{selection_sha}:{run_sha}:{builder_sha}:{protocol_sha}".encode("ascii")
    ).hexdigest()[:24]
    if manifest.get("review_key") != expected_key:
        raise ValueError("Review key does not match the frozen bundle identity")
    if report.get("selection_sha256") != selection_sha:
        raise ValueError("Review report selection identity does not match the manifest")
    if report.get("run_identity_sha256") != run_sha:
        raise ValueError("Review report separation identity does not match the manifest")
    if cohort.get("split") != "train":
        raise ValueError("Review bundle is not restricted to the training split")
    if playback.get("track_order") != list(TRACK_ORDER):
        raise ValueError("Review manifest has an unexpected playback track order")
    if (
        playback.get("sample_rate") != 22050
        or playback.get("channels") != 1
        or playback.get("format") != "PCM16"
    ):
        raise ValueError("Review manifest has an unexpected audio format")
    if any(review_state.get(name) is not False for name in (
        "human_listening_completed",
        "musical_acceptance",
        "instrument_identity_established",
        "ln_tail_ground_truth_established",
    )):
        raise ValueError("Frozen review manifest must not claim completed listening or acceptance")
    if report.get("validation_test_audio_read") is not False:
        raise ValueError("Review report does not confirm validation/test audio exclusion")
    if report.get("model_inference_rerun") is not False:
        raise ValueError("Review report does not confirm inference was not rerun")
    if report.get("human_listening_completed") is not False:
        raise ValueError("Frozen review report must not claim completed listening")
    if report.get("musical_acceptance") is not False:
        raise ValueError("Frozen review report must not claim musical acceptance")

    cases = selection.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Review manifest has no cases")
    case_ids: list[str] = []
    groups: list[str] = []
    clip_paths: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Review manifest contains an invalid case")
        case_id = case.get("case_id")
        group = case.get("group")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Review manifest contains a case without an ID")
        if not isinstance(group, str) or not group:
            raise ValueError(f"Review manifest contains a case without a group: {case_id}")
        case_ids.append(case_id)
        groups.append(group)
        clips = case.get("clips")
        if not isinstance(clips, dict) or set(clips) != set(TRACK_ORDER):
            raise ValueError(f"Review case has an invalid clip map: {case_id}")
        for track in TRACK_ORDER:
            relative_name, _ = _safe_bundle_path(bundle, clips[track])
            clip_paths.append(relative_name)

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Review manifest contains duplicate case IDs")
    if len(groups) != len(set(groups)):
        raise ValueError("Review manifest contains more than one case for a song group")
    if len(clip_paths) != len(set(clip_paths)):
        raise ValueError("Review manifest contains duplicate clip paths")
    if cohort.get("case_count") != len(cases) or cohort.get("group_count") != len(cases):
        raise ValueError("Review manifest case and group counts do not match")
    if report.get("case_count") != len(cases):
        raise ValueError("Review report case count does not match the manifest")
    if report.get("source_track_count") != len(TRACK_ORDER):
        raise ValueError("Review report track count does not match the manifest")
    if report.get("clip_count") != len(cases) * len(TRACK_ORDER):
        raise ValueError("Review report clip count does not match the manifest")

    artifact_hashes = report.get("artifact_sha256")
    required_paths = {
        "manifest.json",
        "index.html",
        "README.md",
        "protocol-snapshot.md",
        "builder-snapshot.py",
        *clip_paths,
    }
    if not isinstance(artifact_hashes, dict) or not required_paths <= set(artifact_hashes):
        raise ValueError("Review report is missing registered bundle artifacts")
    for relative_name, expected_hash in artifact_hashes.items():
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            raise ValueError("Review report contains an invalid artifact hash")
        _, artifact_path = _safe_bundle_path(bundle, relative_name)
        if _sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"Review artifact hash mismatch: {relative_name}")

    for case in cases:
        for track, relative_name in case["clips"].items():
            _, clip_path = _safe_bundle_path(bundle, relative_name)
            if relative_name not in artifact_hashes:
                raise ValueError(f"Review clip has no registered hash: {relative_name}")
            try:
                with wave.open(str(clip_path), "rb") as clip:
                    valid_audio = (
                        clip.getnchannels() == 1
                        and clip.getsampwidth() == 2
                        and clip.getframerate() == 22050
                        and clip.getnframes() > 0
                    )
            except (OSError, wave.Error) as exc:
                raise ValueError(f"Review clip is not a readable WAV: {relative_name}") from exc
            if not valid_audio:
                raise ValueError(
                    f"Review clip has unexpected WAV parameters: {relative_name} ({track})"
                )

    return manifest, cases


def summarize_labels(labels_path: Path, bundle_path: Path = DEFAULT_BUNDLE) -> dict:
    manifest, cases = validate_bundle(bundle_path)
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
    expected_ids = [case["case_id"] for case in cases]
    expected_set = set(expected_ids)
    by_id: dict[str, dict] = {}
    required_fields = {
        "case_id",
        "attack",
        "track_evidence",
        "chart_timing",
        "tail_evidence",
        "note",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != required_fields:
            raise ValueError("Each review row must contain exactly the six exported fields")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Exported review contains an invalid case ID")
        if case_id not in expected_set:
            raise ValueError(f"Exported review contains an unknown case ID: {case_id}")
        if case_id in by_id:
            raise ValueError(f"Exported review contains a duplicate case ID: {case_id}")

        for field, choices in (
            ("attack", ("", *ATTACK_LABELS)),
            ("chart_timing", ("", *CHART_TIMING_LABELS)),
            ("tail_evidence", ("", *TAIL_EVIDENCE_LABELS)),
        ):
            if not isinstance(row[field], str) or row[field] not in choices:
                raise ValueError(f"Invalid {field} label for {case_id}")
        evidence = row["track_evidence"]
        if (
            not isinstance(evidence, list)
            or any(not isinstance(value, str) for value in evidence)
            or len(evidence) != len(set(evidence))
            or any(value not in TRACK_EVIDENCE_LABELS for value in evidence)
        ):
            raise ValueError(f"Invalid track_evidence labels for {case_id}")
        if not isinstance(row["note"], str):
            raise ValueError(f"Invalid note for {case_id}")
        by_id[case_id] = row

    missing_ids = [case_id for case_id in expected_ids if case_id not in by_id]
    if missing_ids:
        raise ValueError(f"Exported review is missing {len(missing_ids)} case rows")

    attack_counts = Counter(row["attack"] for row in by_id.values() if row["attack"])
    chart_timing_counts = Counter(
        row["chart_timing"] for row in by_id.values() if row["chart_timing"]
    )
    tail_evidence_counts = Counter(
        row["tail_evidence"] for row in by_id.values() if row["tail_evidence"]
    )
    track_evidence_counts = Counter(
        value for row in by_id.values() for value in row["track_evidence"]
    )
    incomplete_fields: dict[str, list[str]] = {}
    attack_follow_up_ids: list[str] = []
    chart_timing_follow_up_ids: list[str] = []
    tail_follow_up_ids: list[str] = []
    for case_id in expected_ids:
        row = by_id[case_id]
        missing = [
            field for field, is_filled in (
                ("attack", bool(row["attack"])),
                ("track_evidence", bool(row["track_evidence"])),
                ("chart_timing", bool(row["chart_timing"])),
                ("tail_evidence", bool(row["tail_evidence"])),
            )
            if not is_filled
        ]
        if missing:
            incomplete_fields[case_id] = missing
        if row["attack"] in {"possible", "unclear"}:
            attack_follow_up_ids.append(case_id)
        if row["chart_timing"] in {"early", "late", "no_attack", "unclear"}:
            chart_timing_follow_up_ids.append(case_id)
        if row["tail_evidence"] == "unclear":
            tail_follow_up_ids.append(case_id)

    fully_labeled_count = len(expected_ids) - len(incomplete_fields)
    return {
        "schema": SUMMARY_SCHEMA,
        "review_key": manifest["review_key"],
        "exported_at": timestamp.isoformat(),
        "case_count": len(expected_ids),
        "attack_label_counts": {
            label: attack_counts[label] for label in ATTACK_LABELS
        },
        "track_evidence_counts": {
            label: track_evidence_counts[label] for label in TRACK_EVIDENCE_LABELS
        },
        "chart_timing_counts": {
            label: chart_timing_counts[label] for label in CHART_TIMING_LABELS
        },
        "tail_evidence_counts": {
            label: tail_evidence_counts[label] for label in TAIL_EVIDENCE_LABELS
        },
        "attack_labeled_count": sum(bool(row["attack"]) for row in by_id.values()),
        "track_evidence_labeled_count": sum(
            bool(row["track_evidence"]) for row in by_id.values()
        ),
        "chart_timing_labeled_count": sum(
            bool(row["chart_timing"]) for row in by_id.values()
        ),
        "tail_evidence_labeled_count": sum(
            bool(row["tail_evidence"]) for row in by_id.values()
        ),
        "fully_labeled_count": fully_labeled_count,
        "complete_label_coverage": fully_labeled_count == len(expected_ids),
        "incomplete_case_ids": list(incomplete_fields),
        "incomplete_fields": incomplete_fields,
        "attack_follow_up_case_ids": attack_follow_up_ids,
        "chart_timing_follow_up_case_ids": chart_timing_follow_up_ids,
        "tail_follow_up_case_ids": tail_follow_up_ids,
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
        "case_count": summary["case_count"],
        "fully_labeled_count": summary["fully_labeled_count"],
        "complete_label_coverage": summary["complete_label_coverage"],
        "musical_acceptance": summary["musical_acceptance"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
