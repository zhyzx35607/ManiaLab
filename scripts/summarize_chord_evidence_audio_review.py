"""Validate and summarize labels exported from the chord-evidence listening page."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABEL_SCHEMA = "manialab-chord-evidence-audio-review-labels-v1"
VALID_AUDIBILITY = {"", "audible", "not_audible", "uncertain"}


def summarize_review_labels(
    manifest: dict,
    labels: dict,
    manifest_sha256: str,
) -> dict:
    if manifest.get("schema") != "manialab-chord-evidence-audio-review-v1":
        raise ValueError("Unsupported review manifest schema")
    if labels.get("schema") != LABEL_SCHEMA:
        raise ValueError("Unsupported label export schema")
    if labels.get("review_key") != manifest.get("review_key"):
        raise ValueError("Review key does not match the manifest")
    if labels.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Manifest SHA-256 does not match the label export")
    exported_at = labels.get("exported_at")
    try:
        parsed_time = datetime.fromisoformat(str(exported_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Export timestamp is invalid") from exc
    if parsed_time.tzinfo is None:
        raise ValueError("Export timestamp must include a timezone")

    events = {}
    case_ids = set()
    for case in manifest.get("cases", []):
        case_id = case.get("case_id")
        if not case_id or case_id in case_ids:
            raise ValueError("Manifest has a missing or duplicate case ID")
        case_ids.add(case_id)
        for event in case.get("attacks", []):
            event_id = event.get("id")
            if not event_id or event_id in events:
                raise ValueError("Manifest has a missing or duplicate attack event ID")
            events[event_id] = event

    exported_events = labels.get("events")
    if not isinstance(exported_events, dict):
        raise ValueError("Label export must contain an events object")
    expected_ids = set(events)
    actual_ids = set(exported_events)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unknown = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Event IDs do not match the manifest; missing={missing[:3]}, unknown={unknown[:3]}"
        )

    case_notes = labels.get("caseNotes")
    if not isinstance(case_notes, dict) or set(case_notes) != case_ids:
        raise ValueError("Window notes do not match the manifest case set")
    for note in case_notes.values():
        if not isinstance(note, str) or len(note) > 5000:
            raise ValueError("Window note is invalid or too long")

    totals = Counter()
    by_voice: dict[str, Counter] = defaultdict(Counter)
    by_direction: dict[str, Counter] = defaultdict(Counter)
    for event_id, value in exported_events.items():
        if not isinstance(value, dict):
            raise ValueError(f"Event label must be an object: {event_id}")
        status = value.get("audibility")
        note = value.get("note")
        if status not in VALID_AUDIBILITY:
            raise ValueError(f"Invalid audibility label for {event_id}")
        if not isinstance(note, str) or len(note) > 4000:
            raise ValueError(f"Invalid or overlong event note for {event_id}")
        totals[status or "unlabeled"] += 1
        by_voice[events[event_id]["voice"]][status or "unlabeled"] += 1
        by_direction[events[event_id]["direction"]][status or "unlabeled"] += 1

    return {
        "schema": "manialab-chord-evidence-audio-review-summary-v1",
        "review_key": manifest["review_key"],
        "manifest_sha256": manifest_sha256,
        "case_count": len(case_ids),
        "attack_event_count": len(events),
        "labeled_event_count": sum(count for label, count in totals.items() if label != "unlabeled"),
        "audibility": dict(sorted(totals.items())),
        "by_voice": {key: dict(sorted(value.items())) for key, value in sorted(by_voice.items())},
        "by_direction": {
            key: dict(sorted(value.items())) for key, value in sorted(by_direction.items())
        },
        "window_note_count": sum(bool(note.strip()) for note in case_notes.values()),
        "event_note_count": sum(bool(value["note"].strip()) for value in exported_events.values()),
        "notes_emitted": False,
        "accuracy_computed": False,
        "musical_acceptance": False,
    }


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    try:
        output.relative_to((ROOT / "test-data").resolve())
    except ValueError as exc:
        raise ValueError("Summary output must remain inside ignored test-data") from exc
    if output.exists():
        raise FileExistsError(f"Preserve previous summaries: {output}")

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    labels_path = args.labels if args.labels.is_absolute() else ROOT / args.labels
    manifest_hash = _sha256_file(manifest_path)
    summary = summarize_review_labels(
        _read_json(manifest_path), _read_json(labels_path), manifest_hash,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
