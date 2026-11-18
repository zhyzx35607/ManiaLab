"""Validate and optionally lock piano-listening or lane-playtest CSV reviews."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PIANO_FIELDS = (
    "clip_id",
    "audio_file",
    "audio_start_ms",
    "pitch_midi",
    "audibility",
    "confidence_1_5",
    "comment",
)
LAYOUT_FIELDS = (
    "osz_sha256",
    "keys",
    "play_first",
    "play_second",
    "preferred_A_B_equal_unsure",
    "release_coordination",
    "hand_balance",
    "time_ms",
    "comment",
)

_PIANO_ID = re.compile(r"R[0-9]{3}\Z")
_CLIP_PATH = re.compile(r"clips/R[0-9]{3}\.wav\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PIANO_AUDIBILITY = {"yes", "uncertain", "no", "unclear"}
_PIANO_CONFIDENCE = {"1", "2", "3", "4", "5"}
_LAYOUT_RATING = {"A", "B", "equal", "unsure"}
_LAYOUT_RESPONSE_FIELDS = (
    "preferred_A_B_equal_unsure",
    "release_coordination",
    "hand_balance",
)
_PIANO_RESPONSE_FIELDS = ("audibility", "confidence_1_5")
_LAYOUT_IDENTITY_FIELDS = ("osz_sha256", "keys")


class ReviewError(ValueError):
    """An input or snapshot failure with no participant values in its message."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_csv(data: bytes, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8-sig", errors="strict")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != fields:
            raise ReviewError("CSV header does not match the expected template")
        rows: list[dict[str, str]] = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ReviewError("CSV contains a row with an invalid number of columns")
            rows.append(row)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ReviewError("CSV is not valid UTF-8 CSV") from error
    return rows


def _required_integer(value: str, *, minimum: int, maximum: int | None = None) -> int:
    if not value or len(value) > 16 or not value.isascii() or not value.isdecimal():
        raise ReviewError("CSV contains an invalid numeric field")
    number = int(value)
    if number < minimum or (maximum is not None and number > maximum):
        raise ReviewError("CSV contains an out-of-range numeric field")
    return number


def _piano_rows(data: bytes) -> list[dict[str, str]]:
    rows = _parse_csv(data, PIANO_FIELDS)
    if not rows:
        raise ReviewError("Piano review contains no clips")

    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        clip_id = row["clip_id"]
        if not _PIANO_ID.fullmatch(clip_id) or clip_id in seen:
            raise ReviewError("Piano template contains an invalid or duplicate clip ID")
        if clip_id != f"R{index:03d}":
            raise ReviewError("Piano clip IDs are not in canonical order")
        seen.add(clip_id)
        if not _CLIP_PATH.fullmatch(row["audio_file"]) or row["audio_file"] != f"clips/{clip_id}.wav":
            raise ReviewError("Piano template contains an invalid clip path")
        _required_integer(row["audio_start_ms"], minimum=0)
        _required_integer(row["pitch_midi"], minimum=0, maximum=127)
    return rows


def piano_template_rows(data: bytes) -> list[dict[str, str]]:
    """Parse a piano template and validate its stable clip identity columns."""
    rows = _piano_rows(data)
    if any(row[field] for row in rows for field in _PIANO_RESPONSE_FIELDS + ("comment",)):
        raise ReviewError("Piano template must not contain participant responses")
    return rows


def _piano_review(
    template_data: bytes,
    score_data: bytes,
) -> dict[str, Any]:
    template_rows = piano_template_rows(template_data)
    score_rows = _piano_rows(score_data)
    if len(score_rows) != len(template_rows):
        raise ReviewError("Piano score row count does not match its template")

    started = 0
    complete = 0
    for template, score in zip(template_rows, score_rows):
        if any(score[field] != template[field] for field in PIANO_FIELDS[:4]):
            raise ReviewError("Piano score identities do not match the template")
        audibility = score["audibility"]
        confidence = score["confidence_1_5"]
        if audibility and audibility not in _PIANO_AUDIBILITY:
            raise ReviewError("Piano score contains an invalid audibility value")
        if confidence and confidence not in _PIANO_CONFIDENCE:
            raise ReviewError("Piano score contains an invalid confidence value")
        if len(score["comment"]) > 500:
            raise ReviewError("Piano score comment exceeds the review form limit")
        filled = sum(bool(value) for value in (audibility, confidence))
        has_any_response = filled > 0 or bool(score["comment"])
        if has_any_response:
            started += 1
        if filled == len(_PIANO_RESPONSE_FIELDS):
            complete += 1

    row_count = len(template_rows)
    partial = started - complete
    return {
        "rows": row_count,
        "started_rows": started,
        "complete_rows": complete,
        "partial_rows": partial,
        "unplayed_rows": row_count - started,
        "covered_keys": 0,
        "lockable": row_count > 0 and complete == row_count,
    }


def _layout_rows(data: bytes, *, template: bool) -> list[dict[str, str]]:
    rows = _parse_csv(data, LAYOUT_FIELDS)
    if not rows:
        raise ReviewError("Layout review contains no key rows")

    seen_keys: set[int] = set()
    archive_hash: str | None = None
    for row in rows:
        if not _SHA256.fullmatch(row["osz_sha256"]):
            raise ReviewError("Layout review contains an invalid package hash")
        if archive_hash is None:
            archive_hash = row["osz_sha256"]
        elif row["osz_sha256"] != archive_hash:
            raise ReviewError("Layout review contains mixed package hashes")
        key_count = _required_integer(row["keys"], minimum=6, maximum=10)
        if key_count in seen_keys:
            raise ReviewError("Layout review contains duplicate key rows")
        seen_keys.add(key_count)
        if {row["play_first"], row["play_second"]} != {"A", "B"}:
            raise ReviewError("Layout play order must contain A and B")
        if template and any(row[field] for field in _LAYOUT_RESPONSE_FIELDS + ("time_ms", "comment")):
            raise ReviewError("Layout template must not contain participant responses")

    if seen_keys != set(range(6, 11)):
        raise ReviewError("Layout review must contain one row for each 6K-10K key count")
    return rows


def _layout_review(
    template_data: bytes,
    score_data: bytes,
) -> dict[str, Any]:
    template_rows = _layout_rows(template_data, template=True)
    score_rows = _layout_rows(score_data, template=False)
    if len(score_rows) != len(template_rows):
        raise ReviewError("Layout score row count does not match its template")

    template_by_key = {row["keys"]: row for row in template_rows}
    score_by_key = {row["keys"]: row for row in score_rows}
    started = 0
    complete = 0
    for key_count in sorted(template_by_key, key=int):
        template = template_by_key[key_count]
        score = score_by_key.get(key_count)
        if score is None:
            raise ReviewError("Layout score keys do not match the template")
        if any(score[field] != template[field] for field in _LAYOUT_IDENTITY_FIELDS):
            raise ReviewError("Layout score identities do not match the template")
        expected_order = (template["play_first"], template["play_second"])
        score_order = (score["play_first"], score["play_second"])
        if score_order not in (expected_order, expected_order[::-1]):
            raise ReviewError("Layout play order does not match the template pair")

        ratings = tuple(score[field] for field in _LAYOUT_RESPONSE_FIELDS)
        if any(value and value not in _LAYOUT_RATING for value in ratings):
            raise ReviewError("Layout score contains an invalid rating")
        if score["time_ms"]:
            _required_integer(score["time_ms"], minimum=0)
        filled = sum(bool(value) for value in ratings)
        has_any_response = filled > 0 or bool(score["time_ms"] or score["comment"])
        if has_any_response:
            started += 1
        if filled == len(_LAYOUT_RESPONSE_FIELDS):
            complete += 1

    row_count = len(template_rows)
    partial = started - complete
    return {
        "rows": row_count,
        "started_rows": started,
        "complete_rows": complete,
        "partial_rows": partial,
        "unplayed_rows": row_count - started,
        "covered_keys": complete,
        "lockable": complete > 0 and partial == 0,
    }


def validate_review(
    kind: str,
    template_data: bytes,
    score_data: bytes,
    *,
    expected_template_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate participant CSV values against its frozen template schema."""
    if kind not in {"piano", "layout"}:
        raise ReviewError("Review kind must be piano or layout")
    template_digest = _sha256(template_data)
    if expected_template_sha256 is not None:
        if not _SHA256.fullmatch(expected_template_sha256):
            raise ReviewError("Expected template digest must be a lowercase SHA-256")
        if template_digest != expected_template_sha256:
            raise ReviewError("Template digest does not match the pinned identity")
    counts = (
        _piano_review(template_data, score_data)
        if kind == "piano"
        else _layout_review(template_data, score_data)
    )
    return {
        "schema": 1,
        "kind": kind,
        "template_sha256": template_digest,
        "scores_sha256": _sha256(score_data),
        **counts,
    }


def lock_review_snapshot(
    kind: str,
    template_data: bytes,
    score_data: bytes,
    destination: Path,
    *,
    expected_template_sha256: str | None = None,
) -> dict[str, Any]:
    """Save exact submitted bytes and an integrity-only receipt in a new directory."""
    summary = validate_review(
        kind,
        template_data,
        score_data,
        expected_template_sha256=expected_template_sha256,
    )
    if not summary["lockable"]:
        raise ReviewError("Scores are valid but incomplete; snapshot was not created")

    destination = Path(destination)
    if not destination.parent.is_dir():
        raise ReviewError("Snapshot parent directory does not exist")

    created = False
    scores_created = False
    receipt_created = False
    try:
        destination.mkdir(exist_ok=False)
        created = True
        with (destination / "scores.csv").open("xb") as stream:
            stream.write(score_data)
            scores_created = True
        receipt = {
            **summary,
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": "CSV integrity and workflow snapshot only; not proof of no prior deblinding.",
        }
        with (destination / "receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
            receipt_created = True
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise ReviewError("Snapshot directory already exists; choose a new directory") from error
    except OSError as error:
        if created:
            if receipt_created:
                (destination / "receipt.json").unlink(missing_ok=True)
            if scores_created:
                (destination / "scores.csv").unlink(missing_ok=True)
            try:
                destination.rmdir()
            except OSError:
                pass
        raise ReviewError("Could not write review snapshot") from error
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("piano", "layout"), required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--template-sha256")
    parser.add_argument("--lock-dir", type=Path)
    args = parser.parse_args(argv)

    try:
        template_data = args.template.read_bytes()
        score_data = args.scores.read_bytes()
        if args.lock_dir is None:
            summary = validate_review(
                args.kind,
                template_data,
                score_data,
                expected_template_sha256=args.template_sha256,
            )
        else:
            summary = lock_review_snapshot(
                args.kind,
                template_data,
                score_data,
                args.lock_dir,
                expected_template_sha256=args.template_sha256,
            )
    except (OSError, ReviewError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
