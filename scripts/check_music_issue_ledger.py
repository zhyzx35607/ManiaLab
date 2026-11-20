"""Validate and lock one-row-per-issue listening and playtest records.

Use song keys sakura_grass, sweet_waltz, or choice. Keep times in absolute song
milliseconds; export_sha256 should identify the exact exported .osu difficulty.
Use semicolon-separated checked_tracks from the accepted values enforced below.
When there is no reference chart, leave reference_sha256 blank and explain why in
identity_note. Fixed issues require a fix reference and a clear UTC retest; accepted
issues require a reason and anonymous decision-session ID. The receipt attests only
to CSV structure, declared metadata, and exact bytes; it does not prove listening,
playtesting, reviewer identity, blind status, full coverage, or correctness.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "docs" / "music-issue-ledger-template.csv"
SCHEMA = "manialab-music-issue-ledger-v1"
RECEIPT_SCHEMA = "manialab-music-issue-ledger-lock-v1"
FIELDS = (
    "issue_id",
    "reviewer_session_id",
    "song",
    "reviewed_at_utc",
    "review_mode",
    "source_audio_sha256",
    "analysis_sha256",
    "candidate_sha256",
    "reference_sha256",
    "export_sha256",
    "identity_note",
    "key_count",
    "bookmark_or_section",
    "time_ms",
    "voice_or_part",
    "checked_tracks",
    "category",
    "assessment",
    "audibility",
    "observation",
    "decision",
    "decision_reason",
    "decision_session_id",
    "decision_at_utc",
    "fix_ref",
    "retest_at_utc",
    "retest_result",
)

HASH_FIELDS = (
    "source_audio_sha256",
    "analysis_sha256",
    "candidate_sha256",
    "reference_sha256",
    "export_sha256",
)
REQUIRED_HASH_FIELDS = frozenset({
    "source_audio_sha256", "analysis_sha256", "candidate_sha256", "export_sha256",
})
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|\+00:00)\Z"
)
ISSUE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
SESSION_ID_RE = re.compile(r"anon-[a-z0-9][a-z0-9_-]{7,63}\Z")
SONGS = frozenset({"sakura_grass", "sweet_waltz", "choice"})
REVIEW_MODES = frozenset({"listening", "playtest", "both"})
VOICES = frozenset({"vocals", "piano", "bass", "drums", "guitar", "other", "mixed", "unknown"})
TRACKS = frozenset({
    "mix",
    "stem:vocals", "stem:piano", "stem:bass", "stem:drums", "stem:guitar", "stem:other",
    "candidate_audio", "reference_chart", "exported_chart_playback", "waveform", "spectrum",
})
CATEGORIES = frozenset({
    "detection", "separation_input", "expression_selection", "timing", "voice_clarity",
    "phrase_continuity", "ln_expression", "playability", "structure", "other",
})
ASSESSMENTS = frozenset({"clear", "uncertain", "issue"})
AUDIBILITY = frozenset({"audible", "not_audible", "uncertain", "not_applicable"})
DECISIONS = frozenset({"no_issue", "follow_up", "open", "fixed_and_retested", "accepted"})
RETEST_RESULTS = frozenset({"clear", "uncertain", "issue"})
FIX_ATTEMPT_FIELDS = ("fix_ref", "retest_at_utc", "retest_result")

CLI_HELP_EPILOG = r"""Fill one row per issue, starting from the header-only template (run from the repository root):
  # Create the ignored local parent directory test-data/reviews first.
  Copy-Item docs/music-issue-ledger-template.csv ./test-data/reviews/music-issue-ledger.csv
  # Add and save issue rows without changing the header.
  .\.venv\Scripts\python.exe scripts/check_music_issue_ledger.py --scores ./test-data/reviews/music-issue-ledger.csv --lock-dir ./test-data/reviews/music-issue-ledger-lock

Allowed values (case-sensitive):
  song: sakura_grass, sweet_waltz, choice
  review_mode: listening, playtest, both
  category: detection, separation_input, expression_selection, timing,
    voice_clarity, phrase_continuity, ln_expression, playability, structure, other
  checked_tracks: semicolon-separated mix, stem:vocals, stem:piano, stem:bass,
    stem:drums, stem:guitar, stem:other, candidate_audio, reference_chart,
    exported_chart_playback, waveform, spectrum. Every row requires mix and
    exported_chart_playback. A reference hash requires reference_chart; detection
    and separation_input require at least one stem:* track.
  assessment: clear, uncertain, issue
  audibility: audible, not_audible, uncertain, not_applicable
  decision: no_issue, follow_up, open, fixed_and_retested, accepted
  assessment/decision pairs: clear/no_issue, uncertain/follow_up, or issue with
    open, fixed_and_retested, or accepted.

Required values and formats:
  reviewed_at_utc is required. Timestamps must include UTC as Z or +00:00 and
  decision/retest times cannot precede reviewed_at_utc. accepted requires
  decision_at_utc; fixed_and_retested requires a fix_ref and a clear retest_at_utc.
  reviewer_session_id is required; accepted also requires decision_session_id.
  Both must match anon- followed by 8-64 lowercase letters, digits, _ or -, with
  a letter or digit first. This checks the declared format, not reviewer identity.
  key_count is blank for listening-only rows and 6-10 for playtest/both rows.
  Hash fields are lowercase SHA-256; time_ms is absolute song milliseconds.

The lock receipt records CSV structure, declared metadata, and exact bytes only.
It is not proof that listening or playtesting happened, that reviewers were
anonymous or blind, that review coverage is complete, or that the assessment is correct.
"""


class LedgerError(ValueError):
    """Invalid ledger input or snapshot operation."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_rows(data: bytes, *, require_rows: bool) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8-sig", errors="strict")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise LedgerError("CSV header does not match the ledger template")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise LedgerError("CSV contains a row with an invalid number of columns")
            rows.append(row)
    except (UnicodeDecodeError, csv.Error) as error:
        raise LedgerError("CSV is not valid UTF-8 CSV") from error
    if require_rows and not rows:
        raise LedgerError("Ledger contains no issue rows")
    return rows


def _required_text(row: dict[str, str], field: str, *, maximum: int = 2000) -> str:
    value = row[field]
    if not value.strip() or len(value) > maximum:
        raise LedgerError(f"Invalid or missing {field}")
    return value


def _timestamp(value: str, field: str, *, required: bool = False) -> datetime | None:
    if not value:
        if required:
            raise LedgerError(f"Missing {field}")
        return None
    if len(value) > 40 or not UTC_TIMESTAMP_RE.fullmatch(value):
        raise LedgerError(f"{field} must use an ISO 8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LedgerError(f"Invalid {field}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise LedgerError(f"{field} must be an explicit UTC timestamp")
    return parsed.astimezone(timezone.utc)


def _integer(value: str, field: str, *, minimum: int, maximum: int | None = None) -> int:
    if not value or len(value) > 12 or not value.isascii() or not value.isdecimal():
        raise LedgerError(f"Invalid {field}")
    number = int(value)
    if number < minimum or (maximum is not None and number > maximum):
        raise LedgerError(f"Out-of-range {field}")
    return number


def validate_ledger(
    template_data: bytes,
    scores_data: bytes,
    *,
    expected_template_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate row structure and declared evidence; never opens evidence files."""
    template_rows = _parse_rows(template_data, require_rows=False)
    if template_rows:
        raise LedgerError("Template must contain only the header")
    template_sha256 = _sha256(template_data)
    if expected_template_sha256 is not None:
        if not SHA256_RE.fullmatch(expected_template_sha256):
            raise LedgerError("Expected template digest must be lowercase SHA-256")
        if template_sha256 != expected_template_sha256:
            raise LedgerError("Template digest does not match the pinned identity")

    rows = _parse_rows(scores_data, require_rows=True)
    seen_ids: set[str] = set()
    song_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    assessment_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    sessions: set[str] = set()

    for row in rows:
        issue_id = _required_text(row, "issue_id", maximum=64)
        if not ISSUE_ID_RE.fullmatch(issue_id) or issue_id in seen_ids:
            raise LedgerError("Issue IDs must be valid and unique")
        seen_ids.add(issue_id)

        session_id = _required_text(row, "reviewer_session_id", maximum=69)
        if not SESSION_ID_RE.fullmatch(session_id):
            raise LedgerError("Reviewer session ID must use the anonymous session format")
        sessions.add(session_id)

        song = _required_text(row, "song", maximum=32)
        if song not in SONGS:
            raise LedgerError("Unknown song key")
        song_counts[song] += 1

        reviewed_at = _timestamp(row["reviewed_at_utc"], "reviewed_at_utc", required=True)
        review_mode = row["review_mode"]
        if review_mode not in REVIEW_MODES:
            raise LedgerError("Invalid review_mode")

        for field in HASH_FIELDS:
            value = row[field]
            if not value and field not in REQUIRED_HASH_FIELDS:
                continue
            if not SHA256_RE.fullmatch(value):
                raise LedgerError(f"Invalid {field}")
        if not row["reference_sha256"] and not row["identity_note"].strip():
            raise LedgerError("identity_note is required when reference_sha256 is blank")
        if len(row["identity_note"]) > 1000:
            raise LedgerError("identity_note exceeds the field limit")

        key_count = row["key_count"]
        if review_mode == "listening":
            if key_count:
                raise LedgerError("Listening-only rows must leave key_count blank")
        else:
            _integer(key_count, "key_count", minimum=6, maximum=10)

        _required_text(row, "bookmark_or_section", maximum=120)
        _integer(row["time_ms"], "time_ms", minimum=0, maximum=86_400_000)
        voice = _required_text(row, "voice_or_part", maximum=24)
        if voice not in VOICES:
            raise LedgerError("Invalid voice_or_part")

        track_text = _required_text(row, "checked_tracks", maximum=300)
        tracks = track_text.split(";")
        if any(not track or track not in TRACKS for track in tracks) or len(tracks) != len(set(tracks)):
            raise LedgerError("checked_tracks contains an unknown or duplicate track")
        if not {"mix", "exported_chart_playback"}.issubset(tracks):
            raise LedgerError("Each issue must record mix and exported chart playback")
        if row["reference_sha256"] and "reference_chart" not in tracks:
            raise LedgerError("A referenced chart hash requires reference_chart in checked_tracks")

        category = row["category"]
        if category not in CATEGORIES:
            raise LedgerError("Invalid category")
        category_counts[category] += 1
        if category in {"detection", "separation_input"} and not any(track.startswith("stem:") for track in tracks):
            raise LedgerError("Detection and separation issues must record a checked stem")

        assessment = row["assessment"]
        if assessment not in ASSESSMENTS:
            raise LedgerError("Invalid assessment")
        assessment_counts[assessment] += 1
        audibility = row["audibility"]
        if audibility and audibility not in AUDIBILITY:
            raise LedgerError("Invalid audibility")
        if category in {"detection", "separation_input"} and audibility in {"", "not_applicable"}:
            raise LedgerError("Detection and separation issues require an audibility judgment")

        _required_text(row, "observation")
        decision = row["decision"]
        if decision not in DECISIONS:
            raise LedgerError("Invalid decision")
        decision_counts[decision] += 1
        decision_reason = row["decision_reason"]
        decision_session_id = row["decision_session_id"]
        decision_at = _timestamp(row["decision_at_utc"], "decision_at_utc")
        retest_at = _timestamp(row["retest_at_utc"], "retest_at_utc")
        for timestamp in (decision_at, retest_at):
            if timestamp is not None and timestamp < reviewed_at:
                raise LedgerError("Decision and retest times cannot precede reviewed_at_utc")
        retest_result = row["retest_result"]
        fix_ref = row["fix_ref"]
        if retest_result and retest_result not in RETEST_RESULTS:
            raise LedgerError("Invalid retest_result")
        if len(decision_reason) > 1000 or len(fix_ref) > 128:
            raise LedgerError("Decision or fix reference exceeds the field limit")

        if assessment == "clear" and decision != "no_issue":
            raise LedgerError("Clear assessments must use no_issue")
        if assessment == "uncertain" and decision != "follow_up":
            raise LedgerError("Uncertain assessments must use follow_up")
        if assessment == "issue" and decision not in {"open", "fixed_and_retested", "accepted"}:
            raise LedgerError("Issue assessments need an open, fixed, or accepted decision")
        if decision == "no_issue" and any((decision_reason, decision_session_id, decision_at, fix_ref, retest_at, retest_result)):
            raise LedgerError("no_issue rows cannot include resolution fields")
        if decision == "follow_up" and not decision_reason.strip():
            raise LedgerError("follow_up rows require a decision_reason")
        if decision == "accepted":
            if not decision_reason.strip() or not SESSION_ID_RE.fullmatch(decision_session_id):
                raise LedgerError("Accepted issues require a reason and anonymous decision session ID")
            if decision_at is None or any((fix_ref, retest_at, retest_result)):
                raise LedgerError("Accepted issues require decision_at_utc and cannot claim a fix retest")
        elif decision_session_id or decision_at:
            raise LedgerError("Decision identity fields are only valid for accepted issues")
        if decision == "fixed_and_retested":
            if not fix_ref.strip() or retest_at is None or retest_result != "clear":
                raise LedgerError("fixed_and_retested requires fix_ref and a clear UTC retest")
            if not decision_reason.strip():
                raise LedgerError("fixed_and_retested requires a decision_reason")
        elif decision not in {"open", "fixed_and_retested"} and any((fix_ref, retest_at, retest_result)):
            raise LedgerError("Fix and retest fields require an open or fixed issue")
        if decision == "open":
            attempted = [bool(value) for value in (fix_ref, retest_at, retest_result)]
            if any(attempted) and not all(attempted):
                raise LedgerError("Open issue retest fields must be complete or blank")
            if retest_result == "clear":
                raise LedgerError("A clear retest must use fixed_and_retested")

    return {
        "schema": SCHEMA,
        "template_sha256": template_sha256,
        "scores_sha256": _sha256(scores_data),
        "issue_count": len(rows),
        "distinct_session_count": len(sessions),
        "song_counts": {key: song_counts[key] for key in sorted(song_counts)},
        "category_counts": {key: category_counts[key] for key in sorted(category_counts)},
        "assessment_counts": {key: assessment_counts[key] for key in sorted(assessment_counts)},
        "decision_counts": {key: decision_counts[key] for key in sorted(decision_counts)},
        "evidence_scope": "CSV structure, declared fields, and exact byte snapshot only; not proof of listening, playtesting, reviewer identity, blind status, full coverage, or correctness.",
    }


def lock_review_snapshot(
    template_data: bytes,
    scores_data: bytes,
    destination: Path,
    *,
    expected_template_sha256: str | None = None,
) -> dict[str, Any]:
    """Save exact submitted bytes and a metadata-only receipt in a new directory."""
    summary = validate_ledger(
        template_data,
        scores_data,
        expected_template_sha256=expected_template_sha256,
    )
    destination = Path(destination)
    if not destination.parent.is_dir():
        raise LedgerError("Snapshot parent directory does not exist")
    if destination.exists():
        raise LedgerError("Snapshot directory already exists; choose a new directory")

    created = False
    scores_created = False
    receipt_created = False
    try:
        destination.mkdir(exist_ok=False)
        created = True
        with (destination / "music-issue-ledger.csv").open("xb") as stream:
            scores_created = True
            stream.write(scores_data)
        receipt = {
            "receipt_schema": RECEIPT_SCHEMA,
            **summary,
            "locked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        with (destination / "receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
            receipt_created = True
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except OSError as error:
        if created:
            if receipt_created:
                (destination / "receipt.json").unlink(missing_ok=True)
            if scores_created:
                (destination / "music-issue-ledger.csv").unlink(missing_ok=True)
            try:
                destination.rmdir()
            except OSError:
                pass
        if isinstance(error, FileExistsError):
            raise LedgerError("Snapshot directory or file already exists") from error
        raise LedgerError("Could not write locked snapshot") from error
    return summary


def validate_cli_lock_location(lock_dir: Path, *, repository_root: Path | None = None) -> Path:
    repository_root = Path(ROOT if repository_root is None else repository_root).resolve()
    ignored_root = repository_root / "test-data"
    if ignored_root.resolve() != ignored_root:
        raise LedgerError("Repository test-data directory must not resolve through a symlink")
    resolved_lock_dir = Path(lock_dir).resolve()
    try:
        resolved_lock_dir.relative_to(ignored_root)
    except ValueError as error:
        raise LedgerError("--lock-dir must resolve inside this repository's ignored test-data/ directory") from error
    return resolved_lock_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=CLI_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--lock-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--template-sha256")
    args = parser.parse_args(argv)
    try:
        lock_dir = validate_cli_lock_location(args.lock_dir)
        template_data = args.template.read_bytes()
        scores_data = args.scores.read_bytes()
        summary = lock_review_snapshot(
            template_data,
            scores_data,
            lock_dir,
            expected_template_sha256=args.template_sha256,
        )
    except (OSError, LedgerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "lock_dir": str(lock_dir),
        "issue_count": summary["issue_count"],
        "distinct_session_count": summary["distinct_session_count"],
        "scores_sha256": summary["scores_sha256"],
        "evidence_scope": summary["evidence_scope"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
