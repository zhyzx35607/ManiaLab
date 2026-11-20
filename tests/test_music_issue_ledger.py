from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
from threading import Barrier

import pytest

from scripts import check_music_issue_ledger as ledger_module
from scripts.check_music_issue_ledger import (
    FIELDS,
    LedgerError,
    lock_review_snapshot,
    main,
    validate_ledger,
)


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def _template_bytes():
    return _csv_bytes([])


def _row(**updates):
    row = {
        "issue_id": "ISSUE-001",
        "reviewer_session_id": "anon-12ab34cd",
        "song": "sakura_grass",
        "reviewed_at_utc": "2026-11-20T12:30:00Z",
        "review_mode": "listening",
        "source_audio_sha256": "a" * 64,
        "analysis_sha256": "b" * 64,
        "candidate_sha256": "c" * 64,
        "reference_sha256": "d" * 64,
        "export_sha256": "e" * 64,
        "identity_note": "",
        "key_count": "",
        "bookmark_or_section": "BM-03",
        "time_ms": "12500",
        "voice_or_part": "vocals",
        "checked_tracks": "mix;stem:vocals;reference_chart;exported_chart_playback",
        "category": "detection",
        "assessment": "issue",
        "audibility": "audible",
        "observation": "Synthetic fixture observation.",
        "decision": "open",
        "decision_reason": "Awaiting a human retest.",
        "decision_session_id": "",
        "decision_at_utc": "",
        "fix_ref": "",
        "retest_at_utc": "",
        "retest_result": "",
    }
    row.update(updates)
    return row


def test_valid_rows_summarize_and_lock_exact_bytes_without_review_content(tmp_path):
    template = _template_bytes()
    scores = _csv_bytes([
        _row(),
        _row(
            issue_id="ISSUE-002",
            song="sweet_waltz",
            reviewer_session_id="anon-98zy76xw",
            review_mode="playtest",
            reference_sha256="",
            identity_note="No source reference chart for this song.",
            key_count="8",
            category="playability",
            audibility="not_applicable",
            checked_tracks="mix;exported_chart_playback",
            decision="fixed_and_retested",
            decision_reason="Retested the updated map.",
            fix_ref="d45ddb3",
            retest_at_utc="2026-11-20T13:00:00+00:00",
            retest_result="clear",
        ),
    ])

    summary = validate_ledger(
        template,
        scores,
        expected_template_sha256=hashlib.sha256(template).hexdigest(),
    )
    assert summary["issue_count"] == 2
    assert summary["distinct_session_count"] == 2
    assert summary["song_counts"] == {"sakura_grass": 1, "sweet_waltz": 1}
    assert summary["scores_sha256"] == hashlib.sha256(scores).hexdigest()

    destination = tmp_path / "locked"
    lock_review_snapshot(template, scores, destination)
    assert (destination / "music-issue-ledger.csv").read_bytes() == scores
    receipt_text = (destination / "receipt.json").read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["scores_sha256"] == hashlib.sha256(scores).hexdigest()
    assert receipt["evidence_scope"].startswith("CSV structure, declared fields")
    assert "not proof of listening" in receipt["evidence_scope"]
    assert "full coverage" in receipt["evidence_scope"]
    assert "Synthetic fixture observation" not in receipt_text
    assert "anon-12ab34cd" not in receipt_text


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"source_audio_sha256": "bad"}, "source_audio_sha256"),
        ({"reviewed_at_utc": "2026-11-20T12:00:00"}, "UTC timestamp"),
        ({"reviewed_at_utc": "2026-11-20X12:00:00Z"}, "UTC timestamp"),
        ({"reviewer_session_id": "Alice"}, "anonymous session"),
        ({"checked_tracks": "mix;mix;exported_chart_playback"}, "duplicate track"),
        ({"time_ms": "12.5"}, "time_ms"),
        ({"reference_sha256": ""}, "identity_note"),
    ],
)
def test_rejects_invalid_identity_and_issue_fields(updates, message):
    with pytest.raises(LedgerError, match=message):
        validate_ledger(_template_bytes(), _csv_bytes([_row(**updates)]))


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"assessment": "clear", "decision": "open"}, "Clear assessments"),
        ({"assessment": "uncertain", "decision": "accepted"}, "Uncertain assessments"),
        ({"decision": "fixed_and_retested", "decision_reason": "Fixed."}, "clear UTC retest"),
        ({"decision": "accepted", "decision_reason": "Accepted."}, "decision session ID"),
        ({"decision": "open", "fix_ref": "abc", "retest_at_utc": "", "retest_result": "issue"}, "fields must be complete"),
        ({"decision": "open", "fix_ref": "abc", "retest_at_utc": "2026-11-20T13:00:00Z", "retest_result": "clear"}, "must use fixed_and_retested"),
    ],
)
def test_rejects_inconsistent_assessment_and_resolution(updates, message):
    with pytest.raises(LedgerError, match=message):
        validate_ledger(_template_bytes(), _csv_bytes([_row(**updates)]))


def test_retest_time_cannot_precede_declared_listening_time():
    row = _row(
        decision="fixed_and_retested",
        decision_reason="Synthetic correction.",
        fix_ref="example-commit",
        retest_at_utc="2026-11-19T13:00:00Z",
        retest_result="clear",
    )
    with pytest.raises(LedgerError, match="cannot precede"):
        validate_ledger(_template_bytes(), _csv_bytes([row]))


def test_acceptance_decision_requires_reason_session_and_utc_time():
    row = _row(
        decision="accepted",
        decision_reason="Reviewer accepts this documented variation.",
        decision_session_id="anon-xy12zabc",
        decision_at_utc="2026-11-20T14:00:00Z",
    )
    assert validate_ledger(_template_bytes(), _csv_bytes([row]))["decision_counts"] == {"accepted": 1}


@pytest.mark.parametrize("decision", ["accepted", "fixed_and_retested"])
@pytest.mark.parametrize(
    "reviewed_at, resolved_at, valid",
    [
        ("2026-11-20T12:30:00Z", "2026-11-20T12:30:00.500Z", True),
        ("2026-11-20T12:30:00.500Z", "2026-11-20T12:30:00Z", False),
        ("2026-11-20T12:30:00.500Z", "2026-11-20T12:30:00.500000+00:00", True),
    ],
)
def test_resolution_order_compares_utc_instants_with_fractional_seconds(decision, reviewed_at, resolved_at, valid):
    resolution = (
        dict(decision_session_id="anon-xy12zabc", decision_at_utc=resolved_at)
        if decision == "accepted" else
        dict(fix_ref="abc123", retest_at_utc=resolved_at, retest_result="clear")
    )
    scores = _csv_bytes([_row(
        reviewed_at_utc=reviewed_at,
        decision=decision,
        decision_reason="Synthetic decision timing.",
        **resolution,
    )])
    if valid:
        assert validate_ledger(_template_bytes(), scores)["decision_counts"] == {decision: 1}
    else:
        with pytest.raises(LedgerError, match="cannot precede"):
            validate_ledger(_template_bytes(), scores)


def test_rejects_duplicate_issue_ids_template_changes_and_overwrite(tmp_path):
    template = _template_bytes()
    scores = _csv_bytes([_row(), _row()])
    with pytest.raises(LedgerError, match="unique"):
        validate_ledger(template, scores)
    with pytest.raises(LedgerError, match="pinned identity"):
        validate_ledger(template, _csv_bytes([_row()]), expected_template_sha256="0" * 64)

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(LedgerError, match="already exists"):
        lock_review_snapshot(template, _csv_bytes([_row()]), existing)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_detection_rows_require_audibility_and_separated_track():
    with pytest.raises(LedgerError, match="audibility judgment"):
        validate_ledger(_template_bytes(), _csv_bytes([_row(audibility="not_applicable")]))
    with pytest.raises(LedgerError, match="checked stem"):
        validate_ledger(_template_bytes(), _csv_bytes([_row(
            reference_sha256="",
            identity_note="No source reference chart.",
            checked_tracks="mix;exported_chart_playback",
        )]))


def test_rejects_malformed_or_empty_csv():
    with pytest.raises(LedgerError, match="header"):
        validate_ledger(b"wrong,header\r\n", _csv_bytes([_row()]))
    with pytest.raises(LedgerError, match="no issue rows"):
        validate_ledger(_template_bytes(), _template_bytes())


def test_cli_locks_synthetic_csv_and_prints_integrity_scope(tmp_path, monkeypatch, capsys):
    repository = tmp_path / "repo"
    lock_parent = repository / "test-data" / "reviews"
    lock_parent.mkdir(parents=True)
    monkeypatch.setattr(ledger_module, "ROOT", repository)
    template = tmp_path / "template.csv"
    scores = tmp_path / "scores.csv"
    template.write_bytes(_template_bytes())
    scores.write_bytes(_csv_bytes([_row()]))
    destination = lock_parent / "lock"

    result = main([
        "--template", str(template),
        "--scores", str(scores),
        "--lock-dir", str(destination),
        "--template-sha256", hashlib.sha256(template.read_bytes()).hexdigest(),
    ])

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["issue_count"] == 1
    assert output["scores_sha256"] == hashlib.sha256(scores.read_bytes()).hexdigest()
    assert "not proof of listening" in output["evidence_scope"]
    assert json.loads((destination / "receipt.json").read_text(encoding="utf-8"))["issue_count"] == 1


def test_cli_rejects_lock_outside_ignored_test_data(tmp_path, monkeypatch, capsys):
    repository = tmp_path / "repo"
    (repository / "test-data").mkdir(parents=True)
    monkeypatch.setattr(ledger_module, "ROOT", repository)
    template = tmp_path / "template.csv"
    scores = tmp_path / "scores.csv"
    template.write_bytes(_template_bytes())
    scores.write_bytes(_csv_bytes([_row()]))
    destination = repository / "review-records"

    result = main([
        "--template", str(template),
        "--scores", str(scores),
        "--lock-dir", str(destination),
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "ignored test-data/ directory" in captured.err
    assert not destination.exists()


def _create_directory_symlink_or_skip(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Directory symlinks are unavailable: {error}")


def test_cli_rejects_test_data_root_symlink(tmp_path, monkeypatch, capsys):
    repository = tmp_path / "repo"
    repository.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    test_data = repository / "test-data"
    _create_directory_symlink_or_skip(test_data, external)
    monkeypatch.setattr(ledger_module, "ROOT", repository)
    template = tmp_path / "template.csv"
    scores = tmp_path / "scores.csv"
    template.write_bytes(_template_bytes())
    scores.write_bytes(_csv_bytes([_row()]))
    destination = test_data / "reviews" / "lock"

    result = main([
        "--template", str(template),
        "--scores", str(scores),
        "--lock-dir", str(destination),
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "must not resolve through a symlink" in captured.err
    assert not (external / "reviews").exists()


def test_cli_rejects_symlink_escape_below_test_data(tmp_path, monkeypatch, capsys):
    repository = tmp_path / "repo"
    test_data = repository / "test-data"
    test_data.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    _create_directory_symlink_or_skip(test_data / "reviews", external)
    monkeypatch.setattr(ledger_module, "ROOT", repository)
    template = tmp_path / "template.csv"
    scores = tmp_path / "scores.csv"
    template.write_bytes(_template_bytes())
    scores.write_bytes(_csv_bytes([_row()]))
    destination = test_data / "reviews" / "lock"

    result = main([
        "--template", str(template),
        "--scores", str(scores),
        "--lock-dir", str(destination),
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "must resolve inside this repository's ignored test-data/ directory" in captured.err
    assert not (external / "lock").exists()


def test_cli_help_lists_allowed_values_and_safe_fill_workflow(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    for required_text in (
        "Copy-Item docs/music-issue-ledger-template.csv",
        "--scores ./test-data/reviews/music-issue-ledger.csv --lock-dir ./test-data/reviews/music-issue-ledger-lock",
        "song: sakura_grass, sweet_waltz, choice",
        "review_mode: listening, playtest, both",
        "category: detection, separation_input, expression_selection, timing,",
        "checked_tracks: semicolon-separated",
        "assessment: clear, uncertain, issue",
        "decision: no_issue, follow_up, open, fixed_and_retested, accepted",
        "reviewed_at_utc is required",
        "Both must match anon- followed by 8-64 lowercase letters",
        "not proof that listening or playtesting happened",
        "that review coverage is complete",
    ):
        assert required_text in output


def test_failed_receipt_write_removes_only_new_snapshot_files(tmp_path, monkeypatch):
    original_open = Path.open

    def fail_receipt(path, *args, **kwargs):
        if path.name == "receipt.json":
            raise OSError("synthetic disk failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_receipt)
    destination = tmp_path / "failed-lock"
    with pytest.raises(LedgerError, match="Could not write"):
        lock_review_snapshot(_template_bytes(), _csv_bytes([_row()]), destination)
    assert not destination.exists()


@pytest.mark.parametrize("failed_name", ["music-issue-ledger.csv", "receipt.json"])
def test_partial_snapshot_write_failure_removes_new_files_and_preserves_old_records(tmp_path, monkeypatch, failed_name):
    existing = tmp_path / "existing-lock"
    existing.mkdir()
    sentinel = existing / "music-issue-ledger.csv"
    sentinel.write_bytes(b"existing record")
    original_open = Path.open

    class PartialWriteFailure:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, data):
            self.stream.write(data[:8])
            self.stream.flush()
            raise OSError("synthetic disk failure after partial write")

    def fail_partial_write(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path.name == failed_name and mode in {"x", "xb"}:
            return PartialWriteFailure(stream)
        return stream

    monkeypatch.setattr(Path, "open", fail_partial_write)
    destination = tmp_path / "failed-lock"
    with pytest.raises(LedgerError, match="Could not write"):
        lock_review_snapshot(_template_bytes(), _csv_bytes([_row()]), destination)
    assert not destination.exists()
    assert sentinel.read_bytes() == b"existing record"


def test_concurrent_locks_keep_exactly_one_complete_snapshot(tmp_path, monkeypatch):
    destination = tmp_path / "concurrent-lock"
    barrier = Barrier(2)
    original_mkdir = Path.mkdir

    def simultaneous_mkdir(path, *args, **kwargs):
        if path == destination:
            barrier.wait(timeout=5)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", simultaneous_mkdir)
    submissions = [_csv_bytes([_row(issue_id=f"ISSUE-{index}")]) for index in (1, 2)]

    def attempt(scores):
        try:
            lock_review_snapshot(_template_bytes(), scores, destination)
        except LedgerError as error:
            return False, str(error)
        return True, scores

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(attempt, scores) for scores in submissions]
        results = [future.result(timeout=10) for future in futures]

    winners = [value for successful, value in results if successful]
    failures = [value for successful, value in results if not successful]
    assert len(winners) == len(failures) == 1
    assert "already exists" in failures[0]
    assert (destination / "music-issue-ledger.csv").read_bytes() == winners[0]
    receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["scores_sha256"] == hashlib.sha256(winners[0]).hexdigest()


def test_receipt_file_collision_removes_only_this_lock_csv(tmp_path, monkeypatch):
    original_open = Path.open
    destination = tmp_path / "receipt-collision"

    def claim_receipt(path, mode="r", *args, **kwargs):
        if path.name == "receipt.json" and mode == "x":
            with original_open(path, mode, *args, **kwargs) as stream:
                stream.write("competing receipt")
            raise FileExistsError("receipt claimed by competing lock")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", claim_receipt)
    with pytest.raises(LedgerError, match="already exists"):
        lock_review_snapshot(_template_bytes(), _csv_bytes([_row()]), destination)

    assert destination.exists()
    assert not (destination / "music-issue-ledger.csv").exists()
    assert (destination / "receipt.json").read_text(encoding="utf-8") == "competing receipt"
