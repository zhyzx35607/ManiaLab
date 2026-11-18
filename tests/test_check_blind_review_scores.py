import csv
import hashlib
import io
import json

import pytest

from scripts.check_blind_review_scores import (
    LAYOUT_FIELDS,
    PIANO_FIELDS,
    ReviewError,
    lock_review_snapshot,
    piano_template_rows,
    validate_review,
)


def _csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def _piano_rows():
    return [
        {
            "clip_id": f"R{index:03d}",
            "audio_file": f"clips/R{index:03d}.wav",
            "audio_start_ms": str(index * 1000),
            "pitch_midi": str(59 + index),
            "audibility": "",
            "confidence_1_5": "",
            "comment": "",
        }
        for index in range(1, 3)
    ]


def _piano_bytes(rows=None):
    return _csv_bytes(PIANO_FIELDS, _piano_rows() if rows is None else rows)


def _layout_rows():
    return [
        {
            "osz_sha256": "a" * 64,
            "keys": str(keys),
            "play_first": "A" if keys % 2 == 0 else "B",
            "play_second": "B" if keys % 2 == 0 else "A",
            "preferred_A_B_equal_unsure": "",
            "release_coordination": "",
            "hand_balance": "",
            "time_ms": "",
            "comment": "",
        }
        for keys in range(6, 11)
    ]


def _layout_bytes(rows=None):
    return _csv_bytes(LAYOUT_FIELDS, _layout_rows() if rows is None else rows)


def test_piano_template_rows_validates_stable_identities():
    rows = piano_template_rows(_piano_bytes())

    assert [row["clip_id"] for row in rows] == ["R001", "R002"]
    assert rows[1]["audio_file"] == "clips/R002.wav"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[1].update(clip_id="R001"),
        lambda rows: rows[1].update(clip_id="R003"),
        lambda rows: rows[0].update(audio_file="../R001.wav"),
        lambda rows: rows[0].update(audio_start_ms="-1"),
        lambda rows: rows[0].update(pitch_midi="128"),
    ],
)
def test_piano_template_rejects_noncanonical_identity(mutate):
    rows = _piano_rows()
    mutate(rows)

    with pytest.raises(ReviewError):
        piano_template_rows(_piano_bytes(rows))


def test_piano_requires_all_clips_and_locks_exact_csv_bytes(tmp_path):
    template = _piano_bytes()
    score_rows = _piano_rows()
    score_rows[0].update(
        audibility="yes",
        confidence_1_5="4",
        comment='A note with a comma, a quote " and\nsecond line',
    )
    score_rows[1].update(audibility="unclear", confidence_1_5="1")
    scores = _piano_bytes(score_rows)

    summary = validate_review(
        "piano",
        template,
        scores,
        expected_template_sha256=hashlib.sha256(template).hexdigest(),
    )
    assert summary["complete_rows"] == summary["rows"] == 2
    assert summary["partial_rows"] == 0
    assert summary["lockable"] is True

    destination = tmp_path / "locked"
    locked = lock_review_snapshot("piano", template, scores, destination)

    assert (destination / "scores.csv").read_bytes() == scores
    receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["scores_sha256"] == hashlib.sha256(scores).hexdigest()
    assert receipt["template_sha256"] == hashlib.sha256(template).hexdigest()
    assert receipt["complete_rows"] == locked["complete_rows"]
    assert receipt["evidence_scope"].startswith("CSV integrity and workflow snapshot only")
    assert "A note" not in (destination / "receipt.json").read_text(encoding="utf-8")


def test_piano_partial_scores_are_valid_but_not_lockable(tmp_path):
    template = _piano_bytes()
    rows = _piano_rows()
    rows[0]["audibility"] = "yes"
    scores = _piano_bytes(rows)

    summary = validate_review("piano", template, scores)
    assert summary["started_rows"] == 1
    assert summary["complete_rows"] == 0
    assert summary["partial_rows"] == 1
    assert summary["unplayed_rows"] == 1
    assert summary["lockable"] is False

    destination = tmp_path / "not-created"
    with pytest.raises(ReviewError, match="incomplete"):
        lock_review_snapshot("piano", template, scores, destination)
    assert not destination.exists()


def test_piano_rejects_mismatched_or_invalid_responses():
    template = _piano_bytes()
    rows = _piano_rows()
    rows[0].update(audibility="maybe", confidence_1_5="5")
    with pytest.raises(ReviewError, match="audibility"):
        validate_review("piano", template, _piano_bytes(rows))

    rows = _piano_rows()
    rows[0]["clip_id"] = "R999"
    with pytest.raises(ReviewError, match="canonical"):
        validate_review("piano", template, _piano_bytes(rows))


def test_layout_allows_unplayed_rows_and_reverse_order_for_played_keys(tmp_path):
    template = _layout_bytes()
    rows = _layout_rows()
    rows[0].update(
        play_first=rows[0]["play_second"],
        play_second=rows[0]["play_first"],
        preferred_A_B_equal_unsure="equal",
        release_coordination="A",
        hand_balance="unsure",
        time_ms="12000",
        comment="Only this key count was played.",
    )
    scores = _layout_bytes(rows)

    summary = validate_review("layout", template, scores)
    assert summary["rows"] == 5
    assert summary["complete_rows"] == summary["covered_keys"] == 1
    assert summary["unplayed_rows"] == 4
    assert summary["partial_rows"] == 0
    assert summary["lockable"] is True

    destination = tmp_path / "layout-lock"
    lock_review_snapshot("layout", template, scores, destination)
    assert (destination / "scores.csv").read_bytes() == scores


def test_layout_refuses_partial_rows_and_mixed_package_identity():
    template = _layout_bytes()
    rows = _layout_rows()
    rows[0]["preferred_A_B_equal_unsure"] = "A"

    summary = validate_review("layout", template, _layout_bytes(rows))
    assert summary["partial_rows"] == 1
    assert summary["lockable"] is False

    rows = _layout_rows()
    rows[0]["osz_sha256"] = "b" * 64
    with pytest.raises(ReviewError, match="mixed package"):
        validate_review("layout", _layout_bytes(), _layout_bytes(rows))


def test_layout_rejects_unknown_rating_and_invalid_time():
    template = _layout_bytes()
    rows = _layout_rows()
    rows[0].update(
        preferred_A_B_equal_unsure="C",
        release_coordination="A",
        hand_balance="B",
    )
    with pytest.raises(ReviewError, match="rating"):
        validate_review("layout", template, _layout_bytes(rows))

    rows = _layout_rows()
    rows[0].update(
        preferred_A_B_equal_unsure="A",
        release_coordination="A",
        hand_balance="B",
        time_ms="1.5",
    )
    with pytest.raises(ReviewError, match="numeric"):
        validate_review("layout", template, _layout_bytes(rows))


def test_rejects_header_shape_malformed_csv_and_changed_template_digest():
    piano = _piano_bytes()
    with pytest.raises(ReviewError, match="header"):
        piano_template_rows(b"clip_id,audio_file\r\n")
    with pytest.raises(ReviewError, match="UTF-8 CSV"):
        piano_template_rows(b"\xff")
    with pytest.raises(ReviewError, match="Template digest"):
        validate_review("piano", piano, piano, expected_template_sha256="0" * 64)


def test_snapshot_never_overwrites_existing_directory(tmp_path):
    template = _piano_bytes()
    rows = _piano_rows()
    for row in rows:
        row.update(audibility="yes", confidence_1_5="5")
    scores = _piano_bytes(rows)
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ReviewError, match="already exists"):
        lock_review_snapshot("piano", template, scores, destination)
    assert sentinel.read_text(encoding="utf-8") == "keep"
