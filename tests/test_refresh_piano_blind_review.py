import csv
import hashlib
import json
from zipfile import ZipFile

import pytest

from scripts.build_piano_onset_blind_review import REVIEW_FIELDS, _write_zip
from scripts.refresh_piano_blind_review import refresh_package


def _source_package(root):
    participant = root / "participant"
    (participant / "clips").mkdir(parents=True)
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "clip_id": "R001",
            "audio_file": "clips/R001.wav",
            "audio_start_ms": "1010",
            "pitch_midi": "60",
            "audibility": "",
            "confidence_1_5": "",
            "comment": "",
        },
        {
            "clip_id": "R002",
            "audio_file": "clips/R002.wav",
            "audio_start_ms": "2073",
            "pitch_midi": "64",
            "audibility": "",
            "confidence_1_5": "",
            "comment": "",
        },
    ]
    with (participant / "review-template.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        (participant / row["audio_file"]).write_bytes(row["clip_id"].encode())
    (participant / "README.txt").write_text("Blind review", encoding="utf-8")
    archive = root / "piano-onset-review.zip"
    _write_zip(participant, archive)
    (root / "manifest.json").write_text(
        json.dumps({
            "schema": 1,
            "source_sha256": {"audio": "a" * 64},
            "samples": len(rows),
            "template_sha256": hashlib.sha256(
                (participant / "review-template.csv").read_bytes()
            ).hexdigest(),
            "participant_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    return participant


def test_refresh_rebuilds_page_without_decoding_audio_or_copying_key(tmp_path):
    source = _source_package(tmp_path / "source")
    key = source.parent / "deblind-key.json"
    key.write_text('{"private":"source labels"}', encoding="utf-8")
    template_before = (source / "review-template.csv").read_bytes()
    clip_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source / "clips").glob("*.wav")
    }
    output = tmp_path / "refreshed"

    manifest = refresh_package(source, output)

    assert manifest["samples"] == 2
    assert manifest["source_audio_decoded"] is False
    assert manifest["model_inference_run"] is False
    assert manifest["deblind_key_copied_or_read"] is False
    assert manifest["source_sha256"]["audio"] == "a" * 64
    assert (output / "participant/review-template.csv").read_bytes() == template_before
    page = (output / "participant/index.html").read_text(encoding="utf-8")
    assert "localStorage.setItem" in page
    assert "kong" not in page.casefold()
    assert "transkun" not in page.casefold()
    assert not (output / "deblind-key.json").exists()
    assert key.read_text(encoding="utf-8") == '{"private":"source labels"}'
    assert clip_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source / "clips").glob("*.wav")
    }
    with ZipFile(output / "piano-onset-review.zip") as archive:
        assert archive.testzip() is None
        assert not any("deblind" in name.casefold() for name in archive.namelist())

    second_output = tmp_path / "refreshed-again"
    refresh_package(output / "participant", second_output)
    first_page = (output / "participant/index.html").read_text(encoding="utf-8")
    second_page = (second_output / "participant/index.html").read_text(encoding="utf-8")
    first_key_line = next(line for line in first_page.splitlines() if line.startswith("const draftKey="))
    second_key_line = next(line for line in second_page.splitlines() if line.startswith("const draftKey="))
    assert first_key_line == second_key_line


def test_refresh_rejects_unsafe_or_duplicate_clip_identity_before_output(tmp_path):
    source = _source_package(tmp_path / "source")
    template_path = source / "review-template.csv"
    with template_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["audio_file"] = "../private.wav"
    with template_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "refreshed"

    with pytest.raises(ValueError, match="invalid clip path"):
        refresh_package(source, output)

    assert not output.exists()


@pytest.mark.parametrize("relative_path", ["clips/R001.wav", "review-template.csv"])
def test_refresh_rejects_participant_bytes_changed_after_archive(tmp_path, relative_path):
    source = _source_package(tmp_path / "source")
    changed_file = source / relative_path
    content = changed_file.read_bytes()
    if relative_path == "review-template.csv":
        changed_file.write_bytes(content.replace(b"1010", b"1011"))
    else:
        changed_file.write_bytes(content + b"changed")
    output = tmp_path / "refreshed"

    with pytest.raises(ValueError, match="manifest|archive"):
        refresh_package(source, output)

    assert not output.exists()


def test_refresh_preserves_existing_outputs(tmp_path):
    source = _source_package(tmp_path / "source")
    output = tmp_path / "refreshed"
    output.mkdir()
    (output / "sentinel").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="new output directory"):
        refresh_package(source, output)

    assert (output / "sentinel").read_text(encoding="utf-8") == "keep"
