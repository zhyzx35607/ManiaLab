import json

import numpy as np
import pytest

import scripts.build_piano_onset_blind_review as review
from scripts.build_piano_onset_blind_review import (
    build_package,
    event_differences,
    render_review_html,
    slice_audio,
)


def _note(*, pitch=60, start=1000, audio_start=1010, voice="piano", note_id="n"):
    return {
        "voice": voice,
        "pitch": pitch,
        "start": start,
        "end": start,
        "audio_start": audio_start,
        "id": note_id,
    }


def test_event_differences_use_rendered_notes_and_ignore_other_voices(tmp_path):
    kong_path = tmp_path / "kong.json"
    transkun_path = tmp_path / "transkun.json"
    kong_path.write_text(
        json.dumps([_note(audio_start=1010), _note(voice="drums", pitch=48)]),
        encoding="utf-8",
    )
    transkun_path.write_text(
        json.dumps(
            [
                _note(audio_start=1018, note_id="same-rendered-note"),
                _note(pitch=64, start=2000, audio_start=2012, note_id="added"),
                _note(voice="drums", pitch=49),
            ]
        ),
        encoding="utf-8",
    )

    added, removed = event_differences(kong_path, transkun_path)

    assert [(condition, row["id"]) for condition, row in added] == [("transkun_only", "added")]
    assert removed == []


def test_event_differences_preserve_duplicate_multiplicity(tmp_path):
    kong_path = tmp_path / "kong.json"
    transkun_path = tmp_path / "transkun.json"
    kong_path.write_text(
        json.dumps(
            [_note(audio_start=1010, note_id="first"), _note(audio_start=1020, note_id="second")]
        ),
        encoding="utf-8",
    )
    transkun_path.write_text(
        json.dumps([_note(audio_start=1012, note_id="only")]),
        encoding="utf-8",
    )

    added, removed = event_differences(kong_path, transkun_path)

    assert added == []
    assert len(removed) == 1
    assert removed[0][0] == "kong_only"


def test_piano_event_requires_original_audio_time(tmp_path):
    path = tmp_path / "notes.json"
    note = _note()
    note.pop("audio_start")
    path.write_text(json.dumps([note]), encoding="utf-8")

    with pytest.raises(ValueError, match="source times"):
        event_differences(path, path)


def test_slice_audio_keeps_fixed_length_and_pads_before_start():
    samples = np.arange(10, dtype=np.float32)

    clip = slice_audio(samples, 10, 200, before_ms=400, after_ms=400)

    assert clip.tolist() == [0, 0, 0, 1, 2, 3, 4, 5]


def test_review_html_contains_no_condition_labels():
    content = render_review_html(
        [
            {
                "clip_id": "R001",
                "audio_file": "clips/R001.wav",
                "audio_start_label": "1.010s",
                "audio_start_ms": 1010,
                "pitch": 60,
            }
        ]
    )

    assert 'src="clips/R001.wav"' in content
    assert "MIDI 60" in content
    assert "kong" not in content.casefold()
    assert "transkun" not in content.casefold()


def test_build_package_resolves_and_verifies_report_audio_reference(tmp_path, monkeypatch):
    input_dir = tmp_path / "chart-ab"
    source_dir = tmp_path / "source"
    input_dir.mkdir()
    source_dir.mkdir()
    audio_path = source_dir / "audio.mp3"
    audio_path.write_bytes(b"cached audio")

    kong_notes = [_note(pitch=60, note_id="kong-only")]
    transkun_notes = [_note(pitch=64, note_id="transkun-only")]
    (input_dir / "kong-10k-notes.json").write_text(json.dumps(kong_notes), encoding="utf-8")
    (input_dir / "transkun-10k-notes.json").write_text(json.dumps(transkun_notes), encoding="utf-8")
    (input_dir / "report.json").write_text(
        json.dumps(
            {
                "identity": {
                    "reference_chart_read": False,
                    "direct_listening_performed": False,
                    "source_sha256": {"audio": review.sha256(audio_path)},
                },
                "cases": [
                    {
                        "name": "transkun-10k",
                        "audio_reference": "../source/audio.mp3",
                        "delta_vs_kong": {
                            "added_by_voice": {"piano": 1},
                            "removed_by_voice": {"piano": 1},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    decoded_paths = []

    def fake_decode(path, _work):
        decoded_paths.append(path.resolve())
        return np.zeros(review.SAMPLE_RATE * 3, dtype=np.float32)

    monkeypatch.setattr(review, "decode", fake_decode)
    output_dir = tmp_path / "review-output"

    manifest = build_package(input_dir, output_dir)

    assert decoded_paths == [audio_path.resolve()]
    assert manifest["samples"] == 2
    assert manifest["source_sha256"]["audio"] == review.sha256(audio_path)
    with review.zipfile.ZipFile(output_dir / "piano-onset-review.zip") as archive:
        assert archive.testzip() is None
        participant_files = archive.namelist()
        assert any(name.endswith("index.html") for name in participant_files)
        assert not any("deblind" in name.casefold() for name in participant_files)


def test_build_package_rejects_audio_hash_mismatch(tmp_path):
    input_dir = tmp_path / "chart-ab"
    input_dir.mkdir()
    audio_path = input_dir / "audio.mp3"
    audio_path.write_bytes(b"cached audio")
    (input_dir / "kong-10k-notes.json").write_text(json.dumps([_note()]), encoding="utf-8")
    (input_dir / "transkun-10k-notes.json").write_text(
        json.dumps([_note(pitch=64)]),
        encoding="utf-8",
    )
    (input_dir / "report.json").write_text(
        json.dumps(
            {
                "identity": {
                    "reference_chart_read": False,
                    "direct_listening_performed": False,
                    "source_sha256": {"audio": "0" * 64},
                },
                "cases": [
                    {
                        "name": "transkun-10k",
                        "audio_reference": "audio.mp3",
                        "delta_vs_kong": {
                            "added_by_voice": {"piano": 1},
                            "removed_by_voice": {"piano": 1},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "review-output"
    with pytest.raises(ValueError, match="audio hash differs"):
        build_package(input_dir, output_dir)
    assert not output_dir.exists()
