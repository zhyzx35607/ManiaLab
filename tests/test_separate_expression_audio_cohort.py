import hashlib
import json
import struct

import numpy as np
import pytest

from scripts.separate_expression_audio_cohort import (
    SOURCES,
    canonical_hash,
    load_training_inputs,
    read_float_stereo_wav_frames,
    validate_worker_result,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _make_cohort(root):
    rows = []
    fetch = []
    for beatmap_id, split in ((101, "train"), (202, "validation"), (303, "test")):
        chart = f"chart-{beatmap_id}".encode()
        audio = f"audio-{beatmap_id}".encode()
        row = {
            "beatmap_id": beatmap_id,
            "group": f"group-{beatmap_id}",
            "keys": 7,
            "ln_class": "hybrid",
            "layout_training_split": split,
            "sha256": _sha(chart),
        }
        rows.append(row)
        fetched_reference = {
            "beatmap_id": beatmap_id,
            "group": row["group"],
            "keys": row["keys"],
            "layout_training_split": split,
            "chart_sha256": row["sha256"],
            "audio_sha256": _sha(audio),
        }
        fetch.append({"ok": True, "reference": fetched_reference})
        if split != "train":
            continue
        target = root / str(beatmap_id)
        target.mkdir(parents=True)
        (target / "reference.osu").write_bytes(chart)
        (target / "audio.mp3").write_bytes(audio)
        _write_json(target / "reference.json", {
            **fetched_reference,
            "audio_filename": "audio.mp3",
        })

    _write_json(root / "selection.json", rows)
    _write_json(root / "report.json", {
        "selected_song_groups": len(rows),
        "selection_sha256": canonical_hash(rows),
    })
    _write_json(root / "fetch-report.json", fetch)


def test_loader_verifies_frozen_pairs_and_selects_train_only(tmp_path):
    _make_cohort(tmp_path)

    rows, selection_sha256, fetch_sha256 = load_training_inputs(tmp_path)

    assert [row["beatmap_id"] for row in rows] == [101]
    assert rows[0]["split"] == "train"
    assert rows[0]["audio_sha256"] == _sha(b"audio-101")
    assert len(selection_sha256) == 64
    assert len(fetch_sha256) == 64


def test_canonical_hash_matches_frozen_selector_unicode_encoding():
    value = [{"title": "樱花草", "group": "set:123"}]
    expected = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert canonical_hash(value) == expected


def test_loader_rejects_audio_hash_mismatch(tmp_path):
    _make_cohort(tmp_path)
    (tmp_path / "101" / "audio.mp3").write_bytes(b"changed")

    with pytest.raises(ValueError, match="audio hash mismatch"):
        load_training_inputs(tmp_path)


def test_float_stereo_wav_frame_reader(tmp_path):
    fmt = struct.pack("<HHIIHH", 3, 2, 44100, 44100 * 2 * 4, 8, 32)
    data = bytes(8 * 2 * 4)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(data)) + data
    wav = b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks
    path = tmp_path / "input.wav"
    path.write_bytes(wav)

    assert read_float_stereo_wav_frames(path) == 8


def test_worker_result_checks_model_shape_and_finite_values(tmp_path):
    frames = 2
    sources = {voice: f"{voice}.npy" for voice in SOURCES}
    for voice in SOURCES:
        np.save(tmp_path / sources[voice], np.ones(frames, dtype=np.float32))
    _write_json(tmp_path / "separation.json", {
        "model_sha256": "a" * 64,
        "sample_rate": 22050,
        "frames": frames,
        "sources": sources,
        "reconstruction_rmse": 0.0,
    })

    result = validate_worker_result(tmp_path, "a" * 64, 4)

    assert result["frames"] == frames
    assert set(result["stems"]) == set(SOURCES)
    assert all(len(item["sha256"]) == 64 for item in result["stems"].values())


def test_worker_result_rejects_non_finite_stem(tmp_path):
    sources = {voice: f"{voice}.npy" for voice in SOURCES}
    for voice in SOURCES:
        values = np.ones(2, dtype=np.float32)
        if voice == "other":
            values[0] = np.nan
        np.save(tmp_path / sources[voice], values)
    _write_json(tmp_path / "separation.json", {
        "model_sha256": "b" * 64,
        "sample_rate": 22050,
        "frames": 2,
        "sources": sources,
        "reconstruction_rmse": 0.0,
    })

    with pytest.raises(ValueError, match="Non-finite samples"):
        validate_worker_result(tmp_path, "b" * 64, 4)
