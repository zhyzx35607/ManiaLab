import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

import scripts.build_expression_cohort_stem_review as review
from scripts.separate_expression_audio_cohort import (
    INPUT_PROCESSING,
    MODEL_SHA256,
    SOURCES,
    canonical_hash,
    load_training_inputs,
    record_identity,
    sha256_file,
    validate_worker_result,
)


def _chart():
    return """osu file format v14
[General]
AudioFilename: track.wav
Mode:3
[Metadata]
Title: Fixture
Artist: Test
Creator: Reviewer
Version: Train only
[Difficulty]
CircleSize:6
[TimingPoints]
0,500,4,2,0,70,1,0
[HitObjects]
42,192,0,128,0,2000:0:0:0:0:
128,192,500,128,0,1500:0:0:0:0:
213,192,500,128,0,2500:0:0:0:0:
298,192,750,1,0,0:0:0:0:
384,192,1000,1,0,0:0:0:0:
"""


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path, monkeypatch, stored_audio_filename="track.wav"):
    root = tmp_path / "project"
    cohort = root / "test-data" / "expression-audio-cohort-v3"
    separated = root / "test-data" / "expression-audio-cohort-v3-separated-train-v1"
    docs = root / "docs"
    docs.mkdir(parents=True)
    protocol = docs / "expression-audio-stem-review-protocol.md"
    protocol.write_text("# fixture protocol\n", encoding="utf-8")
    monkeypatch.setattr(review, "PROTOCOL_PATH", protocol)

    chart_bytes = _chart().encode("utf-8")
    chart_hash = hashlib.sha256(chart_bytes).hexdigest()
    audio_path = cohort / "101" / stored_audio_filename
    audio_path.parent.mkdir(parents=True)
    audio = np.zeros((44100 * 10, 2), dtype=np.float32)
    time = np.arange(len(audio), dtype=np.float32) / 44100
    audio[:, 0] = 0.1 * np.sin(2 * np.pi * 220 * time)
    audio[:, 1] = 0.1 * np.sin(2 * np.pi * 330 * time)
    wavfile.write(audio_path, 44100, audio)
    audio_hash = sha256_file(audio_path)
    (cohort / "101" / "reference.osu").write_bytes(chart_bytes)

    selection = []
    for beatmap_id, split in ((101, "train"), (202, "validation"), (303, "test")):
        selection.append({
            "beatmap_id": beatmap_id,
            "group": f"group-{beatmap_id}",
            "keys": 6,
            "ln_class": "hybrid",
            "layout_training_split": split,
            "sha256": chart_hash if split == "train" else f"chart-{beatmap_id}",
        })
    selected = selection[0]
    reference = {
        "beatmap_id": 101,
        "group": selected["group"],
        "keys": 6,
        "layout_training_split": "train",
        "chart_sha256": chart_hash,
        "audio_sha256": audio_hash,
        "audio_filename": stored_audio_filename,
    }
    _write_json(cohort / "101" / "reference.json", reference)
    _write_json(cohort / "selection.json", selection)
    selection_sha = canonical_hash(selection)
    fetch_report = [{"ok": True, "reference": reference}]
    _write_json(cohort / "fetch-report.json", fetch_report)
    _write_json(cohort / "report.json", {
        "selected_song_groups": 3,
        "selection_sha256": selection_sha,
    })

    rows, selection_sha, fetch_sha = load_training_inputs(cohort)
    separated.mkdir(parents=True)
    run_identity = {
        "schema": "manialab-expression-audio-stems-v1",
        "split": "train",
        "selection_sha256": selection_sha,
        "fetch_report_sha256": fetch_sha,
        "model_sha256": MODEL_SHA256,
        "sources": list(SOURCES),
        "profile": "four",
        "input_processing": INPUT_PROCESSING,
        "threads": 2,
        "interop_threads": 1,
        "records": [record_identity(row) for row in rows],
    }
    run_sha = canonical_hash(run_identity)
    _write_json(separated / "run.json", run_identity)
    output_dir = separated / "101"
    output_dir.mkdir()
    sources = {name: f"{name}.npy" for name in SOURCES}
    for index, source in enumerate(SOURCES, start=1):
        value = 2.0 if source == "vocals" else index * 0.1
        np.save(
            output_dir / sources[source],
            np.full(44100 * 5, value, dtype=np.float32),
        )
    _write_json(output_dir / "separation.json", {
        "model_sha256": MODEL_SHA256,
        "sample_rate": 22050,
        "frames": 44100 * 5,
        "sources": sources,
        "reconstruction_rmse": 0.0,
    })
    validation = validate_worker_result(
        output_dir, MODEL_SHA256, input_frames=44100 * 10
    )
    complete = {
        "input": record_identity(rows[0]),
        "model_sha256": MODEL_SHA256,
        "run_identity_sha256": run_sha,
        "input_frames_44100": 44100 * 10,
        "input_seconds": 10.0,
        "validation": validation,
    }
    _write_json(output_dir / "complete.json", complete)
    _write_json(separated / "summary.json", {
        "run_identity_sha256": run_sha,
        "expected_training_groups": 1,
        "completed_training_groups": 1,
        "completed_beatmap_ids": [101],
    })

    ffmpeg = root / "bin" / "ffmpeg.exe"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_bytes(b"fixture")
    monkeypatch.setattr(
        review,
        "_decode_mix",
        lambda _ffmpeg, _source, _destination, _input, frames:
            np.full(frames, 0.1, dtype=np.float32),
    )
    return root, cohort, separated, ffmpeg


def test_clip_bounds_and_pcm16_preserve_shared_gain():
    assert review.clip_sample_bounds(500, 900, 5000) == (0, 52920)
    assert review.clip_sample_bounds(3000, 3400, 5000) == (33075, 108045)
    samples = np.asarray([-0.5, 0.0, 0.5], dtype=np.float32)
    assert review.pcm16(samples, 0.8).tolist() == [-13107, 0, 13107]
    with pytest.raises(ValueError, match="would clip"):
        review.pcm16(np.asarray([1.0]), 1.1)
    with pytest.raises(ValueError, match="starts beyond"):
        review.clip_sample_bounds(5000, 5500, 5000)


def test_script_help_runs_from_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(review.__file__).resolve()), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--cohort" in result.stdout


def test_chart_windows_carry_only_authored_action_targets():
    from manialab.domain import parse_osu

    parsed = parse_osu(_chart())
    windows = review._chart_windows(parsed, duration_frames=220500)

    assert len(windows) == 1
    assert windows[0]["event_count"] == 5
    assert windows[0]["ln_count"] == 3
    assert {
        "含LN", "含2拍以上LN", "含LN和弦", "含错开放手", "含持LN期间tap"
    } <= set(windows[0]["action_tags"])


def test_decode_mix_uses_stereo_mean_and_separation_resampler(tmp_path, monkeypatch):
    rate = 44100
    stereo = np.column_stack((
        np.linspace(-0.5, 0.5, 101, dtype=np.float32),
        np.linspace(0.25, -0.25, 101, dtype=np.float32),
    ))
    decoded = tmp_path / "decoded.wav"
    wavfile.write(decoded, rate, stereo)
    destination = tmp_path / "temporary.wav"

    def fake_ffmpeg(command, **_kwargs):
        import shutil
        shutil.copyfile(decoded, command[-1])
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(review.subprocess, "run", fake_ffmpeg)
    actual = review._decode_mix(
        Path("ffmpeg.exe"), Path("source.ogg"), destination,
        expected_input_frames=len(stereo), expected_output_frames=51,
    )
    mono = np.mean(stereo, axis=1, dtype=np.float32)
    expected = resample_poly(mono, 1, 2).astype(np.float32, copy=False)

    assert np.array_equal(actual, expected)
    assert not destination.exists()


def test_window_selection_prefers_richer_then_earlier_chart_window():
    windows = [
        {"window_index": 3, "_richness": 2, "ln_count": 1, "event_count": 9},
        {"window_index": 4, "_richness": 3, "ln_count": 0, "event_count": 4},
        {"window_index": 1, "_richness": 3, "ln_count": 0, "event_count": 4},
    ]

    assert review._select_case_window(windows)["window_index"] == 1


def test_build_review_bundle_uses_only_train_audio_and_exports_five_tracks(
    tmp_path, monkeypatch
):
    root, cohort, separated, ffmpeg = _fixture(
        tmp_path, monkeypatch, stored_audio_filename="audio.wav"
    )
    output = root / "test-data" / "expression-audio-cohort-v3-stem-review-v1"

    report = review.build_review_bundle(
        cohort_root=cohort,
        separated_root=separated,
        output=output,
        ffmpeg=ffmpeg,
        root=root,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["case_count"] == 1
    assert report["clip_count"] == 5
    assert report["validation_test_audio_read"] is False
    assert manifest["cohort"]["group_count"] == 1
    assert [case["beatmap_id"] for case in manifest["selection"]["cases"]] == [101]
    assert manifest["review"]["musical_acceptance"] is False
    assert manifest["playback"]["per_clip_normalization"] is False
    assert manifest["playback"]["gain_is_global_across_all_cases_and_tracks"] is True
    assert report["global_gain"] == pytest.approx(0.475)
    assert all((output / path).is_file() for path in manifest["selection"]["cases"][0]["clips"].values())
    mix_rate, mix_clip = wavfile.read(
        output / manifest["selection"]["cases"][0]["clips"]["mix"]
    )
    assert mix_rate == 22050
    assert mix_clip[100] == np.rint(0.1 * 0.475 * 32767)
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "谱面时间和LN端点只是作者编排选择" in page
    assert "导出审核 JSON" in page
    assert not (cohort / "202").exists()
    assert not (cohort / "303").exists()

    with pytest.raises(FileExistsError, match="Preserve previous"):
        review.build_review_bundle(
            cohort_root=cohort,
            separated_root=separated,
            output=output,
            ffmpeg=ffmpeg,
            root=root,
        )


def test_build_review_bundle_rejects_audio_extension_mismatch(tmp_path, monkeypatch):
    root, cohort, separated, ffmpeg = _fixture(
        tmp_path, monkeypatch, stored_audio_filename="audio.mp3"
    )
    output = root / "test-data" / "expression-audio-cohort-v3-stem-review-v1"

    with pytest.raises(ValueError, match="Training chart metadata mismatch"):
        review.build_review_bundle(
            cohort_root=cohort,
            separated_root=separated,
            output=output,
            ffmpeg=ffmpeg,
            root=root,
        )

    assert not output.exists()
