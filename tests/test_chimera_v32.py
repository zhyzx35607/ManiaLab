from __future__ import annotations

import zipfile

import numpy as np
import pytest
from scipy.io import wavfile

from manialab import chimera_v32
from manialab.domain import parse_osu, validate


def _chart(keys=10, audio="source.wav"):
    return "\n".join([
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio}",
        "Mode: 3",
        "",
        "[Metadata]",
        "Title: Chimera fixture",
        "Artist: Test",
        "Creator: Test",
        "Version: Source",
        "BeatmapID: 3",
        "BeatmapSetID: 4",
        "",
        "[Difficulty]",
        f"CircleSize: {keys}",
        "OverallDifficulty: 7",
        "",
        "[Events]",
        '0,0,"bg.png",0,0',
        "",
        "[TimingPoints]",
        "1000,500,4,2,0,70,1,0",
        "",
        "[HitObjects]",
        "25,192,100,128,0,900:0:0:0:0:",
        "76,192,1200,1,0,0:0:0:0:",
        "128,192,2000,128,0,2400:0:0:0:0:",
        "",
    ])


def test_cache_environment_requires_d_drive_and_keeps_all_caches_together(tmp_path):
    env = chimera_v32.configure_d_cache_environment(
        {},
        cache_root=r"D:\.cache\chimera-test",
        temp_root=r"D:\.tmp\chimera-test",
    )

    assert env["HF_HOME"].startswith("D:")
    assert env["TORCH_HOME"].startswith("D:")
    assert env["TMP"].startswith("D:")
    assert env["TEMP"].startswith("D:")
    assert env["UV_CACHE_DIR"].startswith("D:")

    with pytest.raises(ValueError, match="D: 盘"):
        chimera_v32.configure_d_cache_environment(
            {},
            cache_root=r"C:\cache",
            temp_root=tmp_path / "tmp",
        )


def test_mapper_command_pins_v32_mania_and_passes_window_controls(tmp_path):
    command = chimera_v32.build_mapper_command(
        audio_path=tmp_path / "excerpt.wav",
        reference_chart=tmp_path / "reference.osu",
        output_dir=tmp_path / "out",
        model_dir=tmp_path / "gamemode=3",
        keycount=7,
        duration_ms=60000,
        bpm=150,
        offset_ms=-1000,
        hold_note_ratio=0.42,
        threads=2,
        start_ms=10000,
        end_ms=40000,
    )

    joined = " ".join(command)
    assert "--config-name v32-mini" in joined
    assert "fast_decoder_loop=true" in command
    assert "gamemode=3" in command
    assert "auto_select_gamemode_model=false" in command
    assert "start_time=10000" in command
    assert "end_time=40000" in command
    assert "keycount=7" in command
    assert "hold_note_ratio=0.4200" in command
    with pytest.raises(ValueError, match="窗口"):
        chimera_v32.build_mapper_command(
            audio_path=tmp_path / "excerpt.wav",
            reference_chart=tmp_path / "reference.osu",
            output_dir=tmp_path / "out",
            model_dir=tmp_path / "gamemode=3",
            keycount=7,
            duration_ms=60000,
            bpm=150,
            offset_ms=0,
            hold_note_ratio=0.42,
            start_ms=40000,
            end_ms=70000,
        )


def test_benchmark_projects_full_song_with_fixed_model_load_cost():
    report = {
        "start_ms": 51181,
        "end_ms": 81181,
        "setup_s": 14.38,
        "load_s": 2.09,
        "generate_s": 115.87,
    }

    assert chimera_v32.benchmark_full_song_seconds(report, 201.6) == pytest.approx(795.116, abs=0.2)


def test_reuse_report_requires_exact_audio_clock_and_model_identity(tmp_path):
    audio = tmp_path / "audio.wav"
    chart = tmp_path / "reference.osu"
    audio.write_bytes(b"audio A")
    chart.write_text(_chart(), encoding="utf-8")
    identity = chimera_v32.inference_identity(
        audio_path=audio,
        reference_chart=chart,
        keycount=7,
        bpm=150,
        offset_ms=-110819,
        hold_note_ratio=0.426,
        threads=6,
        precision="fp32",
        model_dir=r"D:\.cache\model\gamemode=3",
        start_ms=0,
        end_ms=45000,
    )
    report = {"model_revision": chimera_v32.MODEL_REVISION, "input_identity": identity}

    chimera_v32.validate_reusable_inference_report(report, identity)
    changed_clock = identity | {"offset_ms": -110818}
    with pytest.raises(ValueError, match="offset_ms"):
        chimera_v32.validate_reusable_inference_report(report, changed_clock)
    changed_audio = tmp_path / "other.wav"
    changed_audio.write_bytes(b"audio B")
    changed_input = identity | {
        "audio_sha256": chimera_v32._sha256_file(changed_audio),
    }
    with pytest.raises(ValueError, match="audio_sha256"):
        chimera_v32.validate_reusable_inference_report(report, changed_input)
    with pytest.raises(ValueError, match="缺少完整输入身份"):
        chimera_v32.validate_reusable_inference_report(
            {"model_revision": chimera_v32.MODEL_REVISION},
            identity,
        )


def test_memory_sampler_runs_from_the_d_drive_python_environment():
    value, sample = chimera_v32._measured_call(lambda: 7 * 6)

    assert value == 42
    assert sample["elapsed_seconds"] > 0
    assert sample["peak_rss_bytes"] > 0


def test_memory_monitor_falls_back_to_the_current_python(monkeypatch, tmp_path):
    monkeypatch.setattr(chimera_v32, "MAPPER_PYTHON", tmp_path / "missing-python.exe")
    monkeypatch.setattr(chimera_v32.sys, "executable", str(tmp_path / "test-python.exe"))
    captured = {}

    class Monitor:
        pass

    def launch(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Monitor()

    monkeypatch.setattr(chimera_v32.subprocess, "Popen", launch)
    stop_path = tmp_path / "monitor.stop"
    monitor = chimera_v32._start_rss_monitor(123, stop_path, {})

    assert isinstance(monitor, Monitor)
    assert captured["command"][0] == str(tmp_path / "test-python.exe")


def test_climax_picker_selects_the_highest_energy_window():
    rate = 1000
    wave = np.zeros(rate * 80, dtype=np.float32)
    wave[42000:52000] = 0.5
    wave[10000:20000] = 0.1

    start, rms = chimera_v32.choose_climax_window(wave, rate, duration_seconds=10)

    assert 41000 <= start <= 43000
    assert rms == pytest.approx(0.5)


def test_no_vocal_mixer_uses_weighted_non_vocal_stems(tmp_path):
    stems = tmp_path / "stems"
    stems.mkdir()
    np.save(stems / "vocals.npy", np.full(1000, 0.8, dtype=np.float32))
    np.save(stems / "drums.npy", np.full(1000, 0.1, dtype=np.float32))
    np.save(stems / "piano.npy", np.full(1000, 0.1, dtype=np.float32))
    np.save(stems / "bass.npy", np.full(1000, 0.1, dtype=np.float32))
    np.save(stems / "other.npy", np.full(1000, 0.1, dtype=np.float32))
    np.save(stems / "guitar.npy", np.full(1000, 0.1, dtype=np.float32))

    report = chimera_v32.remix_no_vocal(stems, tmp_path / "no_vocal.wav")
    rate, audio = wavfile.read(report["path"])

    assert rate == 22050
    assert audio.shape == (1000, 2)
    assert np.mean(audio) / 32767 == pytest.approx(0.53, abs=0.01)
    assert report["gains"]["vocals"] == 0
    assert report["gains"]["piano"] == 1.2


def test_energy_decoder_preserves_ffmpeg_normalized_float_units(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "manialab.audio_io.decode",
        lambda *_args, **_kwargs: np.array([0.5, -0.25], dtype=np.float32),
    )

    wave, rate = chimera_v32._decode_for_energy(tmp_path / "input.wav", tmp_path / "work")

    assert rate == 22050
    np.testing.assert_array_equal(wave, np.array([0.5, -0.25], dtype=np.float32))


def test_air_ln_trimmer_only_shortens_long_near_silent_tails():
    rate = 1000
    audio = np.zeros(3500, dtype=np.float32)
    audio[:620] = 0.1
    audio[2000:2300] = 0.1
    text = _chart()

    result, report = chimera_v32.trim_air_ln_tails(
        text,
        audio,
        rate,
        grid_bpm=None,
    )
    parsed = parse_osu(result)

    assert report["examined_lns"] == 2
    assert report["trimmed_lns"] == 1
    assert 620 <= parsed["notes"][0].end <= 640
    assert parsed["notes"][1].start == 1200
    assert parsed["notes"][2].end == 2400
    assert validate(parsed["notes"], 10)["valid"]


def test_zero_length_model_ln_keeps_its_attack_as_a_tap():
    malformed = _chart().replace(
        "25,192,100,128,0,900:0:0:0:0:",
        "25,192,100,128,0,100:2:2:-1:30:",
    )

    repaired, report = chimera_v32.sanitize_zero_length_model_holds(malformed)
    parsed = parse_osu(repaired)

    assert report["downgraded"] == 1
    assert parsed["notes"][0].start == 100
    assert not parsed["notes"][0].hold
    assert parsed["notes"][0].raw.endswith(",2:2:-1:30:")
    assert validate(parsed["notes"], 10)["valid"]


def test_native_10k_collision_repair_preserves_object_count_and_timing():
    collided = _chart().replace(
        "76,192,1200,1,0,0:0:0:0:",
        "25,192,500,1,0,0:0:0:0:",
    )
    before = parse_osu(collided)
    assert not validate(before["notes"], 10)["valid"]

    repaired, report = chimera_v32.repair_native_10k_collisions(collided)
    after = parse_osu(repaired)

    assert report["reassigned"] is True
    assert len(after["notes"]) == len(before["notes"])
    assert sorted((n.start, n.end, n.hold) for n in after["notes"]) == sorted(
        (n.start, n.end, n.hold) for n in before["notes"]
    )
    assert validate(after["notes"], 10)["valid"]


def test_grid_snap_moves_only_endpoints_and_restores_confirmed_clock():
    shifted = _chart().replace(
        "25,192,100,128,0,900:0:0:0:0:",
        "25,192,101,128,0,903:0:0:0:0:",
    )

    snapped, report = chimera_v32.snap_chart_to_grid(
        shifted,
        bpm=120,
        offset_ms=0,
    )
    parsed = parse_osu(snapped)

    assert report["adjusted_objects"] >= 1
    assert report["max_endpoint_shift_ms"] <= 7
    assert chimera_v32.validate_map_grid(snapped, bpm=120, offset_ms=0)["off_grid"] == 0
    assert len(parsed["notes"]) == 3


def test_reference_timing_shifts_into_excerpt_and_accepts_confirmed_clock():
    shifted, bpm, offset = chimera_v32.shift_reference_timing(
        _chart(),
        segment_start_ms=3000,
    )
    parsed = parse_osu(shifted)
    assert bpm == pytest.approx(120)
    assert offset == pytest.approx(-2000)
    assert float(parsed["timing"][0].split(",")[0]) == pytest.approx(-2000)

    explicit, bpm, offset = chimera_v32.shift_reference_timing(
        _chart(),
        segment_start_ms=3000,
        bpm=150,
        offset_ms=1000,
    )
    parsed = parse_osu(explicit)
    assert bpm == 150
    assert offset == -2000
    assert parsed["timing"] == ["-2000.000,400.00000000,4,2,0,70,1,0"]


def test_grid_validator_checks_both_starts_and_ln_releases():
    aligned = _chart().replace(
        "1000,500,4,2,0,70,1,0", "0,500,4,2,0,70,1,0",
    ).replace(
        "25,192,100,128,0,900:0:0:0:0:",
        "25,192,500,128,0,1500:0:0:0:0:",
    ).replace(
        "76,192,1200,1,0,0:0:0:0:",
        "76,192,2000,1,0,0:0:0:0:",
    ).replace(
        "128,192,2000,128,0,2400:0:0:0:0:",
        "128,192,2500,128,0,3000:0:0:0:0:",
    )

    assert chimera_v32.validate_map_grid(aligned, bpm=120, offset_ms=0)["off_grid"] == 0
    with pytest.raises(ValueError, match="离线"):
        chimera_v32.validate_map_grid(
            aligned.replace("1500:0:0:0:0:", "1489:0:0:0:0:"),
            bpm=120,
            offset_ms=0,
        )


def test_four_difficulty_package_is_playable_and_keeps_audio_route_per_chart(tmp_path):
    source = tmp_path / "source.osu"
    source.write_text(_chart(keys=7, audio="old.mp3"), encoding="utf-8")
    (tmp_path / "bg.png").write_bytes(b"background")
    (tmp_path / "old.mp3").write_bytes(b"old audio")
    full = tmp_path / "full_mix.wav"
    full.write_bytes(b"full audio")
    labels = chimera_v32._CHART_LABELS
    charts = {label: _chart(keys=10) for label in labels}
    audios = {label: full for label in labels}
    output = tmp_path / "comparison.osz"

    result = chimera_v32.package_four_difficulties(
        output_path=output,
        source_chart_path=source,
        source_chart_member=None,
        chart_texts=charts,
        audio_by_label=audios,
        duration_ms=3000,
    )

    assert output.is_file()
    assert len(result["chart_paths"]) == 4
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert "full_mix.wav" in archive.namelist()
        assert "no_vocal.wav" not in archive.namelist()
        assert "bg.png" in archive.namelist()
        for label, path in zip(labels, result["chart_paths"]):
            parsed = parse_osu(archive.read(path).decode("utf-8"))
            assert parsed["metadata"]["Version"] == label
            assert parsed["metadata"]["BeatmapID"] == "0"
            assert parsed["general"]["AudioFilename"] == "full_mix.wav"
            assert validate(parsed["notes"], 10)["valid"]
