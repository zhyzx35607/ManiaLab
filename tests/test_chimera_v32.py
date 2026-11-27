from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from manialab import chimera_v32, v32_audio, v32_charts, v32_packaging
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


def test_chimera_is_a_portable_compatibility_module():
    source = Path(chimera_v32.__file__).read_text(encoding="utf-8")
    assert len(source.splitlines()) < 80
    assert "D:\\" not in source
    assert chimera_v32.trim_air_ln_tails is v32_charts.trim_air_ln_tails
    assert chimera_v32.snap_chart_to_grid is v32_charts.snap_chart_to_grid
    assert chimera_v32.validate_map_grid is v32_charts.validate_map_grid
    assert chimera_v32.repair_native_10k_collisions is v32_charts.repair_native_10k_collisions
    assert chimera_v32.shift_reference_timing is v32_charts.shift_reference_timing
    assert chimera_v32.package_four_difficulties is v32_packaging.package_four_difficulties


def test_audio_helpers_choose_climax_and_decode_normalized_samples(tmp_path, monkeypatch):
    rate = 1000
    audio = np.zeros(rate * 80, dtype=np.float32)
    audio[42000:52000] = 0.5
    audio[10000:20000] = 0.1
    start, rms = v32_audio.choose_climax_window(audio, rate, duration_seconds=10)
    assert 41000 <= start <= 43000
    assert rms == pytest.approx(0.5)

    monkeypatch.setattr(
        "manialab.audio_io.decode",
        lambda *_args, **_kwargs: np.array([0.5, -0.25], dtype=np.float32),
    )
    decoded, sample_rate = chimera_v32._decode_for_energy(
        tmp_path / "input.wav", tmp_path / "work",
    )
    assert sample_rate == 22050
    np.testing.assert_array_equal(decoded, np.array([0.5, -0.25], dtype=np.float32))


def test_air_ln_trimmer_only_shortens_long_near_silent_tails():
    rate = 1000
    audio = np.zeros(3500, dtype=np.float32)
    audio[:620] = 0.1
    audio[2000:2300] = 0.1
    result, report = v32_charts.trim_air_ln_tails(_chart(), audio, rate, grid_bpm=None)
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
    repaired, report = v32_charts.sanitize_zero_length_model_holds(malformed)
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
    repaired, report = v32_charts.repair_native_10k_collisions(collided)
    after = parse_osu(repaired)
    assert report["reassigned"] is True
    assert len(after["notes"]) == len(before["notes"])
    assert sorted((n.start, n.end, n.hold) for n in after["notes"]) == sorted(
        (n.start, n.end, n.hold) for n in before["notes"]
    )
    assert validate(after["notes"], 10)["valid"]


def test_native_collision_repair_does_not_drop_an_overfull_chord():
    keys = 10
    chart = _chart(keys=keys)
    rows = []
    for lane in range(keys + 1):
        x = int((lane % keys + 0.5) * 512 / keys)
        rows.append(f"{x},192,1000,1,0,0:0:0:0:")
    chart = chart.split("[HitObjects]", 1)[0] + "[HitObjects]\n" + "\n".join(rows) + "\n"

    assert len(parse_osu(chart)["notes"]) == keys + 1
    with pytest.raises(ValueError):
        v32_charts.repair_native_10k_collisions(chart)


def test_grid_snap_moves_only_endpoints_and_restores_confirmed_clock():
    shifted = _chart().replace(
        "25,192,100,128,0,900:0:0:0:0:",
        "25,192,101,128,0,903:0:0:0:0:",
    )
    snapped, report = v32_charts.snap_chart_to_grid(shifted, bpm=120, offset_ms=0)
    parsed = parse_osu(snapped)
    assert report["adjusted_objects"] >= 1
    assert report["max_endpoint_shift_ms"] <= 7
    assert v32_charts.validate_map_grid(snapped, bpm=120, offset_ms=0)["off_grid"] == 0
    assert len(parsed["notes"]) == 3


def test_snapped_model_collision_cleanup_keeps_long_lns_and_unique_tracks():
    keys = 7
    chart = _chart(keys=keys)
    rows = []
    specs = [
        (0, False, 1000),
        (0, True, 2300),
        (1, True, 5000),
        (1, True, 2200),
        (2, False, 1000),
        (3, False, 1000),
        (4, False, 1000),
        (5, False, 1000),
        (6, False, 1000),
    ]
    for lane, hold, end in specs:
        x = int((lane + 0.5) * 512 / keys)
        if hold:
            rows.append(f"{x},192,1000,128,0,{end}:0:0:0:0:")
        else:
            rows.append(f"{x},192,1000,1,0,0:0:0:0:")
    chart = chart.split("[HitObjects]", 1)[0] + "[HitObjects]\n" + "\n".join(rows) + "\n"

    cleaned, report = v32_charts.clean_snapped_model_collisions(
        chart,
        model_keys=keys,
    )
    parsed = parse_osu(cleaned)
    assert report["objects_before"] == 9
    assert report["objects_after"] == 7
    assert report["removed_objects"] == 2
    assert report["duplicate_track_objects_removed"] == 2
    assert report["capacity_objects_removed"] == 0
    assert report["collided_chords"] == 1
    assert report["overfull_chords"] == 1
    assert [note.lane for note in parsed["notes"]] == list(range(7))
    assert [(note.lane, note.end) for note in parsed["notes"] if note.hold] == [
        (0, 2300),
        (1, 5000),
    ]


def test_reference_timing_shifts_into_excerpt_and_accepts_confirmed_clock():
    shifted, bpm, offset = v32_charts.shift_reference_timing(_chart(), segment_start_ms=3000)
    parsed = parse_osu(shifted)
    assert bpm == pytest.approx(120)
    assert offset == pytest.approx(-2000)
    assert float(parsed["timing"][0].split(",")[0]) == pytest.approx(-2000)
    explicit, bpm, offset = v32_charts.shift_reference_timing(
        _chart(), segment_start_ms=3000, bpm=150, offset_ms=1000,
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
    assert v32_charts.validate_map_grid(aligned, bpm=120, offset_ms=0)["off_grid"] == 0
    with pytest.raises(ValueError, match="离线"):
        v32_charts.validate_map_grid(
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
    labels = v32_packaging._CHART_LABELS
    charts = {label: _chart(keys=10) for label in labels}
    audios = {label: full for label in labels}
    output = tmp_path / "comparison.osz"
    result = v32_packaging.package_four_difficulties(
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


def test_four_difficulty_package_accepts_phase04_labels_in_requested_order(tmp_path):
    source = tmp_path / "source.osu"
    source.write_text(_chart(keys=7, audio="old.mp3"), encoding="utf-8")
    (tmp_path / "bg.png").write_bytes(b"background")
    (tmp_path / "old.mp3").write_bytes(b"old audio")
    full = tmp_path / "full_mix.wav"
    full.write_bytes(b"full audio")
    labels = (
        "[1-V32Mini-7K-TorosuDense10K]",
        "[2-V32Full-7K-TorosuDense10K]",
        "[3-V32Full-Native10K-LN55]",
        "[4-V32Full-7K-TorosuMoreDense10K]",
    )
    output = tmp_path / "phase04.osz"
    result = v32_packaging.package_four_difficulties(
        output_path=output,
        source_chart_path=source,
        source_chart_member=None,
        chart_texts={label: _chart(keys=10) for label in labels},
        audio_by_label={label: full for label in labels},
        duration_ms=3000,
        labels=labels,
    )
    assert tuple(Path(path).stem for path in result["chart_paths"]) == labels
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for label, path in zip(labels, result["chart_paths"]):
            parsed = parse_osu(archive.read(path).decode("utf-8"))
            assert parsed["metadata"]["Version"] == label
            assert parsed["general"]["AudioFilename"] == "full_mix.wav"
    with pytest.raises(ValueError, match="唯一"):
        v32_packaging.package_four_difficulties(
            output_path=tmp_path / "invalid.osz",
            source_chart_path=source,
            source_chart_member=None,
            chart_texts={label: _chart(keys=10) for label in labels},
            audio_by_label={label: full for label in labels},
            duration_ms=3000,
            labels=(labels[0], labels[1], labels[2], labels[2]),
        )
