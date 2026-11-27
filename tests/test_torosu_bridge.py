from __future__ import annotations

import json
import sys
import time

import pytest

from manialab import torosu_bridge
from manialab.arrange import assign_lanes
from manialab.domain import Note, export_osu, parse_osu, validate


def _chart(keys=6):
    return "\n".join([
        "osu file format v14",
        "",
        "[General]",
        "AudioFilename: song.ogg",
        "Mode: 3",
        "",
        "[Metadata]",
        "Title: Bridge fixture",
        "Artist: Test",
        "Creator: Test",
        "Version: Source",
        "",
        "[Difficulty]",
        f"CircleSize:{keys}",
        "",
        "[TimingPoints]",
        "0,500,4,1,0,70,1,0",
        "",
        "[HitObjects]",
        "42,192,1000,128,0,1600:0:0:0:0:",
        "128,192,1800,1,0,0:0:0:0:",
        "213,192,2600,1,0,0:0:0:0:",
        "",
    ])


def _convert_fixture(text, keys):
    source = parse_osu(text)
    notes = assign_lanes(
        source["notes"],
        {"keys": keys, "layout_engine": "beam"},
        timing_points=source["timing"],
    )
    return export_osu(notes, keys, None, original=text)


def test_convert_osu_text_passes_cli_options_and_validates_objects(monkeypatch):
    captured = {}

    def run(text, args, *, output_file=False, **_):
        captured.update(text=text, args=args, output_file=output_file)
        return _convert_fixture(text, 10)

    monkeypatch.setattr(torosu_bridge, "_run_with_input", run)

    output = torosu_bridge.convert_osu_text(
        _chart(), 10, playstyle="10K", density="none", keep_jacks=False,
    )

    parsed = parse_osu(output)
    assert parsed["keys"] == 10
    assert len(parsed["notes"]) == 3
    assert sum(note.hold for note in parsed["notes"]) == 1
    assert validate(parsed["notes"], 10)["valid"]
    assert captured["output_file"] is True
    assert captured["args"] == [
        "convert", "--input", "{input}", "--output", "{output}", "--keys", "10",
        "--density", "none", "--strict", "--no-keep-jacks", "--playstyle", "10K",
    ]


def test_strict_conversion_can_request_native_density_expansion(monkeypatch):
    captured = {}

    def run(text, args, **_):
        captured["args"] = args
        return _convert_fixture(text, 10)

    monkeypatch.setattr(torosu_bridge, "_run_with_input", run)
    converted = torosu_bridge.convert_osu_text(_chart(), 10, density="full")

    assert parse_osu(converted)["keys"] == 10
    assert "--strict" in captured["args"]
    assert captured["args"][captured["args"].index("--density") + 1] == "full"


def test_density_conversion_may_add_objects_but_cannot_drop_source_objects(monkeypatch):
    def add_note(text, _args, *, output_file=False, **_):
        output = _convert_fixture(text, 10)
        return output.replace(
            "[HitObjects]\n",
            "[HitObjects]\n281,192,3000,1,0,0:0:0:0:\n",
        )

    monkeypatch.setattr(torosu_bridge, "_run_with_input", add_note)
    result = torosu_bridge.convert_osu_text(
        _chart(), 10, density="dense",
    )
    assert len(parse_osu(result)["notes"]) == 4

    def drop_note(text, _args, *, output_file=False, **_):
        output = _convert_fixture(text, 10)
        return "\n".join(
            line for line in output.splitlines() if ",2600," not in line
        ) + "\n"

    monkeypatch.setattr(torosu_bridge, "_run_with_input", drop_note)
    with pytest.raises(torosu_bridge.TorosuBridgeError, match="丢失"):
        torosu_bridge.convert_osu_text(
            _chart(), 10, density="dense",
        )

    with pytest.raises(ValueError, match="必须启用严格"):
        torosu_bridge.convert_osu_text(_chart(), 10, density="dense", strict=False)


def test_convert_osu_text_rejects_timing_or_object_mutation(monkeypatch):
    def drop_object(text, _args, *, output_file=False, **_):
        lines = text.splitlines()
        lines.remove("213,192,2600,1,0,0:0:0:0:")
        return "\n".join(lines) + "\n"

    monkeypatch.setattr(torosu_bridge, "_run_with_input", drop_object)
    with pytest.raises(torosu_bridge.TorosuBridgeError, match="改变"):
        torosu_bridge.convert_osu_text(_chart(), 6)


def test_convert_osu_text_rejects_hit_sample_and_hit_sound_mutation(monkeypatch):
    def mutate_hit_sound(text, _args, **_):
        lines = _convert_fixture(text, 10).splitlines()
        for index, line in enumerate(lines):
            fields = line.split(",")
            if len(fields) >= 6 and fields[2] == "1000" and int(fields[3]) & 128:
                fields[4] = "1"
                lines[index] = ",".join(fields)
                break
        return "\n".join(lines) + "\n"

    monkeypatch.setattr(torosu_bridge, "_run_with_input", mutate_hit_sound)
    with pytest.raises(torosu_bridge.TorosuBridgeError, match="采样字段"):
        torosu_bridge.convert_osu_text(_chart(), 10)


def test_ln_shape_notes_restores_source_ids_and_only_changes_hold_endpoints(monkeypatch):
    source = parse_osu(_chart())["notes"]

    def shorten_one(osu_text, _args, *, output_file=False, **_kwargs):
        lines = osu_text.splitlines()
        for index, line in enumerate(lines):
            fields = line.split(",")
            if len(fields) >= 6 and int(fields[3]) & 128:
                fields[5] = "1200:0:0:0:0:"
                lines[index] = ",".join(fields)
                break
        return "\n".join(lines) + "\n"

    monkeypatch.setattr(torosu_bridge, "_run_with_input", shorten_one)

    shaped = torosu_bridge.ln_shape_notes(source, 6, "short_ln")

    assert {(note.id, note.lane, note.start) for note in shaped} == {
        (note.id, note.lane, note.start) for note in source
    }
    assert len(shaped) == len(source)
    changed = [note for note in shaped if note.end != next(
        old.end for old in source if old.id == note.id
    )]
    assert len(changed) == 1
    assert changed[0].end == 1200
    assert {note.id for note in shaped} == {note.id for note in source}
    assert sum(note.hold for note in shaped) == sum(note.hold for note in source)


def test_analyze_osu_text_checks_key_count_and_requires_json_object(monkeypatch):
    monkeypatch.setattr(
        torosu_bridge, "_run_with_input",
        lambda _text, args, **_: json.dumps({"patterns": {"Stream": 12}}),
    )
    assert torosu_bridge.analyze_osu_text(_chart(), keys=6)["patterns"]["Stream"] == 12
    with pytest.raises(ValueError, match="键数"):
        torosu_bridge.analyze_osu_text(_chart(), keys=10)

    monkeypatch.setattr(
        torosu_bridge, "_run_with_input", lambda *_args, **_kwargs: "[]",
    )
    with pytest.raises(torosu_bridge.TorosuBridgeError, match="JSON 对象"):
        torosu_bridge.analyze_osu_text(_chart())


def test_convert_notes_restores_note_identity_and_rejects_same_lane_conflicts(monkeypatch):
    notes = [
        Note(1000, 2200, lane=0, id="hold"),
        Note(1800, 1800, lane=1, id="tap-a"),
        Note(2600, 2600, lane=2, id="tap-b"),
    ]

    monkeypatch.setattr(
        torosu_bridge,
        "convert_osu_text",
        lambda text, keys, **_: _convert_fixture(text, keys),
    )
    converted = torosu_bridge.convert_notes(
        notes, 10, source_keys=6, timing_points=["0,500,4,1,0,70,1,0"],
    )
    assert {note.id for note in converted} == {"hold", "tap-a", "tap-b"}
    assert sorted((note.start, note.end) for note in converted) == [
        (1000, 2200), (1800, 1800), (2600, 2600),
    ]
    assert validate(converted, 10)["valid"]

    def conflicting(text, keys, **_):
        converted_text = _convert_fixture(text, keys)
        before, marker, hit_objects = converted_text.partition("[HitObjects]\n")
        rows = hit_objects.splitlines()
        rows[1] = rows[1].replace(rows[1].split(",", 1)[0], rows[0].split(",", 1)[0], 1)
        return before + marker + "\n".join(rows) + "\n"

    monkeypatch.setattr(torosu_bridge, "convert_osu_text", conflicting)
    with pytest.raises(torosu_bridge.TorosuBridgeError, match="结果存在冲突"):
        torosu_bridge.convert_notes(notes, 10, source_keys=6)


def test_convert_notes_density_additions_receive_ids_without_losing_source_identity(monkeypatch):
    notes = [
        Note(1000, 2200, lane=0, id="hold"),
        Note(1800, 1800, lane=1, id="tap-a"),
        Note(2600, 2600, lane=2, id="tap-b"),
    ]
    observed = {}

    def expand(text, keys, **options):
        observed.update(options)
        converted = _convert_fixture(text, keys)
        return converted.replace(
            "[HitObjects]\n",
            "[HitObjects]\n281,192,3000,1,0,0:0:0:0:\n",
        )

    monkeypatch.setattr(torosu_bridge, "convert_osu_text", expand)
    converted = torosu_bridge.convert_notes(
        notes,
        10,
        source_keys=6,
        density="dense",
        keep_jacks=False,
    )

    assert {note.id for note in converted if note.id in {"hold", "tap-a", "tap-b"}} == {
        "hold", "tap-a", "tap-b",
    }
    added = [note for note in converted if note.source == "torosu"]
    assert len(converted) == len(notes) + 1
    assert len(added) == 1
    assert added[0].id.startswith("torosu-")
    assert added[0].hold is False
    assert observed["density"] == "dense"
    assert observed["keep_jacks"] is False
    assert validate(converted, 10)["valid"]


def test_legacy_torosu_assignment_forwards_preset_style_controls(monkeypatch):
    from manialab.arrange import assign_lanes

    note = Note(1000, 1000, lane=0, id="tap")
    observed = {}

    def convert_notes(notes, keys, **options):
        observed.update(options)
        return list(notes)

    monkeypatch.setattr("manialab.torosu_bridge.convert_notes", convert_notes)
    result = assign_lanes(
        [note],
        {
            "keys": 8,
            "key_layout": "8k_3_2_3",
            "layout_engine": "torosu",
            "torosu_keep_jacks": False,
            "torosu_density": "more-dense",
        },
    )

    assert result == [note]
    assert observed["keep_jacks"] is False
    assert observed["density"] == "more-dense"
    assert observed["playstyle"] == "3+2+3"


def test_pitch_only_notes_are_seeded_before_the_native_conversion(monkeypatch):
    observed = {}
    notes = [
        Note(0, 0, lane=-1, pitch=48, source="pitch", id="a"),
        Note(500, 500, lane=-1, pitch=64, source="pitch", id="b"),
        Note(1000, 1500, lane=-1, pitch=72, source="pitch", id="c"),
    ]

    def convert(text, keys, **_):
        source = parse_osu(text)
        observed["source_keys"] = source["keys"]
        return _convert_fixture(text, keys)

    monkeypatch.setattr(torosu_bridge, "convert_osu_text", convert)
    result = torosu_bridge.convert_notes(notes, 10, beat_ms=500)
    assert observed["source_keys"] == 7
    assert {note.id for note in result} == {"a", "b", "c"}
    assert validate(result, 10)["valid"]


def test_six_key_audio_candidates_use_a_five_key_torosu_source(monkeypatch):
    observed = {}
    notes = [
        Note(0, 0, pitch=48, source="pitch", id="a"),
        Note(500, 800, pitch=64, source="pitch", id="b"),
        Note(1000, 1000, pitch=72, source="pitch", id="c"),
    ]

    def convert(text, keys, **_):
        observed["source_keys"] = parse_osu(text)["keys"]
        return _convert_fixture(text, keys)

    monkeypatch.setattr(torosu_bridge, "convert_osu_text", convert)
    result = torosu_bridge.convert_notes(notes, 6, beat_ms=500)

    assert observed["source_keys"] == 5
    assert {note.id for note in result} == {"a", "b", "c"}
    assert validate(result, 6)["valid"]


def test_six_key_full_width_chord_uses_a_valid_native_same_key_source(monkeypatch):
    observed = {}

    def convert(text, keys, **_):
        source = parse_osu(text)
        observed["source_keys"] = source["keys"]
        return export_osu(source["notes"], keys, None, original=text)

    monkeypatch.setattr(torosu_bridge, "convert_osu_text", convert)
    notes = [Note(1000, 1000, pitch=48 + index, id=str(index)) for index in range(6)]

    result = torosu_bridge.convert_notes(notes, 6)

    assert observed["source_keys"] == 6
    assert len(result) == 6
    assert validate(result, 6)["valid"]


def test_audio_candidates_exceeding_target_chord_capacity_are_rejected(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("the native converter should not receive an invalid source chart")

    monkeypatch.setattr(torosu_bridge, "convert_osu_text", unexpected)
    notes = [Note(1000, 1000, id=str(index)) for index in range(7)]
    with pytest.raises(ValueError, match="无法在 Torosu 源谱"):
        torosu_bridge.convert_notes(notes, 6)


def test_simultaneous_notes_keep_distinct_identity_after_row_reordering(monkeypatch):
    notes = [
        Note(1000, 1300, lane=3, source="pitch", voice="bass", id="bass"),
        Note(1000, 1300, lane=0, source="pitch", voice="piano", id="piano"),
        Note(1000, 1300, lane=5, source="pitch", voice="guitar", id="guitar"),
    ]
    monkeypatch.setattr(
        torosu_bridge, "convert_osu_text",
        lambda text, keys, **_: _convert_fixture(text, keys),
    )

    converted = torosu_bridge.convert_notes(notes, 10, source_keys=6)

    assert {note.id: note.voice for note in converted} == {
        "bass": "bass",
        "piano": "piano",
        "guitar": "guitar",
    }


def test_ln_shape_preserves_lanes_and_non_ln_fields_but_allows_endpoint_changes(monkeypatch):
    monkeypatch.setattr(
        torosu_bridge,
        "_run_with_input",
        lambda text, _args, **_: text.replace(
            "42,192,1000,128,0,1600:0:0:0:0:",
            "42,192,1000,128,0,1500:0:0:0:0:",
        ),
    )
    shaped = torosu_bridge.ln_shape_osu_text(_chart(), "short_ln")
    assert parse_osu(shaped)["notes"][0].end == 1500

    monkeypatch.setattr(
        torosu_bridge,
        "_run_with_input",
        lambda text, _args, **_: text.replace(
            "42,192,1000,128,0,1600:",
            "128,192,1000,128,0,1600:",
        ),
    )
    with pytest.raises(torosu_bridge.TorosuBridgeError, match="轨道"):
        torosu_bridge.ln_shape_osu_text(_chart(), "keep")


def test_cli_cancellation_terminates_and_reaps_child_process(monkeypatch):
    monkeypatch.setattr(torosu_bridge, "_cli_path", lambda: __import__("pathlib").Path(sys.executable))
    started = time.monotonic()
    with pytest.raises(InterruptedError, match="已取消"):
        torosu_bridge._run_cli(
            ["-c", "import time; time.sleep(30)"],
            cancelled=lambda: True,
        )
    assert time.monotonic() - started < 3


def test_cli_missing_is_reported_without_falling_back(monkeypatch, tmp_path):
    monkeypatch.delenv("MANIALAB_TOROSU_CLI", raising=False)
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    monkeypatch.setattr(torosu_bridge, "_ENGINE_DIR", tmp_path / "engine")
    monkeypatch.setattr(torosu_bridge.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    with pytest.raises(torosu_bridge.TorosuBridgeError, match="尚未构建"):
        torosu_bridge._cli_path()


def test_frozen_cli_is_loaded_from_internal_bundle(monkeypatch, tmp_path):
    binary = tmp_path / "engines" / "torosu_cli" / "torosu-cli.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native cli")
    monkeypatch.delenv("MANIALAB_TOROSU_CLI", raising=False)
    monkeypatch.setattr(torosu_bridge.sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(torosu_bridge.sys, "frozen", True, raising=False)

    assert torosu_bridge._cli_path() == binary.resolve()


def test_frozen_cli_prefers_internal_bundle_over_build_target(monkeypatch, tmp_path):
    bundled = tmp_path / "engines" / "torosu_cli" / "torosu-cli.exe"
    target = tmp_path / "build" / "debug" / "torosu-cli.exe"
    bundled.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    bundled.write_bytes(b"bundled")
    target.write_bytes(b"build")
    monkeypatch.delenv("MANIALAB_TOROSU_CLI", raising=False)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "build"))
    monkeypatch.setattr(torosu_bridge.sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(torosu_bridge.sys, "frozen", True, raising=False)

    assert torosu_bridge._cli_path() == bundled.resolve()


def test_cli_is_found_in_external_cargo_target_directory(monkeypatch, tmp_path):
    target = tmp_path / "cargo-target"
    binary = target / "x86_64-pc-windows-gnu" / "release" / "torosu-cli.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native cli")
    monkeypatch.delenv("MANIALAB_TOROSU_CLI", raising=False)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(target))
    monkeypatch.setattr(torosu_bridge.sys, "frozen", False, raising=False)

    assert torosu_bridge._cli_path() == binary.resolve()
