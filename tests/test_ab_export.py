from __future__ import annotations

import zipfile

import pytest

from manialab.ab_export import _safe_member, export_comparison
from manialab.arrange import assign_lanes
from manialab.domain import Note, parse_osu, validate, export_osu


def _chart(keys=6):
    return "\n".join([
        "osu file format v14",
        "",
        "[General]",
        "AudioFilename: audio/song.ogg",
        "Mode: 3",
        "",
        "[Metadata]",
        "Title: A B fixture",
        "Artist: Test",
        "Creator: Test",
        "Version: Source",
        "BeatmapID: 123",
        "BeatmapSetID: 456",
        "",
        "[Difficulty]",
        f"CircleSize: {keys}",
        "OverallDifficulty: 7",
        "",
        "[Events]",
        '0,0,"bg.jpg",0,0',
        "",
        "[TimingPoints]",
        "0,500,4,2,0,70,1,0",
        "",
        "[HitObjects]",
        "25,192,1000,128,0,1600:0:0:0:0:",
        "150,192,1800,1,0,0:0:0:0:",
        "300,192,2600,1,0,0:0:0:0:",
        "",
    ])


def _converter(text, keys, **_):
    parsed = parse_osu(text)
    notes = assign_lanes(
        parsed["notes"],
        {"keys": keys, "layout_engine": "beam"},
        timing_points=parsed["timing"],
    )
    return export_osu(notes, keys, None, original=text)


def _analyzer(text, keys=None, playstyle=None):
    parsed = parse_osu(text)
    assert keys == parsed["keys"]
    return {
        "objects": len(parsed["notes"]),
        "ln": sum(note.end > note.start for note in parsed["notes"]),
        "minijack_count": 1,
        "pill_count": 0,
        "hand_balance": 0.9,
        "patterns": {"Stream": {"ratio": 0.25}, "Jack": {"count": 1}},
    }


@pytest.mark.parametrize("keys", range(6, 11))
def test_ab_export_from_osu_preserves_media_and_emits_valid_labeled_lns(
    tmp_path, keys, capsys,
):
    source = tmp_path / "source.osu"
    source.parent.joinpath("audio").mkdir()
    (source.parent / "audio" / "song.ogg").write_bytes(b"audio")
    (source.parent / "bg.jpg").write_bytes(b"background")
    source.write_text(_chart(), encoding="utf-8")
    output = tmp_path / f"comparison-{keys}.osz"

    result = export_comparison(
        source, keys, output, converter=_converter, analyzer=_analyzer,
    )

    assert output.is_file()
    assert result["labels"] == (
        f"[Baseline-OldLayout-{keys}K]", f"[Torosu-{keys}K]",
    )
    report_text = capsys.readouterr().out
    assert "转换耗时" in report_text
    assert "Stream 占比" in report_text and "25.0%" in report_text
    assert "Jack 占比" in report_text and "33.3%" in report_text
    with zipfile.ZipFile(output) as package:
        assert set(package.namelist()) == {
            "audio/song.ogg",
            "bg.jpg",
            *result["chart_paths"],
        }
        for chart_path, expected_version in zip(result["chart_paths"], result["labels"]):
            parsed = parse_osu(package.read(chart_path).decode("utf-8"))
            assert parsed["keys"] == keys
            assert parsed["metadata"]["Version"] == expected_version
            assert parsed["metadata"]["BeatmapID"] == "0"
            assert parsed["metadata"]["BeatmapSetID"] == "-1"
            assert len(parsed["notes"]) == 3
            assert sum(note.end > note.start for note in parsed["notes"]) == 1
            assert validate(parsed["notes"], keys)["valid"]
        assert package.read("audio/song.ogg") == b"audio"
        assert package.read("bg.jpg") == b"background"


def test_ab_export_from_nested_osz_keeps_all_non_chart_members_and_selects_requested_map(
    tmp_path,
):
    source = tmp_path / "source.osz"
    chart = _chart().replace(
        "AudioFilename: audio/song.ogg", "AudioFilename: ../audio/song.ogg",
    ).replace('"bg.jpg"', '"../bg.jpg"')
    with zipfile.ZipFile(source, "w") as package:
        package.writestr("charts/input.osu", chart)
        package.writestr("charts/other.osu", _chart().replace("Version: Source", "Version: Other"))
        package.writestr("audio/song.ogg", b"audio")
        package.writestr("bg.jpg", b"background")
        package.writestr("extra/readme.txt", b"preserved")
    output = tmp_path / "comparison.osz"

    result = export_comparison(
        source, 8, output, chart_member="charts/input.osu",
        converter=_converter, analyzer=_analyzer,
    )

    assert result["source_chart"] == "charts/input.osu"
    with zipfile.ZipFile(output) as package:
        assert set(package.namelist()) == {
            "audio/song.ogg",
            "bg.jpg",
            "extra/readme.txt",
            *result["chart_paths"],
        }
        assert package.read("audio/song.ogg") == b"audio"
        assert package.read("bg.jpg") == b"background"
        assert package.read("extra/readme.txt") == b"preserved"


def test_ab_export_rejects_invalid_torosu_output_without_publishing_package(tmp_path):
    source = tmp_path / "source.osu"
    source.parent.joinpath("audio").mkdir()
    (source.parent / "audio" / "song.ogg").write_bytes(b"audio")
    (source.parent / "bg.jpg").write_bytes(b"background")
    source.write_text(_chart(), encoding="utf-8")
    output = tmp_path / "invalid.osz"

    def conflicting_converter(text, keys, **_):
        lines = text.splitlines()
        first = lines.index("[HitObjects]") + 1
        while not lines[first].strip():
            first += 1
        second = first + 1
        while not lines[second].strip():
            second += 1
        first_fields = lines[first].split(",")
        second_fields = lines[second].split(",")
        second_fields[0] = first_fields[0]
        second_fields[2] = first_fields[2]
        lines[second] = ",".join(second_fields)
        return "\n".join(lines) + "\n"

    with pytest.raises(ValueError, match="轨道冲突"):
        export_comparison(
            source, 6, output, converter=conflicting_converter, analyzer=_analyzer,
        )
    assert not output.exists()


def test_old_layout_baseline_starts_from_scaled_source_columns(tmp_path, monkeypatch):
    source = tmp_path / "source.osu"
    source.parent.joinpath("audio").mkdir()
    (source.parent / "audio" / "song.ogg").write_bytes(b"audio")
    (source.parent / "bg.jpg").write_bytes(b"background")
    source.write_text(_chart(keys=6), encoding="utf-8")
    observed = {}
    original_assign = assign_lanes

    def capture(notes, settings=None, **kwargs):
        observed["lanes"] = [note.lane for note in notes]
        return original_assign(notes, settings, **kwargs)

    monkeypatch.setattr("manialab.ab_export.assign_lanes", capture)
    export_comparison(
        source, 10, tmp_path / "comparison.osz",
        converter=_converter, analyzer=_analyzer,
    )

    assert observed["lanes"] == [0, 2, 5]


def test_ab_export_allows_native_density_extras_without_losing_source_notes(tmp_path):
    source = tmp_path / "source.osu"
    source.parent.joinpath("audio").mkdir()
    (source.parent / "audio" / "song.ogg").write_bytes(b"audio")
    (source.parent / "bg.jpg").write_bytes(b"background")
    source.write_text(_chart(), encoding="utf-8")

    def expanded(text, keys, strict=True, **_):
        assert strict is True
        parsed = parse_osu(text)
        notes = assign_lanes(
            parsed["notes"],
            {"keys": keys, "layout_engine": "beam"},
            timing_points=parsed["timing"],
        )
        notes.append(Note(3000, 3000, lane=keys - 1, id="density-extra"))
        return export_osu(notes, keys, None, original=text)

    result = export_comparison(
        source, 10, tmp_path / "density.osz", density="dense",
        converter=expanded, analyzer=_analyzer,
    )

    assert result["rows"][result["labels"][1]]["objects"] == 4
    with zipfile.ZipFile(result["output"]) as package:
        torosu = parse_osu(package.read(result["chart_paths"][1]).decode("utf-8"))
        assert len(torosu["notes"]) == 4
        assert validate(torosu["notes"], 10)["valid"]


@pytest.mark.parametrize("member", [
    "audio:stream.ogg",
    "nested/CON.txt",
    "nested/LPT1.log",
    "image.png.",
    "trailing-space /song.osu",
    "nested\\..\\audio.ogg",
])
def test_archive_rejects_windows_unsafe_or_aliased_member_names(member):
    with pytest.raises(ValueError):
        _safe_member(member)
