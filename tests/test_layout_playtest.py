import csv
from dataclasses import replace
from hashlib import sha256
import json
from zipfile import ZipFile

import pytest

from manialab.domain import parse_osu
from scripts import build_layout_playtest as playtest


def test_player_pack_hides_engine_names_and_keeps_original_timing_hitsounds_and_ln(tmp_path, monkeypatch):
    source = tmp_path / "source.osz"
    chart = "\n".join(["osu file format v14", "[General]", "Mode:3", "AudioFilename:mix.ogg",
                       "[Metadata]", "Title:Song", "Version:Original", "[Difficulty]", "CircleSize:10",
                       "[TimingPoints]", "0,500,3,2,0,80,1,0", "1200,400,5,2,0,80,1,0",
                       "[HitObjects]", "32,192,1000,128,4,1550:0:0:0:0:", "240,192,1800,1,8,0:0:0:0:"])
    with ZipFile(source, "w") as archive:
        archive.writestr("song/input.osu", chart)
        archive.writestr("song/mix.ogg", b"music")
    def assign(notes, settings, **_):
        lane = 0 if settings["layout_engine"] == "beam" else 1
        return [replace(note, lane=lane) for note in notes]
    monkeypatch.setattr(playtest, "assign_lanes", assign)
    out = tmp_path / "review"
    digest = sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA-256"):
        playtest.build(source, "0" * 64, "song/input.osu", out)
    package = playtest.build(source, digest, "song/input.osu", out)
    with pytest.raises(FileExistsError):
        playtest.build(source, digest, "song/input.osu", out)
    key = json.loads((out / "deblind-after-scoring.json").read_text())
    reference = parse_osu(chart)
    with ZipFile(package) as archive:
        assert len(archive.namelist()) == 11
        assert archive.read("song/mix.ogg") == b"music"
        for k in range(6, 11):
            for letter in ("A", "B"):
                text = archive.read(f"song/Layout-{k}K-{letter}.osu").decode()
                restored = parse_osu(text)
                assert restored["keys"] == k and restored["timing"] == reference["timing"]
                assert restored["general"]["AudioFilename"] == "mix.ogg"
                assert playtest.object_fields(restored["notes"]) == playtest.object_fields(reference["notes"])
                lane = 0 if key["pairs"][str(k)][letter] == "beam" else 1
                assert all(note.lane == lane for note in restored["notes"])
                assert all(label not in text.lower() for label in ("beam", "classic"))
    with (out / "participant/scores.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["keys"]) for row in rows] == list(range(6, 11))
    assert {row["osz_sha256"] for row in rows} == {sha256(package.read_bytes()).hexdigest()}
    assert all(row["preferred_A_B_equal_unsure"] == "" for row in rows)
    assert all({row["play_first"], row["play_second"]} == {"A", "B"} for row in rows)
