from __future__ import annotations

import pytest
import zipfile

import training.prepare_10k_dataset as dataset
from training.prepare_10k_dataset import (
    SayobotClient,
    active_holds_at_windows,
    _archive_member_index,
    _chart_member_index,
    classify_bucket,
    leakage_group_ids,
    parse_osu_chart,
    parse_ffmpeg_audio_metadata,
    split_records,
)


def _chart(*, keycount: int = 10, hit_objects: str | None = None) -> str:
    if hit_objects is None:
        x0 = 512 * 0.5 / keycount
        x1 = 512 * 1.5 / keycount
        x2 = 512 * 2.5 / keycount
        hit_objects = (
            f"{x0},192,1000,128,0,3000:0:0:0:0:\n"
            f"{x1},192,3500,1,0,0:0:0:0:\n"
            f"{x2},192,4500,128,0,7000:0:0:0:0:"
        )
    return f"""osu file format v14

[General]
AudioFilename: song.mp3
Mode: 3

[Metadata]
Title: Song
Artist: Artist
Creator: Mapper
Version: 10K LN
BeatmapID: 123
BeatmapSetID: 456

[Difficulty]
CircleSize: {keycount}
OverallDifficulty: 8

[TimingPoints]
0,500,4,2,0,100,1,0

[HitObjects]
{hit_objects}
"""


def test_parse_mania_lns_and_calculate_real_hold_ratio():
    chart = parse_osu_chart(
        _chart(),
        expected_keycount=10,
        expected_beatmap_id=123,
        expected_set_id=456,
    )

    assert chart.mode == 3
    assert chart.keycount == 10
    assert chart.hit_object_count == 3
    assert chart.hold_count == 2
    assert chart.hold_note_ratio == pytest.approx(2 / 3)
    assert chart.timing_point_count == 1


def test_archive_charts_are_matched_by_embedded_beatmap_id(tmp_path):
    archive_path = tmp_path / "set.osz"
    chart_text = _chart().replace("BeatmapID: 123", "BeatmapID: 99123")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Artist - Song [10K LN].osu", chart_text)
        archive.writestr("audio.mp3", b"audio")

    with zipfile.ZipFile(archive_path) as archive:
        members = _archive_member_index(archive)
        charts = _chart_member_index(archive, members)

    assert charts[99123] == "Artist - Song [10K LN].osu"


def test_parse_rejects_wrong_mode_or_keycount():
    with pytest.raises(ValueError, match="Mania"):
        parse_osu_chart(_chart().replace("Mode: 3", "Mode: 0"))

    with pytest.raises(ValueError, match="键数无效"):
        parse_osu_chart(_chart(keycount=12))


@pytest.mark.parametrize(
    "hit_objects, message",
    [
        ("48,192,1000,128,0,1000:0:0:0:0:", "结束时间无效"),
        (
            "25.6,192,1000,128,0,3000:0:0:0:0:\n"
            "25.6,192,2500,1,0,0:0:0:0:",
            "重叠",
        ),
    ],
)
def test_parse_rejects_invalid_or_overlapping_lns(hit_objects, message):
    with pytest.raises(ValueError, match=message):
        parse_osu_chart(_chart(hit_objects=hit_objects))


def test_cross_window_hold_state_is_explicit_at_window_edge():
    chart = parse_osu_chart(_chart())

    states = active_holds_at_windows(chart, (0, 2000, 3000, 4000, 6000))

    assert states[1]["active_holds"] == [{"column": 0, "start_ms": 1000, "end_ms": 3000}]
    assert states[2]["active_holds"] == []
    assert states[4]["active_holds"] == [{"column": 2, "start_ms": 4500, "end_ms": 7000}]


def test_chart_buckets_use_ln_and_hybrid_thresholds():
    assert classify_bucket(10, 0.45) == "10k_ln"
    assert classify_bucket(10, 0.449) == "10k_hybrid"
    assert classify_bucket(10, 0.15) == "10k_hybrid"
    assert classify_bucket(10, 0.149) == "10k_rice"
    assert classify_bucket(7, 0.45) == "7k_ln_aux"
    assert classify_bucket(9, 0.449) is None


def test_ffmpeg_metadata_keeps_source_codec_not_pcm_null_output_codec():
    text = """Duration: 00:01:37.04, start: 0.000000, bitrate: 200 kb/s
  Stream #0:0: Audio: mp3 (mp3float), 44100 Hz, stereo, fltp, 192 kb/s
  Stream #0:1: Audio: pcm_s16le, 44100 Hz, stereo, s16, 1411 kb/s
"""

    duration, sample_rate, channels, codec = parse_ffmpeg_audio_metadata(text)
    assert duration == pytest.approx(97.04)
    assert (sample_rate, channels, codec) == (44100, 2, "mp3")


def test_leakage_groups_join_same_set_audio_and_song_identity():
    records = [
        {"set_id": 1, "audio_sha256": "a", "artist": "Artist", "title": "Song"},
        {"set_id": 1, "audio_sha256": "b", "artist": "Artist", "title": "Song"},
        {"set_id": 2, "audio_sha256": "b", "artist": "Other", "title": "Track"},
        {"set_id": 3, "audio_sha256": "c", "artist": "Solo", "title": "Separate"},
    ]

    groups = leakage_group_ids(records)

    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_split_records_is_deterministic_and_never_splits_a_song_group():
    records = [
        {
            "set_id": index,
            "audio_sha256": f"audio-{index}",
            "artist": f"Artist {index}",
            "title": f"Song {index}",
        }
        for index in range(20)
    ]
    records.extend(
        [
            {"set_id": 100, "audio_sha256": "shared", "artist": "Same", "title": "Song"},
            {"set_id": 101, "audio_sha256": "shared", "artist": "Same", "title": "Song"},
        ]
    )

    first, groups = split_records(records, train_fraction=0.85, seed="test")
    second, _ = split_records(records, train_fraction=0.85, seed="test")

    assert first == second
    by_group = {}
    for group_id, split in zip(groups, first):
        by_group.setdefault(group_id, set()).add(split)
    assert all(len(splits) == 1 for splits in by_group.values())
    assert set(first) == {"train", "val"}


def test_split_requires_multiple_independent_songs():
    records = [
        {"set_id": 1, "audio_sha256": "same", "artist": "Artist", "title": "Song"},
        {"set_id": 2, "audio_sha256": "same", "artist": "Artist", "title": "Song"},
    ]

    with pytest.raises(ValueError, match="至少需要两个"):
        split_records(records)


def test_sayobot_pagination_uses_endid_and_deduplicates(monkeypatch):
    pages = [
        {
            "status": 0,
            "endid": 2,
            "data": [
                {"sid": 20, "approved": 1},
                {"sid": 21, "approved": 1},
            ],
        },
        {
            "status": 0,
            "endid": 0,
            "data": [
                {"sid": 21, "approved": 1},
                {"sid": 22, "approved": 1},
            ],
        },
    ]
    calls = []

    def fake_api_json(url, *, body=None, timeout=45):
        calls.append((url, body))
        return pages[len(calls) - 1]

    monkeypatch.setattr(dataset, "_api_json", fake_api_json)
    rows = list(
        SayobotClient(request_delay=0).iter_search(
            keywords=("10K",),
            statuses=(1,),
            page_size=2,
        )
    )

    assert [row["sid"] for row in rows] == [20, 21, 22]
    assert [call[1]["offset"] for call in calls] == [0, 2]


def test_sayobot_empty_search_result_is_a_terminal_page(monkeypatch):
    monkeypatch.setattr(
        dataset,
        "_api_json",
        lambda *_args, **_kwargs: {"status": -1, "endid": 0, "data": []},
    )

    assert list(
        SayobotClient(request_delay=0).iter_search(
            keywords=("unsupported query",),
            statuses=(1,),
            page_size=2,
        )
    ) == []
