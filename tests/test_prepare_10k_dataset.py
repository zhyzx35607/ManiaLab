from __future__ import annotations

import pytest
import csv
from pathlib import Path
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
    scan_local_creator_charts,
    tier1_review_rows,
    write_tier1_review,
    read_tier1_review,
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


def test_unicode_song_aliases_join_audio_variants_into_one_leakage_group():
    alias = dataset._normalize_song_key("歌手", "中文歌名")
    records = [
        {
            "set_id": 1,
            "audio_sha256": "english-mix",
            "SongKey": dataset._normalize_song_key("Artist", "Title A"),
            "SongKeyAliases": [dataset._normalize_song_key("Artist", "Title A"), alias],
        },
        {
            "set_id": 2,
            "audio_sha256": "alternate-mix",
            "SongKey": dataset._normalize_song_key("Artist B", "Title B"),
            "SongKeyAliases": [dataset._normalize_song_key("Artist B", "Title B"), alias],
        },
    ]

    assert leakage_group_ids(records)[0] == leakage_group_ids(records)[1]


def test_synthetic_local_ids_are_positive_and_written_into_chart_metadata():
    beatmap_id = dataset._stable_id("local-bid", "review-id")
    set_id = dataset._stable_id("local-set", "folder-id")
    chart = _chart().replace("BeatmapID: 123", "BeatmapID: 0").replace(
        "BeatmapSetID: 456", "BeatmapSetID: 0"
    )

    rewritten = dataset._rewrite_chart_identity(chart, beatmap_id, set_id)
    parsed = parse_osu_chart(
        rewritten,
        expected_keycount=10,
        expected_beatmap_id=beatmap_id,
        expected_set_id=set_id,
    )

    assert beatmap_id > 0
    assert set_id > 0
    assert parsed.beatmap_id == beatmap_id
    assert parsed.beatmapset_id == set_id


def test_dataset_cache_setup_rejects_non_d_drive_before_writing():
    c_drive_output = Path(f"C:/phase07-output-{id(test_dataset_cache_setup_rejects_non_d_drive_before_writing)}") / "dataset"

    with pytest.raises(dataset.DatasetResourceGuardError, match="must stay on D:"):
        dataset._configure_workspace_environment(c_drive_output)

    assert not c_drive_output.parent.exists()


def test_builder_stops_when_system_drive_is_exactly_at_five_gib(monkeypatch):
    monkeypatch.setattr(dataset.os, "name", "nt")
    monkeypatch.setenv("SystemDrive", "C:")
    monkeypatch.setattr(
        dataset.shutil,
        "disk_usage",
        lambda path: type("DiskUsage", (), {"free": 5 * 1024**3})(),
    )

    with pytest.raises(dataset.DatasetResourceGuardError, match="低于或等于 5 GiB"):
        dataset._assert_system_space(5.0)


def test_data_budget_checks_dataset_size_system_floor_and_d_reserve(tmp_path, monkeypatch):
    payload = tmp_path / "chart.osu"
    payload.write_bytes(b"1234")
    checked_thresholds = []
    monkeypatch.setattr(dataset, "_assert_system_space", checked_thresholds.append)

    with pytest.raises(dataset.DatasetResourceGuardError, match="max-dataset-gib"):
        dataset._check_data_budget(
            tmp_path,
            max_dataset_bytes=3,
            min_system_free_gib=5.0,
        )
    assert checked_thresholds == []

    monkeypatch.setattr(
        dataset.shutil,
        "disk_usage",
        lambda _path: type("DiskUsage", (), {"free": dataset.MIN_FREE_BYTES - 1})(),
    )
    with pytest.raises(dataset.DatasetResourceGuardError, match="D 盘剩余空间"):
        dataset._check_data_budget(
            tmp_path,
            max_dataset_bytes=8,
            min_system_free_gib=5.0,
        )
    assert checked_thresholds == [5.0]


def test_parquet_dependencies_are_checked_before_dataset_build(monkeypatch):
    monkeypatch.setattr(
        dataset.importlib.util,
        "find_spec",
        lambda name: None if name == "pyarrow" else object(),
    )

    with pytest.raises(RuntimeError, match="缺少 pyarrow"):
        dataset._require_parquet_dependencies()


def test_builder_exposes_bounded_query_page_budget(tmp_path):
    args = dataset.build_parser().parse_args([
        "--local-songs-dir",
        str(tmp_path),
        "--provisional",
    ])
    assert args.max_pages_per_query == 5
    with pytest.raises(ValueError, match="max-pages-per-query"):
        dataset.build_dataset(
            output_dir=Path("D:/phase07-unused/dataset"),
            candidates=[],
            provisional=True,
            max_pages_per_query=51,
        )


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
    for keycount in range(6, 11):
        assert classify_bucket(keycount, 0.45) == f"{keycount}k_ln"
        assert classify_bucket(keycount, 0.449) == f"{keycount}k_hybrid"
        assert classify_bucket(keycount, 0.15) == f"{keycount}k_hybrid"
        assert classify_bucket(keycount, 0.149) == f"{keycount}k_rice"
    assert classify_bucket(7, 0.45, minimum_aux_ln_ratio=0.45) == "7k_ln_aux"
    assert classify_bucket(9, 0.449, minimum_aux_ln_ratio=0.45) is None


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


def test_split_coverage_keeps_each_keycount_in_both_splits_and_10k_styles():
    records = []
    for keycount in range(6, 11):
        ratios = (0.05, 0.2, 0.6, 0.8) if keycount == 10 else (0.05, 0.2, 0.6, 0.8)
        for index, ratio in enumerate(ratios):
            records.append({
                "set_id": keycount * 100 + index,
                "audio_sha256": f"{keycount}-{index}",
                "artist": f"Artist {keycount}-{index}",
                "title": f"Song {keycount}-{index}",
                "Keycount": keycount,
                "HoldNoteRatio": ratio,
            })

    splits, groups = split_records(
        records,
        require_keycount_coverage=True,
        require_validation_styles=True,
    )
    for keycount in range(6, 11):
        assert {split for row, split in zip(records, splits) if row["Keycount"] == keycount} == {"train", "val"}
    val_ratios = [
        row["HoldNoteRatio"]
        for row, split in zip(records, splits)
        if split == "val" and row["Keycount"] == 10
    ]
    assert any(ratio >= 0.45 for ratio in val_ratios)
    assert any(ratio < 0.45 for ratio in val_ratios)


def test_local_creator_scan_and_review_csv_roundtrip_preserve_manual_decisions(tmp_path):
    songs = tmp_path / "Songs"
    folder = songs / "Set A"
    folder.mkdir(parents=True)
    chart = _chart(keycount=7).replace("Creator: Mapper", "Creator: ZhYzX35607")
    chart = chart.replace("Version: 10K LN", "Version: possible inverse")
    chart = chart.replace("BeatmapID: 123", "BeatmapID: 0").replace("BeatmapSetID: 456", "BeatmapSetID: 0")
    (folder / "chart.OSU").write_text(chart, encoding="utf-8")
    (folder / "song.mp3").write_bytes(b"local-audio-fixture")
    before = (folder / "chart.OSU").read_bytes()

    candidates = scan_local_creator_charts(songs)
    assert len(candidates) == 1
    assert candidates[0]["chart"].keycount == 7
    assert candidates[0]["auto_decision"] == "REVIEW_INVERSE"
    csv_path, md_path = write_tier1_review(tmp_path / "review.csv", candidates)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    assert rows[0]["BeatmapID"] == "0"
    assert rows[0]["AutoDecision"] == "REVIEW_INVERSE"
    assert "ReviewDecision" in rows[0] and rows[0]["ReviewDecision"] == ""
    assert "LN 占比" in md_path.read_text(encoding="utf-8")
    assert (folder / "chart.OSU").read_bytes() == before

    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=dataset.TIER1_REVIEW_COLUMNS)
        writer.writeheader()
        row = rows[0]
        row["ReviewDecision"] = "DROP"
        writer.writerow(row)
    assert read_tier1_review(csv_path, candidates) == {candidates[0]["review_id"]: "DROP"}
    write_tier1_review(csv_path, candidates)
    preserved = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    assert preserved[0]["ReviewDecision"] == "DROP"


def test_local_review_flags_repeated_short_same_lane_ln_tail_gaps():
    chart = _chart(keycount=7)
    rows = []
    for index in range(6):
        lane = index
        start = index * 1000
        end = start + 600
        x = int((lane + 0.5) * 512 / 7)
        rows.append(f"{x},192,{start},128,0,{end}:0:0:0:0:")
        rows.append(f"{x},192,{end + 50},1,0,0:0:0:0:")
    chart_text = chart.split("[HitObjects]", 1)[0] + "[HitObjects]\n" + "\n".join(rows) + "\n"

    parsed = parse_osu_chart(chart_text)
    reasons = dataset._local_review_reasons(parsed)

    assert any("同轨 LN 尾到后继音符" in reason for reason in reasons)


def test_local_review_does_not_flag_only_cross_lane_notes_after_ln_tails():
    chart = _chart(keycount=7)
    rows = []
    for index in range(6):
        ln_lane = index
        tap_lane = (index + 1) % 7
        start = index * 1000
        end = start + 600
        ln_x = int((ln_lane + 0.5) * 512 / 7)
        tap_x = int((tap_lane + 0.5) * 512 / 7)
        rows.append(f"{ln_x},192,{start},128,0,{end}:0:0:0:0:")
        rows.append(f"{tap_x},192,{end + 50},1,0,0:0:0:0:")
    chart_text = chart.split("[HitObjects]", 1)[0] + "[HitObjects]\n" + "\n".join(rows) + "\n"

    parsed = parse_osu_chart(chart_text)

    assert not any("LN 尾到下一音符" in reason for reason in dataset._local_review_reasons(parsed))


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


def test_sayobot_search_can_filter_language_without_affecting_status():
    client = SayobotClient(request_delay=0)
    calls = []

    def fake_api_json(_url, *, body=None, timeout=45):
        calls.append(body)
        return {"status": 0, "endid": 0, "data": []}

    original = dataset._api_json
    dataset._api_json = fake_api_json
    try:
        assert list(client.iter_search(
            keywords=("7K",),
            statuses=(1, 4),
            page_size=10,
            max_pages_per_query=1,
            language=dataset.CHINESE_LANGUAGE_BIT,
        )) == []
    finally:
        dataset._api_json = original

    assert [call["class"] for call in calls] == [1, 4]
    assert all(call["language"] == dataset.CHINESE_LANGUAGE_BIT for call in calls)


def test_sayobot_tier2_accepts_approved_states_but_rejects_pending():
    client = SayobotClient(request_delay=0)
    calls = []

    def fake_api_json(_url, *, body=None, timeout=45):
        calls.append(body)
        status = body["class"]
        return {
            "status": 0,
            "endid": 0,
            "data": [{"sid": status, "approved": status}],
        }

    original = dataset._api_json
    dataset._api_json = fake_api_json
    try:
        rows = list(client.iter_search(
            keywords=("8K LN",),
            statuses=dataset.TIER2_STATUSES,
            page_size=10,
            max_pages_per_query=1,
        ))
    finally:
        dataset._api_json = original

    assert [call["class"] for call in calls] == [1, 2, 3, 4]
    assert [(row["sid"], row["approved"]) for row in rows] == [
        (1, 1), (2, 2), (3, 3), (4, 4)
    ]
    with pytest.raises(ValueError):
        client.list_page(keyword="8K", status=-1, offset=0, limit=10)


def test_tier2_query_filters_chinese_and_reserves_ln_mapsets_by_keycount():
    tier2 = dataset._sayobot_search_options(
        "tier2",
        10,
        page_size=100,
        max_pages_per_query=5,
    )
    tier3 = dataset._sayobot_search_options(
        "tier3",
        10,
        page_size=100,
        max_pages_per_query=5,
    )

    assert tier2["language"] == dataset.CHINESE_LANGUAGE_BIT
    assert tier2["statuses"] == dataset.TIER2_STATUSES
    assert tier2["keywords"][0] == ""
    assert tier2["keywords"][1:4] == ("10K LN", "10 key LN", "10-key LN")
    assert tier3["keywords"][0] == ""
    assert "language" not in tier3
    assert tier3["statuses"] == dataset.RANKED_STATUSES
    assert dataset._tier2_ln_mapset_targets(
        {6: 14, 7: 30, 8: 9, 9: 8, 10: 15}
    ) == {6: 4, 7: 8, 8: 3, 9: 2, 10: 4}
    assert dataset._tier2_ln_keycounts(
        [
            {"Keycount": 6, "Bucket": "6k_ln"},
            {"keycount": 7, "bucket": "7k_rice"},
            {"Keycount": 10, "Bucket": "10k_ln"},
        ]
    ) == {6, 10}


def test_tier2_ln_recovery_does_not_exceed_full_keycount_quota_with_rice():
    eligible, ln_only = dataset._tier2_keycount_eligibility(
        {6, 7},
        dataset.Counter({("tier2", 6): 2, ("tier2", 7): 1}),
        {6: 2, 7: 2},
        {6: set(), 7: set()},
        {6: 1, 7: 1},
    )

    assert eligible == {6, 7}
    assert ln_only == {6}
    assert dataset._is_ln_bucket("6k_ln")
    assert dataset._is_ln_bucket("7k_ln_aux")
    assert not dataset._is_ln_bucket("6k_rice")


def test_deferred_staging_is_not_counted_as_final_dataset_payload(tmp_path):
    root = tmp_path / "staging"
    selected = root / "sayobot_1"
    deferred = root / "sayobot_2"
    selected.mkdir(parents=True)
    deferred.mkdir(parents=True)
    (selected / "audio.mp3").write_bytes(b"123456")
    (selected / "chart.osu").write_bytes(b"78")
    (deferred / "audio.mp3").write_bytes(b"deferred-payload")
    (deferred / "chart.osu").write_bytes(b"deferred")
    records = [{
        "BeatmapSetFolder": "sayobot_1",
        "AudioFile": "audio.mp3",
        "BeatmapFile": "chart.osu",
    }]

    dataset._check_selected_dataset_budget(
        records,
        root,
        max_dataset_bytes=8,
        min_system_free_gib=0,
    )
    with pytest.raises(dataset.DatasetResourceGuardError, match="max-dataset-gib"):
        dataset._check_data_budget(
            root,
            max_dataset_bytes=8,
            min_system_free_gib=0,
        )


def test_review_writer_refuses_unknown_decisions_and_preserves_existing_file(tmp_path):
    songs = tmp_path / "Songs"
    folder = songs / "set"
    folder.mkdir(parents=True)
    (folder / "chart.osu").write_text(
        _chart(keycount=10).replace("Creator: Mapper", "Creator: zhyzx35607"),
        encoding="utf-8",
    )
    (folder / "song.mp3").write_bytes(b"audio")
    candidates = scan_local_creator_charts(songs)
    review_csv, _ = write_tier1_review(tmp_path / "review.csv", candidates)
    original = review_csv.read_bytes()
    rows = list(csv.DictReader(review_csv.open(encoding="utf-8-sig", newline="")))
    rows[0]["ReviewDecision"] = "MAYBE"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=dataset.TIER1_REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    invalid = review_csv.read_bytes()

    with pytest.raises(ValueError, match="无效 ReviewDecision"):
        write_tier1_review(review_csv, candidates)

    assert review_csv.read_bytes() == invalid
    assert original != invalid
