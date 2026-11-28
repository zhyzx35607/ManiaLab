from types import SimpleNamespace

import pytest

from manialab import mapper_metadata as metadata
from manialab.domain import Note


def test_pinyin_kana_and_hangul_are_ascii_and_readable():
    assert metadata.ascii_name("樱花草", "") == "Ying Hua Cao"
    assert metadata.ascii_name("さくら", "") == "Sakura"
    assert metadata.ascii_name("안녕", "") == "Annyeong"


def test_audio_tags_win_and_filename_is_fallback(monkeypatch, tmp_path):
    source = tmp_path / "Artist - 樱花草.mp3"
    source.touch()
    monkeypatch.setattr(metadata, "_ffmetadata", lambda _: {
        "title": "audio", "artist": "unknown", "album": "Album", "date": "2022-01-01"})

    class Completed:
        stdout = b"metadata"

    monkeypatch.setattr("manialab.audio_io.run_hidden", lambda *a, **kw: Completed())
    monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "ffmpeg")
    result = metadata.audio_metadata(source)
    assert (result["TitleUnicode"], result["Title"]) == ("樱花草", "Ying Hua Cao")
    assert result["ArtistUnicode"] == "Artist"
    assert result["Source"] == "Album"


@pytest.mark.parametrize("notes,keys,expected", [
    ([], 10, 8),
    ([Note(0, 0), Note(10, 10)], 10, 8),
    ([Note(0, 20) for _ in range(3)] + [Note(30, 30) for _ in range(7)], 8, 6),
    ([Note(0, 20) for _ in range(3)] + [Note(30, 30) for _ in range(7)], 9, 5),
])
def test_actual_ln_ratio_sets_default_od_hp(notes, keys, expected):
    value = metadata.effective_metadata({"title": "Track", "artist": "Artist", "audio": "track.mp3"},
                                        notes=notes, keys=keys)
    assert value["OverallDifficulty"] == value["HPDrainRate"] == expected


def test_metadata_override_preserves_user_od_hp_and_rejects_paths():
    assert metadata.override_metadata({"OverallDifficulty": 4, "HPDrainRate": 3}) == {
        "OverallDifficulty": 4, "HPDrainRate": 3}
    with pytest.raises(ValueError, match="背景图"):
        metadata.override_metadata({"Background": "../outside.png"})


def test_generated_defaults_have_language_tags_and_background():
    value = metadata.effective_metadata({"title": "樱花草", "artist": "Artist", "audio": "a.mp3"},
                                        {"preset": "ln_focus", "settings": {"torosu_ln_mode": "fln"}},
                                        [], 10)
    assert value["Background"] == "a.png"
    assert "FLN" in value["Tags"] and "10K" in value["Tags"]
    assert "Chinese" in value["Tags"] and "ying hua cao" in value["Tags"]
    assert "inverse" not in value["Tags"].casefold()


def test_generated_background_event_is_separate_from_template_comment_and_overrides_apply():
    from pathlib import Path
    from manialab.domain import export_osu, parse_osu, sections, values

    state = {"title": "原始标题", "artist": "原始歌手", "audio": "track.mp3",
             "export_metadata": {"Title": "Edited Title", "TitleUnicode": "修改标题",
                                 "OverallDifficulty": 4, "HPDrainRate": 3,
                                 "Background": "cover.png"}}
    resolved = metadata.effective_metadata(state, notes=[Note(1000, 1000, lane=0)], keys=7)
    content = export_osu([Note(1000, 1000, lane=0)], 7, Path(__file__).resolve().parents[1]
                         / "vendor" / "mapperatorinator" / "template.osu",
                         audio="track.mp3", track_metadata=resolved)
    event = next(line for line in content.splitlines() if line.startswith('0,0,"'))
    assert event == '0,0,"cover.png",0,0'
    parsed = parse_osu(content)
    assert parsed["metadata"]["Title"] == "Edited Title"
    assert parsed["metadata"]["TitleUnicode"] == "修改标题"
    difficulty = values(sections(content)["Difficulty"])
    assert difficulty["OverallDifficulty"] == "4"
    assert difficulty["HPDrainRate"] == "3"


def test_osu_export_name_uses_ascii_metadata_and_cleans_windows_characters():
    from manialab.export_routes import _export_names

    song, chart = _export_names("[Metadata]\nTitle:樱花草/夏\nArtist:歌手\nCreator:CON\nVersion:LN Focus\n")
    assert song == "Ge Shou - Ying Hua Cao _ Xia (_CON)"
    assert chart == "Ge Shou - Ying Hua Cao _ Xia (_CON) [LN Focus].osu"


def test_online_catalog_is_bounded_and_silent_on_failure(monkeypatch):
    monkeypatch.setattr(metadata, "_LOOKUP_TIMEOUT", .01)

    def never_returns(*_args):
        import time
        time.sleep(.2)
        return []

    monkeypatch.setattr(metadata, "_itunes", never_returns)
    assert metadata.online_metadata({"TitleUnicode": "\u6a31\u82b1\u8349", "ArtistUnicode": "Artist"}) == {}


def test_online_catalog_returns_matched_english_row(monkeypatch):
    monkeypatch.setattr(metadata, "_itunes", lambda query, country, timeout: [{
        "trackId": 1, "trackName": "Ying Hua Cao", "artistName": "Artist",
        "collectionName": "Album", "primaryGenreName": "Pop", "releaseDate": "2020-01-01"}])
    result = metadata.online_metadata({"TitleUnicode": "\u6a31\u82b1\u8349", "ArtistUnicode": "Artist", "Title": "Ying Hua Cao"})
    assert result["Title"] == "Ying Hua Cao" and result["Source"] == "Album"
