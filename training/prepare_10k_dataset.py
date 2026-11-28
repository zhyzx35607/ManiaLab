"""Build a reviewed Mania 6–10K MMRS corpus from local charts and Sayobot."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


SAYOBOT_API = "https://api.sayobot.cn"
SAYOBOT_DOWNLOAD = "https://txy1.sayobot.cn/beatmaps/download/novideo"
RANKED_STATUSES = (1, 4)
# Sayobot's non-negative approved states are safe candidates for the
# Chinese Tier 2 discovery pass.  Tier 3 deliberately remains Ranked/Loved.
TIER2_STATUSES = (1, 2, 3, 4)
SUPPORTED_KEYCOUNTS = (6, 7, 8, 9, 10)
KEYCOUNT_KEYWORDS = {
    7: ("7K", "7 key", "7-key"),
    6: ("6K", "6 key", "6-key"),
    10: ("10K", "10 key", "10-key"),
    8: ("8K", "8 key", "8-key"),
    9: ("9K", "9 key", "9-key"),
}
KEYCOUNT_SHARES = {7: 0.38, 6: 0.20, 10: 0.20, 8: 0.11, 9: 0.11}
CHINESE_LANGUAGE_BIT = 16
MINIMUM_TIER2_STARS = 2.0
MAX_ARCHIVE_BYTES = 200 * 1024**2
MAX_UNPACKED_ARCHIVE_BYTES = 512 * 1024**2
MAX_CHART_BYTES = 4 * 1024**2
MIN_FREE_BYTES = 512 * 1024**2
TIER_SOURCE_WEIGHTS = {"tier1": 4.0, "tier2": 2.0, "tier3": 1.0}
TIER1_CREATOR = "zhyzx35607"
TIER1_REVIEW_COLUMNS = (
    "ReviewId",
    "BeatmapID",
    "BeatmapSetID",
    "SongTitle",
    "Artist",
    "Keycount",
    "DifficultyName",
    "HitObjectCount",
    "LNCount",
    "LNRatio",
    "AutoDecision",
    "AutoReasons",
    "ReviewDecision",
    "SourceRelativePath",
)
INVERSE_KEYWORDS = ("inverse", "反键", "反鍵", "ln mod", "lnmod", "dan", "sv", "gimmick", "test")
STAR_SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
TRAINING_WINDOW_MS = round((2048 - 1) * 128 * 1000 / 16000)
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$", re.IGNORECASE)


class DatasetResourceGuardError(OSError):
    """Stop corpus construction when a configured disk safety boundary is reached."""


@dataclass(frozen=True)
class ManiaObject:
    time_ms: int
    end_ms: int
    column: int
    is_hold: bool


@dataclass(frozen=True)
class ParsedChart:
    beatmap_id: int
    beatmapset_id: int
    mode: int
    keycount: int
    audio_filename: str
    title: str
    title_unicode: str
    artist: str
    artist_unicode: str
    creator: str
    version: str
    overall_difficulty: float
    hit_object_count: int
    hold_count: int
    hold_note_ratio: float
    timing_point_count: int
    objects: tuple[ManiaObject, ...]
    text: str


@dataclass(frozen=True)
class AudioMetadata:
    filename: str
    sha256: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str


def _as_int(value: Any) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _as_finite_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _read_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("谱面不是可读取的 UTF-8 文本")


def _parse_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().casefold()
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line)
    return sections


def _section_values(lines: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().casefold()] = value.strip()
    return values


def _safe_audio_member(filename: str) -> str:
    normalized = filename.replace("\\", "/").strip()
    member = PurePosixPath(normalized)
    if (
        not normalized
        or member.is_absolute()
        or any(part in ("", ".", "..") or ":" in part for part in member.parts)
    ):
        raise ValueError(f"谱面包含不安全的 AudioFilename: {filename!r}")
    return member.as_posix()


def parse_osu_chart(
    content: bytes | str,
    *,
    expected_keycount: int | None = None,
    expected_beatmap_id: int | None = None,
    expected_set_id: int | None = None,
    allow_missing_ids: bool = False,
) -> ParsedChart:
    """Parse and strictly validate a mania chart, including LN syntax/overlaps."""
    text = content if isinstance(content, str) else _read_text(content)
    sections = _parse_sections(text)
    general = _section_values(sections.get("general", []))
    metadata = _section_values(sections.get("metadata", []))
    difficulty = _section_values(sections.get("difficulty", []))

    mode = _as_int(general.get("mode"))
    keycount_value = _as_finite_float(difficulty.get("circlesize"))
    keycount = int(keycount_value) if keycount_value is not None and keycount_value.is_integer() else None
    beatmap_id = _as_int(metadata.get("beatmapid"))
    set_id = _as_int(metadata.get("beatmapsetid"))
    audio_filename = general.get("audiofilename", "")
    if mode != 3:
        raise ValueError(f"目标谱不是 Mania (Mode={mode})")
    if keycount not in SUPPORTED_KEYCOUNTS:
        raise ValueError(f"目标谱键数无效或不支持: {keycount}")
    if expected_keycount is not None and keycount != expected_keycount:
        raise ValueError(f"API 键数 {expected_keycount} 与谱面 CS={keycount} 不一致")
    if expected_beatmap_id is not None and beatmap_id != expected_beatmap_id:
        raise ValueError(f"API 谱面 ID {expected_beatmap_id} 与文件 ID {beatmap_id} 不一致")
    if expected_set_id is not None and set_id != expected_set_id:
        raise ValueError(f"API 谱包 ID {expected_set_id} 与文件 ID {set_id} 不一致")
    if not allow_missing_ids and (beatmap_id is None or set_id is None):
        raise ValueError("谱面缺少有效 BeatmapID 或 BeatmapSetID")
    _safe_audio_member(audio_filename)

    timing_count = 0
    has_uninherited_timing = False
    for line in sections.get("timingpoints", []):
        fields = line.split(",")
        if len(fields) < 6:
            raise ValueError(f"非法 TimingPoint: {line}")
        offset = _as_finite_float(fields[0])
        beat_length = _as_finite_float(fields[1])
        uninherited = _as_int(fields[6]) if len(fields) > 6 else 1
        if offset is None or beat_length is None or uninherited not in (0, 1):
            raise ValueError(f"非法 TimingPoint 数值: {line}")
        if uninherited == 1:
            if beat_length <= 0:
                raise ValueError(f"主 TimingPoint 的 beat length 必须为正: {line}")
            has_uninherited_timing = True
        timing_count += 1
    if timing_count == 0 or not has_uninherited_timing:
        raise ValueError("谱面缺少有效的主 TimingPoint")

    objects: list[ManiaObject] = []
    lane_last_start: dict[int, int] = {}
    lane_last_end: dict[int, int] = {}
    for line in sections.get("hitobjects", []):
        fields = line.split(",")
        if len(fields) < 5:
            raise ValueError(f"非法 HitObject: {line}")
        x = _as_finite_float(fields[0])
        y = _as_finite_float(fields[1])
        start_ms = _as_int(fields[2])
        object_type = _as_int(fields[3])
        if x is None or y is None or start_ms is None or object_type is None:
            raise ValueError(f"HitObject 含非法数值: {line}")
        if not 0 <= x < 512 or not 0 <= y <= 384:
            raise ValueError(f"HitObject 坐标越界: {line}")
        is_hold = bool(object_type & 128)
        if not is_hold and not (object_type & 1):
            raise ValueError(f"Mania HitObject 不是普通键音符或 LN: {line}")
        column = min(keycount - 1, int(x * keycount / 512))
        end_ms = start_ms
        if is_hold:
            if len(fields) < 6:
                raise ValueError(f"LN 缺少结束时间: {line}")
            end_token = fields[5].split(":", 1)[0]
            end_ms = _as_int(end_token)
            if end_ms is None or end_ms <= start_ms:
                raise ValueError(f"LN 结束时间无效: {line}")
        previous_start = lane_last_start.get(column)
        previous_end = lane_last_end.get(column)
        if previous_start == start_ms or (previous_end is not None and start_ms < previous_end):
            raise ValueError(f"同轨音符重叠或重复: {line}")
        lane_last_start[column] = start_ms
        lane_last_end[column] = end_ms
        objects.append(ManiaObject(start_ms, end_ms, column, is_hold))

    if not objects:
        raise ValueError("谱面没有可用的 mania HitObjects")
    od = _as_finite_float(difficulty.get("overalldifficulty"))
    if od is None:
        raise ValueError("谱面缺少有效 OverallDifficulty")
    return ParsedChart(
        beatmap_id=beatmap_id or 0,
        beatmapset_id=set_id or 0,
        mode=mode,
        keycount=keycount,
        audio_filename=audio_filename,
        title=metadata.get("title", ""),
        title_unicode=metadata.get("titleunicode", ""),
        artist=metadata.get("artist", ""),
        artist_unicode=metadata.get("artistunicode", ""),
        creator=metadata.get("creator", ""),
        version=metadata.get("version", ""),
        overall_difficulty=od,
        hit_object_count=len(objects),
        hold_count=sum(obj.is_hold for obj in objects),
        hold_note_ratio=sum(obj.is_hold for obj in objects) / len(objects),
        timing_point_count=timing_count,
        objects=tuple(objects),
        text=text,
    )


def active_holds_at_windows(
    chart: ParsedChart,
    window_edges_ms: Iterable[int],
) -> list[dict[str, Any]]:
    """Return active hold state at each edge for cross-window training checks."""
    edges = sorted(set(int(edge) for edge in window_edges_ms))
    holds = [obj for obj in chart.objects if obj.is_hold]
    return [
        {
            "time_ms": edge,
            "active_holds": [
                {"column": note.column, "start_ms": note.time_ms, "end_ms": note.end_ms}
                for note in holds
                if note.time_ms < edge < note.end_ms
            ],
        }
        for edge in edges
    ]


def classify_bucket(
    keycount: int,
    hold_ratio: float,
    *,
    minimum_aux_ln_ratio: float | None = None,
) -> str | None:
    if keycount not in SUPPORTED_KEYCOUNTS or not math.isfinite(hold_ratio) or not 0 <= hold_ratio <= 1:
        return None
    prefix = f"{keycount}k"
    if minimum_aux_ln_ratio is not None and keycount in (7, 8, 9):
        return f"{prefix}_ln_aux" if hold_ratio >= minimum_aux_ln_ratio else None
    if hold_ratio >= 0.45:
        return f"{prefix}_ln"
    if hold_ratio >= 0.15:
        return f"{prefix}_hybrid"
    return f"{prefix}_rice"


def _normalize_song_key(artist: str, title: str) -> str:
    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())

    artist_key, title_key = normalize(artist), normalize(title)
    if not title_key:
        return ""
    return f"{artist_key}\0{title_key}"


def _song_key_aliases(
    artist: str,
    title: str,
    artist_unicode: str = "",
    title_unicode: str = "",
) -> list[str]:
    artists = {value.strip() for value in (artist, artist_unicode) if value.strip()}
    titles = {value.strip() for value in (title, title_unicode) if value.strip()}
    return sorted({
        key
        for artist_name in artists
        for title_name in titles
        if (key := _normalize_song_key(artist_name, title_name))
    })


def _record_song_keys(record: dict[str, Any]) -> list[str]:
    aliases = record.get("song_key_aliases") or record.get("SongKeyAliases")
    if isinstance(aliases, str):
        aliases = [aliases]
    if aliases:
        return sorted({key for key in aliases if isinstance(key, str) and key})
    song_key = record.get("song_key") or record.get("SongKey")
    if song_key:
        return [str(song_key)]
    return _song_key_aliases(
        str(record.get("artist", record.get("Artist", ""))),
        str(record.get("title", record.get("Title", ""))),
        str(record.get("artist_unicode", record.get("ArtistUnicode", ""))),
        str(record.get("title_unicode", record.get("TitleUnicode", ""))),
    )


class _DisjointSet:
    def __init__(self, count: int):
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def leakage_group_ids(records: list[dict[str, Any]]) -> list[str]:
    """Group records by set, exact audio hash, and all known song-name aliases."""
    dsu = _DisjointSet(len(records))
    first_by_token: dict[str, int] = {}
    tokens_by_index = []
    for index, record in enumerate(records):
        tokens = []
        set_id = record.get("set_id", record.get("BeatmapSetId"))
        folder = record.get("BeatmapSetFolder")
        audio_hash = record.get("audio_sha256") or record.get("AudioSha256")
        song_keys = _record_song_keys(record)
        if set_id is not None:
            tokens.append(f"set:{set_id}")
        if folder:
            tokens.append(f"folder:{folder}")
        if audio_hash:
            tokens.append(f"audio:{str(audio_hash).casefold()}")
        tokens.extend(
            f"song:{song_key}"
            for song_key in song_keys
            if isinstance(song_key, str) and song_key
        )
        tokens_by_index.append(tokens)
        for token in tokens:
            if token in first_by_token:
                dsu.union(index, first_by_token[token])
            else:
                first_by_token[token] = index

    members: dict[int, list[int]] = {}
    for index in range(len(records)):
        members.setdefault(dsu.find(index), []).append(index)
    result = [""] * len(records)
    for indices in members.values():
        identity_tokens = sorted({token for index in indices for token in tokens_by_index[index]})
        group_id = hashlib.sha256("\n".join(identity_tokens).encode("utf-8")).hexdigest()[:20]
        for index in indices:
            result[index] = group_id
    return result


def split_records(
    records: list[dict[str, Any]],
    *,
    train_fraction: float = 0.85,
    seed: str = "manialab-6-10k-v1",
    require_keycount_coverage: bool = False,
    require_validation_styles: bool = False,
) -> tuple[list[str], list[str]]:
    """Assign whole leakage groups to train/val with a deterministic 85/15 split."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction 必须在 0 与 1 之间")
    group_ids = leakage_group_ids(records)
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ValueError("零泄露划分至少需要两个彼此独立的歌曲/音频组")
    val_count = min(len(unique_groups) - 1, max(1, round(len(unique_groups) * (1 - train_fraction))))
    ranked_groups = sorted(
        unique_groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    if require_keycount_coverage or require_validation_styles:
        by_group: dict[str, set[str]] = {}
        groups_by_keycount: dict[str, set[str]] = {}
        required_keys = {f"{keycount}K" for keycount in SUPPORTED_KEYCOUNTS}
        required_styles = {"10K LN", "10K Hybrid/Rice"} if require_validation_styles else set()
        for record, group_id in zip(records, group_ids):
            keycount = int(record.get("Keycount", record.get("keycount", 0)))
            labels = {f"{keycount}K"} if require_keycount_coverage else set()
            if require_validation_styles and keycount == 10:
                hold_ratio = float(record.get("HoldNoteRatio", record.get("hold_note_ratio", 0.0)))
                labels.add("10K LN" if hold_ratio >= 0.45 else "10K Hybrid/Rice")
            by_group.setdefault(group_id, set()).update(labels)
            for label in labels:
                groups_by_keycount.setdefault(label, set()).add(group_id)
        required = required_styles | (required_keys if require_keycount_coverage else set())
        missing = sorted(
            label for label in required
            if len(groups_by_keycount.get(label, ())) < (1 if label in required_styles else 2)
        )
        if missing:
            raise ValueError(
                f"独立歌曲组不足，无法同时覆盖 Train/Val 键数与目标验证风格: {', '.join(missing)}"
            )
        selected: set[str] = set()
        groups_per_keycount = Counter(
            key for labels in by_group.values() for key in labels & required_keys
        )

        def can_select(group: str) -> bool:
            return group not in selected and all(
                groups_per_keycount[keycount] > 1
                for keycount in by_group[group] & required_keys
            )

        def select(group: str) -> None:
            selected.add(group)
            groups_per_keycount.subtract(by_group[group] & required_keys)

        if required_styles:
            # Fixed validation needs two genuinely different songs, not two charts of one song.
            ln_groups = [group for group in ranked_groups if "10K LN" in by_group[group]]
            other_groups = [group for group in ranked_groups if "10K Hybrid/Rice" in by_group[group]]
            val_pair = next((
                (ln, other)
                for ln in ln_groups
                for other in other_groups
                if ln != other and all(
                    groups_per_keycount[key] > int(key in by_group[ln]) + int(key in by_group[other])
                    for key in (by_group[ln] | by_group[other]) & required_keys
                )
            ), None)
            if val_pair is None:
                raise ValueError("需要两首独立的 10K LN 与 Hybrid/Rice 验证歌曲，且训练集需保留 10K")
            for group in val_pair:
                select(group)
        for keycount in sorted(required & required_keys, key=lambda key: (groups_per_keycount[key], key)):
            if any(keycount in by_group[group] for group in selected):
                continue
            chosen = next(
                (group for group in ranked_groups if keycount in by_group[group] and can_select(group)),
                None,
            )
            if chosen is None:
                raise ValueError(f"{keycount} 不能在不清空训练 split 的条件下覆盖验证集")
            select(chosen)
        for group in ranked_groups:
            if len(selected) >= val_count:
                break
            if can_select(group):
                select(group)
        validation_groups = selected
    else:
        validation_groups = set(ranked_groups[:val_count])
    splits = ["val" if group in validation_groups else "train" for group in group_ids]
    return splits, group_ids


def _parse_datetime(value: Any, fallback_epoch: Any = None) -> datetime:
    for candidate in (value, fallback_epoch):
        if candidate is None or candidate == "":
            continue
        if isinstance(candidate, datetime):
            result = candidate
            return result.replace(tzinfo=None) if result.tzinfo else result
        if isinstance(candidate, (int, float)) or (isinstance(candidate, str) and candidate.isdigit()):
            number = float(candidate)
            if number > 1_000_000_000:
                return datetime.fromtimestamp(number, timezone.utc).replace(tzinfo=None)
        if isinstance(candidate, str):
            try:
                result = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
                return result.replace(tzinfo=None) if result.tzinfo else result
            except ValueError:
                pass
    return datetime(2000, 1, 1)


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _find_map_rows(payload: Any) -> list[dict[str, Any]]:
    for item in _walk_dicts(payload):
        rows = item.get("bid_data")
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            return rows
    raise ValueError("Sayobot beatmapinfo 响应缺少 bid_data")


def _status(value: Any) -> int | None:
    status = _as_int(value)
    return status if status in TIER2_STATUSES else None


def _map_star(row: dict[str, Any]) -> float | None:
    for key in ("star", "stars", "StarRating", "difficulty_rating"):
        value = row.get(key)
        number = _as_finite_float(value)
        if number is not None and number > 0:
            return number
        if isinstance(value, dict):
            for nested in value.values():
                number = _as_finite_float(nested)
                if number is not None and number > 0:
                    return number
    return None


def _map_user_id(row: dict[str, Any], details: dict[str, Any]) -> int:
    for source in (row, details):
        for key in ("UserId", "user_id", "uid", "creator_id", "mapper_id"):
            value = _as_int(source.get(key))
            if value is not None and value >= 0:
                return value
    return 0


def _api_json(url: str, *, body: dict[str, Any] | None = None, timeout: float = 45) -> Any:
    data = None
    headers = {"User-Agent": "ManiaLab-10K-dataset-preparer/1.0"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise OSError(f"HTTP {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            last_error = error
            if attempt == 3:
                break
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Sayobot API 请求失败: {url}: {last_error}") from last_error


class SayobotClient:
    def __init__(self, *, request_delay: float = 0.25):
        self.request_delay = max(0.0, request_delay)

    def list_page(
        self,
        *,
        keyword: str,
        status: int,
        offset: int,
        limit: int,
        language: int | None = None,
    ) -> dict[str, Any]:
        if status not in TIER2_STATUSES:
            raise ValueError("只允许检索非负的 Sayobot 状态 (1/2/3/4)")
        body = {
            "cmd": "beatmaplist",
            "type": "search",
            "limit": limit,
            "keyword": keyword,
            "class": status,
            "mode": 8,
            "offset": offset,
        }
        if language is not None:
            body["language"] = language
        payload = _api_json(f"{SAYOBOT_API}/?post", body=body)
        if (
            isinstance(payload, dict)
            and _as_int(payload.get("status")) == -1
            and not payload.get("data")
        ):
            return payload | {"status": 0, "endid": 0, "data": []}
        if not isinstance(payload, dict) or _as_int(payload.get("status")) != 0:
            raise RuntimeError(f"Sayobot beatmaplist 响应失败: {payload!r}")
        return payload

    def mapset_info(self, set_id: int) -> dict[str, Any]:
        if set_id <= 0:
            raise ValueError("beatmapset ID 必须为正数")
        time.sleep(self.request_delay)
        payload = _api_json(f"{SAYOBOT_API}/v2/beatmapinfo?0={set_id}")
        if not isinstance(payload, dict):
            raise RuntimeError("Sayobot beatmapinfo 响应不是对象")
        return payload

    def iter_search(
        self,
        *,
        keywords: Iterable[str],
        statuses: Iterable[int] = RANKED_STATUSES,
        page_size: int = 100,
        max_pages_per_query: int | None = None,
        language: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        seen: set[int] = set()
        for status in statuses:
            for keyword in keywords:
                offset = 0
                pages = 0
                while max_pages_per_query is None or pages < max_pages_per_query:
                    page = self.list_page(
                        keyword=keyword,
                        status=status,
                        offset=offset,
                        limit=page_size,
                        language=language,
                    )
                    rows = page.get("data")
                    if not isinstance(rows, list):
                        raise RuntimeError("Sayobot beatmaplist 响应缺少 data 列表")
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        set_id = _as_int(row.get("sid"))
                        row_status = _status(row.get("approved"))
                        if set_id is None or row_status != status or set_id in seen:
                            continue
                        seen.add(set_id)
                        yield row | {"_search_keyword": keyword, "_search_status": status}
                    next_offset = _as_int(page.get("endid"))
                    if next_offset is None or next_offset <= offset or next_offset == 0 or not rows:
                        break
                    offset = next_offset
                    pages += 1
                    time.sleep(self.request_delay)


def _decode_audio_file(path: Path) -> tuple[float, int, int, str]:
    try:
        import torchaudio
        import torch

        info = torchaudio.info(str(path))
        waveform, sample_rate = torchaudio.load(
            str(path),
            frame_offset=0,
            num_frames=min(int(info.num_frames), int(info.sample_rate)),
        )
        if waveform.numel() == 0 or not bool(torch.isfinite(waveform).all()):
            raise ValueError("解码音频为空或包含 NaN/Inf")
        duration = float(info.num_frames) / float(info.sample_rate)
        return duration, int(info.sample_rate), int(info.num_channels), Path(path).suffix.lstrip(".").casefold()
    except Exception as torchaudio_error:
        try:
            import imageio_ffmpeg

            executable = imageio_ffmpeg.get_ffmpeg_exe()
            probe = subprocess.run(
                [executable, "-hide_banner", "-v", "error", "-i", str(path), "-f", "null", "-"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
            )
            if probe.returncode:
                raise ValueError(probe.stderr.strip()[-400:])
            metadata = subprocess.run(
                [executable, "-hide_banner", "-i", str(path), "-t", "0.001", "-f", "null", "-"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            text = metadata.stderr
            duration, sample_rate, channels, codec = parse_ffmpeg_audio_metadata(text)
            if not duration:
                raise ValueError("ffmpeg 无法提供完整音频元数据")
            return duration, sample_rate, channels, codec
        except Exception as ffmpeg_error:
            raise ValueError(
                f"无法实际解码音频 ({path.name}): torchaudio={torchaudio_error}; ffmpeg={ffmpeg_error}"
            ) from ffmpeg_error


def parse_ffmpeg_audio_metadata(text: str) -> tuple[float, int, int, str]:
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    stream_match = re.search(
        r"Audio:\s*([^,\s]+)(?:\s+\([^)]*\))?,\s*(\d+)\s*Hz,\s*"
        r"(mono|stereo|(\d+(?:\.\d+)?)\s*channels?)",
        text,
        re.IGNORECASE,
    )
    if not duration_match or not stream_match:
        raise ValueError("ffmpeg 无法提供完整音频元数据")
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    channel_type = stream_match.group(3).casefold()
    channels = (
        1
        if channel_type == "mono"
        else 2
        if channel_type == "stereo"
        else int(float(stream_match.group(4)))
    )
    return duration, int(stream_match.group(2)), channels, stream_match.group(1).casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_audio_filename(chart_text: str, audio_name: str) -> str:
    lines = chart_text.splitlines(keepends=True)
    in_general = False
    changed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_general = stripped[1:-1].casefold() == "general"
            continue
        if in_general and stripped.casefold().startswith("audiofilename:"):
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            lines[index] = f"AudioFilename: {audio_name}{ending}"
            changed = True
            break
    if not changed:
        raise ValueError("无法在谱面 [General] 中更新 AudioFilename")
    return "".join(lines)


def _archive_member_index(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        normalized = info.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or any(part in ("", ".", "..") or ":" in part for part in path.parts):
            continue
        members[path.as_posix().casefold()] = info.filename
    return members


def _chart_member_index(
    archive: zipfile.ZipFile,
    members: dict[str, str],
) -> dict[int, str]:
    """Index chart files by their embedded BeatmapID, not their display filename."""
    charts: dict[int, str] = {}
    for normalized, member in members.items():
        if not normalized.endswith(".osu"):
            continue
        try:
            sections = _parse_sections(_read_text(archive.read(member)))
            metadata = _section_values(sections.get("metadata", []))
            beatmap_id = _as_int(metadata.get("beatmapid"))
        except (OSError, UnicodeError, ValueError, zipfile.BadZipFile):
            continue
        if beatmap_id is not None:
            charts.setdefault(beatmap_id, member)
    return charts


def _find_detail_row(rows: list[dict[str, Any]], beatmap_id: int) -> dict[str, Any] | None:
    for row in rows:
        if _as_int(row.get("bid")) == beatmap_id:
            return row
    return None


def _fallback_star(content: bytes) -> float | None:
    try:
        import rosu_pp_py as rosu

        beatmap = rosu.Beatmap(content=_read_text(content))
        difficulty = rosu.Difficulty()
        return float(difficulty.calculate(beatmap).stars)
    except Exception:
        return None


def _stable_id(namespace: str, identity: str) -> int:
    value = int(hashlib.sha256(f"{namespace}:{identity}".encode("utf-8")).hexdigest()[:15], 16)
    return (1 << 61) + value


def _rewrite_chart_identity(chart_text: str, beatmap_id: int, set_id: int) -> str:
    updates = {"beatmapid": beatmap_id, "beatmapsetid": set_id}
    lines = chart_text.splitlines(keepends=True)
    output: list[str] = []
    current_section = ""
    seen: set[str] = set()
    newline = "\r\n" if "\r\n" in chart_text else "\n"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_section == "metadata":
                for key, value in updates.items():
                    if key not in seen:
                        output.append(f"{key}: {value}{newline}")
                        seen.add(key)
            current_section = stripped[1:-1].casefold()
            output.append(line)
            continue
        if current_section == "metadata" and ":" in line:
            name, _, _ = line.lstrip("\ufeff").partition(":")
            key = name.strip().casefold()
            if key in updates:
                ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                output.append(f"{name}: {updates[key]}{ending}")
                seen.add(key)
                continue
        output.append(line)
    if current_section == "metadata":
        for key, value in updates.items():
            if key not in seen:
                output.append(f"{key}: {value}{newline}")
                seen.add(key)
    if seen != set(updates):
        raise ValueError("无法在谱面 [Metadata] 中写入稳定的 BeatmapID/BeatmapSetID")
    return "".join(output)


def _difficulty_flags(version: str) -> list[str]:
    searchable = unicodedata.normalize("NFKC", version).casefold()
    return [
        f"难度名命中关键词：{keyword}"
        for keyword in INVERSE_KEYWORDS
        if (
            keyword in searchable if keyword in {"反键", "反鍵"} else
            re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?:\d+)?(?![a-z0-9])", searchable)
        )
    ]


def _local_review_reasons(chart: ParsedChart) -> list[str]:
    reasons = _difficulty_flags(chart.version)

    lane_objects: dict[int, list[ManiaObject]] = {}
    for obj in chart.objects:
        lane_objects.setdefault(obj.column, []).append(obj)
    tail_gaps = []
    for objects in lane_objects.values():
        objects.sort(key=lambda obj: (obj.time_ms, obj.end_ms))
        for previous, following in zip(objects, objects[1:]):
            if previous.is_hold and following.time_ms >= previous.end_ms:
                tail_gaps.append(following.time_ms - previous.end_ms)
    if len(tail_gaps) >= 6:
        tiny_tail_ratio = sum(gap <= 80 for gap in tail_gaps) / len(tail_gaps)
        if tiny_tail_ratio >= 0.7:
            reasons.append(
                f"同轨 LN 尾到后继音符 ≤80ms 占比异常高：{tiny_tail_ratio:.0%} "
                f"({sum(gap <= 80 for gap in tail_gaps)}/{len(tail_gaps)})"
            )
    if chart.hit_object_count >= 40 and chart.hold_note_ratio >= 0.97:
        reasons.append(f"LN 占比异常高：{chart.hold_note_ratio:.1%}")
    return reasons


def scan_local_creator_charts(
    songs_dir: Path,
) -> list[dict[str, Any]]:
    """Read-only inventory of one creator's supported mania charts."""
    songs_dir = songs_dir.expanduser().resolve()
    if not songs_dir.is_dir():
        raise FileNotFoundError(f"本机 osu! Songs 目录不存在：{songs_dir}")
    rows = []
    for chart_path in sorted(
        (path for path in songs_dir.rglob("*") if path.suffix.casefold() == ".osu"),
        key=lambda path: path.as_posix().casefold(),
    ):
        try:
            resolved = chart_path.resolve(strict=True)
            if (
                not resolved.is_relative_to(songs_dir)
                or not resolved.is_file()
                or resolved.stat().st_size > MAX_CHART_BYTES
            ):
                continue
            content = resolved.read_bytes()
            text = _read_text(content)
        except (OSError, ValueError, RuntimeError):
            continue
        sections = _parse_sections(text)
        general = _section_values(sections.get("general", []))
        metadata = _section_values(sections.get("metadata", []))
        difficulty = _section_values(sections.get("difficulty", []))
        keycount = _as_finite_float(difficulty.get("circlesize"))
        if (
            metadata.get("creator", "").casefold() != TIER1_CREATOR
            or _as_int(general.get("mode")) != 3
            or keycount not in SUPPORTED_KEYCOUNTS
        ):
            continue
        relative = chart_path.relative_to(songs_dir).as_posix()
        review_id = hashlib.sha256(
            (relative + "\0" + hashlib.sha256(content).hexdigest()).encode("utf-8")
        ).hexdigest()[:24]
        chart = None
        invalid_reason = ""
        try:
            chart = parse_osu_chart(text, allow_missing_ids=True)
            audio_parts = PurePosixPath(_safe_audio_member(chart.audio_filename)).parts
            audio_path = resolved.parent.joinpath(*audio_parts).resolve(strict=True)
            if not audio_path.is_relative_to(resolved.parent) or not audio_path.is_file():
                raise ValueError("音频未指向本谱包内的普通文件")
        except (OSError, ValueError, RuntimeError) as exc:
            # Only retain relative, user-facing details; OS errors can contain the profile path.
            invalid_reason = str(exc)[:200] if isinstance(exc, ValueError) else "本谱包的音频无法读取"
        hit_objects = sections.get("hitobjects", [])
        raw_holds = sum(
            bool((_as_int(fields[3]) or 0) & 128)
            for line in hit_objects
            if len(fields := line.split(",")) >= 4
        )
        reasons = _local_review_reasons(chart) if chart else _difficulty_flags(metadata.get("version", ""))
        if invalid_reason:
            reasons.append(f"无法纳入语料：{invalid_reason}")
        rows.append({
            "review_id": review_id,
            "content": content,
            "chart": chart,
            "path": resolved,
            "mapset_path": resolved.parent,
            "songs_root": songs_dir,
            "relative_path": relative,
            "metadata": metadata,
            "keycount": int(keycount),
            "count": chart.hit_object_count if chart else len(hit_objects),
            "holds": chart.hold_count if chart else raw_holds,
            "invalid_reason": invalid_reason,
            "auto_decision": "REVIEW_INVERSE" if reasons else "KEEP",
            "auto_reasons": reasons,
        })
    return rows


def tier1_review_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in candidates:
        metadata = item["metadata"]
        count, holds = item["count"], item["holds"]
        rows.append({
            "ReviewId": item["review_id"],
            "BeatmapID": _as_int(metadata.get("beatmapid")) or 0,
            "BeatmapSetID": _as_int(metadata.get("beatmapsetid")) or 0,
            "SongTitle": metadata.get("titleunicode") or metadata.get("title", ""),
            "Artist": metadata.get("artistunicode") or metadata.get("artist", ""),
            "Keycount": item["keycount"],
            "DifficultyName": metadata.get("version", ""),
            "HitObjectCount": count,
            "LNCount": holds,
            "LNRatio": f"{(holds / count if count else 0.0):.4f}",
            "AutoDecision": item["auto_decision"],
            "AutoReasons": "；".join(item["auto_reasons"]),
            "ReviewDecision": "",
            "SourceRelativePath": item["relative_path"],
        })
    return rows


def write_tier1_review(
    csv_path: Path,
    candidates: list[dict[str, Any]],
) -> tuple[Path, Path]:
    csv_path = csv_path.expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = csv_path.with_suffix(".md")
    rows = tier1_review_rows(candidates)
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
        review_ids = {row.get("ReviewId") for row in existing}
        if len(existing) != len(rows) or review_ids != {row["ReviewId"] for row in rows}:
            raise FileExistsError(
                f"审核 CSV 与本次扫描不同，保留原表（含可能的人工决定）。先移至备份路径再重新生成：{csv_path}"
            )
        decisions = {row["ReviewId"]: row.get("ReviewDecision", "") for row in existing}
        invalid = {
            str(value).strip().upper()
            for value in decisions.values()
            if str(value).strip().upper() not in {"", "KEEP", "DROP"}
        }
        if invalid:
            raise ValueError(f"审核 CSV 含无效 ReviewDecision：{sorted(invalid)}")
        for row in rows:
            row["ReviewDecision"] = decisions[row["ReviewId"]]
    else:
        def csv_cell(value: Any) -> Any:
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                return "'" + value
            return value
        with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TIER1_REVIEW_COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows({key: csv_cell(value) for key, value in row.items()} for row in rows)

    flagged = [row for row in rows if row["AutoDecision"] == "REVIEW_INVERSE"]
    key_counts = {
        keycount: sum(row["Keycount"] == keycount for row in rows)
        for keycount in SUPPORTED_KEYCOUNTS
    }
    lines = [
        "# zhyzx35607 本人谱人工审核",
        "",
        "此清单来自本机 Songs 目录的只读扫描。`REVIEW_INVERSE` 是自动提示，不代表已确认反键；",
        "请逐行填写 `ReviewDecision` 为 `KEEP` 或 `DROP`，尤其检查所有自动标记项。未完成人工核审前，",
        "自动预审数据集只供短 smoke，不是可用于正式长训的最终黄金语料。",
        "",
        f"- Creator：`{TIER1_CREATOR}`",
        f"- 命中 6–10K 谱面：{len(rows)}",
        f"- 自动建议人工复核：{len(flagged)}",
        f"- 键数分布：{', '.join(f'{key}K {count}' for key, count in key_counts.items())}",
        "",
        "## 全部谱面（自动标记项优先）",
        "",
        "| BeatmapID / ReviewId | 曲名 | 键数 | 难度 | 物件数 | LN 占比 | 自动判定 / 原因 | 人工选择 |",
        "|---|---|---:|---|---:|---:|---|---|",
    ]

    def md_cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    for row in flagged + [row for row in rows if row["AutoDecision"] == "KEEP"]:
        title = md_cell(f"{row['Artist']} - {row['SongTitle']}")
        auto = "**REVIEW_INVERSE** " + md_cell(row["AutoReasons"]) if row in flagged else "KEEP"
        lines.append(
            f"| {row['BeatmapID']} / `{row['ReviewId']}` | {title} | {row['Keycount']}K "
            f"| {md_cell(row['DifficultyName'])} | {row['HitObjectCount']} | {float(row['LNRatio']):.1%} "
            f"| {auto} | {md_cell(row['ReviewDecision'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def read_tier1_review(review_csv: Path, candidates: list[dict[str, Any]]) -> dict[str, str]:
    expected = {row["review_id"]: row for row in candidates}
    decisions = {}
    with review_csv.expanduser().open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "ReviewId" not in reader.fieldnames or "ReviewDecision" not in reader.fieldnames:
            raise ValueError("人工审核 CSV 必须包含 ReviewId 与 ReviewDecision 列")
        for row in reader:
            review_id = (row.get("ReviewId") or "").strip()
            decision = (row.get("ReviewDecision") or "").strip().upper()
            if review_id not in expected:
                raise ValueError(f"审核 CSV 含本次扫描不存在的 ReviewId：{review_id}")
            if review_id in decisions:
                raise ValueError(f"审核 CSV 中 ReviewId 重复：{review_id}")
            if decision not in {"KEEP", "DROP"}:
                raise ValueError(
                    f"ReviewId {review_id} 的 ReviewDecision 必须明确填写 KEEP 或 DROP"
                )
            if decision == "KEEP" and expected[review_id]["invalid_reason"]:
                raise ValueError(f"ReviewId {review_id} 未通过结构或音频校验，不能填 KEEP")
            decisions[review_id] = decision
    missing = sorted(set(expected) - set(decisions))
    if missing:
        raise ValueError(
            f"人工审核 CSV 尚未为 {len(missing)} 张谱明确选择 KEEP/DROP，"
            f"首个未审核 ReviewId={missing[0]}"
        )
    return decisions


def _copy_local_creator_charts(
    candidates: list[dict[str, Any]],
    *,
    stage_data_root: Path,
    review_decisions: dict[str, str] | None,
    provisional: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = []
    rejected = []
    copied_audio_by_set: dict[str, dict[str, str]] = {}
    copied_charts: set[str] = set()
    for item in candidates:
        review_id = item["review_id"]
        if review_decisions is not None:
            decision = review_decisions[review_id]
            if decision == "DROP":
                continue
        elif item["auto_decision"] == "REVIEW_INVERSE":
            continue
        chart: ParsedChart = item["chart"]
        source_root: Path = item["mapset_path"]
        try:
            audio_member = _safe_audio_member(chart.audio_filename)
            audio_source = (source_root / Path(*PurePosixPath(audio_member).parts)).resolve(strict=True)
            if not audio_source.is_relative_to(source_root.resolve()) or not audio_source.is_file():
                raise ValueError("AudioFilename 不在本 mapset 内或不是普通文件")
            audio_sha = _sha256_file(audio_source)
            set_identity = str(item["mapset_path"].relative_to(item["songs_root"]).as_posix())
            set_key = hashlib.sha256(set_identity.encode("utf-8")).hexdigest()[:16]
            folder_name = f"local_{set_key}"
            destination = stage_data_root / folder_name
            destination.mkdir(parents=True, exist_ok=True)
            audio_names = copied_audio_by_set.setdefault(folder_name, {})
            audio_name = audio_names.get(audio_sha)
            if audio_name is None:
                suffix = audio_source.suffix.casefold()
                if not suffix or len(suffix) > 10:
                    raise ValueError("音频扩展名无效")
                audio_name = f"audio_{audio_sha[:16]}{suffix}"
                staged_audio = destination / audio_name
                shutil.copy2(audio_source, staged_audio)
                duration, sample_rate, channels, codec = _decode_audio_file(staged_audio)
                audio_names[audio_sha] = audio_name
                audio_metadata = AudioMetadata(
                    filename=audio_name,
                    sha256=audio_sha,
                    size_bytes=staged_audio.stat().st_size,
                    duration_seconds=duration,
                    sample_rate=sample_rate,
                    channels=channels,
                    codec=codec,
                )
            else:
                staged_audio = destination / audio_name
                duration, sample_rate, channels, codec = _decode_audio_file(staged_audio)
                audio_metadata = AudioMetadata(
                    filename=audio_name,
                    sha256=audio_sha,
                    size_bytes=staged_audio.stat().st_size,
                    duration_seconds=duration,
                    sample_rate=sample_rate,
                    channels=channels,
                    codec=codec,
                )
            chart_bytes_hash = hashlib.sha256(item["content"]).hexdigest()
            chart_id = chart.beatmap_id if chart.beatmap_id > 0 else _stable_id("local-bid", review_id)
            set_id = chart.beatmapset_id if chart.beatmapset_id > 0 else _stable_id("local-set", set_key)
            chart_name = f"beatmap_{chart_id}.osu"
            if chart_name in copied_charts:
                chart_name = f"beatmap_{chart_id}_{chart_bytes_hash[:8]}.osu"
            copied_charts.add(chart_name)
            chart_text = _rewrite_chart_identity(chart.text, chart_id, set_id)
            chart_text = _rewrite_audio_filename(chart_text, audio_name)
            parse_osu_chart(
                chart_text,
                expected_keycount=chart.keycount,
                expected_beatmap_id=chart_id,
                expected_set_id=set_id,
            )
            (destination / chart_name).write_text(
                chart_text,
                encoding="utf-8-sig" if chart.text.startswith("\ufeff") else "utf-8",
                newline="",
            )
            star_rating = _fallback_star(item["content"])
            rating_source = "rosu_pp" if star_rating is not None else "overall_difficulty_proxy"
            if star_rating is None:
                star_rating = chart.overall_difficulty
            record = {
                "Id": int(chart_id),
                "BeatmapSetId": int(set_id),
                "ModeInt": 3,
                "Ranked": 0,
                "RankedDate": datetime(2000, 1, 1),
                "SubmittedDate": datetime(2000, 1, 1),
                "DifficultyRating": float(star_rating),
                "DifficultyRatingSource": rating_source,
                "StarRating": [float(star_rating)] * len(STAR_SPEEDS),
                "BeatmapSetFolder": folder_name,
                "BeatmapFile": chart_name,
                "AudioFile": audio_name,
                "UserId": 0,
                "CircleSize": chart.keycount,
                "HoldNoteRatio": chart.hold_note_ratio,
                "Bucket": classify_bucket(chart.keycount, chart.hold_note_ratio),
                "AudioSha256": audio_sha,
                "AudioSizeBytes": audio_metadata.size_bytes,
                "AudioDurationSeconds": audio_metadata.duration_seconds,
                "AudioSampleRate": audio_metadata.sample_rate,
                "AudioChannels": audio_metadata.channels,
                "AudioCodec": audio_metadata.codec,
                "HitObjectCount": chart.hit_object_count,
                "HoldCount": chart.hold_count,
                "Artist": chart.artist,
                "Title": chart.title,
                "ArtistUnicode": chart.artist_unicode,
                "TitleUnicode": chart.title_unicode,
                "Creator": chart.creator,
                "Version": chart.version,
                "SourceStatus": 0,
                "SourceTier": "tier1",
                "SourceTierWeight": TIER_SOURCE_WEIGHTS["tier1"],
                "source_tier": "tier1",
                "tier_weight": TIER_SOURCE_WEIGHTS["tier1"],
                "SourceDescription": "local_creator_review_pending" if provisional else "local_creator_human_reviewed",
                "SourceUrl": "",
                "SearchKeyword": "",
                "SongKey": _normalize_song_key(chart.artist, chart.title),
                "SongKeyAliases": _song_key_aliases(
                    chart.artist,
                    chart.title,
                    chart.artist_unicode,
                    chart.title_unicode,
                ),
                "SetId": int(set_id),
                "BeatmapId": int(chart_id),
                "Keycount": chart.keycount,
                "keycount": chart.keycount,
                "hold_note_ratio": chart.hold_note_ratio,
                "difficulty": float(star_rating),
                "bucket": classify_bucket(chart.keycount, chart.hold_note_ratio),
                "audio_sha256": audio_sha,
                "window_duration_ms": TRAINING_WINDOW_MS,
                "active_holds_by_window": active_holds_at_windows(
                    chart,
                    range(
                        TRAINING_WINDOW_MS,
                        int(audio_metadata.duration_seconds * 1000),
                        TRAINING_WINDOW_MS,
                    ),
                ),
                "local_review_id": review_id,
                "local_review_decision": (
                    review_decisions[review_id] if review_decisions is not None else "AUTO_KEEP"
                ),
                "local_review_auto_decision": item["auto_decision"],
                "split_group_source": "local_mapset",
            }
            records.append(record)
        except Exception as error:
            rejected.append({"review_id": review_id, "reason": str(error)[:400]})
    used_files_by_folder: dict[str, set[str]] = {}
    for record in records:
        used = used_files_by_folder.setdefault(str(record["BeatmapSetFolder"]), set())
        used.update((str(record["AudioFile"]), str(record["BeatmapFile"])))
    if stage_data_root.exists():
        for folder in stage_data_root.iterdir():
            if not folder.is_dir():
                continue
            used_names = used_files_by_folder.get(folder.name, set())
            for child in folder.iterdir():
                if child.is_file() and child.name not in used_names:
                    child.unlink()
            if not any(folder.iterdir()):
                folder.rmdir()
    return records, {"included": len(records), "rejected": len(rejected), "rejected_records": rejected}


def _prepare_mapset(
    *,
    candidate: dict[str, Any],
    detail_payload: dict[str, Any],
    archive_path: Path,
    stage_data_root: Path,
    include_keycounts: set[int],
    ln_only_keycounts: set[int] | None = None,
    minimum_ln_ratio: float | None = None,
    source_tier: str = "tier3",
    minimum_star_rating: float = 0.0,
    require_chinese: bool = False,
) -> list[dict[str, Any]]:
    set_id = _as_int(candidate.get("sid"))
    status = _status(candidate.get("approved"))
    if set_id is None or status is None:
        return []
    details = next((row for row in _walk_dicts(detail_payload) if isinstance(row.get("bid_data"), list)), detail_payload)
    map_rows = _find_map_rows(detail_payload)
    selected_rows: list[tuple[dict[str, Any], int]] = []
    for row in map_rows:
        mode = _as_int(row.get("mode"))
        keycount_value = _as_finite_float(row.get("CS", row.get("cs")))
        if mode != 3 or keycount_value is None or not keycount_value.is_integer():
            continue
        keycount = int(keycount_value)
        if keycount in include_keycounts:
            selected_rows.append((row, keycount))
    if not selected_rows:
        return []

    destination = stage_data_root / f"sayobot_{set_id}"
    destination.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            total_unpacked = sum(info.file_size for info in archive.infolist() if not info.is_dir())
            if total_unpacked > MAX_UNPACKED_ARCHIVE_BYTES:
                raise ValueError(f"OSZ 解压后超过安全上限 ({MAX_UNPACKED_ARCHIVE_BYTES} bytes)")
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"OSZ CRC 校验失败: {bad_member}")
            member_index = _archive_member_index(archive)
            chart_members = _chart_member_index(archive, member_index)
            audio_metadata: dict[str, AudioMetadata] = {}
            for map_row, keycount in selected_rows:
                beatmap_id = _as_int(map_row.get("bid"))
                if beatmap_id is None:
                    continue
                map_member = chart_members.get(beatmap_id)
                if map_member is None:
                    continue
                chart_bytes = archive.read(map_member)
                try:
                    chart = parse_osu_chart(
                        chart_bytes,
                        expected_keycount=keycount,
                        expected_beatmap_id=beatmap_id,
                        expected_set_id=set_id,
                    )
                except ValueError:
                    continue
                row_metadata = _section_values(
                    _parse_sections(chart.text).get("metadata", [])
                )
                if require_chinese and not any(
                    re.search(r"[\u3400-\u9fff]", str(value))
                    for value in (
                        row_metadata.get("title", ""),
                        row_metadata.get("titleunicode", ""),
                        row_metadata.get("artist", ""),
                        row_metadata.get("artistunicode", ""),
                    )
                ):
                    continue
                bucket = classify_bucket(
                    keycount,
                    chart.hold_note_ratio,
                    minimum_aux_ln_ratio=minimum_ln_ratio,
                )
                if bucket is None:
                    continue
                if ln_only_keycounts and keycount in ln_only_keycounts and not _is_ln_bucket(bucket):
                    continue
                star_rating = _map_star(map_row)
                rating_source = "sayobot"
                if star_rating is None:
                    star_rating = _fallback_star(chart_bytes)
                    rating_source = "rosu_pp"
                if star_rating is None or star_rating < minimum_star_rating:
                    continue

                audio_member = _safe_audio_member(chart.audio_filename)
                audio_zip_member = member_index.get(audio_member.casefold())
                if audio_zip_member is None:
                    continue
                audio_bytes = archive.read(audio_zip_member)
                if not audio_bytes:
                    continue
                audio_sha = hashlib.sha256(audio_bytes).hexdigest()
                suffix = Path(audio_member).suffix.casefold()
                if not suffix or len(suffix) > 10:
                    continue
                audio_name = f"audio_{audio_sha[:16]}{suffix}"
                audio_path = destination / audio_name
                if audio_sha not in audio_metadata:
                    audio_path.write_bytes(audio_bytes)
                    try:
                        duration, sample_rate, channels, codec = _decode_audio_file(audio_path)
                    except ValueError:
                        audio_path.unlink(missing_ok=True)
                        continue
                    audio_metadata[audio_sha] = AudioMetadata(
                        filename=audio_name,
                        sha256=audio_sha,
                        size_bytes=len(audio_bytes),
                        duration_seconds=duration,
                        sample_rate=sample_rate,
                        channels=channels,
                        codec=codec,
                    )
                else:
                    audio_name = audio_metadata[audio_sha].filename

                chart_name = f"beatmap_{beatmap_id}.osu"
                chart_text = _rewrite_audio_filename(chart.text, audio_name)
                (destination / chart_name).write_text(chart_text, encoding="utf-8-sig" if chart.text.startswith("\ufeff") else "utf-8", newline="")
                date = _parse_datetime(
                    next((details[key] for key in ("date_submitted", "submitted", "SubmittedDate", "create_time") if key in details), None),
                    candidate.get("lastupdate"),
                )
                records.append(
                    {
                        "Id": beatmap_id,
                        "BeatmapSetId": set_id,
                        "ModeInt": 3,
                        "Ranked": status,
                        "RankedDate": date,
                        "SubmittedDate": date,
                        "DifficultyRating": float(star_rating),
                        "DifficultyRatingSource": rating_source,
                        "StarRating": [float(star_rating)] * len(STAR_SPEEDS),
                        "BeatmapSetFolder": f"sayobot_{set_id}",
                        "BeatmapFile": chart_name,
                        "AudioFile": audio_name,
                        "UserId": _map_user_id(map_row, details),
                        "CircleSize": keycount,
                        "HoldNoteRatio": chart.hold_note_ratio,
                        "Bucket": bucket,
                        "AudioSha256": audio_sha,
                        "AudioSizeBytes": audio_metadata[audio_sha].size_bytes,
                        "AudioDurationSeconds": audio_metadata[audio_sha].duration_seconds,
                        "AudioSampleRate": audio_metadata[audio_sha].sample_rate,
                        "AudioChannels": audio_metadata[audio_sha].channels,
                        "AudioCodec": audio_metadata[audio_sha].codec,
                        "HitObjectCount": chart.hit_object_count,
                        "HoldCount": chart.hold_count,
                        "Artist": chart.artist,
                        "Title": chart.title,
                        "ArtistUnicode": chart.artist_unicode,
                        "TitleUnicode": chart.title_unicode,
                        "Creator": chart.creator,
                        "Version": chart.version,
                        "SourceStatus": status,
                        "SourceTier": source_tier,
                        "SourceTierWeight": TIER_SOURCE_WEIGHTS[source_tier],
                        "source_tier": source_tier,
                        "tier_weight": TIER_SOURCE_WEIGHTS[source_tier],
                        "SourceDescription": candidate.get("_source_description", source_tier),
                        "SourceUrl": f"{SAYOBOT_DOWNLOAD}/{set_id}",
                        "SearchKeyword": candidate.get("_search_keyword", ""),
                        "SongKey": _normalize_song_key(chart.artist, chart.title),
                        "SongKeyAliases": _song_key_aliases(
                            chart.artist,
                            chart.title,
                            chart.artist_unicode,
                            chart.title_unicode,
                        ),
                        "SetId": set_id,
                        "BeatmapId": beatmap_id,
                        "Keycount": keycount,
                        "keycount": keycount,
                        "hold_note_ratio": chart.hold_note_ratio,
                        "difficulty": float(star_rating),
                        "bucket": bucket,
                        "audio_sha256": audio_sha,
                        "window_duration_ms": TRAINING_WINDOW_MS,
                        "active_holds_by_window": active_holds_at_windows(
                            chart,
                            range(
                                TRAINING_WINDOW_MS,
                                int(audio_metadata[audio_sha].duration_seconds * 1000),
                                TRAINING_WINDOW_MS,
                            ),
                        ),
                    }
                )
        if not records:
            shutil.rmtree(destination)
        return records
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise


def _configure_workspace_environment(output_dir: Path) -> Path:
    kit_root = output_dir.resolve().parent
    expected_root = Path(__file__).resolve().parents[1] / "workspace_exp" / "lora_6_10k_kit"
    _require_d_drive(expected_root, "Phase 07 workspace")
    _require_d_drive(kit_root, "dataset/cache directory")
    try:
        kit_root.relative_to(expected_root.resolve())
    except ValueError as error:
        raise ValueError(f"数据集与缓存必须全部位于忽略目录 {expected_root}") from error
    temp_root = kit_root / "tmp"
    paths = {
        "TMP": temp_root,
        "TEMP": temp_root,
        "TMPDIR": temp_root,
        "HF_HOME": kit_root / "hf_home",
        "HF_HUB_CACHE": kit_root / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": kit_root / "hf_home" / "transformers",
        "TORCH_HOME": kit_root / "torch_home",
        "XDG_CACHE_HOME": kit_root / "xdg_cache",
        "PIP_CACHE_DIR": kit_root / "pip_cache",
        "UV_CACHE_DIR": kit_root / "uv_cache",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path.resolve())
    return kit_root


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "生成 MMRS 数据集需要 pandas 与 pyarrow；请使用 Mapperatorinator V32 训练环境运行"
        ) from error
    if not records:
        raise ValueError("没有通过严格校验的谱面，无法写出 metadata.parquet")
    frame = pd.DataFrame(records)
    frame.to_parquet(path, index=True, engine="pyarrow")


def _require_parquet_dependencies() -> None:
    missing = [
        name
        for name in ("pandas", "pyarrow")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "生成 MMRS 数据集需要 pandas 与 pyarrow；"
            f"当前 Python 缺少 {', '.join(missing)}，请使用 D 盘 V32 训练环境运行"
        )


def _count_by(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        label = str(record[key])
        result[label] = result.get(label, 0) + 1
    return dict(sorted(result.items()))


def _sayobot_search_options(
    tier: str,
    keycount: int,
    *,
    page_size: int,
    max_pages_per_query: int,
) -> dict[str, Any]:
    if tier not in ("tier2", "tier3") or keycount not in SUPPORTED_KEYCOUNTS:
        raise ValueError("Sayobot source tier or keycount is unsupported")
    options: dict[str, Any] = {
        "keywords": ("", *KEYCOUNT_KEYWORDS[keycount]),
        "statuses": RANKED_STATUSES,
        "page_size": page_size,
        "max_pages_per_query": max_pages_per_query,
    }
    if tier == "tier2":
        options["statuses"] = TIER2_STATUSES
        options["keywords"] = (
            "",
            f"{keycount}K LN",
            f"{keycount} key LN",
            f"{keycount}-key LN",
            *KEYCOUNT_KEYWORDS[keycount],
        )
        options["language"] = CHINESE_LANGUAGE_BIT
    return options


def _tier2_ln_mapset_targets(per_key_limits: dict[int, int]) -> dict[int, int]:
    return {
        keycount: min(limit, math.ceil(limit * 0.25)) if limit > 0 else 0
        for keycount, limit in per_key_limits.items()
    }


def _tier2_ln_keycounts(records: Iterable[dict[str, Any]]) -> set[int]:
    result = set()
    for record in records:
        bucket = str(record.get("Bucket", record.get("bucket", ""))).casefold()
        if _is_ln_bucket(bucket):
            result.add(int(record.get("Keycount", record.get("keycount"))))
    return result


def _is_ln_bucket(bucket: str) -> bool:
    bucket = str(bucket).casefold()
    return bucket.endswith("_ln") or bucket.endswith("_ln_aux")


def _tier2_keycount_eligibility(
    available_keycounts: Iterable[int],
    accepted_keycount_mapsets: Counter[tuple[str, int]],
    per_key_limits: dict[int, int],
    accepted_ln_mapsets: dict[int, set[int]],
    ln_targets: dict[int, int],
) -> tuple[set[int], set[int]]:
    eligible = set()
    ln_only = set()
    for keycount in available_keycounts:
        if accepted_keycount_mapsets[("tier2", keycount)] < per_key_limits[keycount]:
            eligible.add(keycount)
        elif len(accepted_ln_mapsets[keycount]) < ln_targets[keycount]:
            eligible.add(keycount)
            ln_only.add(keycount)
    return eligible, ln_only


def build_dataset(
    *,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    review_csv: Path | None = None,
    provisional: bool = False,
    max_sets: int = 260,
    max_dataset_gib: float = 4.0,
    page_size: int = 100,
    max_pages_per_query: int = 5,
    train_fraction: float = 0.85,
    seed: str = "manialab-6-10k-v1",
    request_delay: float = 0.25,
    min_system_free_gib: float = 5.0,
) -> dict[str, Any]:
    """Collect source tiers, stage charts, and commit a zero-leakage MMRS manifest."""
    if max_sets < 10:
        raise ValueError("--max-sets must leave room for all five keycounts and three tiers")
    if not math.isfinite(max_dataset_gib) or max_dataset_gib <= 0:
        raise ValueError("--max-dataset-gib must be a positive finite value")
    if not 1 <= page_size <= 100:
        raise ValueError("--page-size must be between 1 and 100")
    if not 1 <= max_pages_per_query <= 50:
        raise ValueError("--max-pages-per-query must be between 1 and 50")
    if not 0 <= min_system_free_gib <= 10_000:
        raise ValueError("--min-system-free-gib must be between 0 and 10000")
    _require_parquet_dependencies()
    output_dir = output_dir.expanduser().resolve()
    kit_root = _configure_workspace_environment(output_dir)
    _assert_system_space(min_system_free_gib)
    kit_root.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"数据集目标目录已非空，不覆盖现有素材：{output_dir}")

    if review_csv:
        decisions = read_tier1_review(review_csv, candidates)
        review_status = "human_reviewed"
        review_path = review_csv.expanduser().resolve()
    elif provisional:
        decisions = None
        review_status = "provisional"
        review_path = kit_root / f"{TIER1_CREATOR}_review.csv"
    else:
        raise ValueError("正式数据集必须使用 --review-csv；短冒烟数据集必须明确指定 --provisional")
    if not review_path.is_relative_to(kit_root.resolve()):
        raise ValueError(f"审核清单必须位于忽略目录 {kit_root}")
    csv_path, md_path = write_tier1_review(review_path, candidates)

    max_dataset_bytes = int(max_dataset_gib * 1024**3)
    download_cache = kit_root / "cache" / "sayobot_osz"
    download_cache.mkdir(parents=True, exist_ok=True)
    build_root = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}-{time.time_ns()}"
    stage_data_root = build_root / "_staging_data"
    stage_data_root.mkdir(parents=True, exist_ok=False)
    client = SayobotClient(request_delay=request_delay)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_mapsets: dict[str, set[int]] = {"tier1": set(), "tier2": set(), "tier3": set()}
    used_mapsets: set[int] = set()
    used_song_keys: set[str] = set()
    used_audio_hashes: set[str] = set()

    try:
        local_records, local_summary = _copy_local_creator_charts(
            candidates,
            stage_data_root=stage_data_root,
            review_decisions=decisions,
            provisional=provisional,
        )
        _check_data_budget(
            stage_data_root,
            max_dataset_bytes=max_dataset_bytes,
            min_system_free_gib=min_system_free_gib,
        )
    except Exception:
        shutil.rmtree(build_root, ignore_errors=True)
        raise
    records.extend(local_records)
    for record in local_records:
        used_mapsets.add(int(record["SetId"]))
        accepted_mapsets["tier1"].add(int(record["SetId"]))
        used_song_keys.update(_record_song_keys(record))
        used_audio_hashes.add(str(record["AudioSha256"]).casefold())
    rejected.extend(local_summary["rejected_records"])
    if review_status == "provisional":
        for item in candidates:
            if item["invalid_reason"] or item["auto_decision"] != "KEEP":
                rejected.append({
                    "review_id": item["review_id"],
                    "reason": item["invalid_reason"] or "自动标记 REVIEW_INVERSE，未纳入预审数据集",
                })

    remaining_sets = max(0, max_sets - len(accepted_mapsets["tier1"]))
    tier2_limit = min(remaining_sets, max(15, math.ceil(max_sets * 0.25)))
    tier3_limit = max(0, remaining_sets - tier2_limit)
    per_key_limits = {
        tier: {
            keycount: max(2, round(targets * KEYCOUNT_SHARES[keycount]))
            for keycount in SUPPORTED_KEYCOUNTS
        }
        for tier, targets in (
            ("tier2", min(tier2_limit, max(1, math.ceil(max_sets * 0.25)))),
            ("tier3", tier3_limit),
        )
    }
    accepted_keycount_mapsets: Counter[tuple[str, int]] = Counter()
    tried_pairs: set[tuple[str, int, int]] = set()
    tier2_ln_mapset_targets = _tier2_ln_mapset_targets(per_key_limits["tier2"])
    accepted_tier2_ln_mapsets: dict[int, set[int]] = {
        keycount: set() for keycount in SUPPORTED_KEYCOUNTS
    }
    deferred_tier2: list[tuple[int, list[dict[str, Any]]]] = []
    mapset_info_cache: dict[int, dict[str, Any]] = {}
    targets_by_tier = {
        "tier2": min(tier2_limit, max(1, math.ceil(max_sets * 0.25))),
        "tier3": tier3_limit,
    }

    def tier2_ln_targets_met() -> bool:
        return all(
            len(accepted_tier2_ln_mapsets[keycount]) >= target
            for keycount, target in tier2_ln_mapset_targets.items()
        )

    def accept_mapset(
        tier: str,
        set_id: int,
        mapset_records: list[dict[str, Any]],
    ) -> None:
        records.extend(mapset_records)
        used_mapsets.add(set_id)
        accepted_mapsets[tier].add(set_id)
        for selected_key in {int(record["Keycount"]) for record in mapset_records}:
            accepted_keycount_mapsets[(tier, selected_key)] += 1
        for record in mapset_records:
            used_song_keys.update(_record_song_keys(record))
            used_audio_hashes.add(str(record["AudioSha256"]).casefold())
        if tier == "tier2":
            for selected_key in _tier2_ln_keycounts(mapset_records):
                accepted_tier2_ln_mapsets[selected_key].add(set_id)

    def process_candidate(candidate: dict[str, Any], tier: str, keycount: int) -> None:
        set_id = _as_int(candidate.get("sid"))
        if set_id is None or set_id in used_mapsets or len(accepted_mapsets[tier]) >= targets_by_tier[tier]:
            return
        pair = (tier, set_id, keycount)
        if pair in tried_pairs:
            return
        tried_pairs.add(pair)
        archive_path = download_cache / f"{set_id}.osz"
        downloaded_archive = False
        try:
            info = mapset_info_cache.get(set_id)
            if info is None:
                info = client.mapset_info(set_id)
                mapset_info_cache[set_id] = info
            map_rows = _find_map_rows(info)
            available = {
                int(value)
                for row in map_rows
                if _as_int(row.get("mode")) == 3
                and (value := _as_finite_float(row.get("CS", row.get("cs")))) is not None
                and value.is_integer()
                and int(value) in SUPPORTED_KEYCOUNTS
            }
            if keycount not in available:
                return
            key_ln_target_met = (
                tier != "tier2"
                or len(accepted_tier2_ln_mapsets[keycount])
                >= tier2_ln_mapset_targets[keycount]
            )
            if (
                accepted_keycount_mapsets[(tier, keycount)] >= per_key_limits[tier][keycount]
                and key_ln_target_met
            ):
                return
            if tier == "tier2":
                eligible_keycounts, ln_only_keycounts = _tier2_keycount_eligibility(
                    available,
                    accepted_keycount_mapsets,
                    per_key_limits[tier],
                    accepted_tier2_ln_mapsets,
                    tier2_ln_mapset_targets,
                )
            else:
                eligible_keycounts = {
                    candidate_keycount
                    for candidate_keycount in available
                    if accepted_keycount_mapsets[(tier, candidate_keycount)]
                    < per_key_limits[tier][candidate_keycount]
                }
                ln_only_keycounts = set()
            if not eligible_keycounts:
                return
            _check_staging_space(stage_data_root, min_system_free_gib=min_system_free_gib)
            if archive_path.exists():
                raise FileExistsError(f"临时 OSZ 已存在，停止覆盖：{archive_path}")
            if shutil.disk_usage(archive_path.parent).free < MIN_FREE_BYTES + MAX_ARCHIVE_BYTES:
                raise DatasetResourceGuardError("D 盘空间不足以安全下载临时 OSZ")
            _download_mapset(set_id, archive_path, max_bytes=MAX_ARCHIVE_BYTES)
            downloaded_archive = True
            _check_staging_space(stage_data_root, min_system_free_gib=min_system_free_gib)
            tier_candidate = candidate | {
                "_source_description": (
                    "Sayobot Chinese-language filter; Chinese Unicode metadata confirmed"
                    if tier == "tier2" else "Sayobot Ranked/Loved; Mania keycount confirmed"
                )
            }
            new_records = _prepare_mapset(
                candidate=tier_candidate,
                detail_payload=info,
                archive_path=archive_path,
                stage_data_root=stage_data_root,
                include_keycounts=eligible_keycounts,
                ln_only_keycounts=ln_only_keycounts,
                source_tier=tier,
                minimum_star_rating=MINIMUM_TIER2_STARS if tier == "tier2" else 0,
                require_chinese=tier == "tier2",
            )
            _check_staging_space(stage_data_root, min_system_free_gib=min_system_free_gib)
            if not new_records:
                rejected.append({"set_id": set_id, "tier": tier, "keycount": keycount, "reason": "no valid matching charts"})
                return
            if any(
                bool(set(_record_song_keys(record)) & used_song_keys)
                or str(record["AudioSha256"]).casefold() in used_audio_hashes
                for record in new_records
            ):
                shutil.rmtree(stage_data_root / f"sayobot_{set_id}", ignore_errors=True)
                rejected.append({"set_id": set_id, "tier": tier, "reason": "song/audio overlaps an already selected Tier 1/2 mapset"})
                return
            candidate_ln_keycounts = _tier2_ln_keycounts(new_records) if tier == "tier2" else set()
            if tier == "tier2" and not tier2_ln_targets_met() and not any(
                len(accepted_tier2_ln_mapsets[ln_keycount])
                < tier2_ln_mapset_targets[ln_keycount]
                for ln_keycount in candidate_ln_keycounts
            ):
                if len(accepted_mapsets[tier]) + len(deferred_tier2) < targets_by_tier[tier]:
                    deferred_tier2.append((set_id, new_records))
                    used_mapsets.add(set_id)
                else:
                    shutil.rmtree(stage_data_root / f"sayobot_{set_id}", ignore_errors=True)
                return
            _check_selected_dataset_budget(
                records + new_records,
                stage_data_root,
                max_dataset_bytes=max_dataset_bytes,
                min_system_free_gib=min_system_free_gib,
            )
            accept_mapset(tier, set_id, new_records)
        except Exception as error:
            if isinstance(error, DatasetResourceGuardError):
                raise
            rejected.append({"set_id": set_id, "tier": tier, "keycount": keycount, "reason": str(error)[:400]})
        finally:
            if downloaded_archive:
                archive_path.unlink(missing_ok=True)

    def accept_deferred_tier2() -> None:
        for set_id, deferred_records in deferred_tier2:
            if len(accepted_mapsets["tier2"]) >= targets_by_tier["tier2"]:
                break
            still_eligible_keycounts = {
                keycount
                for keycount in {int(record["Keycount"]) for record in deferred_records}
                if accepted_keycount_mapsets[("tier2", keycount)]
                < per_key_limits["tier2"][keycount]
            }
            mapset_records = [
                record
                for record in deferred_records
                if int(record["Keycount"]) in still_eligible_keycounts
            ]
            folder = stage_data_root / f"sayobot_{set_id}"
            if not mapset_records:
                shutil.rmtree(folder, ignore_errors=True)
                continue
            if any(
                bool(set(_record_song_keys(record)) & used_song_keys)
                or str(record["AudioSha256"]).casefold() in used_audio_hashes
                for record in mapset_records
            ):
                shutil.rmtree(folder, ignore_errors=True)
                rejected.append({
                    "set_id": set_id,
                    "tier": "tier2",
                    "reason": "deferred Chinese candidate overlaps an already selected song/audio",
                })
                continue
            if len(mapset_records) != len(deferred_records) and folder.is_dir():
                retained_names = {
                    Path(str(record[field])).name
                    for record in mapset_records
                    for field in ("AudioFile", "BeatmapFile")
                }
                for child in folder.iterdir():
                    if child.is_file() and child.name not in retained_names:
                        child.unlink()
            _check_selected_dataset_budget(
                records + mapset_records,
                stage_data_root,
                max_dataset_bytes=max_dataset_bytes,
                min_system_free_gib=min_system_free_gib,
            )
            accept_mapset("tier2", set_id, mapset_records)

    try:
        for tier in ("tier2", "tier3"):
            if tier == "tier3":
                accept_deferred_tier2()
            if targets_by_tier[tier] <= 0:
                continue
            for keycount in (7, 6, 10, 8, 9):
                if len(accepted_mapsets[tier]) >= targets_by_tier[tier]:
                    break
                search_options = _sayobot_search_options(
                    tier,
                    keycount,
                    page_size=page_size,
                    max_pages_per_query=max_pages_per_query,
                )
                for candidate in client.iter_search(**search_options):
                    if len(accepted_mapsets[tier]) >= targets_by_tier[tier]:
                        break
                    key_ln_target_met = (
                        tier != "tier2"
                        or len(accepted_tier2_ln_mapsets[keycount])
                        >= tier2_ln_mapset_targets[keycount]
                    )
                    if (
                        accepted_keycount_mapsets[(tier, keycount)]
                        >= per_key_limits[tier][keycount]
                        and key_ln_target_met
                    ):
                        break
                    process_candidate(candidate, tier, keycount)

        if not records:
            raise RuntimeError("没有通过结构、音频与来源层级校验的训练谱面")
        splits, group_ids = split_records(
            records,
            train_fraction=train_fraction,
            seed=seed,
            require_keycount_coverage=True,
            require_validation_styles=True,
        )
        for record, split, group_id in zip(records, splits, group_ids):
            record["split"] = split
            record["leakage_group_id"] = group_id

        split_roots = {split: build_root / split for split in ("train", "val")}
        for root in split_roots.values():
            (root / "data").mkdir(parents=True, exist_ok=True)
        moved_set_splits: dict[str, str] = {}
        for record, split in zip(records, splits):
            folder = str(record["BeatmapSetFolder"])
            if folder in moved_set_splits:
                if moved_set_splits[folder] != split:
                    raise AssertionError(f"同一个 mapset 被分到不同 split: {folder}")
                continue
            source = stage_data_root / folder
            destination = split_roots[split] / "data" / folder
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            moved_set_splits[folder] = split
        if stage_data_root.exists():
            shutil.rmtree(stage_data_root)

        audio_splits: dict[str, set[str]] = {}
        song_splits: dict[str, set[str]] = {}
        set_splits: dict[str, set[str]] = {}
        for record in records:
            split = str(record["split"])
            audio_splits.setdefault(str(record["AudioSha256"]).casefold(), set()).add(split)
            for song_key in _record_song_keys(record):
                song_splits.setdefault(song_key, set()).add(split)
            set_splits.setdefault(str(record["BeatmapSetFolder"]), set()).add(split)
        leakage = {
            "audio_hash": [key for key, value in audio_splits.items() if len(value) > 1],
            "song": [key for key, value in song_splits.items() if len(value) > 1],
            "mapset": [key for key, value in set_splits.items() if len(value) > 1],
        }
        if any(leakage.values()):
            raise AssertionError(f"发现 Train/Val 泄露: {leakage}")

        for split, root in split_roots.items():
            _write_parquet(root / "metadata.parquet", [row for row in records if row["split"] == split])
        for record in records:
            record["data_relpath"] = f"{record['split']}/data/{record['BeatmapSetFolder']}/{record['BeatmapFile']}"
            record["audio_relpath"] = f"{record['split']}/data/{record['BeatmapSetFolder']}/{record['AudioFile']}"

        split_counts = {
            split: {
                "chart_count": sum(record["split"] == split for record in records),
                "group_count": len({record["leakage_group_id"] for record in records if record["split"] == split}),
            }
            for split in ("train", "val")
        }
        keycount_chart_counts = {
            f"{key}K": sum(record["Keycount"] == key for record in records)
            for key in SUPPORTED_KEYCOUNTS
        }
        keycount_group_counts = {
            f"{key}K": len({record["leakage_group_id"] for record in records if record["Keycount"] == key})
            for key in SUPPORTED_KEYCOUNTS
        }
        tier_chart_counts = _count_by(records, "SourceTier")
        summary = {
            "schema_version": 2,
            "review_status": review_status,
            "local_review": _review_summary(candidates),
            "review_csv": str(csv_path),
            "review_md": str(md_path),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "Tier 1 local reviewed creator + Sayobot Chinese + Sayobot Ranked/Loved",
            "sayobot_api": SAYOBOT_API,
            "model_dataset_layout": "Mapperatorinator MMRS metadata.parquet/data/<BeatmapSetFolder>",
            "target_mapsets": max_sets,
            "max_pages_per_query": max_pages_per_query,
            "tier2_search_language_bit": CHINESE_LANGUAGE_BIT,
            "accepted_mapsets_by_tier": {tier: len(values) for tier, values in accepted_mapsets.items()},
            "tier2_ln_mapset_targets": {
                str(keycount): target
                for keycount, target in tier2_ln_mapset_targets.items()
            },
            "tier2_ln_mapset_counts": {
                str(keycount): len(accepted_tier2_ln_mapsets[keycount])
                for keycount in SUPPORTED_KEYCOUNTS
            },
            "tier2_ln_mapset_shortfalls": {
                str(keycount): max(
                    0,
                    tier2_ln_mapset_targets[keycount]
                    - len(accepted_tier2_ln_mapsets[keycount]),
                )
                for keycount in SUPPORTED_KEYCOUNTS
            },
            "tier2_bucket_chart_counts": _count_by(
                (record for record in records if record["SourceTier"] == "tier2"),
                "Bucket",
            ),
            "chart_count": len(records),
            "song_or_leakage_groups": len(set(group_ids)),
            "split": split_counts,
            "keycount_chart_counts": keycount_chart_counts,
            "keycount_group_counts": keycount_group_counts,
            "tier_chart_counts": tier_chart_counts,
            "bucket_chart_counts": _count_by(records, "Bucket"),
            "reviewed_tier1_charts": sum(record["SourceTier"] == "tier1" for record in records),
            "auto_flagged_tier1_excluded": sum(
                item["auto_decision"] == "REVIEW_INVERSE"
                and (decisions is None or decisions[item["review_id"]] == "DROP")
                for item in candidates
            ),
            "train_fraction_requested": train_fraction,
            "train_chart_fraction_actual": split_counts["train"]["chart_count"] / len(records),
            "train_group_fraction_actual": split_counts["train"]["group_count"] / len(set(group_ids)),
            "zero_audio_hash_leakage": not leakage["audio_hash"],
            "zero_song_group_leakage": not leakage["song"],
            "zero_mapset_leakage": not leakage["mapset"],
            "required_keycount_coverage_train_val": {
                f"{key}K": {
                    split: sum(record["split"] == split and record["Keycount"] == key for record in records) > 0
                    for split in ("train", "val")
                }
                for key in SUPPORTED_KEYCOUNTS
            },
            "rejected_candidates": rejected,
            "provisional_may_train_only_two_smoke_steps": review_status == "provisional",
        }
        manifest_records = []
        for record in records:
            row = {
                key: value
                for key, value in record.items()
                if key != "StarRating"
            }
            for key in ("RankedDate", "SubmittedDate"):
                value = row.get(key)
                if isinstance(value, datetime):
                    row[key] = value.isoformat()
            manifest_records.append(row)
        payload = {
            "schema_version": 2,
            "review_status": review_status,
            "summary": summary,
            "records": manifest_records,
        }
        manifest_path = build_root / "manifest.json"
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _check_data_budget(build_root, max_dataset_bytes=max_dataset_bytes, min_system_free_gib=min_system_free_gib)
        if output_dir.exists():
            if any(output_dir.iterdir()):
                raise FileExistsError(f"数据集目标目录在构建期间变为非空，保留所有内容: {output_dir}")
            output_dir.rmdir()
        build_root.replace(output_dir)
        return summary
    except Exception:
        # Only this process-owned, uniquely named directory is removed on failure.
        if build_root.exists():
            shutil.rmtree(build_root)
        raise


def _valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _download_mapset(set_id: int, destination: Path, *, max_bytes: int = MAX_ARCHIVE_BYTES) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    url = f"{SAYOBOT_DOWNLOAD}/{set_id}"
    request = Request(url, headers={"User-Agent": "ManiaLab-10K-dataset-preparer/1.0"})
    try:
        with urlopen(request, timeout=180) as response, temp_path.open("wb") as target:
            if response.status != 200:
                raise OSError(f"HTTP {response.status}")
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"OSZ 超过下载大小限制 ({max_bytes} bytes)")
                target.write(chunk)
        if total == 0 or not _valid_zip(temp_path):
            raise ValueError(f"Sayobot 返回空文件或损坏 OSZ: {url}")
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _review_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "creator": TIER1_CREATOR,
        "chart_count": len(candidates),
        "auto_keep_count": sum(item["auto_decision"] == "KEEP" for item in candidates),
        "auto_flagged_count": sum(item["auto_decision"] == "REVIEW_INVERSE" for item in candidates),
        "invalid_chart_count": sum(bool(item["invalid_reason"]) for item in candidates),
        "keycount_chart_counts": {
            str(key): sum(item["keycount"] == key for item in candidates)
            for key in SUPPORTED_KEYCOUNTS
        },
    }


def _assert_system_space(min_free_gib: float) -> None:
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("--min-system-free-gib 必须为非负有限值")
    if min_free_gib and os.name == "nt":
        drive = Path(os.getenv("SystemDrive") or Path(sys.executable).drive).anchor
        root = Path(drive + "\\" if drive and not drive.endswith("\\") else drive)
        free = shutil.disk_usage(root).free
        if free <= min_free_gib * 1024**3:
            raise DatasetResourceGuardError(
                f"系统盘 {root} 剩余 {free / 1024**3:.3f} GiB，"
                f"低于或等于 {min_free_gib:g} GiB 停止线"
            )


def _check_data_budget(
    stage_root: Path,
    *,
    max_dataset_bytes: int,
    min_system_free_gib: float,
) -> None:
    total_bytes = sum(
        path.stat().st_size
        for path in stage_root.rglob("*")
        if path.is_file()
    )
    if total_bytes > max_dataset_bytes:
        raise DatasetResourceGuardError(
            f"语料已超过 --max-dataset-gib 限制 ({max_dataset_bytes} bytes)"
        )
    _assert_system_space(min_system_free_gib)
    if shutil.disk_usage(stage_root).free < MIN_FREE_BYTES:
        raise DatasetResourceGuardError("D 盘剩余空间低于 512 MiB，停止构建以保护系统")


def _check_staging_space(stage_root: Path, *, min_system_free_gib: float) -> None:
    _assert_system_space(min_system_free_gib)
    if shutil.disk_usage(stage_root).free < MIN_FREE_BYTES:
        raise DatasetResourceGuardError("D 盘剩余空间低于 512 MiB，停止构建以保护系统")


def _check_selected_dataset_budget(
    records: Iterable[dict[str, Any]],
    stage_root: Path,
    *,
    max_dataset_bytes: int,
    min_system_free_gib: float,
) -> None:
    root = stage_root.resolve()
    payloads = set()
    for record in records:
        folder = str(record["BeatmapSetFolder"])
        for field in ("AudioFile", "BeatmapFile"):
            payload = (root / folder / str(record[field])).resolve()
            if not payload.is_relative_to(root):
                raise ValueError("Staged dataset path escaped its build directory")
            payloads.add(payload)
    total_bytes = sum(path.stat().st_size for path in payloads)
    if total_bytes > max_dataset_bytes:
        raise DatasetResourceGuardError(
            f"已选语料超过 --max-dataset-gib 限制 ({max_dataset_bytes} bytes)"
        )
    _check_staging_space(stage_root, min_system_free_gib=min_system_free_gib)


def _require_d_drive(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.casefold() != "d:":
        raise DatasetResourceGuardError(
            f"{name} must stay on D:; refusing to write under {resolved}"
        )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从本人审核谱和 Sayobot 构建按歌曲零泄露划分的三层 6–10K MMRS 数据集"
    )
    parser.add_argument("--local-songs-dir", type=Path, required=True, help="只读扫描的 osu! Songs 目录")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--review-only", action="store_true", help="仅生成审核 CSV/Markdown，不联网")
    mode.add_argument("--provisional", action="store_true", help="自动剔除疑似谱，只构建供短冒烟的未审版")
    mode.add_argument("--review-csv", type=Path, help="锁定人工审核，要求每行明确填写 KEEP 或 DROP")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="默认写入本仓库 workspace_exp/lora_6_10k_kit 下的 dataset 或 dataset_provisional",
    )
    parser.add_argument("--max-sets", type=int, default=260, help="三层合计谱包数量上限")
    parser.add_argument("--max-dataset-gib", type=float, default=4.0, help="最终数据集磁盘占用上限")
    parser.add_argument("--min-system-free-gib", type=float, default=5.0, help="Windows 系统盘低于指定 GiB 时停止")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages-per-query", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--seed", default="manialab-6-10k-v1")
    parser.add_argument("--request-delay", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        songs_dir = args.local_songs_dir.expanduser().resolve()
        default_kit = _require_d_drive(
            Path(__file__).resolve().parents[1] / "workspace_exp" / "lora_6_10k_kit",
            "Phase 07 workspace",
        )
        dataset_name = "dataset_provisional" if args.provisional else "dataset"
        output_dir = (args.output_dir or (default_kit / dataset_name)).expanduser().resolve()
        if songs_dir == output_dir or output_dir.parent.is_relative_to(songs_dir):
            raise ValueError("输出目录必须与 osu! Songs 来源分离，确保本机素材只读")
        candidates = scan_local_creator_charts(songs_dir)
        if not candidates:
            raise ValueError(f"未找到 Creator={TIER1_CREATOR} 的本机 6–10K 谱面")
        review_csv = args.review_csv.expanduser().resolve() if args.review_csv else None
        if review_csv and not review_csv.is_relative_to(default_kit.resolve()):
            raise ValueError(f"审核 CSV 与其他实验产物必须位于 {default_kit}")
        review_path = review_csv or (default_kit / f"{TIER1_CREATOR}_review.csv")
        csv_path, md_path = write_tier1_review(review_path, candidates)
        if args.provisional or args.review_csv:
            _assert_system_space(args.min_system_free_gib)
            summary = build_dataset(
                output_dir=output_dir,
                candidates=candidates,
                review_csv=review_csv,
                provisional=args.provisional,
                max_sets=args.max_sets,
                max_dataset_gib=args.max_dataset_gib,
                min_system_free_gib=args.min_system_free_gib,
                page_size=args.page_size,
                max_pages_per_query=args.max_pages_per_query,
                train_fraction=args.train_fraction,
                seed=args.seed,
                request_delay=args.request_delay,
            )
        else:
            summary = _review_summary(candidates) | {"review_csv": str(csv_path), "review_md": str(md_path)}
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
