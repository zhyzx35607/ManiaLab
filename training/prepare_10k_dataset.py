"""Collect and validate a portable Mania MMRS dataset from Sayobot."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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
DEFAULT_KEYWORDS = ("10K", "10 key", "10-key")
AUXILIARY_KEYWORDS = ("7K", "8K", "9K")
RANKED_STATUSES = (1, 4)
SUPPORTED_KEYCOUNTS = (7, 8, 9, 10)
STAR_SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
TRAINING_WINDOW_MS = round((2048 - 1) * 128 * 1000 / 16000)
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$", re.IGNORECASE)


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
    artist: str
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
    if beatmap_id is None or set_id is None:
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
        beatmap_id=beatmap_id,
        beatmapset_id=set_id,
        mode=mode,
        keycount=keycount,
        audio_filename=audio_filename,
        title=metadata.get("title", ""),
        artist=metadata.get("artist", ""),
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


def classify_bucket(keycount: int, hold_ratio: float, *, minimum_aux_ln_ratio: float = 0.45) -> str | None:
    if keycount == 10:
        if hold_ratio >= 0.45:
            return "10k_ln"
        if hold_ratio >= 0.15:
            return "10k_hybrid"
        return "10k_rice"
    if keycount in (7, 8, 9) and hold_ratio >= minimum_aux_ln_ratio:
        return f"{keycount}k_ln_aux"
    return None


def _normalize_song_key(artist: str, title: str) -> str:
    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())

    artist_key, title_key = normalize(artist), normalize(title)
    if not title_key:
        return ""
    return f"{artist_key}\0{title_key}"


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
    """Group records by connected set-id, exact audio hash, and song identity."""
    dsu = _DisjointSet(len(records))
    first_by_token: dict[str, int] = {}
    for index, record in enumerate(records):
        tokens = []
        set_id = record.get("set_id") or record.get("BeatmapSetId")
        audio_hash = record.get("audio_sha256") or record.get("AudioSha256")
        song_key = record.get("song_key") or _normalize_song_key(
            str(record.get("artist", record.get("Artist", ""))),
            str(record.get("title", record.get("Title", ""))),
        )
        if set_id is not None:
            tokens.append(f"set:{set_id}")
        if audio_hash:
            tokens.append(f"audio:{str(audio_hash).casefold()}")
        if song_key:
            tokens.append(f"song:{song_key}")
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
        identity_tokens = sorted(
            {
                token
                for index in indices
                for token in (
                    f"set:{records[index].get('set_id') or records[index].get('BeatmapSetId')}",
                    f"audio:{str(records[index].get('audio_sha256') or records[index].get('AudioSha256') or '').casefold()}",
                    "song:" + str(
                        records[index].get("song_key")
                        or _normalize_song_key(
                            str(records[index].get("artist", records[index].get("Artist", ""))),
                            str(records[index].get("title", records[index].get("Title", ""))),
                        )
                    ),
                )
                if token.split(":", 1)[1]
            }
        )
        group_id = hashlib.sha256("\n".join(identity_tokens).encode("utf-8")).hexdigest()[:20]
        for index in indices:
            result[index] = group_id
    return result


def split_records(
    records: list[dict[str, Any]],
    *,
    train_fraction: float = 0.85,
    seed: str = "manialab-10k-v1",
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
    validation_groups = set(ranked_groups[:val_count])
    splits = ["val" if group in validation_groups else "train" for group in group_ids]
    if any(splits[index] != splits[j] for index in range(len(records)) for j in range(index + 1, len(records)) if group_ids[index] == group_ids[j]):
        raise AssertionError("同组歌曲出现跨 Train/Val 泄露")
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
    return status if status in RANKED_STATUSES else None


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
    ) -> dict[str, Any]:
        if status not in RANKED_STATUSES:
            raise ValueError("只允许检索 Ranked (1) 与 Loved (4)")
        payload = _api_json(
            f"{SAYOBOT_API}/?post",
            body={
                "cmd": "beatmaplist",
                "type": "search",
                "limit": limit,
                "keyword": keyword,
                "class": status,
                "mode": 8,
                "offset": offset,
            },
        )
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


def _prepare_mapset(
    *,
    candidate: dict[str, Any],
    detail_payload: dict[str, Any],
    archive_path: Path,
    stage_data_root: Path,
    include_keycounts: set[int],
    minimum_ln_ratio: float,
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
                bucket = classify_bucket(
                    keycount,
                    chart.hold_note_ratio,
                    minimum_aux_ln_ratio=minimum_ln_ratio,
                )
                if bucket is None:
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

                star_rating = _map_star(map_row)
                if star_rating is None:
                    star_rating = _fallback_star(chart_bytes)
                if star_rating is None:
                    continue
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
                        "Creator": chart.creator,
                        "Version": chart.version,
                        "SourceStatus": status,
                        "SourceUrl": f"{SAYOBOT_DOWNLOAD}/{set_id}",
                        "SearchKeyword": candidate.get("_search_keyword", ""),
                        "SongKey": _normalize_song_key(chart.artist, chart.title),
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
        if destination.exists() and not any(destination.iterdir()):
            destination.rmdir()
        raise


def _configure_workspace_environment(output_dir: Path) -> Path:
    kit_root = output_dir.resolve().parent
    cache_root = kit_root / "cache"
    temp_root = kit_root / "tmp"
    paths = {
        "TMP": temp_root,
        "TEMP": temp_root,
        "TMPDIR": temp_root,
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "PIP_CACHE_DIR": cache_root / "pip",
        "UV_CACHE_DIR": cache_root / "uv",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path.resolve())
    return kit_root


def _iter_candidates(
    client: SayobotClient,
    *,
    page_size: int,
    keywords: tuple[str, ...],
) -> Iterable[dict[str, Any]]:
    yield from client.iter_search(keywords=keywords, page_size=page_size)


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


def build_dataset(
    *,
    output_dir: Path,
    max_sets: int = 300,
    page_size: int = 100,
    keywords: tuple[str, ...] = DEFAULT_KEYWORDS,
    auxiliary_ratio: float = 0.05,
    train_fraction: float = 0.85,
    seed: str = "manialab-10k-v1",
    request_delay: float = 0.25,
    resume: bool = False,
    keep_archives: bool = False,
) -> dict[str, Any]:
    if max_sets < 1:
        raise ValueError("max_sets 必须至少为 1")
    if not 0 <= auxiliary_ratio <= 0.25:
        raise ValueError("auxiliary_ratio 必须在 0 与 0.25 之间")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size 必须在 1 与 100 之间")
    output_dir = output_dir.expanduser().resolve()
    kit_root = _configure_workspace_environment(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"数据集目标目录已非空，不覆盖现有素材：{output_dir}"
        )

    download_cache = kit_root / "cache" / "sayobot_osz"
    download_cache.mkdir(parents=True, exist_ok=True)
    build_root = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}-{time.time_ns()}"
    stage_data_root = build_root / "_staging_data"
    stage_data_root.mkdir(parents=True, exist_ok=False)
    client = SayobotClient(request_delay=request_delay)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    tried_set_ids: set[int] = set()
    accepted_primary = 0
    accepted_auxiliary = 0

    def consider_candidate(candidate: dict[str, Any], *, auxiliary: bool) -> None:
        nonlocal accepted_primary, accepted_auxiliary
        set_id = _as_int(candidate.get("sid"))
        if set_id is None or set_id in tried_set_ids:
            return
        tried_set_ids.add(set_id)
        try:
            info = client.mapset_info(set_id)
            map_rows = _find_map_rows(info)
            wanted_keycounts = {7, 8, 9} if auxiliary else {10}
            exact_candidates = [
                row
                for row in map_rows
                if _as_int(row.get("mode")) == 3
                and (_as_finite_float(row.get("CS", row.get("cs"))) in wanted_keycounts)
            ]
            if not exact_candidates:
                return
            archive_path = download_cache / f"{set_id}.osz"
            if not (resume and archive_path.is_file() and _valid_zip(archive_path)):
                _download_mapset(set_id, archive_path)
            new_records = _prepare_mapset(
                candidate=candidate,
                detail_payload=info,
                archive_path=archive_path,
                stage_data_root=stage_data_root,
                include_keycounts=wanted_keycounts,
                minimum_ln_ratio=0.45 if auxiliary else 0.0,
            )
            if not new_records:
                rejected.append({"set_id": set_id, "reason": "no readable valid matching 10K/LN charts"})
                return
            records.extend(new_records)
            if auxiliary:
                accepted_auxiliary += 1
            else:
                accepted_primary += 1
            if not keep_archives and not resume:
                archive_path.unlink(missing_ok=True)
        except Exception as error:
            rejected.append({"set_id": set_id, "reason": str(error)[:400]})

    for candidate in _iter_candidates(
        client,
        page_size=page_size,
        keywords=keywords,
    ):
        if accepted_primary >= max_sets:
            break
        consider_candidate(candidate, auxiliary=False)

    auxiliary_limit = int(math.ceil(accepted_primary * auxiliary_ratio))
    if auxiliary_limit:
        for candidate in client.iter_search(keywords=AUXILIARY_KEYWORDS, page_size=page_size):
            if accepted_auxiliary >= auxiliary_limit:
                break
            consider_candidate(candidate, auxiliary=True)

    if not records:
        raise RuntimeError(
            "Sayobot 候选中没有找到通过音频/谱面/LN 严格校验的 10K；"
            f"候选失败数={len(rejected)}。可查看 {build_root} 与缓存目录后调整 --keyword。"
        )
    splits, group_ids = split_records(
        records,
        train_fraction=train_fraction,
        seed=seed,
    )
    for record, split, group_id in zip(records, splits, group_ids):
        record["split"] = split
        record["leakage_group_id"] = group_id

    split_roots = {
        split: build_root / split
        for split in ("train", "val")
    }
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

    for split, root in split_roots.items():
        _write_parquet(root / "metadata.parquet", [row for row in records if row["split"] == split])

    audio_hash_splits: dict[str, set[str]] = {}
    for record, split in zip(records, splits):
        audio_hash_splits.setdefault(record["AudioSha256"], set()).add(split)
    leaking_audio_hashes = sorted(key for key, values in audio_hash_splits.items() if len(values) > 1)
    if leaking_audio_hashes:
        raise AssertionError(f"发现跨 Train/Val 音频泄露: {len(leaking_audio_hashes)}")

    group_split_members: dict[str, set[str]] = {}
    for group_id, split in zip(group_ids, splits):
        group_split_members.setdefault(group_id, set()).add(split)
    leaking_groups = sorted(group for group, values in group_split_members.items() if len(values) > 1)
    if leaking_groups:
        raise AssertionError(f"发现跨 Train/Val 歌曲组泄露: {len(leaking_groups)}")

    final_records = []
    for record in records:
        manifest_record = {
            key: value
            for key, value in record.items()
            if key not in ("StarRating", "RankedDate", "SubmittedDate")
        }
        manifest_record["RankedDate"] = record["RankedDate"].isoformat()
        manifest_record["SubmittedDate"] = record["SubmittedDate"].isoformat()
        manifest_record["star_ratings_by_speed"] = list(record["StarRating"])
        manifest_record["data_relpath"] = (
            f"{record['split']}/data/{record['BeatmapSetFolder']}/{record['BeatmapFile']}"
        )
        manifest_record["audio_relpath"] = (
            f"{record['split']}/data/{record['BeatmapSetFolder']}/{record['AudioFile']}"
        )
        manifest_record["set_id"] = int(record["BeatmapSetId"])
        manifest_record["bid"] = int(record["Id"])
        manifest_record["status"] = int(record["SourceStatus"])
        final_records.append(manifest_record)

    def count_by(records2: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for record in records2:
            label = str(record[key])
            result[label] = result.get(label, 0) + 1
        return dict(sorted(result.items()))

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Sayobot API + novideo OSZ mirror",
        "sayobot_api": SAYOBOT_API,
        "model_dataset_layout": "Mapperatorinator MMRS metadata.parquet/data/<BeatmapSetFolder>",
        "max_primary_sets": max_sets,
        "accepted_primary_sets": accepted_primary,
        "accepted_auxiliary_sets": accepted_auxiliary,
        "chart_count": len(records),
        "songs_or_leakage_groups": len(set(group_ids)),
        "split_chart_counts": count_by(records, "split"),
        "split_song_counts": {
            split: len({group for group, assigned in zip(group_ids, splits) if assigned == split})
            for split in ("train", "val")
        },
        "bucket_chart_counts": count_by(records, "bucket"),
        "bucket_song_counts": {
            bucket: len(
                {
                    group_id
                    for record, group_id in zip(records, group_ids)
                    if record["bucket"] == bucket
                }
            )
            for bucket in sorted({record["bucket"] for record in records})
        },
        "bucket_split_counts": {
            split: count_by((record for record in records if record["split"] == split), "bucket")
            for split in ("train", "val")
        },
        "status_chart_counts": count_by(records, "SourceStatus"),
        "ten_k_chart_count": sum(record["Keycount"] == 10 for record in records),
        "ten_k_ln_chart_count": sum(
            record["Keycount"] == 10 and record["HoldNoteRatio"] >= 0.45
            for record in records
        ),
        "ten_k_ln_song_count": len(
            {
                group_id
                for record, group_id in zip(records, group_ids)
                if record["Keycount"] == 10 and record["HoldNoteRatio"] >= 0.45
            }
        ),
        "charts_with_cross_window_lns": sum(
            bool(record.get("active_holds_by_window"))
            for record in records
        ),
        "cross_window_active_hold_count": sum(
            sum(len(window["active_holds"]) for window in record.get("active_holds_by_window", []))
            for record in records
        ),
        "train_fraction_requested": train_fraction,
        "train_chart_fraction_actual": sum(split == "train" for split in splits) / len(splits),
        "train_group_fraction_actual": sum(
            1 for group in set(group_ids) if group_split_members[group] == {"train"}
        ) / len(set(group_ids)),
        "zero_audio_hash_leakage": not leaking_audio_hashes,
        "zero_song_group_leakage": not leaking_groups,
        "rejected_candidates": rejected,
        "audio_and_chart_bytes": sum(
            path.stat().st_size
            for path in build_root.rglob("*")
            if path.is_file() and path.name != "metadata.parquet"
        ),
    }
    split_summary = {
        split: {
            "chart_count": summary["split_chart_counts"].get(split, 0),
            "song_count": summary["split_song_counts"].get(split, 0),
            "bucket_chart_counts": summary["bucket_split_counts"].get(split, {}),
        }
        for split in ("train", "val")
    }
    if stage_data_root.exists():
        shutil.rmtree(stage_data_root)
    manifest_path = build_root / "manifest.json"
    for _ in range(3):
        manifest_text = json.dumps(
            {
                "schema_version": 1,
                "summary": summary,
                "splits": split_summary,
                "records": final_records,
            },
            ensure_ascii=False,
            indent=2,
        )
        other_bytes = sum(
            path.stat().st_size
            for path in build_root.rglob("*")
            if path.is_file() and path != manifest_path
        )
        total_bytes = other_bytes + len(manifest_text.encode("utf-8"))
        if summary.get("total_dataset_bytes") == total_bytes:
            break
        summary["total_dataset_bytes"] = total_bytes
        summary["total_dataset_gib"] = round(total_bytes / (1024**3), 3)
    manifest_text = json.dumps(
        {
            "schema_version": 1,
            "summary": summary,
            "splits": split_summary,
            "records": final_records,
        },
        ensure_ascii=False,
        indent=2,
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"数据集目标目录在构建期间变为非空，保留所有内容: {output_dir}")
        output_dir.rmdir()
    build_root.replace(output_dir)
    return summary


def _valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _download_mapset(set_id: int, destination: Path, *, max_bytes: int = 2_000_000_000) -> None:
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


def _parse_keywords(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_KEYWORDS
    keywords = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not keywords:
        raise ValueError("至少需要一个搜索关键词")
    return keywords


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Sayobot 镜像构建严格筛选、按歌曲零泄露划分的 10K MMRS 数据集"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workspace_exp" / "lora_10k_kit" / "dataset",
    )
    parser.add_argument("--max-sets", type=int, default=300, help="最多接受的主 10K mapset 数；辅助 7–9K 另按比例限量")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--keyword", action="append", help="Sayobot 搜索关键词；可重复提供")
    parser.add_argument("--auxiliary-ratio", type=float, default=0.05, help="额外纳入的 7–9K LN mapset 比例上限，默认 5%%")
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--seed", default="manialab-10k-v1")
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true", help="重用已校验过的 Sayobot .osz 下载缓存")
    parser.add_argument("--keep-archives", action="store_true", help="保留下载缓存中的原始 .osz")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = build_dataset(
            output_dir=args.output_dir,
            max_sets=args.max_sets,
            page_size=args.page_size,
            keywords=_parse_keywords(args.keyword),
            auxiliary_ratio=args.auxiliary_ratio,
            train_fraction=args.train_fraction,
            seed=args.seed,
            request_delay=args.request_delay,
            resume=args.resume,
            keep_archives=args.keep_archives,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
