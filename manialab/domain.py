from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
from string import Template

from vendor.mapperatorinator.event import Event, EventType


# These endings describe absent, exhausted or broken model context, not a supported
# musical release. Preserve attacks and raw ends, but do not auto-create LNs.
UNOBSERVED_RELEASES = frozenset(('duration_limit', 'input_boundary',
    'forced_segment_boundary', 'unresolved_or_postprocessed', 'frame_inactive'))


@dataclass
class Note:
    start: int
    end: int
    lane: int = -1
    pitch: int = 60
    strength: float = 1.0
    source: str = "import"
    id: str = ""
    raw: str = ""
    voice: str = "mix"
    phrase: bool = False
    support: tuple = ()
    support_coverage: float = 0.0
    evidence_start: int | None = None
    evidence_end: int | None = None
    release_reason: str = ''

    @property
    def hold(self):
        return self.end > self.start

    def data(self):
        return asdict(self)


def note_from(data):
    n = Note(**{k: v for k, v in data.items() if k in Note.__dataclass_fields__})
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (n.start, n.end, n.lane, n.pitch)):
        raise ValueError("音符时间、轨道和音高必须是整数")
    if n.start < 0 or n.end < n.start or n.end > 24 * 3600 * 1000:
        raise ValueError("音符时间无效")
    return n


def sections(text):
    result = defaultdict(list)
    section = ""
    for line in text.lstrip("\ufeff").splitlines():
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        else:
            result[section].append(line)
    return dict(result)


def values(lines):
    return {k.strip(): v.strip() for line in lines if ":" in line for k, v in [line.split(":", 1)]}


def parse_osu(text):
    s = sections(text)
    general, metadata, difficulty = [values(s.get(k, [])) for k in ("General", "Metadata", "Difficulty")]
    if general.get("Mode") != "3":
        raise ValueError("仅支持原生 osu!mania 谱面（Mode:3）")
    keys = int(float(difficulty.get("CircleSize", "0")))
    if not 1 <= keys <= 18:
        raise ValueError("谱面键数无效")
    notes = []
    for line in s.get("HitObjects", []):
        if not line.strip() or line.startswith("//"):
            continue
        p = line.split(",")
        if len(p) < 5:
            raise ValueError("HitObjects 行不完整")
        x, start, kind = int(p[0]), int(p[2]), int(p[3])
        if not 0 <= x < 512 or start < 0:
            raise ValueError("谱面含无效坐标或负时间")
        if kind & 128:
            end = int(p[5].split(":", 1)[0])
            if end <= start:
                raise ValueError(f"{start}ms 存在长度不为正的 LN")
        elif kind & 1:
            end = start
        else:
            raise ValueError(f"{start}ms 存在不支持的物件类型 {kind}")
        notes.append(Note(start, end, min(keys - 1, x * keys // 512), 60, 1, "import", f"i{len(notes)}", line))
    if not notes:
        raise ValueError("谱面没有音符")
    return {"keys": keys, "metadata": metadata, "general": general, "timing": s.get("TimingPoints", []), "notes": notes}


def upstream_events(notes):
    """Bridge to Mapperatorinator's event vocabulary; pair LN endpoints by lane."""
    events = []
    for n in notes:
        events.append((n.start, n.lane, EventType.HOLD_NOTE if n.hold else EventType.CIRCLE))
        if n.hold:
            events.append((n.end, n.lane, EventType.HOLD_NOTE_END))
    result = []
    for t, lane, kind in sorted(events, key=lambda e: (e[0], e[2] != EventType.HOLD_NOTE_END, e[1])):
        result.extend([Event(kind), Event(EventType.TIME_SHIFT, t), Event(EventType.MANIA_COLUMN, lane)])
    return result


def validate(notes, keys):
    if keys not in range(6, 11):
        raise ValueError("目标键数必须为 6–10")
    issues, busy, previous = [], [-1] * keys, [-1] * keys
    for n in sorted(notes, key=lambda n: (n.start, n.lane, n.end)):
        if not 0 <= n.lane < keys:
            issues.append(f"{n.start}ms：轨道越界")
            continue
        if n.start < 0 or n.end < n.start:
            issues.append(f"{n.start}ms：无效时间")
        if previous[n.lane] == n.start or busy[n.lane] > n.start:
            issues.append(f"{n.start}ms：第 {n.lane + 1} 轨重叠")
        busy[n.lane] = max(busy[n.lane], n.end)
        previous[n.lane] = n.start
    holds = [n for n in notes if n.hold]
    sweep = sorted([(n.start, 1) for n in holds] + [(n.end, -1) for n in holds])
    active = peak = 0
    for _, delta in sweep:
        active += delta
        peak = max(peak, active)
    return {"valid": not issues, "issues": issues[:100], "issue_count": len(issues), "notes": len(notes),
            "ln": len(holds), "ln_ratio": round(len(holds) / max(1, len(notes)), 3),
            "max_held": peak, "longest_ln_ms": max([n.end - n.start for n in holds], default=0),
            "lane_counts": [sum(n.lane == k for n in notes) for k in range(keys)]}


def clean_meta(value):
    return re.sub(r"[\r\n\x00]", " ", str(value))[:200]


def export_osu(notes, keys, template_path: Path, title="Untitled", artist="Unknown", audio="audio.ogg",
               bpm=120.0, offset=0.0, original=None, version="ManiaLab", creator="ManiaLab", grid_division=None, bookmarks=()):
    report = validate(notes, keys)
    if not report["valid"]:
        raise ValueError("谱面未通过结构校验：" + report["issues"][0])
    lines = []
    for n in sorted(notes, key=lambda n: (n.start, n.lane)):
        x = int((n.lane + .5) * 512 / keys)
        if n.raw:
            p = n.raw.split(",")
            p[0] = str(x)
            lines.append(",".join(p))
        else:
            tail = f"{n.end}:0:0:0:0:" if n.hold else "0:0:0:0:"
            lines.append(f"{x},192,{n.start},{128 if n.hold else 1},0,{tail}")
    if original:
        s = sections(original)
        for name, changes in {
            "Difficulty": {"CircleSize": keys},
            "Metadata": {"Version": f"{version} {keys}K", "BeatmapID": 0, "BeatmapSetID": -1},
        }.items():
            retained = [line for line in s.get(name, []) if line.split(":", 1)[0].strip() not in changes]
            s[name] = retained + [f"{k}:{clean_meta(v)}" for k, v in changes.items()]
        s["HitObjects"] = lines
        return "\n".join(["\n".join(s.get("", ["osu file format v14"]))] +
                         [f"[{name}]\n" + "\n".join(body) for name, body in s.items() if name]) + "\n"
    substitutions = dict(audio_filename=clean_meta(audio), preview_time=-1, mode=3, title=clean_meta(title),
        title_unicode=clean_meta(title), artist=clean_meta(artist), artist_unicode=clean_meta(artist),
        creator=clean_meta(creator), version=f"{version} {keys}K", source="", tags="ManiaLab generated",
        hp_drain_rate=5, circle_size=keys, overall_difficulty=7, approach_rate=5,
        slider_multiplier=1.4, slider_tick_rate=1, background_line="",
        timing_points=f"{float(offset):.3f},{60000 / float(bpm):.8f},4,2,0,70,1,0", hit_objects="\n".join(lines))
    content = Template(template_path.read_text(encoding="utf-8")).substitute(substitutions)
    if bookmarks:
        content = re.sub(r'(?m)^Bookmarks:.*$', 'Bookmarks: '+','.join(str(t) for t in sorted(set(bookmarks))), content)
    if grid_division is not None:
        from .timing import grid_report
        content = re.sub(r"(?m)^BeatDivisor:.*$", f"BeatDivisor: {grid_division}", content)
        # Verify serialized hit objects against the serialized red line, not
        # just an earlier in-memory candidate that may differ from the export.
        parsed = parse_osu(content)
        red = next(line.split(",") for line in parsed["timing"] if line.strip())
        check = grid_report(parsed["notes"], dict(bpm=60000 / float(red[1]), offset=float(red[0]), division=grid_division))
        if check["off_grid"]:
            raise ValueError(f"导出落线检查失败：{check['off_grid']} 个起点或 LN 尾端偏离拍线，请重新生成整谱")
    return content


def invariant(notes):
    return Counter((n.id, n.start, n.end, n.raw) for n in notes)
