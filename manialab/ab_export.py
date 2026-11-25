"""Build a playable old-layout vs Torosu comparison beatmap package."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, replace
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import sys
import tempfile
import time
import zipfile

from .arrange import assign_lanes
from .domain import export_osu, parse_osu, sections, validate

_MEDIA_EXTENSIONS = {
    ".avi", ".jpeg", ".jpg", ".m4a", ".mid", ".midi", ".mkv", ".mov",
    ".mp3", ".mp4", ".ogg", ".opus", ".png", ".wav", ".webm", ".webp",
}
_PATTERN_NAMES = (
    "Stream", "Roll", "Trill", "Jumpstream", "Handstream", "Jumptrill",
    "Bracket", "Jack", "Chordjack", "LnStream", "LnChords",
)
_WINDOWS_DEVICE_NAME = re.compile(
    r"^(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|CLOCK\$|"
    r"COM[1-9¹²³]|LPT[1-9¹²³])$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _SourceChart:
    path: Path
    chart_member: str
    text: str
    is_archive: bool


def _safe_member(name: str) -> str:
    if not isinstance(name, str) or not name or any(ord(char) < 32 for char in name):
        raise ValueError(f"输入包含无效的包成员路径：{name!r}")
    normalized = name.replace("\\", "/")
    components = normalized.split("/")
    if any(part in ("", ".", "..") for part in components):
        raise ValueError(f"输入包含不安全的路径：{name!r}")
    for component in components:
        if any(char in '<>:"|?*' for char in component):
            raise ValueError(f"输入包含 Windows 不安全的包成员路径：{name!r}")
        if component.endswith((" ", ".")):
            raise ValueError(f"输入包含 Windows 别名包成员路径：{name!r}")
        device_stem = component.split(".", 1)[0].rstrip(" .")
        if _WINDOWS_DEVICE_NAME.fullmatch(device_stem):
            raise ValueError(f"输入包含 Windows 保留设备名：{name!r}")
    path = PurePosixPath(normalized)
    if not path.parts or path.is_absolute():
        raise ValueError(f"输入包含有不安全的路径：{name!r}")
    return path.as_posix()


def _resolve_asset(chart_member: str, referenced_name: str) -> str:
    referenced = referenced_name.replace("\\", "/")
    reference = PurePosixPath(referenced)
    if not referenced or not reference.parts or reference.is_absolute() or ":" in reference.parts[0]:
        raise ValueError(f"谱面引用了不安全的媒体路径：{referenced_name!r}")
    chart_parent = PurePosixPath(chart_member).parent
    normalized = posixpath.normpath((chart_parent / reference).as_posix())
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"谱面引用的媒体路径越出包根目录：{referenced_name!r}")
    return _safe_member(normalized)


def _media_references(text: str, chart_member: str) -> tuple[str, ...]:
    parsed = parse_osu(text)
    audio_name = parsed["general"].get("AudioFilename", "")
    if not audio_name:
        raise ValueError("谱面缺少 General.AudioFilename，无法生成可播放的 .osz")
    references = {_resolve_asset(chart_member, audio_name)}
    content = sections(text)

    for line in content.get("Events", []):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            fields = next(csv.reader([line], skipinitialspace=True))
        except (csv.Error, StopIteration):
            continue
        for field in fields:
            candidate = field.strip().strip('"')
            if PurePosixPath(candidate).suffix.lower() in _MEDIA_EXTENSIONS:
                references.add(_resolve_asset(chart_member, candidate))

    for line in content.get("HitObjects", []):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        fields = line.split(",")
        if len(fields) < 6:
            continue
        try:
            is_hold = bool(int(fields[3]) & 128)
        except ValueError:
            continue
        sample_field = fields[6] if is_hold and len(fields) > 6 else fields[5]
        sample_name = sample_field.split(":")[-1].strip().strip('"')
        if PurePosixPath(sample_name).suffix.lower() in _MEDIA_EXTENSIONS:
            references.add(_resolve_asset(chart_member, sample_name))
    return tuple(sorted(references))


def _load_source(input_path: Path, chart_member: str | None) -> _SourceChart:
    suffix = input_path.suffix.lower()
    if suffix == ".osu":
        text = input_path.read_text(encoding="utf-8-sig")
        safe_name = _safe_member(input_path.name)
        _media_references(text, safe_name)
        return _SourceChart(input_path, safe_name, text, False)
    if suffix != ".osz":
        raise ValueError("输入文件必须是 .osu 或 .osz")

    with zipfile.ZipFile(input_path, "r") as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        safe_names = [_safe_member(info.filename) for info in members]
        folded: dict[str, zipfile.ZipInfo] = {}
        for info, name in zip(members, safe_names):
            key = name.casefold()
            if key in folded:
                raise ValueError(f".osz 中存在重复成员：{name}")
            folded[key] = info
        charts = [name for name in safe_names if name.lower().endswith(".osu")]
        if not charts:
            raise ValueError(".osz 中没有 .osu 谱面")
        if chart_member:
            selected = _safe_member(chart_member)
            if selected not in charts:
                raise ValueError(f".osz 中不存在指定谱面：{selected}")
        else:
            selected = sorted(charts, key=str.casefold)[0]
        chart_info = folded[selected.casefold()]
        try:
            text = archive.read(chart_info).decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(f"谱面不是有效 UTF-8：{selected}") from error
        referenced = _media_references(text, selected)
        available = set(folded)
        missing = [name for name in referenced if name.casefold() not in available]
        if missing:
            raise ValueError("输入 .osz 缺少谱面引用的媒体文件：" + ", ".join(missing[:5]))
    return _SourceChart(input_path, selected, text, True)


def _set_difficulty_metadata(text: str, version: str) -> str:
    """Set stable, osu!-visible labels and clear inherited ranked IDs."""
    source_sections = sections(text)
    metadata = source_sections.get("Metadata")
    if metadata is None:
        raise ValueError("谱面缺少 [Metadata] 区块")
    replacements = {
        "version": f"Version:{version}",
        "beatmapid": "BeatmapID:0",
        "beatmapsetid": "BeatmapSetID:-1",
    }
    seen = set()
    updated = []
    for line in metadata:
        key = line.split(":", 1)[0].strip().casefold() if ":" in line else ""
        if key in replacements:
            if key not in seen:
                updated.append(replacements[key])
                seen.add(key)
            continue
        updated.append(line)
    for key, line in replacements.items():
        if key not in seen:
            updated.append(line)
    source_sections["Metadata"] = updated
    # export_osu() uses this same section-preserving representation.
    return "\n".join(
        ["\n".join(source_sections.get("", ["osu file format v14"]))]
        + [f"[{name}]\n" + "\n".join(body)
           for name, body in source_sections.items() if name]
    ) + "\n"


def _object_signature(notes) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(
        tuple(field.strip() for field in note.raw.split(",")[1:])
        for note in notes
    ))


def _validate_chart(
    text: str,
    keys: int,
    source_notes,
    *,
    allow_added_objects: bool = False,
) -> dict:
    parsed = parse_osu(text)
    report = validate(parsed["notes"], keys)
    if not report["valid"]:
        raise ValueError("转换谱面存在轨道冲突或越界：" + report["issues"][0])
    if parsed["keys"] != keys:
        raise ValueError(f"转换后键数应为 {keys}K，实际为 {parsed['keys']}K")
    expected = Counter(_object_signature(source_notes))
    actual = Counter(_object_signature(parsed["notes"]))
    if (actual != expected if not allow_added_objects else bool(expected - actual)):
        raise ValueError("键型转换改变了物件起止、标志或采样字段")
    return parsed


def _copy_archive_assets(source: _SourceChart, output: zipfile.ZipFile) -> None:
    with zipfile.ZipFile(source.path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = _safe_member(info.filename)
            if name.lower().endswith(".osu"):
                continue
            with archive.open(info, "r") as reader, output.open(name, "w", force_zip64=True) as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _copy_standalone_assets(source: _SourceChart, output: zipfile.ZipFile) -> None:
    source_text = source.text
    references = _media_references(source_text, source.chart_member)
    root = source.path.parent.resolve()
    for member in references:
        relative = PurePosixPath(member)
        candidate = (root / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"谱面媒体文件越出输入目录：{member}") from error
        if not candidate.is_file():
            raise ValueError(f"谱面引用的媒体文件不存在：{candidate}")
        with candidate.open("rb") as reader, output.open(member, "w", force_zip64=True) as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _find_number(data: dict, names: set[str]):
    for key, value in data.items():
        normalized = str(key).casefold().replace("-", "_")
        if normalized in names and isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    for value in data.values():
        if isinstance(value, dict):
            found = _find_number(value, names)
            if found is not None:
                return found
    return None


def _pattern_values(analysis: dict, object_count: int) -> dict[str, float]:
    result: dict[str, float] = {}
    raw = analysis.get("patterns", {})
    if isinstance(raw, list):
        items = []
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name", item.get("label", item.get("type")))
                if "percentage" in item:
                    value, kind = item["percentage"], "percentage"
                elif "ratio" in item or "share" in item:
                    value, kind = item.get("ratio", item.get("share")), "ratio"
                else:
                    value, kind = item.get("count"), "count"
                if name is not None and isinstance(value, (int, float)):
                    items.append((str(name), value, kind))
    elif isinstance(raw, dict):
        items = []
        for name, value in raw.items():
            if isinstance(value, dict):
                if "percentage" in value:
                    score, kind = value["percentage"], "percentage"
                elif "ratio" in value or "share" in value:
                    score, kind = value.get("ratio", value.get("share")), "ratio"
                else:
                    score, kind = value.get("count"), "count"
                if isinstance(score, (int, float)):
                    items.append((str(name), score, kind))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                items.append((str(name), value, "count"))
    else:
        items = []
    for name, value, kind in items:
        if kind == "ratio":
            percentage = float(value) * 100
        elif kind == "percentage":
            percentage = float(value) * 100 if 0 <= value <= 1 else float(value)
        else:
            percentage = float(value) / max(1, object_count) * 100
        result[name] = percentage
    return result


def _metrics(analysis: dict, parsed: dict) -> dict:
    if not isinstance(analysis, dict):
        raise ValueError("Torosu analyze 必须返回 JSON 对象")
    objects = len(parsed["notes"])
    left = _find_number(analysis, {"left", "left_hand", "left_count", "left_notes"})
    right = _find_number(analysis, {"right", "right_hand", "right_count", "right_notes"})
    hand_balance = _find_number(analysis, {"hand_balance", "hand_balance_ratio", "balance_ratio"})
    if hand_balance is None and left is not None and right is not None:
        hand_balance = min(left, right) / max(1.0, max(left, right))
    return {
        "objects": objects,
        "ln": sum(note.end > note.start for note in parsed["notes"]),
        "minijack": _find_number(analysis, {"minijack", "minijacks", "minijack_count", "minijack_events"}) or 0,
        "pill": _find_number(analysis, {"pill", "pills", "pill_count", "pill_events"}) or 0,
        "hand_balance": hand_balance,
        "patterns": _pattern_values(analysis, objects),
    }


def _format_number(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_report(rows: dict[str, dict], labels: tuple[str, str]) -> None:
    baseline, torosu = labels
    keys = list(dict.fromkeys(
        [*_PATTERN_NAMES]
        + sorted(set(rows[baseline]["patterns"]) | set(rows[torosu]["patterns"]))
    ))
    header = ("指标", baseline, torosu)
    values_to_print = [
        ("转换耗时", f'{rows[baseline]["elapsed_ms"]:.2f} ms',
         f'{rows[torosu]["elapsed_ms"]:.2f} ms'),
        ("物件数", rows[baseline]["objects"], rows[torosu]["objects"]),
        ("LN 数", rows[baseline]["ln"], rows[torosu]["ln"]),
        ("minijack", rows[baseline]["minijack"], rows[torosu]["minijack"]),
        ("pill", rows[baseline]["pill"], rows[torosu]["pill"]),
        ("双手均衡", _format_number(rows[baseline]["hand_balance"]),
         _format_number(rows[torosu]["hand_balance"])),
    ]
    values_to_print.extend(
        (f"{name} 占比", f'{rows[baseline]["patterns"].get(name, 0):.1f}%',
         f'{rows[torosu]["patterns"].get(name, 0):.1f}%')
        for name in keys
    )
    widths = [max(len(str(cell)) for cell in [header[index], *(row[index] for row in values_to_print)])
              for index in range(3)]
    print(" | ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(header)))
    print("-+-".join("-" * width for width in widths))
    for row in values_to_print:
        print(" | ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))


def export_comparison(
    input_path: str | Path,
    keys: int,
    output_path: str | Path,
    *,
    chart_member: str | None = None,
    playstyle: str | None = None,
    density: str = "none",
    keep_jacks: bool = True,
    converter=None,
    analyzer=None,
) -> dict:
    """Write an A/B .osz and return its chart names and metrics."""
    if type(keys) is not int or keys not in range(6, 11):
        raise ValueError("目标键数必须为 6–10")
    source_path = Path(input_path)
    output = Path(output_path)
    if output.suffix.lower() != ".osz":
        raise ValueError("输出文件扩展名必须是 .osz")
    if source_path.resolve() == output.resolve():
        raise ValueError("输出路径不能覆盖输入谱面")
    if converter is None or analyzer is None:
        from . import torosu_bridge
        converter = converter or torosu_bridge.convert_osu_text
        analyzer = analyzer or torosu_bridge.analyze_osu_text

    source = _load_source(source_path, chart_member)
    parsed_source = parse_osu(source.text)
    original_notes = parsed_source["notes"]
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)

    labels = (f"[Baseline-OldLayout-{keys}K]", f"[Torosu-{keys}K]")
    baseline_started = time.perf_counter()
    source_keys = parsed_source["keys"]
    baseline_seed = [
        replace(
            note,
            lane=min(keys - 1, int((note.lane + .5) * keys / source_keys)),
        )
        for note in original_notes
    ]
    baseline_notes = assign_lanes(
        baseline_seed,
        {"keys": keys, "layout_engine": "beam"},
        timing_points=parsed_source["timing"],
    )
    baseline_text = export_osu(
        baseline_notes, keys, None, original=source.text, version=labels[0],
    )
    baseline_text = _set_difficulty_metadata(baseline_text, labels[0])
    baseline_parsed = _validate_chart(baseline_text, keys, original_notes)
    baseline_elapsed = (time.perf_counter() - baseline_started) * 1000

    torosu_started = time.perf_counter()
    torosu_text = converter(
        source.text,
        keys,
        playstyle=playstyle,
        density=density,
        keep_jacks=keep_jacks,
        strict=True,
    )
    if not isinstance(torosu_text, str):
        raise ValueError("Torosu convert 必须返回 .osu 文本")
    torosu_text = _set_difficulty_metadata(torosu_text, labels[1])
    torosu_parsed = _validate_chart(
        torosu_text,
        keys,
        original_notes,
        allow_added_objects=density.casefold().replace("_", "-") != "none",
    )
    torosu_elapsed = (time.perf_counter() - torosu_started) * 1000

    report = {
        labels[0]: _metrics(analyzer(baseline_text, keys=keys, playstyle=playstyle), baseline_parsed)
        | {"elapsed_ms": baseline_elapsed},
        labels[1]: _metrics(analyzer(torosu_text, keys=keys, playstyle=playstyle), torosu_parsed)
        | {"elapsed_ms": torosu_elapsed},
    }
    output_chart_paths = tuple(
        (PurePosixPath(source.chart_member).parent / f"{label}.osu").as_posix()
        for label in labels
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".tmp", dir=output.parent,
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(
            temporary_name, "w", compression=zipfile.ZIP_DEFLATED,
        ) as package:
            if source.is_archive:
                _copy_archive_assets(source, package)
            else:
                _copy_standalone_assets(source, package)
            package.writestr(output_chart_paths[0], baseline_text.encode("utf-8"))
            package.writestr(output_chart_paths[1], torosu_text.encode("utf-8"))
        with zipfile.ZipFile(temporary_name, "r") as package:
            bad_member = package.testzip()
            if bad_member is not None:
                raise ValueError(f"输出 .osz CRC 校验失败：{bad_member}")
            names = package.namelist()
            for chart_path in output_chart_paths:
                if names.count(chart_path) != 1:
                    raise ValueError(f"输出 .osz 缺少唯一对比谱面：{chart_path}")
            for chart_path in output_chart_paths:
                parsed = parse_osu(package.read(chart_path).decode("utf-8"))
                if parsed["keys"] != keys or not validate(parsed["notes"], keys)["valid"]:
                    raise ValueError(f"输出 .osz 内谱面结构校验失败：{chart_path}")
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    _print_report(report, labels)
    return {
        "output": output,
        "chart_paths": output_chart_paths,
        "labels": labels,
        "rows": report,
        "source_chart": source.chart_member,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导出 ManiaLab 旧 layout 与 Torosu 的同曲 .osz 实打对比包",
    )
    parser.add_argument("--input", required=True, help="输入 .osu 或 .osz")
    parser.add_argument("--keys", required=True, type=int, choices=range(6, 11))
    parser.add_argument("--output", required=True, help="输出 .osz 路径")
    parser.add_argument("--chart", help="多谱面 .osz 内指定的 .osu 成员路径")
    parser.add_argument("--playstyle", help="Torosu playstyle")
    parser.add_argument("--density", default="none", help="Torosu density profile")
    parser.add_argument(
        "--keep-jacks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保留重复键型（默认开启）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = export_comparison(
            args.input,
            args.keys,
            args.output,
            chart_member=args.chart,
            playstyle=args.playstyle,
            density=args.density,
            keep_jacks=args.keep_jacks,
        )
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"导出失败：{error}", file=sys.stderr)
        return 2
    print(f"已生成：{result['output']}")
    print(f"输入谱面：{result['source_chart']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
