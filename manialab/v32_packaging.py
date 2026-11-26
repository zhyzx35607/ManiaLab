"""Validated .osz packaging helpers for V32 charts."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from .domain import parse_osu, sections, validate
from .v32_charts import _set_field


_CHART_LABELS = (
    "[1-Old-BasicPitch-MILP]",
    "[2-V32Mini-Native10K-FullMix]",
    "[3-V32Mini-7K-TorosuDense10K-FullMix]",
    "[4-V32Mini-7K-TorosuDense10K-NoVocal]",
)


def _set_chart_metadata(text: str, *, version: str, audio_filename: str) -> str:
    from .v32_charts import _format_sections

    content = sections(text)
    content["General"] = _set_field(content.get("General", []), "AudioFilename", audio_filename)
    content["General"] = _set_field(content["General"], "Mode", "3")
    metadata = content.get("Metadata", [])
    for key, value in (
        ("Version", version),
        ("BeatmapID", "0"),
        ("BeatmapSetID", "-1"),
    ):
        metadata = _set_field(metadata, key, value)
    content["Metadata"] = metadata
    return _format_sections(content)


def _copy_template_media(
    source_path: Path,
    chart_member: str,
    source_text: str,
    output: zipfile.ZipFile,
    *,
    source_is_archive: bool,
) -> set[str]:
    from .ab_export import _media_references, _resolve_asset, _safe_member

    parsed = parse_osu(source_text)
    source_audio = _resolve_asset(chart_member, parsed["general"].get("AudioFilename", ""))
    references = {
        _safe_member(name)
        for name in _media_references(source_text, chart_member)
        if _safe_member(name).casefold() != source_audio.casefold()
    }
    copied = set()
    if source_is_archive:
        with zipfile.ZipFile(source_path, "r") as archive:
            members = {}
            for info in archive.infolist():
                if info.is_dir() or info.filename.lower().endswith(".osu"):
                    continue
                name = _safe_member(info.filename)
                members[name.casefold()] = (name, info)
            for member in sorted(references, key=str.casefold):
                item = members.get(member.casefold())
                if item is None:
                    raise ValueError(f"参考 .osz 缺少媒体文件：{member}")
                name, info = item
                output.writestr(name, archive.read(info))
                copied.add(name.casefold())
    else:
        root = source_path.parent.resolve()
        for member in sorted(references, key=str.casefold):
            candidate = (root / Path(*PurePosixPath(member).parts)).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise ValueError(f"媒体路径越出参考谱面目录：{member}") from error
            if not candidate.is_file():
                raise ValueError(f"参考谱面引用的媒体不存在：{candidate}")
            output.write(candidate, member)
            copied.add(member.casefold())
    return copied


def package_four_difficulties(
    *,
    output_path: str | Path,
    source_chart_path: str | Path,
    source_chart_member: str | None,
    chart_texts: dict[str, str],
    audio_by_label: dict[str, str | Path],
    duration_ms: int,
    labels: tuple[str, ...] = _CHART_LABELS,
) -> dict:
    """Package four validated charts in an explicit order.

    The legacy labels remain the default for existing experiment callers.
    """
    from .ab_export import _load_source, _media_references, _resolve_asset, _safe_member

    if (
        not isinstance(labels, (tuple, list))
        or len(labels) != 4
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != 4
        or any("/" in label or "\\" in label or any(ord(char) < 32 for char in label) for label in labels)
    ):
        raise ValueError("labels 必须包含 4 个唯一且安全的难度标签")
    labels = tuple(labels)
    if tuple(chart_texts) != labels:
        raise ValueError("四难度顺序或标签与验收约定不一致")
    if set(audio_by_label) != set(labels):
        raise ValueError("每个难度都必须显式绑定自己的播放音频")
    source = _load_source(Path(source_chart_path), source_chart_member)
    chart_parent = PurePosixPath(source.chart_member).parent
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有谱包：{output}")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".tmp", dir=output.parent,
    )
    os.close(fd)
    chart_paths = []
    metrics = {}
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            copied = _copy_template_media(
                source.path, source.chart_member, source.text, archive,
                source_is_archive=source.is_archive,
            )
            audio_members = {}
            for label, audio_path in audio_by_label.items():
                audio = Path(audio_path)
                member = _safe_member((chart_parent / audio.name).as_posix())
                if member.casefold() not in copied:
                    archive.write(audio, member)
                    copied.add(member.casefold())
                audio_members[label] = member
            for label in labels:
                audio_member = audio_members[label]
                relative_audio = PurePosixPath(audio_member).relative_to(chart_parent).as_posix()
                text = _set_chart_metadata(
                    chart_texts[label], version=label, audio_filename=relative_audio,
                )
                parsed = parse_osu(text)
                report = validate(parsed["notes"], 10)
                if parsed["keys"] != 10 or not report["valid"]:
                    raise ValueError(f"{label} 存在键数错误、同轨冲突或非法 LN")
                if any(note.end > duration_ms + 100 for note in parsed["notes"]):
                    raise ValueError(f"{label} 含超出音频片段的物件")
                _resolve_asset((chart_parent / f"{label}.osu").as_posix(), relative_audio)
                chart_path = _safe_member((chart_parent / f"{label}.osu").as_posix())
                archive.writestr(chart_path, text.encode("utf-8"))
                chart_paths.append(chart_path)
                metrics[label] = {
                    "objects": len(parsed["notes"]),
                    "ln": report["ln"],
                    "peak_held": report["max_held"],
                    "lane_counts": report["lane_counts"],
                }
        with zipfile.ZipFile(temporary, "r") as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f".osz CRC 校验失败：{bad}")
            names = {name.casefold() for name in archive.namelist()}
            for chart_path in chart_paths:
                chart_text = archive.read(chart_path).decode("utf-8")
                parsed = parse_osu(chart_text)
                audio_ref = _resolve_asset(chart_path, parsed["general"]["AudioFilename"])
                if audio_ref.casefold() not in names:
                    raise ValueError(f"{chart_path} 引用的音频不在 .osz 中")
                missing = [
                    member for member in _media_references(chart_text, chart_path)
                    if member.casefold() not in names
                ]
                if missing:
                    raise ValueError(
                        f"{chart_path} 引用的媒体不在 .osz 中：" + ", ".join(missing[:5])
                    )
                if not validate(parsed["notes"], 10)["valid"]:
                    raise ValueError(f"{chart_path} 回读后结构无效")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"output": output, "chart_paths": tuple(chart_paths), "metrics": metrics}
