"""Strict subprocess bridge to the read-only Torosu-derived Rust CLI."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Thread
import time

from .domain import Note, parse_osu, validate

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engines" / "torosu_cli"
_SUPPORTED_KEYS = range(6, 11)


class TorosuBridgeError(ValueError):
    """The native Torosu CLI is unavailable or returned an invalid result."""


def _cli_path() -> Path:
    configured = os.environ.get("MANIALAB_TOROSU_CLI")
    frozen_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    configured_target = os.environ.get("CARGO_TARGET_DIR")

    def root_candidates(root: Path) -> list[Path]:
        return [
            root / triple / profile / executable
            for triple in ("x86_64-pc-windows-msvc", "x86_64-pc-windows-gnu")
            for profile in ("release", "debug")
            for executable in ("torosu-cli.exe", "torosu-cli")
        ] + [
            root / profile / executable
            for profile in ("release", "debug")
            for executable in ("torosu-cli.exe", "torosu-cli")
        ]

    bundled_candidates = [
        frozen_root / "engines" / "torosu_cli" / "torosu-cli.exe",
        frozen_root / "engines" / "torosu_cli" / "torosu-cli",
    ]
    if configured:
        candidates = [Path(configured)]
    elif getattr(sys, "frozen", False):
        candidates = bundled_candidates + (
            root_candidates(Path(configured_target)) if configured_target else []
        )
    else:
        candidates = (
            root_candidates(Path(configured_target)) if configured_target else []
        ) + bundled_candidates + root_candidates(_ENGINE_DIR / "target")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise TorosuBridgeError(
        "Torosu CLI 尚未构建。请按 engines/torosu_cli/README.md 在 D 盘配置 "
        "Rust 工具链并运行 cargo build，"
        "或设置 MANIALAB_TOROSU_CLI 指向已构建的可执行文件。"
    )


def _timeout_seconds() -> float:
    try:
        timeout = float(os.environ.get("MANIALAB_TOROSU_TIMEOUT", "30"))
    except ValueError as error:
        raise TorosuBridgeError("MANIALAB_TOROSU_TIMEOUT 必须是正数") from error
    if not 0 < timeout <= 300:
        raise TorosuBridgeError("MANIALAB_TOROSU_TIMEOUT 必须大于 0 且不超过 300 秒")
    return timeout


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_cli(args: list[str], *, cancelled=lambda: False) -> str:
    binary = _cli_path()
    timeout = _timeout_seconds()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    try:
        process = subprocess.Popen(
            [str(binary), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            cwd=_ENGINE_DIR.parent.parent,
        )
    except OSError as error:
        raise TorosuBridgeError(f"无法启动 Torosu CLI：{error}") from error

    def drain(stream, destination):
        destination.append(stream.read())
        stream.close()

    readers = [
        Thread(target=drain, args=(process.stdout, stdout_parts), daemon=True),
        Thread(target=drain, args=(process.stderr, stderr_parts), daemon=True),
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if cancelled():
                _stop_process(process)
                raise InterruptedError("Torosu 定轨已取消")
            if time.monotonic() >= deadline:
                _stop_process(process)
                raise TorosuBridgeError(f"Torosu CLI 超过 {timeout:g} 秒仍未完成")
            time.sleep(0.05)
        returncode = process.wait()
    finally:
        if process.poll() is None:
            _stop_process(process)
        for reader in readers:
            reader.join(timeout=2)
    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if returncode:
        detail = (stderr or stdout).strip()
        raise TorosuBridgeError(
            f"Torosu CLI 退出码 {returncode}"
            + (f"：{detail[-2000:]}" if detail else "")
        )
    return stdout


def _temporary_osu(text: str):
    """Keep development scratch files in the ignored native-engine directory."""
    scratch = (
        Path(os.environ["MANIALAB_TOROSU_TEMP"])
        if os.environ.get("MANIALAB_TOROSU_TEMP")
        else _ENGINE_DIR / ".tmp"
        if not getattr(sys, "frozen", False)
        else Path(tempfile.gettempdir()) / "ManiaLab" / "Torosu"
    )
    scratch.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="manialab-torosu-", dir=scratch)


def _run_with_input(
    text: str,
    args: list[str],
    *,
    output_file: bool = False,
    cancelled=lambda: False,
):
    with _temporary_osu(text) as directory:
        root = Path(directory)
        input_path = root / "input.osu"
        input_path.write_text(text, encoding="utf-8", newline="\n")
        output_path = root / "output.osu"
        expanded = [
            part.replace("{input}", str(input_path)).replace("{output}", str(output_path))
            for part in args
        ]
        stdout = _run_cli(expanded, cancelled=cancelled)
        if output_file:
            if not output_path.is_file():
                raise TorosuBridgeError("Torosu CLI 没有生成输出谱面")
            try:
                return output_path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError) as error:
                raise TorosuBridgeError(f"无法读取 Torosu 输出谱面：{error}") from error
        return stdout


def _validate_conversion(
    source_text: str,
    output_text: str,
    keys: int,
    *,
    strict: bool,
    allow_added_objects: bool,
) -> None:
    source = parse_osu(source_text)
    output = parse_osu(output_text)
    if output["keys"] != keys:
        raise TorosuBridgeError(
            f"Torosu 输出应为 {keys}K，实际为 {output['keys']}K"
        )
    report = validate(output["notes"], keys)
    if not report["valid"]:
        raise TorosuBridgeError(
            "Torosu 输出存在同轨冲突或非法 LN："
            + report["issues"][0]
        )
    source_objects = sorted(_object_payload(note.raw) for note in source["notes"])
    output_objects = sorted(_object_payload(note.raw) for note in output["notes"])
    preserves_source = strict and not (Counter(source_objects) - Counter(output_objects))
    if not allow_added_objects:
        preserves_source = preserves_source and output_objects == source_objects
    if not preserves_source:
        raise TorosuBridgeError(
            "Torosu 转换丢失或改变了源物件起止、标志或采样字段；拒绝保存结果"
        )


def _object_payload(raw: str) -> tuple[str, ...]:
    fields = raw.split(",")
    if len(fields) < 6:
        raise TorosuBridgeError("谱面 HitObject 字段不完整")
    return tuple(field.strip() for field in fields[1:])


def _ln_shape_signature(notes) -> tuple[tuple, ...]:
    signature = []
    for note in notes:
        fields = note.raw.split(",")
        if len(fields) < 6:
            raise TorosuBridgeError("谱面 HitObject 字段不完整")
        kind = int(fields[3])
        normalized_kind = (kind & ~128) | 1
        sample = fields[5].split(":")
        sample = sample[1:] if note.hold else sample
        signature.append((
            note.lane,
            note.start,
            fields[1].strip(),
            normalized_kind,
            fields[4].strip(),
            tuple(part.strip() for part in sample),
        ))
    return tuple(sorted(signature))


def convert_osu_text(
    osu_text: str,
    keys: int,
    *,
    playstyle: str | None = None,
    density: str = "none",
    keep_jacks: bool = True,
    strict: bool = True,
    cancelled=lambda: False,
) -> str:
    """Convert an osu!mania chart and reject any timing/object mutation."""
    if type(keys) is not int or keys not in _SUPPORTED_KEYS:
        raise ValueError("目标键数必须为 6–10")
    if not strict:
        raise ValueError("ManiaLab 桥接必须启用严格 Torosu 输出，以保留原物件字段")
    if not isinstance(osu_text, str):
        raise TypeError("osu_text 必须是字符串")
    parse_osu(osu_text)
    density_name = str(density).casefold().replace("_", "-")
    if density_name in {"moredense", "more-dense"}:
        density_name = "more-dense"
    if density_name not in {"none", "dense", "more-dense", "full"}:
        raise ValueError("density 必须为 none、dense、more-dense 或 full")
    args = [
        "convert",
        "--input",
        "{input}",
        "--output",
        "{output}",
        "--keys",
        str(keys),
        "--density",
        density_name,
    ]
    if strict:
        args.append("--strict")
    args.append("--keep-jacks" if keep_jacks else "--no-keep-jacks")
    if playstyle:
        args.extend(("--playstyle", playstyle))
    output = _run_with_input(
        osu_text, args, output_file=True, cancelled=cancelled,
    )
    _validate_conversion(
        osu_text,
        output,
        keys,
        strict=strict,
        allow_added_objects=density_name != "none",
    )
    return output


def analyze_osu_text(
    osu_text: str,
    *,
    keys: int | None = None,
    playstyle: str | None = None,
) -> dict:
    """Return Torosu pattern and modifier analysis for an osu!mania chart."""
    parsed = parse_osu(osu_text)
    if keys is not None and (type(keys) is not int or parsed["keys"] != keys):
        raise ValueError("分析键数与输入谱面不一致")
    args = ["analyze", "--input", "{input}", "--json"]
    if playstyle:
        args.extend(("--playstyle", playstyle))
    stdout = _run_with_input(osu_text, args)
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise TorosuBridgeError("Torosu analyze 没有输出有效 JSON") from error
    if not isinstance(result, dict):
        raise TorosuBridgeError("Torosu analyze 必须输出 JSON 对象")
    return result


def ln_shape_osu_text(osu_text: str, mode: str) -> str:
    """Apply Torosu's FLN, short-LN, or keep mode to an osu!mania chart."""
    if mode not in {"fln", "short_ln", "keep"}:
        raise ValueError("LN 模式必须为 fln、short_ln 或 keep")
    source = parse_osu(osu_text)
    output = _run_with_input(
        osu_text,
        ["ln-shape", "--input", "{input}", "--output", "{output}", "--mode", mode],
        output_file=True,
    )
    parsed = parse_osu(output)
    if parsed["keys"] != source["keys"]:
        raise TorosuBridgeError("LN 转换意外改变了谱面键数")
    report = validate(parsed["notes"], parsed["keys"])
    if not report["valid"]:
        raise TorosuBridgeError("LN 转换输出结构非法：" + report["issues"][0])
    if _ln_shape_signature(source["notes"]) != _ln_shape_signature(parsed["notes"]):
        raise TorosuBridgeError("LN 转换改变了物件数量、轨道或非 LN 音效字段")
    return output


def _temporary_chart(
    notes: list[Note],
    keys: int,
    timing_points,
    beat_ms: float,
    offset_ms: float,
    meter: int,
) -> str:
    if not 1 <= keys <= 18:
        raise ValueError("Torosu 源谱键数必须为 1–18")
    lines = [
        "osu file format v14",
        "",
        "[General]",
        "Mode:3",
        "AudioFilename:temporary-audio.ogg",
        "",
        "[Metadata]",
        "Title:ManiaLab Torosu bridge",
        "Artist:ManiaLab",
        "Creator:ManiaLab",
        "Version:Temporary source",
        "",
        "[Difficulty]",
        f"CircleSize:{keys}",
        "HPDrainRate:5",
        "OverallDifficulty:7",
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "",
        "[TimingPoints]",
    ]
    valid_points = [
        line.strip() for line in timing_points
        if isinstance(line, str) and line.strip() and len(line.split(",")) >= 2
    ]
    if valid_points:
        lines.extend(valid_points)
    else:
        if not isinstance(beat_ms, (int, float)) or beat_ms <= 0:
            raise ValueError("beat_ms 必须是正数")
        if not isinstance(offset_ms, (int, float)):
            raise ValueError("offset_ms 必须为数字")
        if type(meter) is not int or meter <= 0:
            raise ValueError("meter 必须为正整数")
        lines.append(f"{float(offset_ms):.3f},{float(beat_ms):.8f},{meter},1,0,70,1,0")
    lines.extend(("", "[HitObjects]"))
    for note in notes:
        if not 0 <= note.lane < keys:
            raise ValueError(f"{note.start}ms 的源轨道不在 {keys}K 范围内")
        x = int((note.lane + 0.5) * 512 / keys)
        if note.raw:
            fields = note.raw.split(",")
            if len(fields) >= 6:
                fields[0] = str(x)
                lines.append(",".join(fields))
                continue
        tail = (
            f"{note.end}:0:0:0:0"
            if note.hold else "0:0:0:0:"
        )
        lines.append(f"{x},192,{note.start},{128 if note.hold else 1},0,{tail}")
    return "\n".join(lines) + "\n"


def _tag_sample_filename(sample_field: str, hold: bool, token: str) -> str:
    parts = sample_field.split(":")
    filename_index = 5 if hold else 4
    if len(parts) <= filename_index:
        parts.extend([""] * (filename_index + 1 - len(parts)))
    parts[filename_index] = token
    return ":".join(parts)


def _sample_token(raw: str, hold: bool) -> str | None:
    fields = raw.split(",")
    if len(fields) < 6:
        return None
    sample_parts = fields[5].split(":")
    filename_index = 5 if hold else 4
    return sample_parts[filename_index] if len(sample_parts) > filename_index else None


def _assign_unmapped_notes(
    notes: list[Note],
    target_keys: int,
    beat_ms: float,
    timing_points,
    cancelled=lambda: False,
) -> tuple[list[Note], int]:
    """Seed audio candidates, then let Torosu perform the target conversion."""
    by_start: dict[int, list[Note]] = {}
    for note in notes:
        by_start.setdefault(note.start, []).append(note)
    first_source_keys = min(7, target_keys - 1)
    last_error = None
    for source_keys in range(first_source_keys, target_keys + 1):
        if cancelled():
            raise InterruptedError("Torosu 定轨已取消")
        if source_keys >= 6:
            from .arrange_config import parameters
            from .layout import search_layout

            try:
                prefs = parameters({"keys": source_keys, "layout_engine": "beam"})
                return search_layout(
                    notes,
                    prefs,
                    beat_ms=beat_ms,
                    cancelled=cancelled,
                    timing_points=timing_points,
                ), source_keys
            except ValueError as error:
                last_error = error
                continue
        busy = [-1] * source_keys
        result = []
        try:
            for start, chord in sorted(by_start.items()):
                available = [lane for lane, end in enumerate(busy) if end <= start]
                if len(chord) > len(available):
                    raise ValueError(f"{start}ms")
                ordered = sorted(chord, key=lambda note: (note.pitch, note.id))
                for note, lane in zip(ordered, available):
                    result.append(replace(note, lane=lane))
                    busy[lane] = note.end
            return result, source_keys
        except ValueError as error:
            last_error = error
    raise ValueError(
        f"无法在 Torosu 源谱中安排所有同时按住物件（{last_error}）；"
        "请降低候选密度或改用其他定轨引擎"
    )


def convert_notes(
    notes,
    keys: int,
    *,
    source_keys: int | None = None,
    timing_points=(),
    beat_ms: float = 500,
    playstyle: str | None = None,
    density: str = "none",
    keep_jacks: bool = True,
    offset_ms: float = 0,
    meter: int = 4,
    cancelled=lambda: False,
    strict: bool = True,
) -> list[Note]:
    """Convert Note objects while preserving identity, timing, and LN type."""
    if type(keys) is not int or keys not in _SUPPORTED_KEYS:
        raise ValueError("目标键数必须为 6–10")
    source_notes = list(notes)
    if not source_notes:
        return []
    if not strict:
        raise ValueError("阶段 #02 只支持保时长、保物件的严格 Torosu 转换")
    if cancelled():
        raise InterruptedError("Torosu 定轨已取消")
    if all(note.lane < 0 for note in source_notes):
        source_notes, source_keys = _assign_unmapped_notes(
            source_notes, keys, beat_ms, timing_points, cancelled,
        )
    elif any(note.lane < 0 for note in source_notes):
        raise ValueError("Torosu 定轨不接受已映射与未映射轨道混合的输入")
    elif source_keys is None:
        source_keys = max(note.lane for note in source_notes) + 1
    if type(source_keys) is not int or not 1 <= source_keys <= 18:
        raise ValueError("源谱键数必须为 1–18")

    identity_by_token = {}
    tagged_notes = []
    for index, note in enumerate(source_notes):
        token = f"ml-{index:08x}.wav"
        identity_by_token[token] = note
        x = int((note.lane + 0.5) * 512 / source_keys)
        if note.raw:
            fields = note.raw.split(",")
            fields[0] = str(x)
            fields[5] = _tag_sample_filename(fields[5], note.hold, token)
            tagged_notes.append(replace(note, raw=",".join(fields)))
        else:
            sample = _tag_sample_filename(
                f"{note.end}:0:0:0:0:" if note.hold else "0:0:0:0:",
                note.hold,
                token,
            )
            tagged_notes.append(replace(
                note,
                raw=f"{x},192,{note.start},{128 if note.hold else 1},0,{sample}",
            ))
    source_text = _temporary_chart(
        tagged_notes, source_keys, timing_points, beat_ms, offset_ms, meter,
    )
    if cancelled():
        raise InterruptedError("Torosu 定轨已取消")
    output_text = convert_osu_text(
        source_text,
        keys,
        playstyle=playstyle,
        density=density,
        keep_jacks=keep_jacks,
        strict=strict,
        cancelled=cancelled,
    )
    if cancelled():
        raise InterruptedError("Torosu 定轨已取消")
    output = parse_osu(output_text)["notes"]
    result = []
    for converted in output:
        token = _sample_token(converted.raw, converted.hold)
        original = identity_by_token.pop(token, None)
        if original is None:
            raise TorosuBridgeError("Torosu 输出无法映射回原始音符身份")
        result.append(replace(original, lane=converted.lane))
    if identity_by_token:
        raise TorosuBridgeError("Torosu 输出遗漏了原始音符")
    report = validate(result, keys)
    if not report["valid"]:
        raise TorosuBridgeError("Torosu 定轨结果存在冲突：" + report["issues"][0])
    return result
