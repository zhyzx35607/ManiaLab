import io
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

from flask import request, send_file

from .domain import export_osu, note_from, parse_osu
from .storage import AUDIO, resolve_project_reference, safe_path


def _audio_export_sha256(variant):
    """Return the candidate's recorded source audio identity for OSZ validation."""
    if "source_audio_sha256" in variant:
        value = variant["source_audio_sha256"]
    else:
        value = None
        separation = variant.get("source_separation")
        if isinstance(separation, dict):
            details = separation.get("separation")
            if isinstance(details, dict):
                value = details.get("audio_sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("该候选未保存有效的原始音频 SHA-256，无法安全导出 OSZ；请重新分析并生成候选")
    return value.lower()


def _validate_osz_audio_name(audio_name, allowed_extensions=None):
    if not audio_name:
        raise ValueError("谱面没有音频引用，无法验证 OSZ 中的音频成员")
    suffix = PurePosixPath(str(audio_name).replace("\\", "/")).suffix.lower()
    if suffix == ".osu":
        raise ValueError("谱面的 AudioFilename 不能引用 .osu 谱面文件")
    if allowed_extensions is not None and suffix not in allowed_extensions:
        raise ValueError("remap 谱面的 AudioFilename 必须使用支持的音频格式")


def _verify_osz_audio(archive, base, folder, audio_name, expected_sha256,
                      allowed_extensions=None):
    _validate_osz_audio_name(audio_name, allowed_extensions)
    referenced = resolve_project_reference(base, folder, audio_name)
    expected_name = referenced.relative_to(base).as_posix().casefold()
    matches = [info for info in archive.infolist()
               if not info.is_dir() and info.filename.replace("\\", "/").casefold() == expected_name]
    if len(matches) != 1:
        raise ValueError("OSZ 中的谱面音频引用缺失或重复，无法验证音频成员")

    if expected_sha256 is None:
        return

    digest = hashlib.sha256()
    with archive.open(matches[0], "r") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("OSZ 中的原始音频与候选分析时的音频不一致；请恢复原音频或重新分析并生成候选")


def register_export_routes(app, projects, resources, current_audio):
    @app.get("/api/projects/<pid>/export/<kind>")
    def export(pid, kind):
        state = projects.load(pid)
        variant = state["variants"][state["active"]] if state["active"] >= 0 else None
        expected_id = request.args.get("candidate_id")
        current_id = variant["candidate_id"] if variant is not None else ""
        if expected_id is not None and expected_id != current_id:
            raise ValueError("当前候选已经改变，请刷新项目后再导出")
        if kind == "json":
            if not state.get("analysis") and variant is None:
                raise ValueError("请先分析音频或生成候选，再下载分析 JSON")
            evidence = {"title": state["title"],
                        "variant": dict(variant) if variant is not None else None,
                        "analysis": state.get("analysis"), "plan": state.get("plan"),
                        "reference_style": state.get("reference_style"),
                        "expression": state.get("expression"), "reference_marks": state.get("reference_marks")}
            if evidence["variant"] is not None:
                evidence["variant"].pop("_edit_journal_head", None)
            return send_file(io.BytesIO(json.dumps(evidence, ensure_ascii=False, indent=2).encode()),
                             download_name="ManiaLab-analysis.json", as_attachment=True,
                             mimetype="application/json")
        if variant is None:
            raise ValueError("请先生成谱面")
        notes = [note_from(n) for n in variant["notes"]]
        if variant["mode"] == "audio" and variant.get("timing_revision", 0) < 3:
            raise ValueError("该候选来自旧版，未通过完整落线检查；请重新分析音频并生成整谱后导出")
        base = projects.path(pid) / "files"
        original = None
        folder = Path()
        if variant["mode"] == "remap":
            original = safe_path(base, state["original"]).read_text(encoding="utf-8-sig")
            folder = PurePosixPath(state["original"]).parent
        content = export_osu(notes, variant["settings"]["keys"], resources / "vendor" / "mapperatorinator" / "template.osu",
                             title=state["title"], artist=state["artist"], audio=state.get("audio") or "audio.ogg",
                             bpm=variant["bpm"], offset=variant["offset"], original=original, meter=variant.get("meter", 4),
                             grid_division=variant["division"] if variant["mode"] == "audio" else None,
                             version=f'ManiaLab {variant["settings"]["seed"]}',
                             bookmarks=[max(0, math.floor(t-variant.get('audio_shift_ms', 0)+.5)) for t in
                                 variant.get('selection', {}).get('structure', {}).get('boundaries', [])[1:-1]])
        if variant.get('provided_timing'):
            from .upstream import verify_fixed_timing

            verify_fixed_timing(content, variant['provided_timing'])
        if kind == "osu":
            return send_file(io.BytesIO(content.encode("utf-8")), download_name="ManiaLab.osu",
                             as_attachment=True, mimetype="application/octet-stream")
        if kind != "osz":
            raise ValueError("不支持的导出格式")
        audio_name = parse_osu(content)["general"].get("AudioFilename", "")
        allowed_audio_extensions = AUDIO if variant["mode"] == "remap" else None
        _validate_osz_audio_name(audio_name, allowed_audio_extensions)
        current_audio(state)
        expected_audio_sha256 = _audio_export_sha256(variant) if variant["mode"] == "audio" else None
        buffer = tempfile.TemporaryFile(mode="w+b")
        try:
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr((folder / "ManiaLab.osu").as_posix(), content)
                for file in base.rglob("*"):
                    if file.is_file() and file.suffix.lower() != ".osu":
                        archive.write(file, file.relative_to(base).as_posix())
                # A separately uploaded audio file can satisfy a nested chart by basename.
                # Keep the chart's timing metadata while closing its audio reference in the archive.
                audio_name = parse_osu(content)['general'].get('AudioFilename', '')
                if audio_name:
                    referenced = resolve_project_reference(base, folder, audio_name)
                    archived_names = {
                        info.filename.replace("\\", "/").casefold()
                        for info in archive.filelist if not info.is_dir()
                    }
                    reference_name = referenced.relative_to(base).as_posix()
                    if not referenced.is_file() and reference_name.casefold() not in archived_names:
                        archive.write(current_audio(state), referenced.relative_to(base).as_posix())
            with zipfile.ZipFile(buffer, "r") as archive:
                audio_name = parse_osu(content)["general"].get("AudioFilename", "")
                _verify_osz_audio(archive, base, folder, audio_name, expected_audio_sha256,
                                  allowed_audio_extensions)
            buffer.seek(0)
            response = send_file(buffer, download_name="ManiaLab.osz", as_attachment=True,
                                 mimetype="application/octet-stream")
            response.call_on_close(buffer.close)
            return response
        except BaseException:
            buffer.close()
            raise
