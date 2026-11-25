import io
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import unicodedata
import zipfile

from flask import request, send_file

from .domain import export_osu, note_from, parse_osu
from .storage import AUDIO, MAX_IMPORT_BYTES, MAX_IMPORT_ENTRIES, resolve_project_reference, safe_path


MAX_BATCH_EXPORT_ITEMS = 10
MAX_BATCH_EXPORT_BYTES = MAX_IMPORT_BYTES * MAX_BATCH_EXPORT_ITEMS
_BATCH_EXPORT_DISK_RESERVE_BYTES = 256 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_CANDIDATE_ID = re.compile(r"^[0-9a-f]{32}$")


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


def _candidate_osu_content(projects, resources, project_id, state, variant):
    if not isinstance(variant, dict):
        raise ValueError("请先为歌曲生成候选谱面")
    if variant["mode"] == "audio" and variant.get("timing_revision", 0) < 3:
        raise ValueError("该候选来自旧版，未通过完整落线检查；请重新分析并生成整谱后导出")
    notes = [note_from(note) for note in variant["notes"]]
    base = projects.path(project_id) / "files"
    original = None
    folder = Path()
    if variant["mode"] == "remap":
        original = safe_path(base, state["original"]).read_text(encoding="utf-8-sig")
        folder = PurePosixPath(state["original"]).parent
    content = export_osu(
        notes,
        variant["settings"]["keys"],
        resources / "vendor" / "mapperatorinator" / "template.osu",
        title=state["title"],
        artist=state["artist"],
        audio=state.get("audio") or "audio.ogg",
        bpm=variant["bpm"],
        offset=variant["offset"],
        original=original,
        meter=variant.get("meter", 4),
        grid_division=variant["division"] if variant["mode"] == "audio" else None,
        version=f'ManiaLab {variant["settings"]["seed"]}',
        bookmarks=[
            max(0, math.floor(time - variant.get("audio_shift_ms", 0) + .5))
            for time in variant.get("selection", {}).get("structure", {}).get("boundaries", [])[1:-1]
        ],
    )
    if variant.get("provided_timing"):
        from .upstream import verify_fixed_timing

        verify_fixed_timing(content, variant["provided_timing"])
    return content, base, folder


def _build_osz_file(projects, current_audio, project_id, state, variant, content,
                    base, folder, *, temp_dir=None):
    audio_name = parse_osu(content)["general"].get("AudioFilename", "")
    allowed_audio_extensions = AUDIO if variant["mode"] == "remap" else None
    _validate_osz_audio_name(audio_name, allowed_audio_extensions)
    current_audio(state)
    expected_audio_sha256 = _audio_export_sha256(variant) if variant["mode"] == "audio" else None
    buffer = tempfile.TemporaryFile(mode="w+b", dir=temp_dir)
    try:
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr((folder / "ManiaLab.osu").as_posix(), content)
            for file in base.rglob("*"):
                if file.is_symlink():
                    raise ValueError("项目文件不能包含符号链接")
                if file.is_file() and file.suffix.lower() != ".osu":
                    archive.write(file, file.relative_to(base).as_posix())
            # A separately uploaded audio file can satisfy a nested chart by basename.
            # Keep the chart's timing metadata while closing its audio reference in the archive.
            audio_name = parse_osu(content)["general"].get("AudioFilename", "")
            if audio_name:
                referenced = resolve_project_reference(base, folder, audio_name)
                archived_names = {
                    info.filename.replace("\\", "/").casefold()
                    for info in archive.filelist if not info.is_dir()
                }
                reference_name = referenced.relative_to(base).as_posix()
                if not referenced.is_file() and reference_name.casefold() not in archived_names:
                    archive.write(current_audio(state), reference_name)
        with zipfile.ZipFile(buffer, "r") as archive:
            audio_name = parse_osu(content)["general"].get("AudioFilename", "")
            _verify_osz_audio(archive, base, folder, audio_name, expected_audio_sha256,
                              allowed_audio_extensions)
        buffer.seek(0)
        return buffer
    except BaseException:
        buffer.close()
        raise


def _estimate_osz_uncompressed_size(base, folder, content):
    """Bound each generated OSZ and the full batch before allocating its bundle."""
    names = set()
    total = len(content.encode("utf-8"))
    entry_count = 1
    for path in base.rglob("*"):
        if path.is_symlink():
            raise ValueError("项目文件不能包含符号链接")
        if not path.is_file() or path.suffix.lower() == ".osu":
            continue
        entry_count += 1
        if entry_count > MAX_IMPORT_ENTRIES:
            raise ValueError("项目文件数量超过批量谱包安全上限")
        names.add(path.relative_to(base).as_posix().casefold())
        total += path.stat().st_size
        if total > MAX_IMPORT_BYTES:
            raise ValueError("单首歌曲的 OSZ 超过 512MB 批量导出上限")
    audio_name = parse_osu(content)["general"].get("AudioFilename", "")
    if audio_name:
        referenced = resolve_project_reference(base, folder, audio_name)
        reference_name = referenced.relative_to(base).as_posix()
        if reference_name.casefold() not in names:
            if not referenced.is_file():
                raise ValueError("项目引用的音频文件不存在，无法导出 OSZ")
            total += referenced.stat().st_size
    if total > MAX_IMPORT_BYTES:
        raise ValueError("单首歌曲的 OSZ 超过 512MB 批量导出上限")
    return total


def _safe_osz_member_name(title, project_id, index, used):
    title = unicodedata.normalize("NFKC", str(title or "歌曲"))
    title = "".join(
        "_" if char in '<>:"/\\|?*' or ord(char) < 32 else char
        for char in title
    ).strip(" .")[:80] or "歌曲"
    stem = f"{index:02d}-{title}-{project_id[:8]}"
    name, suffix = f"{stem}.osz", 2
    while name.casefold() in used:
        name, suffix = f"{stem}-{suffix}.osz", suffix + 1
    used.add(name.casefold())
    return name


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
        content, base, folder = _candidate_osu_content(
            projects, resources, pid, state, variant,
        )
        if kind == "osu":
            return send_file(io.BytesIO(content.encode("utf-8")), download_name="ManiaLab.osu",
                             as_attachment=True, mimetype="application/octet-stream")
        if kind != "osz":
            raise ValueError("不支持的导出格式")
        buffer = _build_osz_file(
            projects, current_audio, pid, state, variant, content, base, folder,
        )
        response = send_file(buffer, download_name="ManiaLab.osz", as_attachment=True,
                             mimetype="application/octet-stream")
        response.call_on_close(buffer.close)
        return response

    @app.route("/api/batch/export-osz", methods=["GET", "POST"])
    def export_batch_osz():
        if request.method == "GET":
            encoded_items = request.args.get("items", "")
            if len(encoded_items) > 2048:
                raise ValueError("批量谱包下载请求过长")
            items = []
            if encoded_items:
                for value in encoded_items.split(","):
                    parts = value.split(":")
                    if len(parts) != 2:
                        raise ValueError("批量谱包候选列表无效")
                    items.append({"project": parts[0], "candidate_id": parts[1]})
        else:
            payload = request.get_json()
            items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH_EXPORT_ITEMS:
            raise ValueError(f"批量谱包需要 1–{MAX_BATCH_EXPORT_ITEMS} 首歌曲")
        task_service = app.extensions.get("task_service")
        if task_service is None:
            raise ValueError("当前服务不支持批量谱包")

        normalized_items = []
        seen_projects = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("批量谱包项目无效")
            project_id, expected_id = item.get("project"), item.get("candidate_id")
            if (not isinstance(project_id, str)
                    or not isinstance(expected_id, str)
                    or not _CANDIDATE_ID.fullmatch(expected_id)):
                raise ValueError("批量谱包项目 ID 或候选身份无效")
            if project_id in seen_projects:
                raise ValueError("批量谱包不能重复包含同一项目")
            seen_projects.add(project_id)
            projects.path(project_id)
            normalized_items.append((project_id, expected_id))

        snapshots = []
        estimated_total = 0
        for project_id, expected_id in normalized_items:
            if task_service.has_active_for(project_id):
                raise ValueError("批量谱包中的项目有任务正在运行，请等待后重新导出")
            state = projects.load(project_id)
            if state.get("id") != project_id:
                raise ValueError("项目身份无效")
            active = state.get("active", -1)
            variants = state.get("variants", [])
            variant = (variants[active] if isinstance(active, int)
                       and 0 <= active < len(variants) else None)
            if not isinstance(variant, dict) or variant.get("candidate_id") != expected_id:
                raise ValueError("候选已改变或不是当前候选，请刷新批次后重新导出")
            if variant.get("mode") != "audio":
                raise ValueError("批量谱包只支持音频制谱候选")
            content, base, folder = _candidate_osu_content(
                projects, resources, project_id, state, variant,
            )
            estimated_total += _estimate_osz_uncompressed_size(base, folder, content)
            if estimated_total > MAX_BATCH_EXPORT_BYTES:
                raise ValueError("批量 OSZ 总展开大小超过安全上限")
            export_variant = {"mode": variant["mode"]}
            if variant["mode"] == "audio":
                export_variant["source_audio_sha256"] = _audio_export_sha256(variant)
            snapshots.append(dict(
                project=project_id,
                candidate_id=expected_id,
                title=state.get("title") or project_id,
                state={"id": project_id, "audio": state.get("audio")},
                variant=export_variant,
                content=content,
                base=base,
                folder=folder,
            ))

        output_directory = Path(projects.root)
        if shutil.disk_usage(output_directory).free < (
                estimated_total + _BATCH_EXPORT_DISK_RESERVE_BYTES):
            raise ValueError("磁盘可用空间不足，无法安全创建批量谱包")
        outer_buffer = tempfile.TemporaryFile(mode="w+b", dir=output_directory)
        used_names = set()
        try:
            with zipfile.ZipFile(outer_buffer, "w", compression=zipfile.ZIP_STORED,
                                 allowZip64=True) as outer:
                for index, snapshot in enumerate(snapshots, start=1):
                    if task_service.has_active_for(snapshot["project"]):
                        raise ValueError("项目任务已启动，请等待后重新导出批量谱包")
                    inner = _build_osz_file(
                        projects, current_audio, snapshot["project"], snapshot["state"],
                        snapshot["variant"], snapshot["content"], snapshot["base"],
                        snapshot["folder"], temp_dir=output_directory,
                    )
                    try:
                        inner.seek(0, 2)
                        inner_size = inner.tell()
                        member_name = _safe_osz_member_name(
                            snapshot["title"], snapshot["project"], index, used_names,
                        )
                        info = zipfile.ZipInfo(member_name)
                        info.compress_type = zipfile.ZIP_STORED
                        info.file_size = inner_size
                        with outer.open(info, "w", force_zip64=True) as destination:
                            inner.seek(0)
                            shutil.copyfileobj(inner, destination, _COPY_CHUNK_BYTES)
                    finally:
                        inner.close()

                # Do not return a bundle if an item changed while its ZIP was built.
                for snapshot in snapshots:
                    if task_service.has_active_for(snapshot["project"]):
                        raise ValueError("项目任务已启动，请重新生成批量谱包")
                    latest = projects.load(snapshot["project"])
                    active = latest.get("active", -1)
                    variants = latest.get("variants", [])
                    current = (variants[active] if isinstance(active, int)
                               and 0 <= active < len(variants) else None)
                    if (latest.get("id") != snapshot["project"]
                            or not isinstance(current, dict)
                            or current.get("candidate_id") != snapshot["candidate_id"]):
                        raise ValueError("候选在打包时已改变，请刷新批次后重新导出")
            outer_buffer.seek(0)
            response = send_file(
                outer_buffer,
                download_name="ManiaLab-批量谱包.zip",
                as_attachment=True,
                mimetype="application/zip",
            )
            response.call_on_close(outer_buffer.close)
            return response
        except BaseException:
            outer_buffer.close()
            raise
