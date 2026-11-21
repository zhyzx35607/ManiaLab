from pathlib import Path
import json
import shutil
from copy import deepcopy

from flask import jsonify, request

from .arrange_config import DEFAULTS
from .chart_workspace import select_chart
from .domain import parse_osu
from .meter import project_with_meter
from .storage import (
    AUDIO,
    MAX_IMPORT_BYTES,
    MAX_IMPORT_ENTRIES,
    extract_osz,
    safe_path,
    validate_import_paths,
)


def register_project_routes(app, projects, task_service, *, parameters, describe_chart):
    def import_files():
        files = request.files.getlist("files")
        if not files:
            raise ValueError("请选择音频、.osu 或 .osz 文件")
        if len(files) > MAX_IMPORT_ENTRIES:
            raise ValueError("导入文件数量超过安全上限")
        pid, path = projects.create()
        try:
            state = dict(id=pid, title="Untitled", artist="Unknown", mode="audio", settings=DEFAULTS.copy(),
                         variants=[], active=-1, charts=[], audio=None, original=None, input_notes=[])
            imported_bytes = 0
            imported_entries = 0
            for file in files:
                name = Path(file.filename.replace("\\", "/")).name
                suffix = Path(name).suffix.lower()
                if suffix == ".osz":
                    added_bytes, added_entries = extract_osz(
                        file.stream,
                        path / "files",
                        max_uncompressed_bytes=MAX_IMPORT_BYTES - imported_bytes,
                        max_entries=MAX_IMPORT_ENTRIES - imported_entries,
                    )
                    imported_bytes += added_bytes
                    imported_entries += added_entries
                elif suffix in AUDIO | {".osu"}:
                    position = file.stream.tell()
                    file.stream.seek(0, 2)
                    added_bytes = file.stream.tell()
                    file.stream.seek(position)
                    if imported_bytes + added_bytes > MAX_IMPORT_BYTES:
                        raise ValueError("导入展开后的总数据量超过 512MB 安全上限")
                    if imported_entries >= MAX_IMPORT_ENTRIES:
                        raise ValueError("导入文件数量超过安全上限")
                    target, = validate_import_paths(path / "files", [(name, False)])
                    file.stream.seek(0)
                    with target.open("xb") as destination:
                        file.save(destination)
                    imported_bytes += added_bytes
                    imported_entries += 1
                else:
                    raise ValueError("支持 MP3/WAV/FLAC/OGG/M4A/OPUS、.osu 和 .osz")
            candidates = list((path / "files").rglob("*"))
            audio_candidates = [
                item for item in candidates
                if item.is_file() and item.suffix.lower() in AUDIO
            ]
            audio = audio_candidates[0] if audio_candidates else None
            if audio:
                state.update(audio=audio.relative_to(path / "files").as_posix(), title=audio.stem)
            for chart in sorted(item for item in candidates if item.is_file() and item.suffix.lower() == ".osu"):
                try:
                    parsed = parse_osu(chart.read_text(encoding="utf-8-sig"))
                    state["charts"].append({"path": chart.relative_to(path / "files").as_posix(),
                                            "label": f'{parsed["keys"]}K · {parsed["metadata"].get("Version", chart.stem)}'})
                except (ValueError, UnicodeError, IndexError):
                    continue
            if state["charts"]:
                state = select_chart(path, state, state["charts"][0]["path"], initial=True)
            elif len(audio_candidates) > 1:
                raise ValueError("没有可用谱面且包含多个音频文件，请只上传一个音频文件")
            elif not state["audio"]:
                raise ValueError("没有可用音频或原生 mania 谱面")
            projects.save(state)
            return jsonify(project_with_meter(state))
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            raise

    def get_project(pid):
        state = projects.load(pid)
        if state.get("original") and "input_timing" not in state:
            source = safe_path(projects.path(pid) / "files", state["original"])
            state["input_timing"] = parse_osu(source.read_text(encoding="utf-8-sig"))["timing"]
        return jsonify(project_with_meter(state))

    def choose_chart(pid):
        with task_service.lock:
            task_service.ensure_project_idle(pid)
            state = projects.load(pid)
            relative = request.json["path"]
            if relative not in [chart["path"] for chart in state["charts"]]:
                raise ValueError("未知谱面")
            if (relative or None) != state.get("original"):
                state = select_chart(projects.path(pid), state, relative)
                projects.save(state)
            return jsonify(project_with_meter(state))

    def choose_variant(pid):
        with task_service.lock:
            task_service.ensure_project_idle(pid)
            state = projects.load(pid)
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValueError("候选版本无效")
            index = payload.get("index")
            if type(index) is not int or not 0 <= index < len(state["variants"]):
                raise ValueError("候选版本无效")
            variant = state["variants"][index]
            candidate_id = payload.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id != variant.get("candidate_id"):
                raise ValueError("候选版本已经改变或页面版本过旧，请刷新后再切换")
            state["active"] = index
            state["settings"] = variant["settings"].copy()
            selection = variant.get("selection")
            expression = selection.get("expression") if isinstance(selection, dict) else None
            analysis = state.get("analysis") or {}
            if isinstance(expression, dict):
                from .expression import signature

                if expression.get("evidence_signature") == signature(analysis):
                    state["expression"] = deepcopy(expression)
                else:
                    state.pop("expression", None)
            else:
                state.pop("expression", None)
            projects.save(state)
            return jsonify(project_with_meter(state))

    def reference_style(pid):
        with task_service.lock:
            task_service.ensure_project_idle(pid)
            state = projects.load(pid)
            file = request.files.get("file")
            if not file or Path(file.filename or "").suffix.lower() != ".osu":
                raise ValueError("请选择一份 .osu 参考谱面")
            raw = file.stream.read(10 * 1024 * 1024 + 1)
            if len(raw) > 10 * 1024 * 1024:
                raise ValueError("参考谱面超过 10MB")
            try:
                profile = describe_chart(parse_osu(raw.decode("utf-8-sig")))
            except UnicodeError:
                raise ValueError("参考谱面必须为 UTF-8 文本") from None
            draft = json.loads(request.form.get("settings", "{}"))
            if not isinstance(draft, dict):
                raise ValueError("当前偏好必须为参数对象")
            prefs = parameters(parameters(state["settings"] | draft) | profile["preferences"])
            state["reference_style"] = profile
            # The current target key count remains the user's explicit choice.
            state["settings"] = prefs
            projects.save(state)
        return jsonify(project_with_meter(state))

    app.add_url_rule("/api/import", endpoint="import_files", view_func=import_files, methods=["POST"])
    app.add_url_rule("/api/projects/<pid>", endpoint="get_project", view_func=get_project, methods=["GET"])
    app.add_url_rule("/api/projects/<pid>/chart", endpoint="choose_chart", view_func=choose_chart,
                     methods=["POST"])
    app.add_url_rule("/api/projects/<pid>/variant", endpoint="choose_variant", view_func=choose_variant,
                     methods=["POST"])
    app.add_url_rule("/api/projects/<pid>/reference-style", endpoint="reference_style",
                     view_func=reference_style, methods=["POST"])
