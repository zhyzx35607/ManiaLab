from pathlib import Path
import json
import shutil

from flask import jsonify, request

from .arrange_config import DEFAULTS
from .chart_workspace import select_chart
from .domain import parse_osu
from .meter import project_with_meter
from .storage import AUDIO, extract_osz, safe_path, validate_import_paths


def register_project_routes(app, projects, task_service, *, parameters, describe_chart):
    def import_files():
        files = request.files.getlist("files")
        if not files:
            raise ValueError("请选择音频、.osu 或 .osz 文件")
        pid, path = projects.create()
        try:
            state = dict(id=pid, title="Untitled", artist="Unknown", mode="audio", settings=DEFAULTS.copy(),
                         variants=[], active=-1, charts=[], audio=None, original=None, input_notes=[])
            for file in files:
                name = Path(file.filename.replace("\\", "/")).name
                suffix = Path(name).suffix.lower()
                if suffix == ".osz":
                    extract_osz(file.stream, path / "files")
                elif suffix in AUDIO | {".osu"}:
                    target, = validate_import_paths(path / "files", [(name, False)])
                    with target.open("xb") as destination:
                        file.save(destination)
                else:
                    raise ValueError("支持 MP3/WAV/FLAC/OGG/M4A/OPUS、.osu 和 .osz")
            candidates = list((path / "files").rglob("*"))
            audio = next((item for item in candidates if item.is_file() and item.suffix.lower() in AUDIO), None)
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
            index = request.json["index"]
            if type(index) is not int or not 0 <= index < len(state["variants"]):
                raise ValueError("候选版本无效")
            state["active"] = index
            state["settings"] = state["variants"][index]["settings"].copy()
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
