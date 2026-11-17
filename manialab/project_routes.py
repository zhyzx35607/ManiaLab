from pathlib import Path
import shutil

from flask import jsonify, request

from .arrange import DEFAULTS
from .domain import parse_osu
from .storage import AUDIO, extract_osz, safe_path, validate_import_paths


def register_project_routes(app, projects, task_service):
    def set_chart(state, relative):
        previous_audio = state.get("audio")
        base = projects.path(state["id"]) / "files"
        source = safe_path(base, relative)
        content = source.read_text(encoding="utf-8-sig")
        parsed = parse_osu(content)
        from .structure import reference_marks

        state["reference_marks"] = reference_marks(content)
        state.pop("expression", None)
        state.pop("plan", None)
        state.update(mode="remap", original=relative, input_notes=[note.data() for note in parsed["notes"]],
                     title=parsed["metadata"].get("Title", source.stem),
                     artist=parsed["metadata"].get("Artist", "Unknown"), input_keys=parsed["keys"],
                     input_timing=parsed["timing"], variants=[], active=-1)
        state["settings"]["keys"] = min(10, max(6, parsed["keys"]))
        audio_name = parsed["general"].get("AudioFilename", "")
        audio = safe_path(base, (Path(relative).parent / audio_name).as_posix()) if audio_name else None
        if audio and audio.is_file():
            state["audio"] = audio.relative_to(base).as_posix()
        elif state.get("audio") and Path(state["audio"]).name.casefold() == Path(audio_name).name.casefold():
            pass
        else:
            state["audio"] = None
        if state.get("audio") != previous_audio:
            state.pop("provided_stems", None)
        state.pop("analysis", None)
        for file in ("activations.npz", "preview.wav"):
            (projects.path(state["id"]) / file).unlink(missing_ok=True)

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
                set_chart(state, state["charts"][0]["path"])
            elif not state["audio"]:
                raise ValueError("没有可用音频或原生 mania 谱面")
            projects.save(state)
            return jsonify(state)
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
            raise

    def get_project(pid):
        state = projects.load(pid)
        if state.get("original") and "input_timing" not in state:
            source = safe_path(projects.path(pid) / "files", state["original"])
            state["input_timing"] = parse_osu(source.read_text(encoding="utf-8-sig"))["timing"]
        return jsonify(state)

    def choose_chart(pid):
        with task_service.lock:
            task_service.ensure_project_idle(pid)
            state = projects.load(pid)
            relative = request.json["path"]
            if relative not in [chart["path"] for chart in state["charts"]]:
                raise ValueError("未知谱面")
            set_chart(state, relative)
            projects.save(state)
            return jsonify(state)

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
            return jsonify(state)

    app.add_url_rule("/api/import", endpoint="import_files", view_func=import_files, methods=["POST"])
    app.add_url_rule("/api/projects/<pid>", endpoint="get_project", view_func=get_project, methods=["GET"])
    app.add_url_rule("/api/projects/<pid>/chart", endpoint="choose_chart", view_func=choose_chart,
                     methods=["POST"])
    app.add_url_rule("/api/projects/<pid>/variant", endpoint="choose_variant", view_func=choose_variant,
                     methods=["POST"])
