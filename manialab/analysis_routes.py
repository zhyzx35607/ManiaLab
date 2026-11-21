import os
import secrets
import shutil
from pathlib import Path

from flask import request

from .chart_workspace import audio_work
from .stems import VOICES
from .storage import AUDIO, safe_path


def register_analysis_routes(app, *, projects, task_service, resources, current_audio, submit,
                             requested_shift, separator, piano_detector, guitar_detector,
                             instrument_detector, services):
    app.extensions["analysis_services"] = services

    @app.post("/api/projects/<pid>/analyze")
    def analyze_audio(pid):
        payload = request.json
        if not isinstance(payload, dict):
            raise ValueError("采音请求必须是对象")
        prefs = services.parameters(payload.get("settings", {}))
        analysis_engine = payload.get("engine", "basic")
        piano_engine = payload.get("piano_engine", "general")
        guitar_engine = payload.get("guitar_engine", "general")
        allow_guitar_releases = payload.get("allow_guitar_predicted_releases", False)
        instrument_engine = payload.get("instrument_engine", "general")
        allow_releases = payload.get("allow_predicted_releases", False)
        if analysis_engine not in ("basic", "deep", "deep6", "provided"):
            raise ValueError("采音引擎无效")
        if piano_engine not in ("general", "specialist"):
            raise ValueError("钢琴采音方式无效")
        if guitar_engine not in ("general", "fl"):
            raise ValueError("吉他采音方式无效")
        if type(allow_guitar_releases) is not bool:
            raise ValueError("吉他模型音尾选项必须是布尔值")
        if allow_guitar_releases and guitar_engine != "fl":
            raise ValueError("请先选择吉他专用采音，再启用吉他模型音尾")
        if instrument_engine not in ("general", "tsumugi"):
            raise ValueError("贝斯与伴奏采音方式无效")
        if type(allow_releases) is not bool:
            raise ValueError("模型音尾选项必须是布尔值")
        if allow_releases and instrument_engine != "tsumugi":
            raise ValueError("请先选择贝斯与其他伴奏专用采音，再启用模型音尾")
        if instrument_engine == "tsumugi":
            if analysis_engine not in ("deep", "deep6", "provided"):
                raise ValueError("专用采音需要分离或导入的贝斯、其他伴奏声轨")
            if not instrument_detector.status()["available"]:
                raise ValueError("当前安装包未包含贝斯与其他伴奏专用组件")
            if analysis_engine == "provided":
                with task_service.lock:
                    provided = projects.load(pid).get("provided_stems", {})
                    if not any(voice in ("bass", "other") for voice in provided.get("voices", [])):
                        raise ValueError("已导入声轨中没有贝斯或其他伴奏")
        if guitar_engine == "fl":
            if analysis_engine not in ("deep6", "provided"):
                raise ValueError("吉他专用采音需要六声部分离或已导入的吉他声轨")
            if not guitar_detector.status()["available"]:
                raise ValueError("当前安装包未包含吉他专用采音模型")
            if analysis_engine == "provided":
                with task_service.lock:
                    provided = projects.load(pid).get("provided_stems", {})
                    if "guitar" not in provided.get("voices", []):
                        raise ValueError("已导入声轨中没有吉他")
        if piano_engine == "specialist":
            if analysis_engine not in ("deep6", "provided"):
                raise ValueError("钢琴专用采音需要六声部分离或已导入的钢琴声轨")
            piano_status = piano_detector.status()
            if not piano_status["available"]:
                raise ValueError(piano_status.get("message", "当前安装包未包含钢琴专用采音模型"))

        def operation(state, progress, cancelled):
            work = audio_work(projects.path(pid), state)
            if analysis_engine in ("deep", "deep6", "provided"):
                selected_separator = (
                    services.ProvidedStemSeparator(state.get("provided_stems"))
                    if analysis_engine == "provided"
                    else services.Separator(resources, "six") if analysis_engine == "deep6" else separator
                )
                state["analysis"] = services.analyze_deep(
                    current_audio(state), work, resources / "assets/basic-pitch.onnx",
                    prefs, selected_separator, progress, cancelled,
                    piano_detector=piano_detector if piano_engine == "specialist" else None,
                    guitar_detector=guitar_detector if guitar_engine == "fl" else None,
                    allow_guitar_predicted_releases=allow_guitar_releases,
                    instrument_detector=instrument_detector if instrument_engine == "tsumugi" else None,
                    allow_predicted_releases=allow_releases,
                )
            else:
                state["analysis"] = services.analyze(
                    current_audio(state), work, resources / "assets" / "basic-pitch.onnx",
                    prefs, progress, cancelled,
                )
            state["settings"] = prefs
            state["analysis_options"] = dict(
                engine=analysis_engine, piano_engine=piano_engine,
                instrument_engine=instrument_engine, allow_predicted_releases=allow_releases,
                guitar_engine=guitar_engine, allow_guitar_predicted_releases=allow_guitar_releases,
            )
            state["audio_shift_ms"] = requested_shift(payload, state)
            from .expression import propose_expression

            if state["analysis"].get("structure"):
                state["expression"] = propose_expression(state["analysis"], prefs)
            else:
                state.pop("expression", None)
            # Local and external plans refer to the previous analysis sections.
            state.pop("plan", None)
            # Keep old variants selectable, but never mask a fresh timing result
            # with the previously selected candidate's BPM / Offset.
            if state["mode"] == "audio":
                state["active"] = -1
            return {"candidates": len(state["analysis"]["candidates"])}

        return submit(
            pid, operation,
            identity=dict(
                engine=analysis_engine, piano_engine=piano_engine,
                instrument_engine=instrument_engine, allow_predicted_releases=allow_releases,
                guitar_engine=guitar_engine, allow_guitar_predicted_releases=allow_guitar_releases,
            ),
        )

    @app.post("/api/projects/<pid>/provided-stems")
    def import_provided_stems(pid):
        with task_service.lock:
            task_service.ensure_project_idle(pid)
            task_service.ensure_idle("已有计算任务运行，请完成后再导入声轨")
            state = projects.load(pid)
            current_audio(state)
            if not request.files or set(request.files) - set(VOICES):
                raise ValueError("请选择并标明声部文件")
            # Preserve old sets and analyses; each upload has an isolated path.
            work = audio_work(projects.path(pid), state)
            folder = work / "provided-uploads" / secrets.token_hex(16)
            uploads_to_save = []
            for voice in request.files:
                uploads = request.files.getlist(voice)
                if len(uploads) != 1:
                    raise ValueError("每个声部只能选择一个文件")
                upload = uploads[0]
                name = Path(upload.filename.replace("\\", "/")).name
                if Path(name).suffix.lower() not in AUDIO:
                    raise ValueError("声部文件必须是音频")
                target = safe_path(folder / voice, name)
                uploads_to_save.append((voice, upload, target))
            sources = {}
            new_decoded_folders = []
            replaced_decoded_folders = []

            def remove_path(path):
                target = Path(path)
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)

            def cleanup(committed):
                failures = []
                if committed:
                    targets = [backup for _, backup in replaced_decoded_folders]
                else:
                    targets = [folder, *new_decoded_folders]
                    for canonical, backup in reversed(replaced_decoded_folders):
                        try:
                            remove_path(canonical)
                            if backup.exists():
                                os.replace(backup, canonical)
                        except OSError as error:
                            if backup.exists():
                                error.add_note(
                                    f"ManiaLab恢复提示：旧声轨缓存备份仍在 {backup}；"
                                    f"目标目录为 {canonical}。请保留备份并核对项目状态。")
                            if isinstance(error, FileNotFoundError) and not backup.exists():
                                continue
                            failures.append(error)
                for target in targets:
                    try:
                        remove_path(target)
                    except OSError as error:
                        if committed and Path(target).exists():
                            error.add_note(
                                f"ManiaLab恢复提示：任务结果已保存；旧声轨缓存备份"
                                f"尚未清理，路径为 {target}。")
                        if isinstance(error, FileNotFoundError) and not Path(target).exists():
                            continue
                        failures.append(error)
                if failures:
                    raise ExceptionGroup('声轨导入资源清理失败', failures)

            def operation(state, progress, cancelled):
                def track_new_folder(path, *, backup=None):
                    target = Path(path)
                    if backup is not None:
                        replacement = target, Path(backup)
                        if replacement not in replaced_decoded_folders:
                            replaced_decoded_folders.append(replacement)
                    elif target not in new_decoded_folders:
                        new_decoded_folders.append(target)

                report = services.prepare_provided_stems(
                    current_audio(state), sources, work, progress, cancelled,
                    on_new_folder=track_new_folder,
                )
                state["provided_stems"] = report
                return {"voices": report["voices"], "duration_seconds": report["duration_seconds"]}

            try:
                for voice, upload, target in uploads_to_save:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as destination:
                        upload.save(destination)
                    sources[voice] = target
                return submit(pid, operation, cleanup=cleanup)
            except Exception:
                cleanup(False)
                raise
