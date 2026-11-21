import math
import re
import secrets
import time

from flask import request

from .arrange import layout_metrics, parameters
from .candidate_identity import new_candidate_id
from .audio_io import decode
from .chart_workspace import archive_active_chart, audio_work, chart_workspace_snapshot_path
from .domain import clean_meta, parse_osu, upstream_events, validate
from .ln_patterns import analyze_ln_patterns
from .review import review_chart
from .structure import reference_marks
from .timing import grid_report
from .upstream import fixed_timing_output


def register_upstream_routes(app, *, engine, projects, submit, current_audio, requested_tempo):
    @app.post("/api/projects/<pid>/engine-install")
    def engine_install(pid):
        if not engine.status().get("install_available"):
            raise ValueError("当前便携配置不含原模型安装器")
        installation = {"ready_invalidated": False}

        def mark_ready_invalidated():
            installation["ready_invalidated"] = True

        def operation(state, progress, cancelled):
            engine.install(progress, cancelled, on_ready_invalidated=mark_ready_invalidated)
            return engine.status()

        def cleanup(committed):
            if installation["ready_invalidated"] and not committed:
                engine.invalidate_ready_marker()

        return submit(pid, operation, cleanup=cleanup)

    @app.post("/api/projects/<pid>/engine-generate")
    def engine_generate(pid):
        if not engine.status().get("generate_available"):
            raise ValueError("当前便携配置不含原模型运行器；请使用支持该组件的构建")
        payload = request.get_json() or {}
        prefs = parameters(payload.get("settings", {}))
        seconds = payload.get("seconds", 30)
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 1200:
            raise ValueError("时长应为 0–1200 秒；0 表示全曲")

        publication = {}

        def cleanup(committed):
            snapshot_path = publication.get("snapshot_path")
            snapshot_backup = publication.get("snapshot_backup")
            if committed:
                if snapshot_backup is not None:
                    try:
                        snapshot_backup.unlink(missing_ok=True)
                    except OSError as error:
                        if snapshot_backup.exists():
                            error.add_note(
                                f"ManiaLab恢复提示：结果已保存；旧工作区快照备份尚未清理，"
                                f"路径为 {snapshot_backup}。")
                        raise
                return
            failures = []
            chart_path = publication.get("chart_path")
            if publication.get("chart_created") and chart_path is not None:
                try:
                    chart_path.unlink(missing_ok=True)
                except OSError as error:
                    failures.append(error)
            if snapshot_path is not None:
                try:
                    if snapshot_backup is not None and snapshot_backup.exists():
                        snapshot_backup.replace(snapshot_path)
                    elif publication.get("archive_attempted") and not publication["snapshot_existed"]:
                        snapshot_path.unlink(missing_ok=True)
                except OSError as error:
                    if snapshot_backup is not None and snapshot_backup.exists():
                        error.add_note(
                            f"ManiaLab恢复提示：旧工作区快照未能恢复；备份仍在 "
                            f"{snapshot_backup}，目标路径为 {snapshot_path}。请保留备份并核对项目状态。")
                    failures.append(error)
                if not publication["snapshot_parent_existed"]:
                    try:
                        snapshot_path.parent.rmdir()
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        failures.append(error)
            if failures:
                raise ExceptionGroup("原模型发布资源回滚失败", failures)

        def operation(state, progress, cancelled):
            def check_cancelled():
                if cancelled():
                    raise InterruptedError("已取消，原有谱面已保留")

            # Supply decoded WAV so upstream pydub needs no global ffmpeg installation.
            project = projects.path(pid)
            work = audio_work(project, state)
            if not (work / "preview.wav").exists():
                decode(current_audio(state), work, cancelled=cancelled)
            check_cancelled()
            tempo = requested_tempo(payload, state) if payload.get("use_timing") is True else None
            if tempo is not None and not (tempo.get("stable") or payload.get("timing_confirmed") is True):
                raise ValueError("请先检查并确认当前 BPM / Offset，再传入原模型")
            raw = engine.generate(
                work / "preview.wav", work / ("engine-" + secrets.token_hex(4)),
                prefs, progress, cancelled, seconds, tempo,
            )
            check_cancelled()
            text = raw.read_text(encoding="utf-8-sig")
            check_cancelled()
            text = re.sub(r"(?m)^AudioFilename:.*$", lambda _: "AudioFilename: " + state["audio"], text)
            for field, value in {
                "Title": state["title"], "TitleUnicode": state["title"],
                "Artist": state["artist"], "ArtistUnicode": state["artist"],
            }.items():
                text = re.sub(
                    r"(?m)^" + field + r":.*$",
                    lambda _, f=field, v=value: f + ":" + clean_meta(v),
                    text,
                )
            alignment = None
            if tempo:
                text, model_notes, alignment = fixed_timing_output(text, tempo)
            check_cancelled()
            parsed = parse_osu(text)
            check_cancelled()
            selected = model_notes if tempo else parsed["notes"]
            # This is the model comparison path. Reassigning every column here
            # erases the learned LN layout before the user can inspect it.
            # The ordinary remap action remains available after generation.
            assigned = selected
            report = validate(assigned, prefs["keys"])
            check_cancelled()
            beat = 60000 / tempo["bpm"] if tempo else 500
            report["layout"] = layout_metrics(assigned, prefs["keys"], beat, parsed["timing"])
            check_cancelled()
            report["review"] = review_chart(assigned, prefs["keys"], beat, parsed["timing"])
            check_cancelled()
            report["ln_patterns"] = analyze_ln_patterns(assigned, prefs["keys"], beat, parsed["timing"])
            check_cancelled()
            if tempo:
                report["grid"] = grid_report(assigned, tempo)
                check_cancelled()
            warnings = getattr(engine, "last_warnings", [])
            message = "已保留 mania 模型生成的键型。当前应用仍使用原版权重；需要重新定轨时可另行生成候选。"
            if not report["valid"]:
                message += f" 原始键型存在 {report['issue_count']} 项结构冲突，请先检查或重新定轨。"
            if tempo:
                message += (
                    f" 已按确认的 BPM / Offset 及细分量化模型输出：移动 {alignment['moved']} 个起点、"
                    f"{alignment['tails_moved']} 个音尾，{alignment['collapsed_holds']} 个不足一格的 LN 转为短键。"
                    f" {alignment['large_attack_shifts']} 个起音移动超过 30ms，需要回听；落线不代表采音正确。"
                )
            if warnings:
                message += "\n原模型警告（可能包含未配对并被上游丢弃的 LN）：\n" + "\n".join(warnings)
            variant = dict(
                candidate_id=new_candidate_id(), notes=[note.data() for note in assigned],
                settings=prefs, mode="remap", report=report,
                selection={
                    "message": message, "upstream_warnings": warnings, "alignment": alignment,
                    "layout_origin": "mapperatorinator",
                },
                bpm=tempo["bpm"] if tempo else 120,
                offset=tempo["offset"] if tempo else 0,
                division=tempo.get("division", 4) if tempo else 4,
                audio_shift_ms=tempo.get("audio_shift_ms", 0) if tempo else 0,
                timing_confirmed=bool(tempo),
                provided_timing=tempo,
                created=time.time(),
                region=None,
                event_count=len(upstream_events(assigned)),
            )
            check_cancelled()

            name = "Mapperatorinator-" + secrets.token_hex(4) + ".osu"
            chart_path = project / "files" / name
            snapshot_path = chart_workspace_snapshot_path(project, state.get("original"))
            publication.update(
                chart_path=chart_path,
                chart_created=False,
                snapshot_path=snapshot_path,
                snapshot_existed=snapshot_path.is_file(),
                snapshot_backup=None,
                snapshot_parent_existed=snapshot_path.parent.exists(),
            )
            with chart_path.open("x", encoding="utf-8") as output:
                publication["chart_created"] = True
                output.write(text)
            check_cancelled()
            if publication["snapshot_existed"]:
                snapshot_backup = snapshot_path.with_name(
                    f".{snapshot_path.name}.{secrets.token_hex(4)}.rollback"
                )
                publication["snapshot_backup"] = snapshot_backup
                snapshot_path.replace(snapshot_backup)
            publication["archive_attempted"] = True
            archive_active_chart(project, state)
            check_cancelled()
            state.update(
                original=name,
                reference_marks=reference_marks(text),
                input_notes=[note.data() for note in selected],
                input_keys=parsed["keys"],
                input_timing=parsed["timing"],
                mode="remap",
                settings=prefs,
            )
            state["charts"].append({"path": name, "label": f'Mapperatorinator V32 · {prefs["keys"]}K'})
            state["variants"] = [variant]
            state["active"] = 0
            return report

        return submit(pid, operation, cleanup=cleanup)
