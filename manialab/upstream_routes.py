import math
import re
import secrets
import time

from flask import request

from .arrange import layout_metrics, parameters
from .audio_io import decode
from .chart_workspace import archive_active_chart, audio_work
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
        def operation(state, progress, cancelled):
            engine.install(progress, cancelled)
            return engine.status()

        return submit(pid, operation)

    @app.post("/api/projects/<pid>/engine-generate")
    def engine_generate(pid):
        if not engine.status().get("generate_available"):
            raise ValueError("当前便携配置不含原模型运行器；请使用支持该组件的构建")
        payload = request.get_json() or {}
        prefs = parameters(payload.get("settings", {}))
        seconds = payload.get("seconds", 30)
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 1200:
            raise ValueError("时长应为 0–1200 秒；0 表示全曲")

        def operation(state, progress, cancelled):
            # Supply decoded WAV so upstream pydub needs no global ffmpeg installation.
            project = projects.path(pid)
            work = audio_work(project, state)
            if not (work / "preview.wav").exists():
                decode(current_audio(state), work, cancelled=cancelled)
            tempo = requested_tempo(payload, state) if payload.get("use_timing") is True else None
            if tempo is not None and not (tempo.get("stable") or payload.get("timing_confirmed") is True):
                raise ValueError("请先检查并确认当前 BPM / Offset，再传入原模型")
            raw = engine.generate(
                work / "preview.wav", work / ("engine-" + secrets.token_hex(4)),
                prefs, progress, cancelled, seconds, tempo,
            )
            text = raw.read_text(encoding="utf-8-sig")
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
            parsed = parse_osu(text)
            selected = model_notes if tempo else parsed["notes"]
            # This is the model comparison path. Reassigning every column here
            # erases the learned LN layout before the user can inspect it.
            # The ordinary remap action remains available after generation.
            assigned = selected
            name = "Mapperatorinator-" + secrets.token_hex(4) + ".osu"
            (project / "files" / name).write_text(text, encoding="utf-8")
            archive_active_chart(project, state)
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
            report = validate(assigned, prefs["keys"])
            beat = 60000 / tempo["bpm"] if tempo else 500
            report["layout"] = layout_metrics(assigned, prefs["keys"], beat, parsed["timing"])
            report["review"] = review_chart(assigned, prefs["keys"], beat, parsed["timing"])
            report["ln_patterns"] = analyze_ln_patterns(assigned, prefs["keys"], beat, parsed["timing"])
            warnings = getattr(engine, "last_warnings", [])
            message = "已保留 mania 模型生成的键型。当前应用仍使用原版权重；需要重新定轨时可另行生成候选。"
            if not report["valid"]:
                message += f" 原始键型存在 {report['issue_count']} 项结构冲突，请先检查或重新定轨。"
            if tempo:
                report["grid"] = grid_report(assigned, tempo)
                message += (
                    f" 已按确认的 BPM / Offset 及细分量化模型输出：移动 {alignment['moved']} 个起点、"
                    f"{alignment['tails_moved']} 个音尾，{alignment['collapsed_holds']} 个不足一格的 LN 转为短键。"
                    f" {alignment['large_attack_shifts']} 个起音移动超过 30ms，需要回听；落线不代表采音正确。"
                )
            if warnings:
                message += "\n原模型警告（可能包含未配对并被上游丢弃的 LN）：\n" + "\n".join(warnings)
            state["variants"] = [
                dict(
                    notes=[note.data() for note in assigned], settings=prefs, mode="remap", report=report,
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
            ]
            state["active"] = 0
            return report

        return submit(pid, operation)
