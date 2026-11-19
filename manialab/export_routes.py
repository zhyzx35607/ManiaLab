import io
import json
from pathlib import Path
import tempfile
import zipfile

from flask import send_file

from .domain import export_osu, note_from, parse_osu
from .storage import safe_path


def register_export_routes(app, projects, resources, current_audio):
    @app.get("/api/projects/<pid>/export/<kind>")
    def export(pid, kind):
        state = projects.load(pid)
        if kind == "json":
            if not state.get("analysis") and state["active"] < 0:
                raise ValueError("请先分析音频或生成候选，再下载分析 JSON")
            evidence = {"title": state["title"],
                        "variant": state["variants"][state["active"]] if state["active"] >= 0 else None,
                        "analysis": state.get("analysis"), "plan": state.get("plan"),
                        "reference_style": state.get("reference_style"),
                        "expression": state.get("expression"), "reference_marks": state.get("reference_marks")}
            return send_file(io.BytesIO(json.dumps(evidence, ensure_ascii=False, indent=2).encode()),
                             download_name="ManiaLab-analysis.json", as_attachment=True,
                             mimetype="application/json")
        if state["active"] < 0:
            raise ValueError("请先生成谱面")
        variant = state["variants"][state["active"]]
        notes = [note_from(n) for n in variant["notes"]]
        if variant["mode"] == "audio" and variant.get("timing_revision", 0) < 3:
            raise ValueError("该候选来自旧版，未通过完整落线检查；请重新分析音频并生成整谱后导出")
        base = projects.path(pid) / "files"
        original = None
        folder = Path()
        if variant["mode"] == "remap":
            original = safe_path(base, state["original"]).read_text(encoding="utf-8-sig")
            folder = Path(state["original"]).parent
        content = export_osu(notes, variant["settings"]["keys"], resources / "vendor" / "mapperatorinator" / "template.osu",
                             title=state["title"], artist=state["artist"], audio=state.get("audio") or "audio.ogg",
                             bpm=variant["bpm"], offset=variant["offset"], original=original, meter=variant.get("meter", 4),
                             grid_division=variant["division"] if variant["mode"] == "audio" else None,
                             version=f'ManiaLab {variant["settings"]["seed"]}',
                             bookmarks=[max(0, round(t-variant.get('audio_shift_ms', 0))) for t in
                                 variant.get('selection', {}).get('structure', {}).get('boundaries', [])[1:-1]])
        if variant.get('provided_timing'):
            from .upstream import verify_fixed_timing

            verify_fixed_timing(content, variant['provided_timing'])
        if kind == "osu":
            return send_file(io.BytesIO(content.encode("utf-8")), download_name="ManiaLab.osu",
                             as_attachment=True, mimetype="application/octet-stream")
        if kind != "osz":
            raise ValueError("不支持的导出格式")
        current_audio(state)
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
                    referenced = safe_path(base, (folder / audio_name).as_posix())
                    if not referenced.is_file():
                        archive.write(current_audio(state), referenced.relative_to(base).as_posix())
            buffer.seek(0)
            response = send_file(buffer, download_name="ManiaLab.osz", as_attachment=True,
                                 mimetype="application/octet-stream")
            response.call_on_close(buffer.close)
            return response
        except BaseException:
            buffer.close()
            raise
