"""Keep each difficulty's active work and each audio file's derived media separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from .domain import parse_osu
from .arrange_config import DEFAULT_KEY_LAYOUTS
from .storage import resolve_project_reference, safe_path


_PROJECT_FIELDS = ("id", "charts", "cache_origin_audio", "track_metadata", "export_metadata")


def _key(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def audio_work(project, state):
    audio = state.get("audio")
    if audio == state.get("cache_origin_audio", audio):
        return project
    return project / "audio-cache" / _key(audio or "")


def chart_workspace_snapshot_path(project, source):
    return project / "chart-workspaces" / (_key(source or "") + ".json")


def archive_active_chart(project, state):
    # Existing projects keep using their original project-root audio caches.
    state.setdefault("cache_origin_audio", state.get("audio"))
    source = state.get("original")
    path = chart_workspace_snapshot_path(project, source)
    path.parent.mkdir(exist_ok=True)
    snapshot = {key: value for key, value in state.items() if key not in _PROJECT_FIELDS}
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except BaseException as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(
                f"ManiaLab恢复提示：谱面快照临时文件未能清理，路径：{temporary}；"
                f"清理错误：{cleanup_error!r}"
            )
        raise
    try:
        temporary.unlink(missing_ok=True)
    except OSError as cleanup_error:
        cleanup_error.add_note(
            f"ManiaLab恢复提示：谱面快照临时文件未能清理，路径：{temporary}"
        )
        raise
    if source is None and not any(c["path"] == "" for c in state["charts"]):
        state["charts"].insert(0, {"path": "", "label": "原始音频项目"})


def _chart_audio(base, relative, audio_name):
    if not audio_name:
        return None
    name = PurePosixPath(str(audio_name).replace("\\", "/"))
    referenced = resolve_project_reference(base, PurePosixPath(relative).parent, audio_name)
    if referenced.is_file():
        return referenced.relative_to(base).as_posix()
    # A separately uploaded audio file is stored at the root of the import.
    uploaded = resolve_project_reference(base, PurePosixPath(), name.name)
    if uploaded.is_file():
        return uploaded.relative_to(base).as_posix()
    return None


def _new_chart(project, state, relative):
    from .structure import reference_marks

    base = project / "files"
    path = safe_path(base, relative)
    text = path.read_text(encoding="utf-8-sig")
    parsed = parse_osu(text)
    audio = _chart_audio(base, relative, parsed["general"].get("AudioFilename"))
    target_keys = min(10, max(6, parsed["keys"]))
    chart = dict(
        mode="remap", original=relative, audio=audio,
        title=parsed["metadata"].get("Title", path.stem),
        artist=parsed["metadata"].get("Artist", "Unknown"), input_keys=parsed["keys"],
        input_notes=[note.data() for note in parsed["notes"]], input_timing=parsed["timing"],
        reference_marks=reference_marks(text), variants=[], active=-1,
        settings=state["settings"] | {
            "keys": target_keys,
            "key_layout": DEFAULT_KEY_LAYOUTS[target_keys],
        },
    )
    if "reference_style" in state:
        chart["reference_style"] = state["reference_style"]
    if audio and audio == state.get("audio"):
        for field in ("provided_stems", "audio_shift_ms"):
            if field in state:
                chart[field] = state[field]
    return chart


def select_chart(project, state, relative, *, initial=False):
    source = relative or None
    saved = chart_workspace_snapshot_path(project, source)
    if saved.is_file():
        chart = json.loads(saved.read_text(encoding="utf-8"))
        if chart.get("original") != source:
            raise ValueError("保存的谱面与所选难度不符")
    elif source:
        chart = _new_chart(project, state, source)
    else:
        raise ValueError("没有已保存的原始音频项目")
    if not initial:
        archive_active_chart(project, state)
    shared = {key: state[key] for key in _PROJECT_FIELDS if key in state}
    shared.setdefault("cache_origin_audio", chart["audio"])
    active = shared | chart
    audio_work(project, active).mkdir(parents=True, exist_ok=True)
    return active
