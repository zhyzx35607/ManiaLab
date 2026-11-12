"""Build a local, provenance-checked listening pack for train-only chart actions."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manialab.domain import parse_osu
from manialab.sequence_prior import sequence_beat_positions
from scripts.research_expression_actions import TARGET_NAMES, extract_chart_windows
from scripts.separate_expression_audio_cohort import (
    INPUT_PROCESSING,
    MODEL_SHA256,
    SOURCES,
    canonical_hash,
    load_training_inputs,
    read_json,
    record_identity,
    sha256_file,
    verify_completed,
)
SAMPLE_RATE = 22050
INPUT_RATE = 44100
WINDOW_BEATS = 8.0
PADDING_MS = 1500
TARGET_PEAK = 0.95
DEFAULT_COHORT = ROOT / "test-data" / "expression-audio-cohort-v3"
DEFAULT_SEPARATED = ROOT / "test-data" / "expression-audio-cohort-v3-separated-train-v1"
DEFAULT_OUTPUT = ROOT / "test-data" / "expression-audio-cohort-v3-stem-review-v1"
PROTOCOL_PATH = ROOT / "docs" / "expression-audio-stem-review-protocol.md"
TRACKS = ("mix", *SOURCES)
TRACK_LABELS = {
    "mix": "原混音",
    "drums": "Demucs drums",
    "bass": "Demucs bass",
    "other": "Demucs other",
    "vocals": "Demucs vocals",
}
ACTION_LABELS = {
    "ln_fraction": "含LN",
    "ln_long_fraction_2beat": "含2拍以上LN",
    "ln_chord_fraction": "含LN和弦",
    "staggered_release_fraction": "含错开放手",
    "hybrid_tap_fraction": "含持LN期间tap",
    "inverse_ln_fraction": "含近续LN起音",
    "release_only_fraction": "含独立LN尾",
}
ACTION_TARGETS = tuple(ACTION_LABELS)


def _write_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def pcm16(samples: np.ndarray, gain: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError("Review clip contains non-finite samples")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Playback gain must be finite and positive")
    scaled = samples * gain
    if np.max(np.abs(scaled), initial=0.0) > 1.0 + 1e-12:
        raise ValueError("Playback gain would clip a review clip")
    return np.rint(np.clip(scaled, -1.0, 1.0) * 32767).astype(np.int16)


def clip_sample_bounds(
    start_ms: float,
    end_ms: float,
    duration_ms: float,
    sample_rate: int = SAMPLE_RATE,
    padding_ms: int = PADDING_MS,
) -> tuple[int, int]:
    values = (start_ms, end_ms, duration_ms)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Chart and audio times must be finite")
    if (
        start_ms < 0 or end_ms <= start_ms or duration_ms <= 0
        or sample_rate <= 0 or padding_ms < 0
    ):
        raise ValueError("Review interval, duration, sample rate, or padding is invalid")
    if start_ms >= duration_ms:
        raise ValueError("Chart window starts beyond the source audio")
    left_ms = max(0.0, start_ms - padding_ms)
    right_ms = min(duration_ms, end_ms + padding_ms)
    left = max(0, round(left_ms * sample_rate / 1000))
    right = min(round(duration_ms * sample_rate / 1000), round(right_ms * sample_rate / 1000))
    if right <= left:
        raise ValueError("Chart window produced an empty audio clip")
    return left, right


def _chart_windows(parsed: dict, duration_frames: int) -> list[dict]:
    _, targets = extract_chart_windows(parsed, window_beats=WINDOW_BEATS)
    notes = sorted(parsed["notes"], key=lambda note: (note.start, note.lane, note.end))
    times = sorted({time for note in notes for time in (note.start, note.end)})
    positions = sequence_beat_positions(times, parsed.get("timing", ()), fallback=500.0)
    beat_by_time = dict(zip(times, positions.tolist()))
    origin = min(beat_by_time[note.start] for note in notes)
    by_window: dict[int, list] = {}
    for note in notes:
        index = int(math.floor(
            (beat_by_time[note.start] - origin) / WINDOW_BEATS + 1e-9
        ))
        by_window.setdefault(index, []).append(note)
    indices = sorted(by_window)
    if len(indices) != len(targets):
        raise ValueError("Chart action windows do not match the parsed chart")

    duration_ms = duration_frames / SAMPLE_RATE * 1000
    result = []
    for index, window_targets in zip(indices, targets):
        events = by_window[index]
        chart_start = min(note.start for note in events)
        chart_end = max(note.end for note in events)
        if chart_start >= duration_ms:
            continue
        clip_start, clip_stop = clip_sample_bounds(
            chart_start, max(chart_end, chart_start + 1), duration_ms
        )
        target_values = {
            name: float(value)
            for name, value in zip(TARGET_NAMES, window_targets)
            if math.isfinite(float(value))
        }
        action_tags = [
            label for name, label in ACTION_LABELS.items()
            if target_values.get(name, 0.0) > 0
        ]
        note_rows = []
        for note in sorted(events, key=lambda item: (item.start, item.lane, item.end)):
            length_beats = (
                beat_by_time[note.end] - beat_by_time[note.start]
                if note.hold else 0.0
            )
            note_rows.append({
                "lane": note.lane + 1,
                "start_ms": int(note.start),
                "end_ms": int(note.end),
                "kind": "LN" if note.hold else "tap",
                "length_beats": round(float(length_beats), 4) if note.hold else None,
            })
        result.append({
            "window_index": index,
            "chart_start_ms": int(chart_start),
            "chart_end_ms": int(chart_end),
            "chart_end_beyond_audio": chart_end > duration_ms,
            "clip_start_sample": clip_start,
            "clip_stop_sample": clip_stop,
            "clip_start_ms": round(clip_start / SAMPLE_RATE * 1000, 3),
            "clip_end_ms": round(clip_stop / SAMPLE_RATE * 1000, 3),
            "event_count": len(events),
            "ln_count": sum(note.hold for note in events),
            "action_tags": action_tags,
            "action_targets": {
                name: round(target_values[name], 6)
                for name in TARGET_NAMES if name in target_values
            },
            "notes": note_rows,
            "_richness": len(action_tags),
        })
    return result


def _select_case_window(windows: list[dict]) -> dict:
    if not windows:
        raise ValueError("Training chart has no window that overlaps its audio")
    return max(
        windows,
        key=lambda row: (
            row["_richness"],
            row["ln_count"],
            row["event_count"],
            -row["window_index"],
        ),
    )


def _decode_mix(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    expected_input_frames: int,
    expected_output_frames: int,
) -> np.ndarray:
    command = [
        str(ffmpeg), "-v", "error", "-nostdin", "-y", "-i", str(source),
        "-vn", "-ac", "2", "-ar", str(INPUT_RATE), "-c:a", "pcm_f32le",
        str(destination),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise ValueError(f"FFmpeg could not decode training audio {source.name}")
    try:
        rate, stereo = wavfile.read(destination)
        if (
            rate != INPUT_RATE or stereo.dtype != np.float32
            or stereo.ndim != 2 or stereo.shape[1] != 2
            or len(stereo) != expected_input_frames
            or not np.isfinite(stereo).all()
        ):
            raise ValueError("Decoded audio does not match the frozen stereo PCM input")
        mono = np.mean(stereo, axis=1, dtype=np.float32)
        downsampled = resample_poly(mono, 1, 2).astype(np.float32, copy=False)
        if len(downsampled) != expected_output_frames:
            raise ValueError("Decoded mix frame count does not match the separated stems")
        return downsampled
    finally:
        Path(destination).unlink(missing_ok=True)


def _verify_run(separated_root: Path, rows: list[dict],
                selection_sha256: str) -> tuple[str, list[dict]]:
    run_identity = read_json(separated_root / "run.json")
    summary = read_json(separated_root / "summary.json")
    expected_ids = [row["beatmap_id"] for row in rows]
    if (
        run_identity.get("schema") != "manialab-expression-audio-stems-v1"
        or run_identity.get("split") != "train"
        or run_identity.get("selection_sha256") != selection_sha256
        or run_identity.get("model_sha256") != MODEL_SHA256
        or run_identity.get("sources") != list(SOURCES)
        or run_identity.get("profile") != "four"
        or run_identity.get("input_processing") != INPUT_PROCESSING
        or run_identity.get("threads") != 2
        or run_identity.get("interop_threads") != 1
        or run_identity.get("records") != [record_identity(row) for row in rows]
    ):
        raise ValueError("Separated run does not match the frozen training cohort")
    run_sha256 = canonical_hash(run_identity)
    if (
        summary.get("run_identity_sha256") != run_sha256
        or summary.get("expected_training_groups") != len(rows)
        or summary.get("completed_training_groups") != len(rows)
        or summary.get("completed_beatmap_ids") != expected_ids
    ):
        raise ValueError("Separated summary does not match the frozen training cohort")

    completed = [
        verify_completed(
            separated_root / str(row["beatmap_id"]), row, MODEL_SHA256, run_sha256
        )
        for row in rows
    ]
    numeric_dirs = {
        int(path.name) for path in separated_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    }
    if numeric_dirs != set(expected_ids):
        raise ValueError("Separated output contains unexpected or missing chart IDs")
    return run_sha256, completed


def _case_rows(cohort_root: Path, rows: list[dict], completed: list[dict]) -> list[dict]:
    result = []
    for row, complete in zip(rows, completed):
        chart_path = cohort_root / str(row["beatmap_id"]) / "reference.osu"
        parsed = parse_osu(chart_path.read_text(encoding="utf-8-sig"))
        chart_audio = str(parsed["general"].get("AudioFilename", ""))
        chart_extension = PurePosixPath(chart_audio.replace("\\", "/")).suffix.casefold()
        stored_extension = PurePosixPath(
            str(row["audio_filename"]).replace("\\", "/")
        ).suffix.casefold()
        if (
            parsed["keys"] != row["keys"]
            or not chart_extension
            or chart_extension != stored_extension
        ):
            raise ValueError(f"Training chart metadata mismatch for {row['beatmap_id']}")
        validation = complete["validation"]
        windows = _chart_windows(parsed, validation["frames"])
        selected = dict(_select_case_window(windows))
        selected.pop("_richness")
        selected.update({
            "case_id": f"{row['keys']}k-{row['ln_class']}-{row['beatmap_id']}",
            "beatmap_id": row["beatmap_id"],
            "group": row["group"],
            "keys": row["keys"],
            "ln_class": row["ln_class"],
            "audio_filename": row["audio_filename"],
            "audio_sha256": row["audio_sha256"],
            "chart_sha256": row["chart_sha256"],
            "model_sha256": complete["model_sha256"],
            "duration_frames": validation["frames"],
            "sample_rate": validation["sample_rate"],
        })
        result.append(selected)
    return sorted(result, key=lambda item: (
        item["keys"], item["ln_class"], item["beatmap_id"]
    ))


def _render_html(cases: list[dict], review_key: str) -> str:
    body_rows = []
    for index, case in enumerate(cases, start=1):
        case_id = html.escape(case["case_id"], quote=True)
        tags = "".join(
            f'<span class="tag">{html.escape(tag)}</span>'
            for tag in case["action_tags"]
        ) or '<span class="muted">此窗没有LN动作标签</span>'
        note_rows = "".join(
            "<li>"
            f'{note["start_ms"]} ms · lane {note["lane"]} · '
            f'{html.escape(note["kind"])}'
            + (
                f' · {note["end_ms"]} ms end · {note["length_beats"]:.3f} beat'
                if note["kind"] == "LN" else ""
            )
            + "</li>"
            for note in case["notes"]
        )
        buttons = "".join(
            f'<button type="button" data-src="{html.escape(case["clips"][track], quote=True)}" '
            f'data-kind="{html.escape(TRACK_LABELS[track], quote=True)}">'
            f'{html.escape(TRACK_LABELS[track])}</button>'
            for track in TRACKS
        )
        tail_status = (
            '<span class="warning">谱面尾端晚于音频末尾</span>'
            if case["chart_end_beyond_audio"] else ""
        )
        body_rows.append(f"""
<tr data-id="{case_id}" data-key="{case['keys']}" data-class="{html.escape(case['ln_class'], quote=True)}"
    data-tags="{html.escape(' '.join(case['action_tags']), quote=True)}">
  <td class="number">{index:02d}</td>
  <td><strong>{case['beatmap_id']} · {case['keys']}K</strong>
    <small>{html.escape(case['ln_class'])} · 窗口 {case['window_index']}</small>
    <small>谱面时刻 {case['chart_start_ms']}–{case['chart_end_ms']} ms</small>{tail_status}</td>
  <td class="actions">{tags}<details><summary>{case['event_count']} 个物件 · {case['ln_count']} 条LN</summary>
    <ul>{note_rows}</ul></details></td>
  <td class="play">{buttons}</td>
  <td><select data-field="attack" aria-label="{case_id} 起音听感">
    <option value="">未判断</option><option value="clear">清楚可听</option>
    <option value="possible">可能可听</option><option value="not_distinct">没有独立起音</option>
    <option value="unclear">无法判断</option></select></td>
  <td class="tracks">{''.join(
      f'<label><input type="checkbox" data-field="track_evidence" value="{track}">{html.escape(TRACK_LABELS[track])}</label>'
      for track in SOURCES
  )}<label><input type="checkbox" data-field="track_evidence" value="unclear">无法判断</label></td>
  <td><select data-field="chart_timing" aria-label="{case_id} 谱面时间关系">
    <option value="">未判断</option><option value="rough">大致对齐</option>
    <option value="early">声音明显早于谱面</option><option value="late">声音明显晚于谱面</option>
    <option value="no_attack">无清楚起音</option><option value="unclear">无法判断</option></select>
    <select data-field="tail_evidence" aria-label="{case_id} LN延续听感">
    <option value="">未判断</option><option value="past_chart_end">可听到延续越过谱面尾端</option>
    <option value="before_chart_end">声音在谱面尾端前已衰减或消失</option>
    <option value="unclear">无法判断</option><option value="not_applicable">此窗无LN</option></select></td>
  <td><textarea data-field="note" rows="2" aria-label="{case_id} 备注"></textarea></td>
</tr>""")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ManiaLab · 训练集声学证据复核</title>
<style>
:root{{font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif;color:#20272b;background:#f5f7f6}}
*{{box-sizing:border-box}} body{{margin:0;padding:20px;max-width:1800px;margin-inline:auto}}
h1{{font-size:22px;margin:0 0 10px}} .scope{{max-width:120ch;color:#39494a;margin:0 0 14px}}
.toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 0;border-block:1px solid #c9d4d0}}
.toolbar label{{display:flex;gap:6px;align-items:center}} select,input,textarea,button{{font:inherit}}
.toolbar select,.toolbar input{{padding:5px 7px}} button{{padding:5px 8px;border:1px solid #9dafaa;border-radius:4px;background:#fff;color:#233532;cursor:pointer}}
button:hover{{background:#e8f0ed}} #summary{{margin-left:auto;color:#3e5650}}
.player{{position:sticky;top:0;z-index:2;background:#f5f7f6;border-bottom:1px solid #c9d4d0;padding:8px 0}}
.player audio{{width:min(100%,700px);vertical-align:middle}} #now-playing{{margin-right:12px;font-weight:600}}
.table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;min-width:1500px;table-layout:fixed;background:#fff}}
th,td{{padding:8px;border-bottom:1px solid #dde4e1;text-align:left;vertical-align:top}}
th{{position:sticky;top:58px;background:#e8eeeb;z-index:1;font-weight:600}}
td small{{display:block;color:#5e706d;font-size:12px;margin-top:2px}} .number{{color:#596b67;width:46px}}
td:nth-child(2){{width:180px}} td:nth-child(3){{width:260px}} td:nth-child(4){{width:260px}}
td:nth-child(5){{width:130px}} td:nth-child(6){{width:170px}} td:nth-child(7){{width:240px}} td:nth-child(8){{width:210px}}
td select,td textarea{{width:100%;margin:2px 0}} td textarea{{resize:vertical;min-height:46px}}
.play{{display:flex;gap:5px;flex-wrap:wrap}} .tag{{display:inline-block;margin:0 4px 4px 0;padding:2px 6px;border-radius:3px;background:#e5eee7;color:#27483b;font-size:12px}}
.muted{{color:#6a7774;font-size:12px}} details{{margin-top:5px}} details ul{{margin:5px 0;padding-left:18px;max-height:160px;overflow:auto;font-size:12px}}
.tracks label{{display:inline-flex;gap:4px;align-items:center;margin:0 6px 5px 0;white-space:nowrap}}
.warning{{display:block;color:#9b4d2b;font-size:12px}} tr[hidden]{{display:none}}
@media(max-width:700px){{body{{padding:12px}} .toolbar{{align-items:flex-start}} #summary{{margin-left:0}}}}
</style>
</head>
<body>
<h1>训练集声学证据复核</h1>
<p class="scope">每首训练谱选一个作者动作较丰富的8拍窗口；这是有目的的诊断样本，不是随机评测。
谱面时间和LN端点只是作者编排选择，不是声学真值。原混音是主要听觉依据；Demucs四轨是弱分离证据，
轨道名不代表已确认的乐器身份。听到的衰减也不等于物理按键释放。</p>
<div class="toolbar">
  <label>键数 <select id="key-filter"><option value="all">全部</option>{''.join(f'<option value="{key}">{key}K</option>' for key in range(6, 11))}</select></label>
  <label>谱面类别 <select id="class-filter"><option value="all">全部</option>
    <option value="ln_majority">ln_majority</option><option value="hybrid">hybrid</option>
    <option value="tap_majority">tap_majority</option></select></label>
  <label>审核状态 <select id="status-filter"><option value="all">全部</option>
    <option value="unreviewed">待复核</option><option value="reviewed">已填写</option></select></label>
  <label>动作筛选 <input id="search" type="search" placeholder="LN和弦、错开放手等"></label>
  <button type="button" id="export">导出审核 JSON</button><span id="summary"></span>
</div>
<div class="player"><span id="now-playing">尚未选择音频</span><audio controls preload="none"></audio></div>
<div class="table-wrap"><table>
<thead><tr><th>#</th><th>训练样本 / 谱面时间</th><th>作者谱动作</th><th>试听轨道</th><th>起音听感</th>
<th>提供可听证据的分轨</th><th>时间 / LN延续</th><th>备注</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table></div>
<script>
const STORE_KEY = "ManiaLab-expression-stem-review-v1:{review_key}";
const rows = [...document.querySelectorAll("tbody tr[data-id]")];
const audio = document.querySelector("audio");
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(STORE_KEY) || "{{}}"); }} catch (_) {{}}
function saveRow(row) {{
  const entry = {{}};
  for (const el of row.querySelectorAll("[data-field]")) {{
    if (el.type === "checkbox") continue;
    entry[el.dataset.field] = el.value;
  }}
  entry.track_evidence = [...row.querySelectorAll('[data-field="track_evidence"]:checked')]
    .map(el => el.value);
  state[row.dataset.id] = entry;
  try {{ localStorage.setItem(STORE_KEY, JSON.stringify(state)); }} catch (_) {{}}
  updateView();
}}
for (const row of rows) {{
  const entry = state[row.dataset.id] || {{}};
  for (const input of row.querySelectorAll("[data-field]")) {{
    if (input.type === "checkbox") {{
      input.checked = (entry.track_evidence || []).includes(input.value);
    }} else {{
      input.value = entry[input.dataset.field] || "";
    }}
    input.addEventListener(input.tagName === "TEXTAREA" ? "input" : "change", () => saveRow(row));
  }}
  row.querySelectorAll("button[data-src]").forEach(button => button.addEventListener("click", () => {{
    document.querySelector("#now-playing").textContent =
      row.dataset.id + " · " + button.dataset.kind;
    audio.pause();
    audio.src = button.dataset.src;
    audio.load();
    audio.play().catch(() => {{}});
  }}));
}}
function isReviewed(entry) {{
  return !!(entry.attack || entry.chart_timing || entry.tail_evidence ||
    (entry.track_evidence || []).length || entry.note);
}}
function updateView() {{
  const key = document.querySelector("#key-filter").value;
  const chartClass = document.querySelector("#class-filter").value;
  const status = document.querySelector("#status-filter").value;
  const query = document.querySelector("#search").value.trim().toLowerCase();
  let visible = 0, reviewed = 0;
  for (const row of rows) {{
    const entry = state[row.dataset.id] || {{}};
    const hasReview = isReviewed(entry);
    const match = (key === "all" || row.dataset.key === key)
      && (chartClass === "all" || row.dataset.class === chartClass)
      && (status === "all" || (status === "reviewed" ? hasReview : !hasReview))
      && (!query || (row.dataset.tags + " " + row.dataset.id).toLowerCase().includes(query));
    row.hidden = !match;
    if (match) visible++;
    if (hasReview) reviewed++;
  }}
  document.querySelector("#summary").textContent =
    `已填写 ${{reviewed}} / ${{rows.length}} · 当前显示 ${{visible}}`;
}}
for (const id of ["key-filter", "class-filter", "status-filter", "search"]) {{
  document.querySelector("#" + id).addEventListener("input", updateView);
  document.querySelector("#" + id).addEventListener("change", updateView);
}}
document.querySelector("#export").addEventListener("click", () => {{
  const review = rows.map(row => {{
    const entry = state[row.dataset.id] || {{}};
    return {{
      case_id: row.dataset.id,
      attack: entry.attack || "",
      track_evidence: entry.track_evidence || [],
      chart_timing: entry.chart_timing || "",
      tail_evidence: entry.tail_evidence || "",
      note: entry.note || ""
    }};
  }});
  const blob = new Blob([JSON.stringify({{
    schema: "manialab-expression-stem-review-v1",
    review_key: "{review_key}",
    exported_at: new Date().toISOString(),
    review
  }}, null, 2)], {{type: "application/json"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "expression-audio-stem-review-v1-labels.json";
  link.click();
  URL.revokeObjectURL(link.href);
}});
updateView();
</script>
</body></html>"""


def build_review_bundle(
    cohort_root: Path = DEFAULT_COHORT,
    separated_root: Path = DEFAULT_SEPARATED,
    output: Path = DEFAULT_OUTPUT,
    ffmpeg: Path | None = None,
    root: Path = ROOT,
) -> dict:
    root = Path(root).resolve()
    cohort_root = Path(cohort_root).resolve()
    separated_root = Path(separated_root).resolve()
    output = Path(output).resolve()
    test_data = (root / "test-data").resolve()
    if output == test_data or not output.is_relative_to(test_data):
        raise ValueError("Review output must be a new ignored directory under test-data")
    if output.exists():
        raise FileExistsError(f"Preserve previous review artifacts: {output}")
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"Review protocol is missing: {PROTOCOL_PATH}")

    rows, selection_sha256, fetch_report_sha256 = load_training_inputs(cohort_root)
    run_sha256, completed = _verify_run(separated_root, rows, selection_sha256)
    cases = _case_rows(cohort_root, rows, completed)
    if len(cases) != len(rows):
        raise ValueError("Expected one review case for every frozen training group")
    ffmpeg = Path(ffmpeg).resolve() if ffmpeg else None
    if ffmpeg is None:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = Path(get_ffmpeg_exe()).resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"FFmpeg binary is missing: {ffmpeg}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        clips_dir = staging / "clips"
        clips_dir.mkdir()
        row_by_id = {row["beatmap_id"]: row for row in rows}
        complete_by_id = {
            item["input"]["beatmap_id"]: item for item in completed
        }
        arrays_by_case: dict[str, dict[str, np.ndarray]] = {}
        peak = 0.0
        for index, case in enumerate(cases, start=1):
            row = row_by_id[case["beatmap_id"]]
            complete = complete_by_id[case["beatmap_id"]]
            expected_frames = case["duration_frames"]
            mix = _decode_mix(
                ffmpeg,
                row["audio_path"],
                staging / f"decode-{case['beatmap_id']}.wav",
                complete["input_frames_44100"],
                expected_frames,
            )
            left, right = case["clip_start_sample"], case["clip_stop_sample"]
            selected_arrays = {"mix": np.asarray(mix[left:right], dtype=np.float32).copy()}
            for source in SOURCES:
                stem_path = separated_root / str(case["beatmap_id"]) / f"{source}.npy"
                stem = np.load(stem_path, mmap_mode="r", allow_pickle=False)
                if (
                    stem.dtype != np.float32 or stem.shape != (expected_frames,)
                    or not np.isfinite(stem[left:right]).all()
                ):
                    raise ValueError(
                        f"Stem slice failed validation for {case['beatmap_id']} {source}"
                    )
                selected_arrays[source] = np.asarray(stem[left:right]).copy()
            if any(not len(values) for values in selected_arrays.values()):
                raise ValueError(f"Empty review clip for {case['beatmap_id']}")
            peak = max(
                peak,
                *(float(np.max(np.abs(values), initial=0.0))
                  for values in selected_arrays.values()),
            )
            case["clips"] = {}
            for source in TRACKS:
                case["clips"][source] = (
                    f"clips/{index:02d}-{case['beatmap_id']}-{source}.wav"
                )
            arrays_by_case[case["case_id"]] = selected_arrays

        gain = min(1.0, TARGET_PEAK / peak) if peak else 1.0
        clip_count = 0
        for case in cases:
            selected_arrays = arrays_by_case[case["case_id"]]
            for source in TRACKS:
                relative = case["clips"][source]
                path = staging / relative
                expected = pcm16(selected_arrays[source], gain)
                wavfile.write(path, SAMPLE_RATE, expected)
                check_rate, restored = wavfile.read(path)
                if (
                    check_rate != SAMPLE_RATE or restored.dtype != np.int16
                    or restored.shape != expected.shape
                    or not np.array_equal(restored, expected)
                ):
                    raise ValueError(f"Written review clip failed verification: {relative}")
                clip_count += 1
            del selected_arrays

        script_path = Path(__file__).resolve()
        script_sha256 = sha256_file(script_path)
        protocol_sha256 = sha256_file(PROTOCOL_PATH)
        review_key = hashlib.sha256(
            f"{selection_sha256}:{run_sha256}:{script_sha256}:{protocol_sha256}".encode()
        ).hexdigest()[:24]
        manifest = {
            "schema": "manialab-expression-stem-review-manifest-v1",
            "purpose": (
                "Human listening aid for one action-rich chart window per frozen "
                "training group; not a random evaluation sample."
            ),
            "review_key": review_key,
            "cohort": {
                "selection_sha256": selection_sha256,
                "fetch_report_sha256": fetch_report_sha256,
                "split": "train",
                "group_count": len(rows),
                "case_count": len(cases),
                "groups": [record_identity(row) for row in rows],
            },
            "separation": {
                "run_identity_sha256": run_sha256,
                "model_sha256": MODEL_SHA256,
                "sources": list(SOURCES),
                "directory": str(separated_root),
            },
            "selection": {
                "window_beats": WINDOW_BEATS,
                "policy": (
                    "One non-empty 8-beat start-event window per training chart, "
                    "maximizing the count of distinct charted LN/action targets; "
                    "ties favor more LNs, more events, then the earlier window."
                ),
                "targets_are_chart_actions_not_acoustic_labels": True,
                "charts_read": "train only",
                "cases": cases,
            },
            "playback": {
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "format": "PCM16",
                "track_order": list(TRACKS),
                "linear_gain": gain,
                "gain_is_global_across_all_cases_and_tracks": True,
                "per_clip_normalization": False,
                "source_mix_decode": {
                    "sample_rate": INPUT_RATE,
                    "channels": 2,
                    "codec": "pcm_f32le",
                    "downmix": "arithmetic mean of stereo channels",
                    "resample": "scipy.signal.resample_poly(up=1, down=2)",
                },
                "padding_ms_each_side": PADDING_MS,
            },
            "review": {
                "human_listening_completed": False,
                "musical_acceptance": False,
                "instrument_identity_established": False,
                "ln_tail_ground_truth_established": False,
            },
            "builder_sha256": script_sha256,
            "protocol_sha256": protocol_sha256,
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "index.html").write_text(
            _render_html(cases, review_key), encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# 表达研究训练集声学证据复核\n\n"
            "打开 `index.html`，以原混音为主试听每个训练样本，并把Demucs输出当作辅助证据。"
            "页面审核状态可暂存于当前浏览器；请导出JSON保存人工判断。\n\n"
            "每首谱只取一个作者动作较丰富的8拍窗口，因此该包是有目的的诊断样本，"
            "不能用于估算准确率或代表完整歌曲。谱面时刻和LN端点不是声学真值；"
            "分轨名称不证明真实乐器身份，听觉衰减也不等于物理按键释放。\n\n"
            "输入、选择策略、时钟、共享增益、音频切片和产物SHA见`manifest.json`与`report.json`。"
            "本包只含train组的短PCM16片段，不包含完整音频、validation/test样本或模型权重。\n",
            encoding="utf-8",
        )
        shutil.copyfile(script_path, staging / "builder-snapshot.py")
        shutil.copyfile(PROTOCOL_PATH, staging / "protocol-snapshot.md")
        artifact_hashes = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        report = {
            "schema": "manialab-expression-stem-review-report-v1",
            "selection_sha256": selection_sha256,
            "run_identity_sha256": run_sha256,
            "case_count": len(cases),
            "source_track_count": len(TRACKS),
            "clip_count": clip_count,
            "clip_audio_seconds": round(sum(
                (case["clip_stop_sample"] - case["clip_start_sample"]) / SAMPLE_RATE
                for case in cases
            ), 3),
            "global_gain": gain,
            "source_mix_decode_count": len(cases),
            "validation_test_audio_read": False,
            "model_inference_rerun": False,
            "human_listening_completed": False,
            "musical_acceptance": False,
            "artifact_sha256": artifact_hashes,
        }
        _write_json(staging / "report.json", report)
        if output.exists():
            raise FileExistsError(f"Preserve previous review artifacts: {output}")
        staging.rename(output)
        report["report_sha256"] = sha256_file(output / "report.json")
        report["output"] = str(output)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--separated", type=Path, default=DEFAULT_SEPARATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    result = build_review_bundle(
        cohort_root=args.cohort,
        separated_root=args.separated,
        output=args.output,
        ffmpeg=args.ffmpeg,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
