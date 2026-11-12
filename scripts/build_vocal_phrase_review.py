"""Build a local, provenance-checked listening pack for frozen vocal phrases."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 22050
PADDING_MS = 1500
TARGET_PEAK = 0.95
DEFAULT_OUTPUT = ROOT / "test-data" / "sakura-benchmark" / "vocal-phrase-audio-review-v1"
SOURCE_ROOT = ROOT / "test-data" / "sakura-benchmark"
ANALYSIS_PATH = SOURCE_ROOT / "piano-full-pipeline-v1" / "general-analysis.json"
IDENTITY_PATH = SOURCE_ROOT / "piano-full-pipeline-v1" / "identity.json"
AUDIT_PATH = SOURCE_ROOT / "raw-pcm-audit-v1" / "report.json"
VOCALS_PATH = SOURCE_ROOT / "raw-pcm-audit-v1" / "vocals.wav"
MIX_PATH = SOURCE_ROOT / "raw-pcm-audit-v1" / "mix.wav"
PROVENANCE_PATH = ROOT / "test-data" / "vocal-phrase-release-review-v1" / "report.json"
PROTOCOL_PATH = ROOT / "docs" / "vocal-phrase-audio-review-protocol.md"
EXPECTED_PROVENANCE_SHA256 = "bf1db1bf07eb83a72b1d4916f0143f991409fd269f96e6cd40369e67b315e3ac"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_bounds_ms(start_ms: float, end_ms: float, duration_ms: float,
                   padding_ms: int = PADDING_MS) -> tuple[float, float]:
    values = (start_ms, end_ms, duration_ms)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Phrase and audio times must be finite")
    if start_ms < 0 or end_ms <= start_ms or duration_ms <= 0 or padding_ms < 0:
        raise ValueError("Phrase interval, duration, or padding is invalid")
    if end_ms > duration_ms:
        raise ValueError("Phrase extends beyond the source audio")
    return max(0.0, start_ms - padding_ms), min(duration_ms, end_ms + padding_ms)


def pcm16(samples: np.ndarray, gain: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError("Audio clip contains non-finite samples")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Playback gain must be finite and positive")
    scaled = samples * gain
    if np.max(np.abs(scaled), initial=0.0) > 1.0 + 1e-12:
        raise ValueError("Playback gain would clip the audio")
    return np.rint(np.clip(scaled, -1.0, 1.0) * 32767).astype(np.int16)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _recorded_path(root: Path, recorded: str) -> Path:
    normalized = str(recorded).replace("\\", "/")
    path = Path(normalized)
    return path if path.is_absolute() else root / path


def _check_provenance_inputs(root: Path, provenance: dict) -> dict[str, str]:
    inputs = provenance["real"]["inputs"]
    checked = {}
    for recorded_path, expected_hash in inputs.items():
        path = _recorded_path(root, recorded_path)
        if not path.is_file():
            raise ValueError(f"Frozen provenance input is missing: {recorded_path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen provenance input changed: {recorded_path}")
        checked[recorded_path.replace("\\", "/")] = actual_hash
    return checked


def _validate_sources(root: Path, analysis: dict, identity: dict, audit: dict,
                      provenance: dict, analysis_path: Path, vocals_path: Path,
                      mix_path: Path, provenance_path: Path
                      ) -> tuple[np.ndarray, np.ndarray, int, float, dict]:
    checked_inputs = _check_provenance_inputs(root, provenance)
    real = provenance["real"]
    if not real.get("frozen_geometry_matched") or not real.get("phrase_geometry_unchanged"):
        raise ValueError("The frozen phrase geometry was not independently matched")
    if real.get("phrase_count") != len(analysis.get("phrase_candidates", [])):
        raise ValueError("Phrase count does not match the frozen provenance")
    analysis_key = analysis_path.relative_to(root).as_posix()
    if checked_inputs.get(analysis_key) != sha256_file(analysis_path):
        raise ValueError("Analysis JSON is not part of the frozen input set")

    audit_rows = {row["role"]: row for row in audit["rows"]}
    vocal_row, mix_row = audit_rows["vocals"], audit_rows["mix"]
    separation = analysis["separation"]
    if identity["sources"]["vocals"] != vocal_row["original_sha256"]:
        raise ValueError("Nova vocal source identity does not match the raw PCM audit")
    if identity["mix_sha256"] != mix_row["original_sha256"]:
        raise ValueError("Mix source identity does not match the raw PCM audit")
    if separation["audio_sha256"] != identity["mix_sha256"]:
        raise ValueError("Phrase analysis uses a different mix clock")
    if sha256_file(vocals_path) != vocal_row["raw_float_sha256"]:
        raise ValueError("Vocal PCM does not match the audited raw float source")
    if sha256_file(mix_path) != mix_row["raw_float_sha256"]:
        raise ValueError("Mix PCM does not match the audited raw float source")

    rate_vocals, vocals = wavfile.read(vocals_path, mmap=True)
    rate_mix, mix = wavfile.read(mix_path, mmap=True)
    if (
        rate_vocals != SAMPLE_RATE or rate_mix != SAMPLE_RATE
        or separation["sample_rate"] != SAMPLE_RATE
        or vocals.dtype != np.float32 or mix.dtype != np.float32
        or vocals.ndim != 1 or mix.ndim != 1
        or len(vocals) != len(mix)
        or len(vocals) != vocal_row["samples"]
        or len(mix) != mix_row["samples"]
        or not np.isfinite(vocals).all() or not np.isfinite(mix).all()
    ):
        raise ValueError("Expected complete, finite mono float32 PCM at 22050 Hz")

    duration_ms = len(vocals) / rate_vocals * 1000
    declared_duration_ms = float(analysis["duration"]) * 1000
    identity_duration_ms = float(separation["duration_seconds"]) * 1000
    if not math.isclose(duration_ms, declared_duration_ms, abs_tol=0.05):
        raise ValueError("Analysis duration does not match the source PCM")
    if not math.isclose(duration_ms, identity_duration_ms, abs_tol=0.05):
        raise ValueError("Separation identity duration does not match the source PCM")

    max_peak = max(float(np.max(np.abs(vocals))), float(np.max(np.abs(mix))))
    gain = min(1.0, TARGET_PEAK / max_peak) if max_peak else 1.0
    identity_report = {
        "analysis_sha256": sha256_file(analysis_path),
        "vocal_pcm_sha256": sha256_file(vocals_path),
        "mix_pcm_sha256": sha256_file(mix_path),
        "vocal_original_source_sha256": vocal_row["original_sha256"],
        "mix_original_source_sha256": mix_row["original_sha256"],
        "sample_rate": rate_vocals,
        "samples": len(vocals),
        "duration_seconds": len(vocals) / rate_vocals,
        "vocal_peak": float(np.max(np.abs(vocals))),
        "mix_peak": float(np.max(np.abs(mix))),
        "global_playback_gain": gain,
        "phrase_provenance_sha256": sha256_file(provenance_path),
        "checked_provenance_inputs": checked_inputs,
    }
    return vocals, mix, rate_vocals, gain, identity_report


def _segment_for(start_ms: float, segments: list[dict]) -> dict | None:
    return next(
        (segment for segment in segments
         if segment["start_ms"] <= start_ms < segment["end_ms"]),
        None,
    )


def _render_html(rows: list[dict], sections: list[dict], review_key: str) -> str:
    section_options = ['<option value="all">全部段落</option>']
    for section in sections:
        index = section["index"]
        section_options.append(
            f'<option value="{index}">{html.escape(section["label"])}</option>'
        )
    table_rows = []
    for row in rows:
        label = html.escape(row["section_label"])
        candidate_id = html.escape(row["id"], quote=True)
        vocal_clip = html.escape(row["clips"]["vocals"], quote=True)
        mix_clip = html.escape(row["clips"]["mix"], quote=True)
        time_text = f'{row["start_ms"] / 1000:.3f}–{row["end_ms"] / 1000:.3f} s'
        table_rows.append(f"""
<tr data-id="{candidate_id}" data-section="{row['section_index']}"
    data-search="{html.escape(candidate_id + ' ' + label, quote=True)}">
  <td>{row['index']:02d}</td>
  <td><strong>{html.escape(time_text)}</strong><small>{html.escape(label)} · {row['duration_ms']:.0f} ms</small></td>
  <td>{row['pitch']}<small>{row['support_count']} 支持音 · coverage {row['support_coverage']}</small></td>
  <td class="play">
    <button type="button" data-src="{vocal_clip}" data-kind="Nova 人声轨">播放人声</button>
    <button type="button" data-src="{mix_clip}" data-kind="原混音">播放混音</button>
  </td>
  <td>
    <select data-field="source" aria-label="候选 {row['index']} 声源判断">
      <option value="">未审核</option>
      <option value="clear_voice">明确人声</option>
      <option value="possible_voice">疑似人声</option>
      <option value="other_or_bleed">非人声或疑似串音</option>
      <option value="unclear">无法判断</option>
    </select>
  </td>
  <td>
    <select data-field="boundary" aria-label="候选 {row['index']} 边界判断">
      <option value="">未审核</option>
      <option value="reasonable">起止基本吻合</option>
      <option value="start_review">起点需复核</option>
      <option value="end_review">音尾需复核</option>
      <option value="both_review">两端需复核</option>
      <option value="unclear">无法判断</option>
    </select>
  </td>
  <td><textarea data-field="note" rows="1" aria-label="候选 {row['index']} 备注"></textarea></td>
</tr>""")
    section_rows = []
    current_section = None
    for section in sections:
        section_rows.append(
            f'<tr class="section-row" data-section-heading="{section["index"]}">'
            f'<th colspan="7">{html.escape(section["label"])}'
            f' <span>{section["start_ms"] / 1000:.3f}–{section["end_ms"] / 1000:.3f} s</span>'
            f'</th></tr>'
        )
        section_rows.extend(
            item for item, row in zip(table_rows, rows)
            if row["section_index"] == section["index"]
        )
    section_rows.extend(
        item for item, row in zip(table_rows, rows) if row["section_index"] is None
    )
    if any(row["section_index"] is None for row in rows):
        section_rows.insert(
            0, '<tr class="section-row"><th colspan="7">未映射段落</th></tr>'
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>樱花草 · 人声乐句候选复核</title>
<style>
:root{{font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif;color:#20252b;background:#fff}}
body{{max-width:1500px;margin:0 auto;padding:20px}}
h1{{font-size:22px;margin:0 0 8px}} .notice{{max-width:100ch;color:#4b5563}}
.toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px 0;border-block:1px solid #d7dce1}}
input,select,textarea,button{{font:inherit}} button{{padding:6px 9px;cursor:pointer}}
.toolbar input,.toolbar select{{padding:6px}} #summary{{margin-left:auto;color:#4b5563}}
.player{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #d7dce1;padding:8px 0}}
.player audio{{width:min(100%,700px);vertical-align:middle}} #now-playing{{margin-right:12px}}
table{{border-collapse:collapse;width:100%;table-layout:fixed}}
th,td{{padding:8px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}}
th{{font-weight:600}} th span,td small{{display:block;color:#68717c;font-size:12px}}
.section-row th{{background:#f2f4f5;padding-top:12px}}
td:nth-child(1){{width:44px}} td:nth-child(2){{width:185px}} td:nth-child(3){{width:125px}}
td:nth-child(4){{width:175px}} td:nth-child(5),td:nth-child(6){{width:160px}}
td:nth-child(7){{width:220px}} td select,td textarea{{width:100%;box-sizing:border-box}}
td textarea{{min-height:34px;resize:vertical}} .play{{display:flex;gap:5px;flex-wrap:wrap}}
@media(max-width:900px){{body{{padding:12px}} table{{min-width:1040px}} .table-wrap{{overflow-x:auto}} #summary{{margin-left:0}}}}
</style>
</head>
<body>
<h1>樱花草 · 97 条连续人声乐句候选</h1>
<p class="notice">这些是待核候选，不是人声真值。分别试听 Nova 人声轨与原混音；分离标签、音高和覆盖率都不能单独证明声源或边界正确。Reference 谱没有人声音符，不作为原曲无人声的证据。</p>
<div class="toolbar">
  <label>段落 <select id="section-filter">{''.join(section_options)}</select></label>
  <label>复核状态 <select id="status-filter"><option value="all">全部</option><option value="unreviewed">待复核</option><option value="reviewed">已标声源</option></select></label>
  <label>筛选 <input id="search" type="search" placeholder="候选 ID / 段落"></label>
  <button type="button" id="export">导出审核记录</button>
  <span id="summary"></span>
</div>
<div class="player"><span id="now-playing">选择候选回放</span><audio controls preload="none"></audio></div>
<div class="table-wrap"><table>
<thead><tr><th>#</th><th>原音时间</th><th>音高 / 支持</th><th>回放</th><th>声源判断</th><th>边界判断</th><th>备注</th></tr></thead>
<tbody>{''.join(section_rows)}</tbody>
</table></div>
<script>
const STORE_KEY = "ManiaLab-vocal-phrase-review-v1:{review_key}";
const rows = [...document.querySelectorAll("tbody tr[data-id]")];
const audio = document.querySelector("audio");
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(STORE_KEY) || "{{}}"); }} catch (_) {{}}
function saveRow(row) {{
  const entry = {{}};
  row.querySelectorAll("[data-field]").forEach(el => entry[el.dataset.field] = el.value);
  state[row.dataset.id] = entry;
  try {{ localStorage.setItem(STORE_KEY, JSON.stringify(state)); }} catch (_) {{}}
  updateView();
}}
for (const row of rows) {{
  for (const input of row.querySelectorAll("[data-field]")) {{
    input.value = (state[row.dataset.id] || {{}})[input.dataset.field] || "";
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
function updateView() {{
  const section = document.querySelector("#section-filter").value;
  const status = document.querySelector("#status-filter").value;
  const query = document.querySelector("#search").value.trim().toLowerCase();
  let shown = 0, marked = 0;
  for (const row of rows) {{
    const entry = state[row.dataset.id] || {{}};
    const sectionMatch = section === "all" || row.dataset.section === section;
    const statusMatch = status === "all" || (status === "unreviewed" ? !entry.source : !!entry.source);
    const searchMatch = !query || row.dataset.search.toLowerCase().includes(query);
    const visible = sectionMatch && statusMatch && searchMatch;
    row.hidden = !visible;
    if (visible) shown++;
    if (entry.source) marked++;
  }}
  document.querySelectorAll("tr[data-section-heading]").forEach(heading => {{
    const sectionMatch = section === "all" || heading.dataset.sectionHeading === section;
    const hasVisible = rows.some(row => row.dataset.section === heading.dataset.sectionHeading && !row.hidden);
    heading.hidden = !(sectionMatch && hasVisible);
  }});
  document.querySelector("#summary").textContent =
    `声源已标记 ${{marked}} / ${{rows.length}} · 当前显示 ${{shown}}`;
}}
for (const id of ["section-filter", "status-filter", "search"]) {{
  document.querySelector("#" + id).addEventListener("input", updateView);
  document.querySelector("#" + id).addEventListener("change", updateView);
}}
document.querySelector("#export").addEventListener("click", () => {{
  const review = rows.map(row => ({{
    id: row.dataset.id,
    ...((state[row.dataset.id]) || {{source:"", boundary:"", note:""}})
  }}));
  const blob = new Blob([JSON.stringify({{
    schema:"manialab-vocal-phrase-review-v1",
    review_key:"{review_key}",
    exported_at:new Date().toISOString(),
    review
  }}, null, 2)], {{type:"application/json"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "vocal-phrase-review-v1-labels.json";
  link.click();
  URL.revokeObjectURL(link.href);
}});
updateView();
</script>
</body></html>"""


def build_review_bundle(output: Path = DEFAULT_OUTPUT, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Preserve previous audio review artifacts: {output}")
    if output.parent.resolve() == output:
        raise ValueError("Output directory cannot be a filesystem root")

    source_root = root / "test-data" / "sakura-benchmark"
    analysis_path = source_root / "piano-full-pipeline-v1" / "general-analysis.json"
    identity_path = source_root / "piano-full-pipeline-v1" / "identity.json"
    audit_path = source_root / "raw-pcm-audit-v1" / "report.json"
    vocals_path = source_root / "raw-pcm-audit-v1" / "vocals.wav"
    mix_path = source_root / "raw-pcm-audit-v1" / "mix.wav"
    provenance_path = root / "test-data" / "vocal-phrase-release-review-v1" / "report.json"
    protocol_path = root / "docs" / "vocal-phrase-audio-review-protocol.md"
    runner_path = Path(__file__).resolve()

    analysis = _read_json(analysis_path)
    identity = _read_json(identity_path)
    audit = _read_json(audit_path)
    provenance = _read_json(provenance_path)
    if root == ROOT and sha256_file(provenance_path) != EXPECTED_PROVENANCE_SHA256:
        raise ValueError("Frozen vocal phrase provenance report changed")
    if not protocol_path.is_file():
        raise ValueError(f"Review protocol is missing: {protocol_path}")

    vocals, mix, rate, gain, source_identity = _validate_sources(
        root, analysis, identity, audit, provenance, analysis_path, vocals_path, mix_path,
        provenance_path,
    )
    duration_ms = len(vocals) / rate * 1000
    segments = analysis.get("segments")
    phrases = analysis.get("phrase_candidates")
    if not isinstance(segments, list) or not isinstance(phrases, list) or not phrases:
        raise ValueError("Analysis has no frozen segments or phrase candidates")
    ids = [str(phrase["id"]) for phrase in phrases]
    if len(ids) != len(set(ids)):
        raise ValueError("Phrase candidate IDs must be unique")

    source_identity["phrase_provenance_sha256"] = sha256_file(provenance_path)
    source_identity["analysis_duration_seconds"] = float(analysis["duration"])
    source_identity["phrase_count"] = len(phrases)
    source_identity["section_count"] = len(segments)
    source_identity["clip_padding_ms"] = PADDING_MS

    ordered_phrases = sorted(phrases, key=lambda phrase: (phrase["start"], phrase["end"], phrase["id"]))
    section_indices = {id(section): index + 1 for index, section in enumerate(segments)}
    prepared = []
    for index, phrase in enumerate(ordered_phrases, start=1):
        start_ms, end_ms = float(phrase["start"]), float(phrase["end"])
        clip_start_ms, clip_end_ms = clip_bounds_ms(start_ms, end_ms, duration_ms)
        left = round(clip_start_ms * rate / 1000)
        right = round(clip_end_ms * rate / 1000)
        if right <= left:
            raise ValueError(f"Phrase {phrase['id']} produced an empty clip")
        segment = _segment_for(start_ms, segments)
        section_index = section_indices[id(segment)] if segment is not None else None
        section_label = str(segment["label"]) if segment is not None else "未映射段落"
        prepared.append({
            "index": index,
            "id": str(phrase["id"]),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "pitch": int(phrase["pitch"]),
            "support_count": len(phrase.get("support") or []),
            "support_coverage": phrase.get("support_coverage"),
            "section_index": section_index,
            "section_label": section_label,
            "clip_start_ms": left / rate * 1000,
            "clip_end_ms": right / rate * 1000,
            "clip_start_sample": left,
            "clip_stop_sample": right,
            "clips": {
                "vocals": f"clips/{index:03d}-vocals.wav",
                "mix": f"clips/{index:03d}-mix.wav",
            },
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        clips_dir = staging / "clips"
        clips_dir.mkdir()
        summaries = []
        for row, phrase in zip(prepared, ordered_phrases):
            left, right = row["clip_start_sample"], row["clip_stop_sample"]
            for kind, samples in (("vocals", vocals), ("mix", mix)):
                path = staging / row["clips"][kind]
                wavfile.write(path, rate, pcm16(samples[left:right], gain))
                check_rate, restored = wavfile.read(path)
                expected_shape = (right - left,)
                if check_rate != rate or restored.shape != expected_shape or restored.dtype != np.int16:
                    raise ValueError(f"Written clip failed format verification: {path.name}")
                expected = np.rint(
                    np.asarray(samples[left:right], dtype=np.float64) * gain * 32767
                ).astype(np.int16)
                if not np.array_equal(restored, expected):
                    raise ValueError(f"Written clip failed sample verification: {path.name}")
            summaries.append({
                **row,
                "model_phrase_flag": bool(phrase.get("phrase")),
                "source_label": phrase.get("source"),
            })

        review_key = hashlib.sha256(
            (source_identity["analysis_sha256"] + source_identity["vocal_pcm_sha256"]).encode()
        ).hexdigest()[:20]
        section_counts = []
        for index, section in enumerate(segments, start=1):
            section_counts.append({
                "index": index,
                "label": str(section["label"]),
                "start_ms": float(section["start_ms"]),
                "end_ms": float(section["end_ms"]),
                "phrase_candidates": sum(row["section_index"] == index for row in summaries),
            })
        manifest = {
            "schema": "manialab-vocal-phrase-audio-review-v1",
            "purpose": "Human listening aid for frozen vocal phrase candidates; no automatic labels.",
            "review_key": review_key,
            "sources": source_identity,
            "sections": section_counts,
            "phrases": summaries,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (staging / "index.html").write_text(
            _render_html(summaries, section_counts, review_key), encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# 樱花草人声乐句音频复核\n\n"
            "打开 `index.html`，逐条试听 Nova 人声轨与原混音。页面标签只保存在当前浏览器，"
            "关闭前请用“导出审核记录”保存 JSON。输入源、时间轴、切片和 SHA-256 见 "
            "`manifest.json` 与 `report.json`。\n\n"
            "音频为原始 float PCM 经过同一个线性增益后转为 PCM16；未做逐片段响度归一、"
            "滤波、压缩、重采样、时间移动或模型推理。片段左右各含 1.5 秒上下文，"
            "所有时间仍是原曲绝对时钟。\n\n"
            "人工试听尚未完成。候选、分离标签、音高、内部 coverage 和 Reference 谱均不是"
            "声学真值；不能由该审核包本身推出识别准确率或音乐质量提升。\n",
            encoding="utf-8",
        )
        shutil.copyfile(runner_path, staging / "runner-snapshot.py")
        shutil.copyfile(protocol_path, staging / "protocol-snapshot.md")
        shutil.copyfile(provenance_path, staging / "phrase-provenance-snapshot.json")
        artifact_hashes = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        report = {
            "schema": "manialab-vocal-phrase-audio-review-report-v1",
            "input_identity": source_identity,
            "candidate_count": len(summaries),
            "section_counts": section_counts,
            "clip_count": len(summaries) * 2,
            "playback": {
                "sample_rate": rate,
                "channels": 1,
                "format": "PCM16",
                "global_linear_gain": gain,
                "peak_after_gain": max(source_identity["vocal_peak"], source_identity["mix_peak"]) * gain,
                "per_clip_normalization": False,
                "padding_ms_each_side": PADDING_MS,
            },
            "model_inference_rerun": False,
            "human_listening_completed": False,
            "musical_acceptance": False,
            "artifact_sha256": artifact_hashes,
        }
        report_path = staging / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise ValueError(f"Preserve previous audio review artifacts: {output}")
        staging.rename(output)
        report["report_sha256"] = sha256_file(output / "report.json")
        report["output"] = str(output)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: list[str] | None = None) -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = cli.parse_args(argv)
    report = build_review_bundle(args.output, args.root)
    print(json.dumps({
        "output": report["output"],
        "candidate_count": report["candidate_count"],
        "clip_count": report["clip_count"],
        "report_sha256": report["report_sha256"],
        "human_listening_completed": report["human_listening_completed"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
