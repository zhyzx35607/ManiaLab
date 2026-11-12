"""Build a local listening aid for frozen Sakura selection differences."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manialab.audition import render_audition


TIE_ROOT = Path("test-data/chord-evidence-ties-v1")
BASELINE_ROOT = Path("test-data/articulation-interval-gaps-v3")
ANALYSIS_PATH = Path("test-data/sakura-tsumugi-expression-onset-v3/analysis.json")
MIX_PATH = Path("test-data/sakura-benchmark/raw-pcm-audit-v1/mix.wav")
MIX_AUDIT_PATH = Path("test-data/sakura-benchmark/raw-pcm-audit-v1/report.json")
PROTOCOL_PATH = Path("docs/chord-evidence-audio-review-protocol.md")
DEFAULT_OUTPUT = Path("test-data/sakura-chord-evidence-audio-review-v1")

EXPECTED_TIE_REPORT_SHA256 = "fe55e2a9a116953a7cb9ac53e443fd0f0bfae24cc7c6fd8ceaf39cab60553577"
EXPECTED_BASELINE_REPORT_SHA256 = "1fcf7a60dbf820c21a49efc866a56a035821706dc7a6ba609e02eb4d2fd8d279"
EXPECTED_ANALYSIS_SHA256 = "26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a"
EXPECTED_MIX_ORIGINAL_SHA256 = "2a82d46c43e43acb4f8e21de71573be1ae6b71b8cf5485364c9ed1d5d3b12765"
EXPECTED_MIX_PCM_SHA256 = "dfd2e1581124db8255a347c8414411d1978e279d06116603df64dcfa60e8745e"
EXPECTED_SAMPLE_RATE = 22050
EXPECTED_SAMPLE_COUNT = 6258672

WINDOWS_MS = (
    (8000, 16000),
    (16000, 24000),
    (40000, 48000),
    (96000, 104000),
    (136000, 144000),
    (200000, 208000),
)
MUSIC_GAIN = 0.50
CLICK_GAIN = 0.12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _decision_map(selection: dict) -> dict[str, dict]:
    decisions = selection.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Selection file does not contain a decisions list")
    result = {}
    for row in decisions:
        note_id = row.get("id")
        if not note_id or note_id in result:
            raise ValueError("Selection decisions contain a missing or duplicate ID")
        result[note_id] = row
    return result


def build_review_cases(
    changes: dict,
    candidates: list[dict],
    old_selection: dict,
    new_selection: dict,
    old_selected: list[dict],
    new_selected: list[dict],
    windows_ms: tuple[tuple[int, int], ...] = WINDOWS_MS,
) -> list[dict]:
    candidate_by_id = {row["id"]: row for row in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("Candidate IDs are not unique")

    old_ids = {row["id"] for row in old_selected}
    new_ids = {row["id"] for row in new_selected}
    removed_ids = set(changes.get("removed", []))
    added_ids = set(changes.get("added", []))
    if removed_ids != old_ids - new_ids or added_ids != new_ids - old_ids:
        raise ValueError("Changed candidate IDs do not match frozen selections")
    if removed_ids & added_ids:
        raise ValueError("A candidate cannot be both removed and added")

    old_decisions = _decision_map(old_selection)
    new_decisions = _decision_map(new_selection)
    all_changed_ids = removed_ids | added_ids
    missing = all_changed_ids - candidate_by_id.keys()
    if missing:
        raise ValueError(f"Changed candidates are missing from the frozen list: {sorted(missing)[:3]}")
    if not all_changed_ids <= old_decisions.keys() or not all_changed_ids <= new_decisions.keys():
        raise ValueError("Changed candidate is missing an old or new decision")

    candidate_changes = []
    for direction, ids in (("removed", removed_ids), ("added", added_ids)):
        for note_id in ids:
            note = candidate_by_id[note_id]
            old = old_decisions[note_id]
            new = new_decisions[note_id]
            candidate_changes.append({
                "id": note_id,
                "direction": direction,
                "voice": note["voice"],
                "candidate_start_ms": int(note["start"]),
                "candidate_end_ms": int(note["end"]),
                "evidence_start_ms": note.get("evidence_start"),
                "evidence_offset_ms": (
                    int(note["evidence_start"]) - int(note["start"])
                    if note.get("evidence_start") is not None else None
                ),
                "pitch": note.get("pitch"),
                "source": note.get("source"),
                "strength": note.get("strength"),
                "old": {
                    "decision": old.get("decision"),
                    "reason": old.get("reason"),
                    "chart_start_ms": old.get("chart_start"),
                },
                "new": {
                    "decision": new.get("decision"),
                    "reason": new.get("reason"),
                    "chart_start_ms": new.get("chart_start"),
                },
            })

    attack_changes = []
    for direction, pairs, ids, decision_rows in (
        ("removed", changes.get("removed_attacks", []), removed_ids, old_decisions),
        ("added", changes.get("added_attacks", []), added_ids, new_decisions),
    ):
        seen = set()
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Changed attack record must be [voice, time_ms]")
            voice, time_ms = str(pair[0]), int(pair[1])
            key = (voice, time_ms)
            if key in seen:
                raise ValueError("Changed attack records contain duplicates")
            seen.add(key)
            matched_ids = [
                note_id for note_id in ids
                if decision_rows[note_id].get("voice") == voice
                and decision_rows[note_id].get("chart_start") == time_ms
            ]
            if len(matched_ids) != 1:
                raise ValueError(
                    f"Changed attack {voice}@{time_ms}ms maps to {len(matched_ids)} candidate IDs"
                )
            candidate = candidate_by_id[matched_ids[0]]
            attack_changes.append({
                "id": f"{direction}:{voice}:{time_ms}",
                "direction": direction,
                "voice": voice,
                "time_ms": time_ms,
                "candidate_ids": matched_ids,
                "candidate_start_ms": int(candidate["start"]),
            })

    all_attacks = {(row["direction"], row["voice"], row["time_ms"]) for row in attack_changes}
    if len(all_attacks) != len(attack_changes):
        raise ValueError("Removed and added attack changes contain duplicate records")

    cases = []
    assigned_attacks = []
    for begin_ms, end_ms in windows_ms:
        if begin_ms < 0 or end_ms <= begin_ms:
            raise ValueError("Review windows must be positive and ordered")
        key = f"{begin_ms:06d}-{end_ms:06d}"
        case_attacks = [
            row for row in attack_changes if begin_ms <= row["time_ms"] < end_ms
        ]
        assigned_attacks.extend(case_attacks)
        case_candidates = [
            row for row in candidate_changes
            if begin_ms <= row["candidate_start_ms"] < end_ms
        ]
        cases.append({
            "case_id": key,
            "start_ms": begin_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - begin_ms,
            "attacks": sorted(case_attacks, key=lambda row: (
                row["time_ms"], row["direction"], row["voice"]
            )),
            "candidate_changes": sorted(case_candidates, key=lambda row: (
                row["candidate_start_ms"], row["direction"], row["id"]
            )),
            "clips": {},
        })
    if len(assigned_attacks) != len(attack_changes):
        raise ValueError("Review windows do not cover all changed attack records")
    return cases


def _write_audition(
    source: Path,
    destination: Path,
    start_ms: int,
    duration_ms: int,
    events: list[tuple[int, str]],
) -> dict:
    buffer, metadata = render_audition(
        source, start_ms, duration_ms, events,
        music_volume=MUSIC_GAIN, click_volume=CLICK_GAIN,
    )
    if metadata["gain"] != 1.0:
        raise ValueError("Fixed review gain unexpectedly invoked peak normalization")
    with destination.open("xb") as stream:
        stream.write(buffer.getvalue())
    rate, restored = wavfile.read(destination)
    expected_samples = round(duration_ms * rate / 1000)
    if (
        rate != EXPECTED_SAMPLE_RATE
        or restored.shape != (expected_samples, 2)
        or restored.dtype != np.int16
        or np.max(np.abs(restored.astype(np.int32)), initial=0) >= 32767
    ):
        raise ValueError(f"Rendered review audio failed validation: {destination.name}")
    return metadata | {
        "file": destination.name,
        "sha256": sha256_file(destination),
        "channels": 2,
        "format": "PCM16",
    }


def _render_html(manifest: dict, manifest_sha256: str) -> str:
    def esc(value) -> str:
        return html.escape(str(value), quote=True)

    event_options = (
        ("", "待标"),
        ("audible", "听到对应起音"),
        ("not_audible", "未听到"),
        ("uncertain", "不确定"),
    )
    sections = []
    for case in manifest["cases"]:
        tracks = []
        labels = {
            "source": "原混音，无定位 click",
            "removed_clicks": "旧版落选起音 click",
            "added_clicks": "新版新增起音 click",
        }
        for kind, label in labels.items():
            filename = case["clips"].get(kind)
            if filename:
                tracks.append(
                    f'<div class="track"><label>{esc(label)}</label>'
                    f'<audio controls preload="none" src="clips/{esc(filename)}"></audio></div>'
                )
        if not case["attacks"]:
            tracks.append(
                '<p class="track-note">此窗起音集合没有变化；请先听原混音，'
                '再参考下方候选音高与取舍记录。</p>'
            )

        attack_rows = []
        for event in case["attacks"]:
            options = "".join(
                f'<option value="{esc(value)}">{esc(label)}</option>'
                for value, label in event_options
            )
            ids = ", ".join(event["candidate_ids"])
            attack_rows.append(
                f'<tr data-event-id="{esc(event["id"])}">'
                f'<td>{event["time_ms"] / 1000:.3f}</td>'
                f'<td>{esc(event["direction"])}</td>'
                f'<td>{esc(event["voice"])}</td>'
                f'<td><code>{esc(ids)}</code></td>'
                f'<td><select aria-label="起音听感" data-label>{options}</select></td>'
                f'<td><input type="text" aria-label="起音备注" data-note></td>'
                '</tr>'
            )
        if not attack_rows:
            attack_table = '<p class="muted">本窗无增删起音记录。</p>'
        else:
            attack_table = (
                '<div class="table-wrap"><table><thead><tr><th>时间(s)</th>'
                '<th>变化</th><th>候选声部</th><th>候选ID</th><th>原混音听感</th>'
                '<th>备注</th></tr></thead><tbody>'
                + "".join(attack_rows)
                + "</tbody></table></div>"
            )

        candidate_rows = []
        for row in case["candidate_changes"]:
            delta = row["evidence_offset_ms"]
            delta_text = "" if delta is None else f"{delta:+d} ms"
            strength = row["strength"]
            strength_text = "" if strength is None else f"{float(strength):.3f}"
            old = row["old"]
            new = row["new"]
            candidate_rows.append(
                "<tr>"
                f'<td><code>{esc(row["id"])}</code></td>'
                f'<td>{esc(row["direction"])}</td>'
                f'<td>{esc(row["voice"])}</td>'
                f'<td>{row["candidate_start_ms"] / 1000:.3f}</td>'
                f'<td>{esc(row["pitch"])}</td>'
                f'<td>{esc(row["source"])}</td>'
                f'<td>{esc(strength_text)}</td>'
                f'<td>{esc(delta_text)}</td>'
                f'<td>{esc(old["decision"])} / {esc(old["reason"])}</td>'
                f'<td>{esc(new["decision"])} / {esc(new["reason"])}</td>'
                "</tr>"
            )
        candidate_table = (
            '<details><summary>候选决策差异（'
            f'{len(candidate_rows)}项）</summary>'
            '<div class="table-wrap"><table><thead><tr><th>ID</th><th>增删</th>'
            '<th>候选声部</th><th>原始起音(s)</th><th>MIDI</th><th>来源</th>'
            '<th>strength</th><th>证据偏移</th><th>旧决策/原因</th>'
            '<th>新决策/原因</th></tr></thead><tbody>'
            + "".join(candidate_rows)
            + "</tbody></table></div></details>"
        )
        sections.append(
            f'<section class="window" data-case-id="{esc(case["case_id"])}">'
            f'<h2>{case["start_ms"] / 1000:.3f}–{case["end_ms"] / 1000:.3f}秒</h2>'
            f'<div class="track-grid">{"".join(tracks)}</div>'
            f'{attack_table}'
            f'<details class="window-note"><summary>窗口备注</summary>'
            f'<textarea aria-label="窗口备注" data-case-note></textarea></details>'
            f'{candidate_table}'
            '</section>'
        )

    embedded = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>樱花草候选差异听审</title>
  <style>
    :root {{ color-scheme: light; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #202a34; background: #f7f9fa; }}
    * {{ box-sizing: border-box; }}
    body {{ max-width: 1240px; margin: 0 auto; padding: 20px; font-size: 14px; line-height: 1.45; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    p {{ margin: 8px 0; }}
    .lede {{ max-width: 900px; color: #43505d; }}
    .notice {{ border-left: 3px solid #c87827; background: #fff; padding: 10px 12px; margin: 16px 0; }}
    .actions {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 16px 0; }}
    button {{ min-height: 36px; padding: 6px 12px; border: 1px solid #9daab5; border-radius: 4px; background: #fff; color: #202a34; cursor: pointer; }}
    button:hover {{ background: #eaf0f3; }}
    .window {{ border-top: 1px solid #b9c4cc; padding: 18px 0 22px; }}
    .track-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 8px 0 16px; }}
    .track label {{ display: block; font-weight: 600; margin-bottom: 4px; }}
    audio {{ display: block; width: 100%; height: 40px; }}
    .track-note, .muted {{ color: #586675; }}
    .table-wrap {{ overflow-x: auto; margin: 10px 0 14px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; }}
    th, td {{ border-bottom: 1px solid #d4dce1; padding: 7px 8px; text-align: left; vertical-align: top; white-space: nowrap; }}
    th {{ background: #edf1f3; font-weight: 600; }}
    code {{ font-size: 12px; }}
    select, input[type="text"], textarea {{ font: inherit; min-height: 32px; border: 1px solid #9daab5; border-radius: 3px; background: #fff; color: #202a34; padding: 4px 6px; }}
    input[type="text"] {{ min-width: 150px; width: 100%; }}
    details {{ margin: 8px 0; }}
    summary {{ cursor: pointer; font-weight: 600; padding: 6px 0; }}
    .window-note textarea {{ width: min(720px, 100%); min-height: 64px; display: block; }}
    #status {{ color: #43505d; }}
    @media (max-width: 760px) {{
      body {{ padding: 14px; }}
      .track-grid {{ grid-template-columns: 1fr; gap: 8px; }}
      th, td {{ padding: 6px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>《樱花草》10K 候选差异听审</h1>
    <p class="lede">比较冻结的旧/新选择在原混音中的可辨起音。系统声部标签和 MIDI 候选不是声学真值；请先听原混音，再用 click 预览定位时间。</p>
  </header>
  <p class="notice">原混音片段使用统一 0.50 增益；定位 click 为 0.12，未做逐片响度归一、滤波、重采样或时间移动。click 只标候选起音，不证明音高、声源或音乐价值。此复核不替代 osu! 试打。</p>
  <div class="actions">
    <button id="export" type="button">导出标签 JSON</button>
    <button id="clear" type="button">清除本机暂存</button>
    <span id="status" role="status"></span>
  </div>
  <main>{"".join(sections)}</main>
  <script id="manifest-data" type="application/json">{embedded}</script>
  <script>
    const manifest = JSON.parse(document.getElementById("manifest-data").textContent);
    const manifestSha256 = "{manifest_sha256}";
    const storageKey = "manialab-chord-evidence-audio-review:" + manifest.review_key;
    const status = document.getElementById("status");
    const rows = [...document.querySelectorAll("tr[data-event-id]")];
    const cases = [...document.querySelectorAll("section[data-case-id]")];
    function collect() {{
      const events = {{}};
      for (const row of rows) {{
        events[row.dataset.eventId] = {{
          audibility: row.querySelector("[data-label]").value,
          note: row.querySelector("[data-note]").value
        }};
      }}
      const caseNotes = {{}};
      for (const section of cases) {{
        caseNotes[section.dataset.caseId] = section.querySelector("[data-case-note]").value;
      }}
      return {{ events, caseNotes }};
    }}
    function save() {{
      const value = collect();
      try {{
        localStorage.setItem(storageKey, JSON.stringify(value));
        updateStatus(value);
      }} catch (error) {{
        status.textContent = "浏览器暂存不可用，请填写后立即导出 JSON。";
      }}
    }}
    function updateStatus(value) {{
      const labels = Object.values(value.events).filter(row => row.audibility).length;
      status.textContent = `已标记 ${{labels}} / ${{rows.length}} 条起音`;
    }}
    function restore() {{
      try {{
        const saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
        for (const row of rows) {{
          const value = saved.events?.[row.dataset.eventId];
          if (!value) continue;
          row.querySelector("[data-label]").value = value.audibility || "";
          row.querySelector("[data-note]").value = value.note || "";
        }}
        for (const section of cases) {{
          section.querySelector("[data-case-note]").value = saved.caseNotes?.[section.dataset.caseId] || "";
        }}
        updateStatus(collect());
      }} catch (error) {{
        status.textContent = "读取暂存失败；请导出当前填写内容。";
      }}
    }}
    document.addEventListener("input", save);
    document.addEventListener("change", save);
    document.getElementById("clear").addEventListener("click", () => {{
      if (!window.confirm("清除当前浏览器中的本地听审标签？已导出的 JSON 不受影响。")) return;
      localStorage.removeItem(storageKey);
      for (const row of rows) {{
        row.querySelector("[data-label]").value = "";
        row.querySelector("[data-note]").value = "";
      }}
      for (const section of cases) section.querySelector("[data-case-note]").value = "";
      updateStatus(collect());
    }});
    document.getElementById("export").addEventListener("click", () => {{
      const state = collect();
      const payload = {{
        schema: "manialab-chord-evidence-audio-review-labels-v1",
        review_key: manifest.review_key,
        manifest_sha256: manifestSha256,
        exported_at: new Date().toISOString(),
        ...state
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "sakura-chord-evidence-audio-review-v1-labels.json";
      anchor.click();
      URL.revokeObjectURL(url);
    }});
    restore();
  </script>
</body>
</html>
"""


def build_review_bundle(output: Path = DEFAULT_OUTPUT, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    ignored_root = (root / "test-data").resolve()
    try:
        output.relative_to(ignored_root)
    except ValueError as exc:
        raise ValueError("Review output must remain inside ignored test-data") from exc
    if output.exists():
        raise FileExistsError(f"Preserve previous and partial reviews: {output}")

    tie_root = root / TIE_ROOT
    baseline_root = root / BASELINE_ROOT
    current_report_path = tie_root / "report.json"
    current_plan_path = tie_root / "execution-plan.json"
    baseline_report_path = baseline_root / "report.json"
    baseline_plan_path = baseline_root / "execution-plan.json"
    analysis_path = root / ANALYSIS_PATH
    mix_path = root / MIX_PATH
    mix_audit_path = root / MIX_AUDIT_PATH

    if sha256_file(current_report_path) != EXPECTED_TIE_REPORT_SHA256:
        raise ValueError("Frozen chord-evidence report changed")
    if sha256_file(baseline_report_path) != EXPECTED_BASELINE_REPORT_SHA256:
        raise ValueError("Frozen baseline report changed")
    if sha256_file(analysis_path) != EXPECTED_ANALYSIS_SHA256:
        raise ValueError("Frozen Sakura analysis changed")

    current_report = read_json(current_report_path)
    current_plan = read_json(current_plan_path)
    baseline_report = read_json(baseline_report_path)
    baseline_plan = read_json(baseline_plan_path)
    if sha256_file(current_plan_path) != current_report["execution_plan_sha256"]:
        raise ValueError("Current chord-evidence execution plan changed")
    if sha256_file(baseline_plan_path) != baseline_report["execution_plan_sha256"]:
        raise ValueError("Frozen baseline execution plan changed")
    analysis_key = ANALYSIS_PATH.as_posix()
    if current_plan["inputs"].get(analysis_key) != EXPECTED_ANALYSIS_SHA256:
        raise ValueError("Chord-evidence plan does not reference the frozen analysis")

    current_case = next(row for row in current_report["cases"] if row["keys"] == 10)
    baseline_case = next(row for row in baseline_report["cases"] if row["keys"] == 10)
    current_case_root = tie_root / "10k"
    baseline_case_root = baseline_root / "10k"
    checked_inputs = {}
    for name, expected_hash in current_case["artifacts"].items():
        path = current_case_root / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen 10K chord artifact changed: {name}")
        checked_inputs[path.relative_to(root).as_posix()] = expected_hash
    for name in ("selected.json", "selection.json"):
        expected_hash = baseline_case["artifacts"][name]
        path = baseline_case_root / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen 10K baseline artifact changed: {name}")
        checked_inputs[path.relative_to(root).as_posix()] = expected_hash

    audit = read_json(mix_audit_path)
    analysis = read_json(analysis_path)
    mix_row = next(row for row in audit["rows"] if row["role"] == "mix")
    if (
        mix_row["original_sha256"] != EXPECTED_MIX_ORIGINAL_SHA256
        or mix_row["raw_float_sha256"] != EXPECTED_MIX_PCM_SHA256
        or mix_row["samples"] != EXPECTED_SAMPLE_COUNT
        or sha256_file(mix_path) != EXPECTED_MIX_PCM_SHA256
        or analysis.get("separation", {}).get("audio_sha256") != EXPECTED_MIX_ORIGINAL_SHA256
    ):
        raise ValueError("Original-mix PCM does not match its frozen audio audit")

    rate, pcm = wavfile.read(mix_path, mmap=True)
    if (
        rate != EXPECTED_SAMPLE_RATE
        or pcm.dtype != np.float32
        or pcm.ndim != 1
        or len(pcm) != EXPECTED_SAMPLE_COUNT
        or not np.isfinite(pcm).all()
    ):
        raise ValueError("Original-mix PCM format or sample count changed")

    current_selected = read_json(current_case_root / "selected.json")
    old_selected = read_json(baseline_case_root / "selected.json")
    current_selection = read_json(current_case_root / "selection.json")
    old_selection = read_json(baseline_case_root / "selection.json")
    candidates = read_json(current_case_root / "candidates.json")
    changes = read_json(current_case_root / "changes.json")
    cases = build_review_cases(
        changes, candidates, old_selection, current_selection,
        old_selected, current_selected,
    )

    builder_path = Path(__file__).resolve()
    protocol_path = root / PROTOCOL_PATH
    input_identity = {
        "source_paths": {
            "analysis": ANALYSIS_PATH.as_posix(),
            "mix": MIX_PATH.as_posix(),
            "mix_audit": MIX_AUDIT_PATH.as_posix(),
        },
        "current_report_sha256": sha256_file(current_report_path),
        "current_execution_plan_sha256": sha256_file(current_plan_path),
        "baseline_report_sha256": sha256_file(baseline_report_path),
        "baseline_execution_plan_sha256": sha256_file(baseline_plan_path),
        "analysis_sha256": sha256_file(analysis_path),
        "original_mix_sha256": mix_row["original_sha256"],
        "raw_float_mix_sha256": sha256_file(mix_path),
        "mix_audit_sha256": sha256_file(mix_audit_path),
        "raw_float_mix_samples": int(len(pcm)),
        "sample_rate": int(rate),
        "checked_artifacts": checked_inputs,
        "builder_sha256": sha256_file(builder_path),
        "audition_helper_sha256": sha256_file(root / "manialab" / "audition.py"),
        "protocol_sha256": sha256_file(protocol_path),
    }
    key_material = json.dumps(
        {"inputs": input_identity, "windows_ms": WINDOWS_MS},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    review_key = hashlib.sha256(key_material).hexdigest()[:20]

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        clips_dir = stage / "clips"
        clips_dir.mkdir()
        for case in cases:
            begin_ms, end_ms = case["start_ms"], case["end_ms"]
            prefix = case["case_id"]
            source_name = f"{prefix}-source.wav"
            source_meta = _write_audition(
                mix_path, clips_dir / source_name,
                begin_ms, end_ms - begin_ms, [],
            )
            case["clips"]["source"] = source_name
            case["audio"] = {"source": source_meta}
            for direction, kind in (("removed", "removed_clicks"), ("added", "added_clicks")):
                attacks = [row for row in case["attacks"] if row["direction"] == direction]
                if not attacks:
                    continue
                clip_name = f"{prefix}-{direction}-clicks.wav"
                events = [(row["time_ms"], "head") for row in attacks]
                metadata = _write_audition(
                    mix_path, clips_dir / clip_name,
                    begin_ms, end_ms - begin_ms, events,
                )
                case["clips"][kind] = clip_name
                case["audio"][kind] = metadata

        manifest = {
            "schema": "manialab-chord-evidence-audio-review-v1",
            "purpose": "Human listening aid for frozen old/new selection differences.",
            "review_key": review_key,
            "source": "original mix; no separated/model voice is treated as ground truth",
            "input_identity": input_identity,
            "playback": {
                "sample_rate": EXPECTED_SAMPLE_RATE,
                "channels": 2,
                "format": "PCM16",
                "music_gain": MUSIC_GAIN,
                "click_gain": CLICK_GAIN,
                "peak_normalization": False,
                "time_shift_ms": 0,
                "filtering": False,
                "resampling": False,
            },
            "cases": cases,
            "neural_inference": False,
            "reference_chart_read": False,
            "automatic_timing_acceptance": False,
            "musical_acceptance": False,
            "listened": False,
            "limits": [
                "The source-mix playback is the evidence; click overlays only mark candidate attack times.",
                "Voice names, model classes, pitch candidates and strength are not acoustic truth.",
                "Audibility labels do not establish musical value or correct chart arrangement.",
                "Human listening and osu! playtesting are still required.",
            ],
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_hash = sha256_file(manifest_path)
        (stage / "index.html").write_text(
            _render_html(manifest, manifest_hash), encoding="utf-8"
        )
        (stage / "README.md").write_text(
            "# 樱花草候选差异听审\n\n"
            "在本地 HTTP 服务中打开 `index.html`。先听每窗原混音，再用旧版/新版差异起音 "
            "click 定位。标记每条差异起音在原混音中的可辨程度，并导出 JSON；页面暂存只在"
            "当前浏览器中有效。\n\n"
            "本包包含六个冻结窗口的原混音 PCM16 片段，未复制整首音频。所有片段使用固定"
            "0.50 音乐增益和0.12 click增益，输出不触发峰值归一化。声部、音高和模型强度"
            "仅用于候选定位，不能当作真实乐器或音乐价值结论。\n\n"
            "听审与osu!试打尚未完成。标签导出后可由 "
            "`scripts/summarize_chord_evidence_audio_review.py` 校验和汇总；"
            "汇总器不会计算准确率或自动判定音乐质量。\n",
            encoding="utf-8",
        )
        shutil.copyfile(builder_path, stage / "runner-snapshot.py")
        shutil.copyfile(protocol_path, stage / "protocol-snapshot.md")

        artifact_hashes = {
            path.relative_to(stage).as_posix(): sha256_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        report = {
            "schema": "manialab-chord-evidence-audio-review-report-v1",
            "review_key": review_key,
            "manifest_sha256": manifest_hash,
            "case_count": len(cases),
            "attack_event_count": sum(len(case["attacks"]) for case in cases),
            "candidate_change_count": sum(len(case["candidate_changes"]) for case in cases),
            "clip_count": sum(len(case["audio"]) for case in cases),
            "input_identity": input_identity,
            "artifact_sha256": artifact_hashes,
            "neural_inference": False,
            "reference_chart_read": False,
            "musical_acceptance": False,
            "listened": False,
        }
        (stage / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"Review output appeared during build: {output}")
        stage.replace(output)
    except Exception:
        if stage.exists() and stage.parent.resolve() == output.parent.resolve():
            shutil.rmtree(stage)
        raise
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_review_bundle(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
