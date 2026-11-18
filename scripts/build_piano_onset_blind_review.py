"""Build a blinded listening package from the cached 10K piano chart A/B."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import random
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manialab.audio_config import SR
from manialab.audio_io import decode


DEFAULT_INPUT = ROOT / "test-data/audio-transcription-chart-ab-20261118"
DEFAULT_OUTPUT = ROOT / "test-data/piano-onset-blind-review-20261118"
CLIP_BEFORE_MS = 1000
CLIP_AFTER_MS = 1500
SAMPLE_RATE = SR
REVIEW_FIELDS = ("clip_id", "audio_file", "audio_start_ms", "pitch_midi", "audibility", "confidence_1_5", "comment")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value


def piano_events(path: Path) -> dict[tuple[int, int, int], list[dict]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a note list: {path}")

    grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or row.get("voice") != "piano":
            continue
        pitch, start, end, audio_start = (
            row.get("pitch"),
            row.get("start"),
            row.get("end"),
            row.get("audio_start"),
        )
        if any(type(value) is not int or value < 0 for value in (pitch, start, end, audio_start)):
            raise ValueError("Piano event needs nonnegative integer pitch and source times")
        if pitch > 127 or end < start:
            raise ValueError("Piano event has an invalid pitch or duration")
        grouped[(pitch, start, end)].append(row)

    if not grouped:
        raise ValueError(f"No piano events found: {path}")
    for event_rows in grouped.values():
        event_rows.sort(key=lambda row: (row["audio_start"], str(row.get("id", ""))))
    return grouped


def event_differences(
    kong_path: Path,
    transkun_path: Path,
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """Return unmatched rendered piano events and their private conditions."""
    kong, transkun = piano_events(kong_path), piano_events(transkun_path)
    added = []
    removed = []
    for key in sorted(kong.keys() | transkun.keys()):
        kong_rows = kong.get(key, [])
        transkun_rows = transkun.get(key, [])
        shared = min(len(kong_rows), len(transkun_rows))
        added.extend(("transkun_only", row) for row in transkun_rows[shared:])
        removed.extend(("kong_only", row) for row in kong_rows[shared:])
    return added, removed


def slice_audio(
    samples: np.ndarray,
    sample_rate: int,
    center_ms: int,
    *,
    before_ms: int = CLIP_BEFORE_MS,
    after_ms: int = CLIP_AFTER_MS,
) -> np.ndarray:
    if sample_rate <= 0 or center_ms < 0 or before_ms < 0 or after_ms <= 0:
        raise ValueError("Invalid audio clip timing")
    audio = np.asarray(samples, dtype=np.float32)
    center = round(center_ms * sample_rate / 1000)
    if center >= len(audio):
        raise ValueError("Source event is outside the decoded audio")

    before = round(before_ms * sample_rate / 1000)
    after = round(after_ms * sample_rate / 1000)
    target_start = center - before
    target_end = center + after
    result = np.zeros(before + after, dtype=np.float32)
    source_start = max(target_start, 0)
    source_end = min(target_end, len(audio))
    destination_start = source_start - target_start
    destination_end = destination_start + source_end - source_start
    result[destination_start:destination_end] = audio[source_start:source_end]
    return result


def render_review_html(rows: list[dict]) -> str:
    body_rows = []
    for row in rows:
        clip_id = html.escape(row["clip_id"], quote=True)
        audio_file = html.escape(row["audio_file"], quote=True)
        target_time = html.escape(row["audio_start_label"])
        audio_start_ms = html.escape(str(row["audio_start_ms"]), quote=True)
        pitch = html.escape(str(row["pitch"]))
        body_rows.append(
            "<tr>"
            f"<th scope=\"row\" data-field=\"clip_id\" data-value=\"{clip_id}\">{clip_id}</th>"
            f"<td data-field=\"audio_start_ms\" data-value=\"{audio_start_ms}\">{target_time}</td>"
            f"<td data-field=\"pitch_midi\" data-value=\"{pitch}\">MIDI {pitch}</td>"
            f"<td data-field=\"audio_file\" data-value=\"{audio_file}\"><audio controls preload=\"none\" src=\"{audio_file}\"></audio></td>"
            "<td><select data-field=\"audibility\">"
            "<option value=\"\"></option>"
            "<option value=\"yes\">听到目标音高起音</option>"
            "<option value=\"uncertain\">不确定</option>"
            "<option value=\"no\">没有听到</option>"
            "<option value=\"unclear\">无法判断</option>"
            "</select></td>"
            "<td><select data-field=\"confidence_1_5\">"
            "<option value=\"\"></option>"
            + "".join(f"<option value=\"{score}\">{score}</option>" for score in range(1, 6))
            + "</select></td>"
            "<td><textarea data-field=\"comment\" rows=\"1\" maxlength=\"500\"></textarea></td>"
            "</tr>"
        )
    return """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>盲听复核</title>
<style>
body{font:15px/1.45 system-ui,sans-serif;margin:24px;color:#202124}
main{max-width:1200px;margin:auto}table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid #d9dce1;padding:9px;text-align:left;vertical-align:middle}
thead{position:sticky;top:0;background:#fff}audio{width:260px;max-width:38vw}
select,textarea{font:inherit;max-width:100%}textarea{min-width:140px;resize:vertical}
button{font:inherit;padding:8px 12px;margin:12px 0}
</style>
</head>
<body>
<main>
<h1>盲听复核</h1>
<p>根据片段判断目标音高是否有清楚的钢琴起音。先完成并导出复核表，再查看解盲表。</p>
<button id="export" type="button">导出复核 CSV</button>
<table>
<thead><tr><th>编号</th><th>原音时间</th><th>目标音高</th><th>片段</th><th>听感</th><th>把握度</th><th>备注</th></tr></thead>
<tbody>""" + "".join(body_rows) + """</tbody>
</table>
</main>
<script>
const fields=""" + json.dumps(REVIEW_FIELDS) + """;
const csvCell=value=>'"'+String(value??'').replaceAll('"','""')+'"';
document.getElementById('export').addEventListener('click',()=>{
  const rows=[fields];
  for(const tr of document.querySelectorAll('tbody tr')){
    rows.push(fields.map(name=>{
      const field=tr.querySelector(`[data-field="${name}"]`);
      return field.dataset.value??field.value;
    }));
  }
  const csv='\\ufeff'+rows.map(row=>row.map(csvCell).join(',')).join('\\r\\n')+'\\r\\n';
  const link=document.createElement('a');
  link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  link.download='blind-review-results.csv';
  link.click();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000);
});
</script>
</body>
</html>
"""


def _write_zip(participant_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(participant_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = Path("piano-onset-review") / path.relative_to(participant_dir)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Blind review archive CRC validation failed")
        if any(
            term in name.casefold()
            for name in archive.namelist()
            for term in ("kong", "transkun", "deblind", "answer-key")
        ):
            raise ValueError("Blind review archive contains a source label")
        for name in archive.namelist():
            if name.endswith((".html", ".txt", ".csv")):
                content = archive.read(name).decode("utf-8").casefold()
                if "kong" in content or "transkun" in content:
                    raise ValueError("Blind review content contains a source label")


def build_package(input_dir: Path, output_dir: Path) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Preserve existing review packages by choosing a new output directory")

    report_path = input_dir / "report.json"
    kong_path = input_dir / "kong-10k-notes.json"
    transkun_path = input_dir / "transkun-10k-notes.json"
    for path in (report_path, kong_path, transkun_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = read_json(report_path)
    identity = report.get("identity", {})
    if identity.get("reference_chart_read") is not False:
        raise ValueError("Expected the chart A/B report that did not read Reference")
    if identity.get("direct_listening_performed") is not False:
        raise ValueError("The source A/B report already records direct listening")
    transkun_case = next(
        (case for case in report.get("cases", []) if case.get("name") == "transkun-10k"),
        None,
    )
    if transkun_case is None:
        raise ValueError("The source report is missing its 10K Transkun case")
    audio_reference = transkun_case.get("audio_reference")
    if not isinstance(audio_reference, str) or not audio_reference:
        raise ValueError("The source report is missing its chart audio reference")
    audio_path = (input_dir / audio_reference).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if identity.get("source_sha256", {}).get("audio") != sha256(audio_path):
        raise ValueError("Cached A/B audio hash differs from the report")

    added, removed = event_differences(kong_path, transkun_path)
    delta = transkun_case.get("delta_vs_kong", {})
    expected_added = delta.get("added_by_voice", {}).get("piano")
    expected_removed = delta.get("removed_by_voice", {}).get("piano")
    if len(added) != expected_added or len(removed) != expected_removed:
        raise ValueError("Piano event differences do not match the cached A/B report")

    with tempfile.TemporaryDirectory(prefix="manialab-blind-audio-") as work:
        samples = decode(audio_path, Path(work))
    if len(samples) < SAMPLE_RATE // 4:
        raise ValueError("Decoded source audio is too short")

    private_rows = [
        {"condition": condition, "note": note}
        for condition, note in added + removed
    ]
    seed = int(sha256(report_path)[:16], 16)
    random.Random(seed).shuffle(private_rows)

    participant_dir = output_dir / "participant"
    clips_dir = participant_dir / "clips"
    clips_dir.mkdir(parents=True)
    deblind = {}
    review_rows = []
    for index, private in enumerate(private_rows, start=1):
        clip_id = f"R{index:03d}"
        note = private["note"]
        clip_name = f"{clip_id}.wav"
        clip_path = clips_dir / clip_name
        clipped = slice_audio(samples, SAMPLE_RATE, note["audio_start"])
        pcm = (np.clip(clipped, -1, 1) * 32767).astype(np.int16)
        wavfile.write(clip_path, SAMPLE_RATE, pcm)
        review_row = {
            "clip_id": clip_id,
            "audio_file": f"clips/{clip_name}",
            "audio_start_label": f"{note['audio_start'] / 1000:.3f}s",
            "audio_start_ms": note["audio_start"],
            "pitch": note["pitch"],
        }
        review_rows.append(review_row)
        deblind[clip_id] = {
            "condition": private["condition"],
            "chart_start_ms": note["start"],
            "audio_start_ms": note["audio_start"],
            "pitch": note["pitch"],
            "event_id": note.get("id"),
            "clip_sha256": sha256(clip_path),
        }

    (participant_dir / "index.html").write_text(
        render_review_html(review_rows),
        encoding="utf-8",
    )
    with (participant_dir / "review-template.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(dict(clip_id=row["clip_id"], audio_file=row["audio_file"],
                                 audio_start_ms=row["audio_start_ms"], pitch_midi=row["pitch"]))
    (participant_dir / "README.txt").write_text(
        "盲听复核\n\n"
        "打开 index.html，在每条片段中判断目标 MIDI 音高是否有清楚的钢琴起音，"
        "记录听感、把握度和备注。片段按盲化顺序排列；不要在完成评分前查找解盲表。\n\n"
        "所有片段来自同一份钢琴输入音频。每段保留目标时刻前 1.0 秒和后 1.5 秒，"
        "使用原幅度，不做逐段归一化。时间使用原音频时钟，目标时刻位于片段中点前 1 秒处。\n\n"
        "样本只覆盖两张 10K 输出谱中不相同的钢琴音符，不代表所有模型候选或整首曲子的准确率。"
        "评分只记录听者对当前片段中目标起音的判断，不把它当作物理音高或按键松开真值。\n\n"
        "完成后点“导出复核 CSV”，或填写 review-template.csv。保留导出的盲评表；"
        "解盲表单独存放在包外。\n",
        encoding="utf-8",
    )

    archive_path = output_dir / "piano-onset-review.zip"
    _write_zip(participant_dir, archive_path)
    key_path = output_dir / "deblind-key.json"
    key_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "review_protocol": "rendered_10k_piano_event_differences",
                "items": deblind,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": 1,
        "source_sha256": {
            "report": sha256(report_path),
            "kong_notes": sha256(kong_path),
            "transkun_notes": sha256(transkun_path),
            "audio": sha256(audio_path),
        },
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "clip_before_ms": CLIP_BEFORE_MS,
            "clip_after_ms": CLIP_AFTER_MS,
            "normalization": "none",
        },
        "samples": len(review_rows),
        "source_differences": {
            "transkun_only_piano_events": len(added),
            "kong_only_piano_events": len(removed),
        },
        "participant_archive_sha256": sha256(archive_path),
        "deblind_key_outside_archive": True,
        "reference_chart_read": False,
        "model_inference_run": False,
        "participant_package_model_labels": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_package(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
