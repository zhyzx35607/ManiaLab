"""Describe section-level LN and interaction patterns in the Sakura charts.

This is a chart-expression audit, not an audio transcription or quality score.
Only the Reference chart has the author-confirmed source-to-lane mapping.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manialab.domain import parse_osu


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "test-data" / "sakura-benchmark"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "expression-profile-v2" / "report.json"


def redlines(timing_points):
    points = {}
    for line in timing_points:
        fields = line.split(",")
        try:
            offset = float(fields[0])
            period = float(fields[1])
            is_red = len(fields) < 7 or fields[6].strip() == "1"
        except (ValueError, IndexError):
            continue
        if math.isfinite(offset) and math.isfinite(period) and period > 0:
            points[offset] = period
    ordered = sorted(points.items())
    return [row[0] for row in ordered], [row[1] for row in ordered]


def duration_beats(start_ms, end_ms, timing_points):
    """Integrate a chart interval over its red-line beat periods."""
    if end_ms <= start_ms:
        return 0.0
    offsets, periods = redlines(timing_points)
    if not offsets:
        return (end_ms - start_ms) / 500.0

    cursor = float(start_ms)
    end_ms = float(end_ms)
    beats = 0.0
    while cursor < end_ms:
        index = max(0, bisect_right(offsets, cursor) - 1)
        stop = end_ms
        if index + 1 < len(offsets):
            stop = min(stop, offsets[index + 1])
        beats += (stop - cursor) / periods[index]
        cursor = stop
    return beats


def profile_segment(notes, keys, timing_points, start_ms, end_ms, lane_sources=None):
    segment_notes = [note for note in notes if start_ms <= note.start < end_ms]
    holds = [note for note in segment_notes if note.hold]
    taps = [note for note in segment_notes if not note.hold]

    groups_by_start = {}
    for note in segment_notes:
        groups_by_start.setdefault(note.start, []).append(note)
    onset_groups = list(groups_by_start.values())
    chord_groups = [group for group in onset_groups if len(group) > 1]
    hold_chords = [
        group for group in onset_groups
        if sum(note.hold for note in group) > 1
    ]
    staggered_hold_chords = [
        group for group in hold_chords
        if len({note.end for note in group if note.hold}) > 1
    ]

    hybrid_taps = [
        note for note in taps
        if any(
            hold.lane != note.lane and hold.start < note.start < hold.end
            for hold in notes
            if hold.hold
        )
    ]
    intersecting_holds = [
        note for note in notes
        if note.hold and note.start < end_ms and note.end > start_ms
    ]
    occupancy_events = []
    for note in intersecting_holds:
        occupancy_events.extend((
            (max(start_ms, note.start), 1),
            (min(end_ms, note.end), -1),
        ))
    active = peak_held = 0
    for _, change in sorted(occupancy_events, key=lambda item: (item[0], item[1])):
        active += change
        peak_held = max(peak_held, active)

    lengths = [
        duration_beats(note.start, note.end, timing_points)
        for note in holds
    ]
    result = {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "objects": len(segment_notes),
        "onset_groups": len(onset_groups),
        "chord_groups": len(chord_groups),
        "chord_group_fraction": len(chord_groups) / len(onset_groups) if onset_groups else 0.0,
        "ln": len(holds),
        "ln_fraction": len(holds) / len(segment_notes) if segment_notes else 0.0,
        "ln_median_beats": median(lengths) if lengths else None,
        "ln_p90_beats": sorted(lengths)[math.ceil(0.9 * len(lengths)) - 1] if lengths else None,
        "ln_max_beats": max(lengths, default=None),
        "ln_at_least_2_beats": sum(length >= 2.0 for length in lengths),
        "ln_at_least_4_beats": sum(length >= 4.0 for length in lengths),
        "ln_chord_groups": len(hold_chords),
        "staggered_ln_chord_groups": len(staggered_hold_chords),
        "staggered_ln_chord_fraction": (
            len(staggered_hold_chords) / len(hold_chords) if hold_chords else 0.0
        ),
        "taps_during_other_lane_ln": len(hybrid_taps),
        "taps_during_other_lane_ln_fraction": (
            len(hybrid_taps) / len(taps) if taps else 0.0
        ),
        "peak_held": peak_held,
        "ln_crossing_segment_end": sum(note.end > end_ms for note in holds),
        "lane_objects": [
            sum(note.lane == lane for note in segment_notes)
            for lane in range(keys)
        ],
        "lane_ln": [
            sum(note.lane == lane and note.hold for note in segment_notes)
            for lane in range(keys)
        ],
    }
    if lane_sources is not None:
        result["source_groups"] = {
            source: {
                "objects": sum(note.lane + 1 in lanes for note in segment_notes),
                "ln": sum(note.lane + 1 in lanes and note.hold for note in segment_notes),
            }
            for source, lanes in lane_sources.items()
        }
    return result


def build_profile(source_root=DEFAULT_SOURCE):
    source_root = Path(source_root)
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    chart_records = manifest["charts"]
    source_lanes = manifest["lane_sources"]
    duration_ms = round(
        manifest["audio"][0]["samples"] / manifest["audio"][0]["sample_rate"] * 1000
    )
    bookmarks = chart_records["reference"]["marks"]["bookmarks"]
    boundaries = [0, *bookmarks, duration_ms]
    if boundaries != sorted(set(boundaries)):
        raise ValueError("共同书签必须严格递增且不重复")

    charts = {}
    for role in ("reference", "finished"):
        record = chart_records[role]
        path = source_root / f"{role}.osu"
        chart_bytes = path.read_bytes()
        sha256 = hashlib.sha256(chart_bytes).hexdigest()
        if sha256 != record["chart_sha256"]:
            raise ValueError(f"{role} 谱面 SHA256 与冻结清单不符")
        chart = parse_osu(chart_bytes.decode("utf-8-sig"))
        if int(chart["metadata"].get("BeatmapID", "0")) != record["beatmap_id"]:
            raise ValueError(f"{role} BeatmapID 与冻结清单不符")
        if chart["keys"] != 10:
            raise ValueError(f"{role} 不是 10K 谱")
        charts[role] = {
            "beatmap_id": record["beatmap_id"],
            "version": record["version"],
            "sha256": sha256,
            "objects": len(chart["notes"]),
            "ln": sum(note.hold for note in chart["notes"]),
            "ln_fraction": sum(note.hold for note in chart["notes"]) / len(chart["notes"]),
            "timing_points": chart["timing"],
            "notes": chart["notes"],
        }

    segments = []
    for start_ms, end_ms in zip(boundaries, boundaries[1:]):
        segment = {"start_ms": start_ms, "end_ms": end_ms}
        for role, chart in charts.items():
            lane_map = source_lanes if role == "reference" else None
            segment[role] = profile_segment(
                chart["notes"], 10, chart["timing_points"],
                start_ms, end_ms, lane_sources=lane_map,
            )
        segments.append(segment)

    return {
        "schema": "sakura-expression-profile-v2",
        "purpose": (
            "Describe section-level chart expression. Does not establish audio "
            "support, instrument identity in the finished chart, or quality."
        ),
        "beatmapset_id": manifest["beatmapset_id"],
        "bookmarks_ms": bookmarks,
        "duration_ms": duration_ms,
        "reference_source_mapping": source_lanes,
        "charts": {
            role: {key: value for key, value in chart.items()
                   if key not in ("notes", "timing_points")}
            for role, chart in charts.items()
        },
        "finished_source_mapping": None,
        "segments": segments,
        "limitations": [
            "Reference source labels are author-confirmed; they are not complete MIDI truth.",
            "Finished 10K lanes are not assigned instrument identities.",
            "Section statistics describe one chart and are not a universal style rule.",
            "A chart LN endpoint is not, by itself, evidence of audible sustain.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    report = build_profile(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(
        "Reference: "
        f"{report['charts']['reference']['objects']} objects, "
        f"{report['charts']['reference']['ln']} LN; Finished: "
        f"{report['charts']['finished']['objects']} objects, "
        f"{report['charts']['finished']['ln']} LN; "
        f"{len(report['segments'])} shared-bookmark sections."
    )


if __name__ == "__main__":
    main()
