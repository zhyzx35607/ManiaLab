"""Compare ppy's decoded mania objects and timing with ManiaLab's OSZ reader."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import posixpath
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from manialab.domain import parse_osu


def check(condition, label):
    if not condition:
        raise ValueError(label)


def digest_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update((",".join(str(cell) for cell in row) + "\n").encode("ascii"))
    return digest.hexdigest()


def timing_records(lines):
    red, speeds = {}, {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        fields = line.split(",")
        time, period = map(float, fields[:2])
        meter = int(fields[2]) if len(fields) > 2 and int(fields[2]) > 0 else 4
        inherited = len(fields) >= 7 and fields[6].strip() == "0"
        if not inherited:
            red[time] = (period, meter)
            speeds.setdefault(time, 1.0)
        else:
            speeds[time] = min(10, max(.1, -100 / period)) if period < 0 else 1.0
    return red, speeds


def compare_timing(lines, official):
    expected_red, expected_velocities = timing_records(lines)
    actual_red = {p["time"]: (p["period"], p["meter"]) for p in official["timing"]}
    check(expected_red.keys() == actual_red.keys(), "Red line positions differ")
    for time, (period, meter) in expected_red.items():
        actual_period, actual_meter = actual_red[time]
        check(math.isclose(period, actual_period, rel_tol=1e-10, abs_tol=1e-8) and meter == actual_meter,
              f"Red line differs at {time} ms")
    # The game omits consecutive identical velocities, including the initial default of 1x.
    actual_velocities = {p["time"]: p["value"] for p in official["velocities"]}
    actual = expected = 1.0
    for time in sorted(expected_velocities.keys() | actual_velocities.keys()):
        actual = actual_velocities.get(time, actual)
        expected = expected_velocities.get(time, expected)
        check(math.isclose(expected, actual, rel_tol=1e-9), f"Scroll velocity differs at {time} ms")


def compare(official):
    totals = Counter()
    summaries = []
    for archive in official["archives"]:
        path = Path(archive["path"])
        with path.open("rb") as source:
            check(hashlib.file_digest(source, "sha256").hexdigest() == archive["sha256"], str(path))
        difficulties = []
        with ZipFile(path) as bundle:
            check(bundle.testzip() is None, f"{path}: CRC")
            names = bundle.namelist()
            check(len(names) == len(set(names)), f"{path}: duplicate archive name")
            entries = sorted(name for name in names if name.lower().endswith(".osu"))
            check(entries == sorted(entry["entry"] for entry in archive["difficulties"]),
                  f"{path}: difficulty entries differ")
            for entry in archive["difficulties"]:
                name = entry["entry"]
                chart = parse_osu(bundle.read(name).decode("utf-8-sig"))
                rows = sorted((n.start, n.end, n.lane, "ln" if n.hold else "tap") for n in chart["notes"])
                check(entry["mode"] == "mania" and entry["keys"] == chart["keys"], name + ": ruleset or keys")
                check(len(rows) == entry["objects"] and sum(n.hold for n in chart["notes"]) == entry["holds"],
                      name + ": object type counts")
                check(digest_rows(rows) == entry["rows_sha256"], name + ": lane, type or note endpoints")
                check(digest_rows((a, b, kind) for a, b, _, kind in rows) == entry["endpoints_sha256"],
                      name + ": timing or type")
                check(chart["general"]["AudioFilename"] == entry["audio"]
                      and chart["metadata"]["Version"] == entry["version"], name + ": audio or difficulty metadata")
                audio_name = posixpath.normpath(posixpath.join(posixpath.dirname(name), entry["audio"].replace("\\", "/")))
                with bundle.open(audio_name) as sound:
                    audio_sha = hashlib.file_digest(sound, "sha256").hexdigest()
                compare_timing(chart["timing"], entry)
                summary = dict(entry, audio_sha256=audio_sha, endpoints=len(rows) + entry["holds"])
                difficulties.append(summary)
                totals.update(difficulties=1, objects=len(rows), holds=entry["holds"], endpoints=summary["endpoints"],
                              red_lines=len(entry["timing"]), velocity_changes=len(entry["velocities"]))
        summaries.append(dict(archive=path.name, sha256=archive["sha256"], difficulties=difficulties))
    return dict(game=official["game"], ruleset=official["ruleset"], packages=len(summaries),
                totals=dict(totals), archives=summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_report", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    source = args.official_report.read_bytes()
    summary = compare(json.loads(source))
    summary["official_report_sha256"] = hashlib.sha256(source).hexdigest()
    with args.summary.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(summary["totals"]))
