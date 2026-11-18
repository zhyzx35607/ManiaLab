"""Build a local blind comparison of the two production lane-assignment modes."""
import argparse
from collections import Counter
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from secrets import SystemRandom
import shutil
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.arrange import assign_lanes
from manialab.domain import export_osu, parse_osu, sections, validate
from manialab.storage import AUDIO, extract_osz, safe_path


def object_fields(notes):
    return Counter(n.raw.partition(",")[2] for n in notes)


def build(package, expected_sha, chart, out):
    if sha256(package.read_bytes()).hexdigest() != expected_sha.lower():
        raise ValueError("Input package SHA-256 mismatch")
    out.mkdir(parents=True, exist_ok=False)
    player = out / "participant"
    player.mkdir()
    archive = player / "Layout-AB-6-10K.osz"
    keys = list(range(6, 11))
    shuffle = SystemRandom()
    beam_first = set(shuffle.sample(keys, 2 + shuffle.randrange(2)))
    play_b_first = set(shuffle.sample(keys, 2 + shuffle.randrange(2)))
    evidence = {}
    with TemporaryDirectory() as temp:
        base = Path(temp)
        with package.open("rb") as stream:
            extract_osz(stream, base)
        path = safe_path(base, chart)
        text = path.read_text(encoding="utf-8-sig")
        original = parse_osu(text)
        audio = safe_path(base, (Path(chart).parent / original["general"]["AudioFilename"]).as_posix())
        if not audio.is_file() or audio.suffix.lower() not in AUDIO:
            raise ValueError("Source chart must have a playable audio reference")
        data = sections(text)
        # Give the player a separate osu! set from the source comparison.
        data["Metadata"] = [line for line in data["Metadata"] if
                            line.partition(":")[0].strip() not in ("Title", "TitleUnicode")]
        title = original["metadata"].get("Title", "ManiaLab") + " [Layout A-B]"
        data["Metadata"].extend(["Title:" + title, "TitleUnicode:" + title])
        common = "\n".join("\n".join(([f"[{key}]"] if key else []) + value) for key, value in data.items())
        with ZipFile(archive, "x", ZIP_DEFLATED) as zipped:
            for asset in base.rglob("*"):
                if asset.is_file() and asset.suffix.lower() != ".osu":
                    zipped.write(asset, asset.relative_to(base).as_posix())
            for k in keys:
                selected = [replace(n, lane=min(k - 1, int((n.lane + .5) * k / original["keys"])))
                            for n in original["notes"]]
                engines = ("beam", "classic") if k in beam_first else ("classic", "beam")
                signatures = []
                for label, engine in zip(("A", "B"), engines):
                    notes = assign_lanes(selected, {"keys": k, "seed": 42, "layout_engine": engine,
                                                   "pattern_prior": 0}, timing_points=original["timing"])
                    chart_text = export_osu(notes, k, ROOT / "vendor/mapperatorinator/template.osu",
                                            original=common, version="Layout " + label)
                    restored = parse_osu(chart_text)
                    if (restored["timing"] != original["timing"] or object_fields(restored["notes"])
                            != object_fields(original["notes"]) or not validate(restored["notes"], k)["valid"]):
                        raise ValueError("Layout changed the source timing, objects or capacity")
                    signatures.append(Counter((n.lane, n.raw.partition(",")[2]) for n in restored["notes"]))
                    zipped.writestr((Path(chart).parent / f"Layout-{k}K-{label}.osu").as_posix(), chart_text)
                changed = sum((signatures[0] - signatures[1]).values())
                if not changed:
                    raise ValueError(f"The two {k}K layouts are identical")
                evidence[k] = {"A": engines[0], "B": engines[1], "changed_objects": changed}
    bundle_sha = sha256(archive.read_bytes()).hexdigest()
    with ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Playtest archive CRC mismatch")
    with (player / "scores.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["osz_sha256", "keys", "play_first", "play_second", "preferred_A_B_equal_unsure",
                         "release_coordination", "hand_balance", "time_ms", "comment"])
        for k in keys:
            first, second = ("B", "A") if k in play_b_first else ("A", "B")
            writer.writerow([bundle_sha, k, first, second, "", "", "", "", ""])
    shutil.copyfile(ROOT / "docs/layout-playtest.md", player / "playtest.md")
    (out / "deblind-after-scoring.json").write_text(json.dumps(dict(
        input_sha256=expected_sha, input_chart=chart, input_keys=original["keys"], notes=len(original["notes"]),
        ln=sum(n.hold for n in original["notes"]), osz_sha256=bundle_sha, seed=42, pattern_prior=0, pairs=evidence),
        ensure_ascii=False, indent=2), encoding="utf-8")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--chart", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.package, args.sha256, args.chart, args.out))
