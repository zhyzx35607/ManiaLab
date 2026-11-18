"""Check editing, reopening and exporting each difficulty of a local mania package."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.domain import parse_osu, validate
from manialab.server import create_app


def checked(response):
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.get_json(silent=True)}")
    return response


def run_job(client, base, headers, action, payload):
    job = checked(client.post(f"{base}/{action}", headers=headers, json=payload)).json
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        job = checked(client.get("/api/jobs/" + job["id"])).json
        if job["status"] == "done":
            return checked(client.get(base)).json
        if job["status"] not in ("queued", "running"):
            raise RuntimeError(f"{action}: {job['message']}")
        time.sleep(.1)
    raise TimeoutError(action)


def change_lane(variant, keys):
    notes = variant["notes"]
    for note in notes:
        occupied = {n["lane"] for n in notes if n["start"] == note["start"] or
                    (n["start"] < note["end"] and n["end"] > note["start"])}
        for lane in range(keys):
            if lane != note["lane"] and lane not in occupied:
                return {"op": "update", "base_created": variant["created"],
                        "note_id": note["id"], "values": {"lane": lane}}
    raise ValueError("No legal lane change in this difficulty")


def variants_digest(state):
    encoded = json.dumps(state["variants"], sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def verify(package, expected_sha, out):
    if sha256(package.read_bytes()).hexdigest() != expected_sha.lower():
        raise ValueError("Source package SHA-256 mismatch")
    out.mkdir(parents=True, exist_ok=False)
    saved = {}
    app = create_app(ROOT, out)
    try:
        client = app.test_client()
        headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
        with package.open("rb") as source:
            state = checked(client.post("/api/import", data={"files": (source, "input.osz")}, headers=headers)).json
        base = "/api/projects/" + state["id"]
        for chart in state["charts"]:
            name = chart["path"]
            state = checked(client.post(base + "/chart", headers=headers, json={"path": name})).json
            keys = state["input_keys"]
            generated = run_job(client, base, headers, "generate", {"mode": "remap", "settings": {"keys": keys}})
            original = generated["variants"][generated["active"]]
            edit = change_lane(original, keys)
            edited = run_job(client, base, headers, "edit", edit)
            version = edited["variants"][edited["active"]]
            assert next(n for n in version["notes"] if n["id"] == edit["note_id"])["lane"] == edit["values"]["lane"]
            assert edited["active"] == 1 and edited["variants"][0] == original
            saved[name] = variants_digest(edited)
            print(f"Saved {name}", flush=True)
    finally:
        app.extensions["executor"].shutdown()

    rows = []
    app = create_app(ROOT, out)
    try:
        client = app.test_client()
        headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
        with ZipFile(package) as source:
            for name in reversed(saved):
                state = checked(client.post(base + "/chart", headers=headers, json={"path": name})).json
                assert state["active"] == 1 and variants_digest(state) == saved[name]
                original = parse_osu(source.read(name).decode("utf-8-sig"))
                plain = checked(client.get(base + "/export/osu")).data
                parsed = parse_osu(plain.decode("utf-8-sig"))
                assert parsed["keys"] == original["keys"] and parsed["timing"] == original["timing"]
                assert Counter(n.raw.partition(",")[2] for n in original["notes"]) == Counter(
                    n.raw.partition(",")[2] for n in parsed["notes"])
                expected = state["variants"][1]["notes"]
                assert sorted((n.start, n.end, n.lane) for n in parsed["notes"]) == sorted(
                    (n["start"], n["end"], n["lane"]) for n in expected)
                report = validate(parsed["notes"], parsed["keys"])
                assert report["valid"], report["issues"]
                with ZipFile(BytesIO(checked(client.get(base + "/export/osz")).data)) as exported:
                    assert exported.testzip() is None
                    entry, = (p for p in exported.namelist() if p.lower().endswith(".osu"))
                    assert exported.read(entry) == plain
                    audio = PurePosixPath(entry).parent / parsed["general"]["AudioFilename"].replace("\\", "/")
                    assert sha256(exported.read(str(audio))).digest() == sha256(source.read(state["audio"])).digest()
                rows.append(dict(chart=name, keys=parsed["keys"], notes=report["notes"], ln=report["ln"],
                                 variants=2, variants_sha256=saved[name], osu_sha256=sha256(plain).hexdigest()))
    finally:
        app.extensions["executor"].shutdown()
    (out / "report.json").write_text(json.dumps(dict(package_sha256=expected_sha, charts=rows), indent=2),
                                     encoding="utf-8")
    print(f"Passed {len(rows)} difficulties", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.package, args.sha256, args.out)
