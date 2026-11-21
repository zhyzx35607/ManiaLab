"""Exercise the public project APIs against two fixed external music excerpts."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REVISION = "5285c94e33c74bbe97501f5c9fa95e5e2ce55d48"
RECORDINGS = {
    "sweet-waltz": (
        "147793__setuniman__sweet-waltz-0i-22mi",
        "248f93b7c729bc0bd6c4cbc107e6b67ec4d60e40c111e313d4d69f77748da26a",
    ),
    "choice": (
        "admiralbob77_-_Choice_-_Drum-bass",
        "57b4d95473b92a8441c1d3ab20f836a3e0cbf501bf3afe1ce7a5d0d98d7d4576",
    ),
}
PROTOCOL = dict(
    upstream="librosa/data", revision=REVISION, sources=RECORDINGS,
    excerpt="First 24 seconds, or complete audio if shorter. Stereo PCM16 at 44100 Hz.",
    engine="deep", voices=["drums", "bass", "other", "vocals"], beat_model="Beat This!",
    intra_threads=2, inter_threads=1, parallel_inference_jobs=1,
    keys=[6, 7, 8, 9, 10], settings="Application defaults", seed=42, audio_shift_ms=22.5,
    checks=["Record BPM, confidence, stability, voice/section counts and the timing-check response.",
            "Try 6K without explicit confirmation. Unstable timing must reject without saving a candidate.",
            "If rejected, use the unchanged estimated grid with a harness override only to test export.",
            "Every keycount must retain a nonempty valid chart, zero off-grid endpoints, "
            "intact audio, and identical lanes and note endpoints after OSZ reimport.",
            "Render an eight-second beat audition and chart audition from each project."],
    pending=["Human listening for notes, voice balance, structure and grid alignment",
             "Independent osu! player assessment for each keycount and LN release"],
)


def check(condition, description):
    if not condition:
        raise ValueError(description)


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download(relative):
    path = f"repos/librosa/data/contents/{relative}?ref={REVISION}"
    item = json.loads(subprocess.check_output(["gh", "api", path], timeout=120))
    if item["encoding"] != "base64":
        item = json.loads(subprocess.check_output(["gh", "api", item["git_url"]], timeout=120))
    check(item["encoding"] == "base64", f"Missing Base64 GitHub contents for {relative}")
    return base64.b64decode(item["content"])


def prepare(work):
    import imageio_ffmpeg

    protocol = work / "protocol.json"
    if work.exists():
        check(protocol.is_file() and read_json(protocol) == json.loads(json.dumps(PROTOCOL))
              and not (work / "sources.json").exists(), "Choose a new output directory")
    else:
        work.mkdir(parents=True)
        write_json(protocol, PROTOCOL)
    registry_path = work / "registry.txt"
    registry = download("index/registry.txt")
    if not registry_path.exists():
        registry_path.write_bytes(registry)
    check(registry_path.read_bytes() == registry, "Upstream registry differs from saved registry")
    registered = dict(line.split() for line in registry.decode().splitlines() if line.strip())
    locked = {}
    for name, (stem, expected) in RECORDINGS.items():
        folder = work / name
        folder.mkdir(exist_ok=True)
        source_name = stem + ".hq.ogg"
        check(registered[source_name] == expected, f"{name}: upstream registry hash changed")
        for filename in (source_name, stem + ".txt", stem + ".toml"):
            path = folder / filename
            raw = path.read_bytes() if path.exists() else download("audio/" + filename)
            check(digest(raw) == registered[filename], f"{filename}: upstream checksum mismatch")
            if not path.exists():
                path.write_bytes(raw)
        clip = folder / (name + ".wav")
        if not clip.exists():
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-n",
                            "-i", str(folder / source_name), "-t", "24", "-vn", "-ac", "2",
                            "-ar", "44100", "-c:a", "pcm_s16le", str(clip)], check=True, timeout=60)
        with wave.open(str(clip)) as audio:
            check(audio.getnchannels() == 2 and audio.getframerate() == 44100 and
                  0 < audio.getnframes() <= 44100 * 24, f"{name}: clip geometry")
            duration = audio.getnframes() / audio.getframerate()
        locked[name] = dict(file=clip.name, source_sha256=expected, clip_sha256=digest(clip.read_bytes()),
                            start_seconds=0, duration_seconds=duration)
    write_json(work / "sources.json", locked)
    print(json.dumps(locked, indent=2), flush=True)


def successful(response):
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}: {response.get_data(as_text=True)[:300]}")
    return response


def variant_selection_payload(saved, index):
    candidate_id = saved["variants"][index].get("candidate_id")
    check(isinstance(candidate_id, str) and candidate_id,
          "Saved candidate has no persistent identity")
    return {"index": index, "candidate_id": candidate_id}


def task(client, url, payload, headers):
    request = successful(client.post(url, json=payload, headers=headers)).get_json()
    last = -1
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        result = successful(client.get("/api/jobs/" + request["id"])).get_json()
        if result["status"] in ("done", "error", "cancelled"):
            return result
        if result["progress"] // 10 > last:
            last = result["progress"] // 10
            print(f"{url.rsplit('/', 1)[-1]}: {result['progress']}% {result['message']}", flush=True)
        time.sleep(.25)
    successful(client.post(f"/api/jobs/{request['id']}/cancel", json={}, headers=headers))
    raise TimeoutError(f"{url}: exceeded 15 minutes")


def imported(client, headers, filename, data):
    return successful(client.post("/api/import", headers=headers,
                                  data={"files": (io.BytesIO(data), filename)})).get_json()


def signature(notes):
    return sorted((n["start"], n["end"], n["lane"]) for n in notes)


def run_case(client, headers, work, name, locked):
    from manialab.domain import note_from, parse_osu, validate
    from manialab.timing import grid_report

    folder = work / name
    clip = folder / locked["file"]
    sound = clip.read_bytes()
    check(digest(sound) == locked["clip_sha256"], name + ": clip changed")
    id_path = folder / "project.json"
    if not id_path.is_file():
        project = imported(client, headers, clip.name, sound)
        write_json(id_path, {"id": project["id"]})
    project_id = read_json(id_path)["id"]
    base = "/api/projects/" + project_id
    state = lambda: successful(client.get(base)).get_json()
    input_state = state()
    if not input_state.get("analysis"):
        result = task(client, base + "/analyze", {"engine": "deep", "audio_shift_ms": 22.5}, headers)
        check(result["status"] == "done", result["message"])
    analysis = state()["analysis"]
    check(analysis["separation"]["profile"] == "four" and sorted(v["voice"] for v in analysis["voices"])
          == sorted(PROTOCOL["voices"]), "Expected the four-voice separator")
    tempo = analysis["tempo"]
    clock = {"audio_shift_ms": 22.5}
    diagnostic = successful(client.post(base + "/timing-check", headers=headers, json=clock)).get_json()
    payload = dict(clock, snap=True, mode="audio", timing_confirmed=False, settings={"keys": 6})
    gate_file = folder / "automatic-gate.json"
    if not gate_file.is_file():
        before = len(state()["variants"])
        attempt = task(client, base + "/generate", payload, headers)
        accepted = attempt["status"] == "done"
        check(accepted == bool(tempo["stable"]), f"{name}: automatic timing gate disagrees with its status")
        if not accepted:
            check("Timing" in attempt["message"] and len(state()["variants"]) == before,
                  "Unconfirmed run must reject the clock without persisting notes")
        write_json(gate_file, dict(accepted=accepted, status=attempt["status"], message=attempt["message"]))
    gate = read_json(gate_file)
    keys_report = {}
    for keys in PROTOCOL["keys"]:
        saved = state()
        index = next((i for i, variant in enumerate(saved["variants"]) if variant["settings"]["keys"] == keys), None)
        if index is None:
            payload.update(settings={"keys": keys}, timing_confirmed=not gate["accepted"])
            result = task(client, base + "/generate", payload, headers)
            check(result["status"] == "done", f"{name}-{keys}K: {result['message']}")
        elif index != saved["active"]:
            successful(client.post(base + "/variant", headers=headers,
                                   json=variant_selection_payload(saved, index)))
        saved = state()
        variant = saved["variants"][saved["active"]]
        notes = variant["notes"]
        expected = signature(notes)
        check(variant["settings"]["keys"] == keys and variant["report"]["valid"] and expected,
              f"{name}-{keys}: invalid or empty chart")
        archive_path = folder / f"{name}-{keys}K.osz"
        # A resumed check must validate the exact bytes that the listener receives.
        bundle = archive_path.read_bytes() if archive_path.is_file() else successful(client.get(base + "/export/osz")).data
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            check(archive.testzip() is None, "Invalid archive CRC")
            chart_name, = (s for s in archive.namelist() if s.endswith(".osu"))
            parsed = parse_osu(archive.read(chart_name).decode("utf-8-sig"))
            sound_name = parsed["general"]["AudioFilename"]
            check(archive.read(sound_name) == sound, "Archived audio differs")
            red, = (line.split(",") for line in parsed["timing"] if line.strip() and line.split(",")[6] == "1")
            roundtrip, chart_keys = parsed["notes"], parsed["keys"]
            grid = grid_report(roundtrip, dict(bpm=60000 / float(red[1]), offset=float(red[0]),
                                               division=variant["division"]))
            check(keys == chart_keys and grid["off_grid"] == 0 and validate(roundtrip, keys)["valid"] and
                  signature([n.data() for n in roundtrip]) == expected, "Saved notes differ after parsing")
        if not archive_path.exists():
            with archive_path.open("xb") as stream:
                stream.write(bundle)
        reopened = imported(client, headers, archive_path.name, bundle)
        check(reopened["input_keys"] == keys and signature(reopened["input_notes"]) == expected and
              validate([note_from(n) for n in reopened["input_notes"]], keys)["valid"], "OSZ reimport mismatch")
        keys_report[keys] = dict(notes=len(notes), ln=variant["report"]["ln"], grid=grid,
                                 max_held=variant["report"]["max_held"], lanes=variant["report"]["lane_counts"],
                                 selected_by_voice=dict(Counter(n["voice"] for n in notes)),
                                 selected_ln_by_voice=dict(Counter(n["voice"] for n in notes if n["end"] > n["start"])),
                                 export=archive_path.name, export_sha256=digest(bundle), reimport_id=reopened["id"])
    for mode in ("beats", "chart"):
        response = successful(client.post(base + "/audition", headers=headers, json=dict(
            clock, mode=mode, start_ms=0, duration_ms=8000, beat_clock="fixed")))
        with wave.open(io.BytesIO(response.data)) as preview:
            check(preview.getnframes() == 8 * preview.getframerate() and preview.getnchannels() == 2,
                  "Audition duration/channel count mismatch")
        path = folder / (mode + "-audition.wav")
        if not path.is_file():
            path.write_bytes(response.data)
    summary = dict(project_id=project_id, clip=locked, engine=analysis["engine"], cached=analysis["cached"],
                   analysis_seconds=analysis["elapsed_seconds"], voices=analysis["voices"],
                   tempo={key: tempo.get(key) for key in ("bpm", "offset", "division", "confidence", "stable",
                                                          "evidence_source", "reliability")},
                   timing_check=diagnostic, gate=gate, clock_source="automatic" if gate["accepted"]
                   else "unchanged estimate, explicitly enabled by harness for engineering checks only",
                   sections=[(s["start_ms"], s["end_ms"]) for s in analysis["structure"]["segments"]],
                   charts=keys_report, human_listening=False, player_playtest=False)
    write_json(folder / "report.json", summary)
    return summary


def verify(work):
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "2"
    from manialab.server import create_app
    from manialab.separation import BEAT_MODEL_SHA256, MODEL_SHA256

    check(read_json(work / "protocol.json") == json.loads(json.dumps(PROTOCOL)), "Protocol changed")
    sources = read_json(work / "sources.json")
    deep = ROOT / "assets/deep"
    check(digest((deep / "hdemucs_high_trained.pt").read_bytes()) == MODEL_SHA256, "Wrong Demucs checkpoint")
    check(digest((deep / "beat-this-final0.ckpt").read_bytes()) == BEAT_MODEL_SHA256, "Wrong beat checkpoint")
    runtime = ROOT / ".venv-audio/Scripts/python.exe"
    if runtime.is_file():
        os.environ["MANIALAB_DEEP_PYTHON"] = str(runtime)
    app = create_app(ROOT, work / "runtime")
    try:
        client = app.test_client()
        headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
        summaries = {}
        for name in RECORDINGS:
            path = work / name / "report.json"
            if path.exists():
                summaries[name] = read_json(path)
            else:
                print(f"Starting {name}", flush=True)
                summaries[name] = run_case(client, headers, work, name, sources[name])
        write_json(work / "report.json", dict(protocol_sha256=digest((work / "protocol.json").read_bytes()),
                    basic_pitch_sha256=digest((ROOT / "assets/basic-pitch.onnx").read_bytes()),
                    demucs_sha256=MODEL_SHA256, beat_sha256=BEAT_MODEL_SHA256,
                    songs=summaries, pending=PROTOCOL["pending"]))
    finally:
        app.extensions["executor"].shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify"))
    parser.add_argument("work", type=Path, help="New local directory for the protocol, sources and exports")
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else verify)(args.work.resolve())
