import io
from pathlib import Path
import zipfile

import numpy as np
import pytest
from scipy.io import wavfile

from manialab.audition import render_audition
from manialab.domain import Note, parse_osu
from manialab.meter import detected_meter, meter_for_clock
from manialab.server import create_app
from tests.test_server import wait_job


ROOT = Path(__file__).resolve().parents[1]


def analysis(meter=3, bars=6):
    beats = [115 + i * 400 for i in range(meter * bars + 1)]
    return dict(revision=3, duration=(beats[-1] + 50) / 1000,
                tempo=dict(bpm=150, offset=115, division=4, stable=True),
                separation=dict(beat_evidence=dict(beats_ms=beats, downbeats_ms=beats[::meter])))


@pytest.mark.parametrize("meter", [2, 3, 4, 5, 7])
def test_meter_uses_measured_bars_and_matches_the_shifted_red_line(meter):
    result = analysis(meter)
    evidence = result["separation"]["beat_evidence"]
    evidence["beats_ms"] = [t + (-1)**i * 14 for i, t in enumerate(evidence["beats_ms"])]
    evidence["downbeats_ms"] = evidence["beats_ms"][::meter]
    assert detected_meter(result) == meter
    assert meter_for_clock(result, dict(bpm=150, offset=93, audio_shift_ms=22.5)) == meter
    assert meter_for_clock(result, dict(bpm=150, offset=153, audio_shift_ms=-37.5)) == meter
    assert meter_for_clock(result, dict(bpm=150, offset=115, audio_shift_ms=22.5)) == 4
    assert meter_for_clock(result, dict(bpm=75, offset=93, audio_shift_ms=22.5)) == 4


@pytest.mark.parametrize(("audio_offset", "shift", "chart_offset"), [
    (115, 22.5, 93),
    (50, 99.5, -49),
])
def test_meter_clock_matches_chart_rounding_at_positive_and_negative_half_ms(audio_offset, shift, chart_offset):
    result = analysis()
    beats = [audio_offset + i * 400 for i in range(19)]
    result["tempo"]["offset"] = audio_offset
    result["duration"] = (beats[-1] + 50) / 1000
    result["separation"]["beat_evidence"] = dict(beats_ms=beats, downbeats_ms=beats[::3])
    assert meter_for_clock(result, dict(bpm=150, offset=chart_offset, audio_shift_ms=shift)) == 3


@pytest.mark.parametrize("defect", [
    "one-bar-late", "other-phase", "short", "new-meter", "missing-pulse", "off-grid",
    "wrong-pulse", "short-excerpt", "unstable", "old-drums", "no-evidence", "nonfinite",
])
def test_ambiguous_or_incomplete_evidence_retains_four_beats(defect):
    result = analysis()
    evidence = result["separation"]["beat_evidence"]
    if defect == "one-bar-late":
        evidence["downbeats_ms"] = evidence["downbeats_ms"][1:]
    elif defect == "other-phase":
        evidence["downbeats_ms"] = evidence["beats_ms"][1::3]
    elif defect == "short":
        result = analysis(bars=2)
    elif defect == "new-meter":
        evidence["downbeats_ms"][-2] += 400
    elif defect == "missing-pulse":
        evidence["beats_ms"].pop(7)
    elif defect == "off-grid":
        evidence["beats_ms"][5] += 40
    elif defect == "wrong-pulse":
        evidence["beats_ms"][3] -= 22
        evidence["downbeats_ms"][1] += 22
    elif defect == "short-excerpt":
        result["duration"] *= 10
    elif defect == "unstable":
        result["tempo"]["stable"] = False
    elif defect == "old-drums":
        result["tempo"]["evidence_source"] = "drums"
    elif defect == "no-evidence":
        del result["separation"]
    elif defect == "nonfinite":
        evidence["downbeats_ms"][0] = float("nan")
    assert detected_meter(result) == 4


@pytest.mark.parametrize("tempo", [None, {}, {"bpm": 150}, {"bpm": 150, "offset": None},
                                   {"bpm": 150, "offset": float("nan")}, {"bpm": float("inf"), "offset": 0}])
def test_partial_legacy_clock_uses_four_even_when_its_bpm_matches(tempo):
    data = analysis() | {"tempo": tempo}
    assert detected_meter(data) == 4
    assert meter_for_clock(data, {"bpm": 150, "offset": 93, "audio_shift_ms": 22.5}) == 4


def test_fixed_meter_survives_edit_reanalysis_selection_region_and_both_export_formats(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    projects = app.extensions["projects"]
    try:
        imported = client.post("/api/import", headers=headers,
                               data={"files": (io.BytesIO(b"fixture audio"), "song.wav")}).json
        pid = imported["id"]
        base = "/api/projects/" + pid
        state = projects.load(pid)
        notes = [Note(515 + i*400, 715 + i*400, source="pitch", id=f"n{i}").data() for i in range(14)]
        state.update(audio_shift_ms=22.5, analysis=analysis() | {"candidates": notes, "timing_attacks": notes})
        projects.save(state)
        preview = projects.path(pid) / "preview.wav"
        wavfile.write(preview, 8000, np.zeros(8000 * 8, np.int16))

        def run(action, payload):
            submitted = client.post(base + "/" + action, json=payload, headers=headers)
            assert submitted.status_code == 200
            job = wait_job(client, submitted.json)
            assert job["status"] == "done", job
            project = client.get(base).json
            return project["variants"][project["active"]]

        def audition(meter, expected_offset=93, **clock):
            response = client.post(base + "/audition", json=dict(mode="beats", duration_ms=5000, **clock),
                                   headers=headers)
            expected = [(expected_offset + i * 400, "accent" if i % meter == 0 else "beat")
                        for i in range(13) if expected_offset + i * 400 < 5000]
            reference, _ = render_audition(preview, 0, 5000, expected)
            assert response.status_code == 200 and response.data == reference.getvalue()

        def exported(meter):
            text = client.get(base + "/export/osu").data
            with zipfile.ZipFile(io.BytesIO(client.get(base + "/export/osz").data)) as archive:
                assert archive.read("ManiaLab.osu") == text
            red, = (line.split(",") for line in parse_osu(text.decode())["timing"] if line.strip())
            assert [float(red[0]), float(red[1]), int(red[2])] == [93, 400, meter]

        assert client.get(base).json["detected_meter"] == 3
        audition(3)
        candidate = run("generate", {"mode": "audio", "settings": {"keys": 6}})
        assert (candidate["bpm"], candidate["offset"], candidate["division"], candidate["meter"]) == (150, 93, 4, 3)
        exported(3)
        # Simulate reanalysis on the same beat clock with different downbeat evidence.
        state = projects.load(pid)
        state["analysis"]["separation"] = analysis(meter=4)["separation"]
        projects.save(state)
        assert client.get(base).json["detected_meter"] == 4
        audition(3)
        note = candidate["notes"][0]
        edited = run("edit", dict(base_created=candidate["created"], op="update", note_id=note["id"],
                                  values={"lane": (note["lane"] + 1) % 6}))
        assert edited["meter"] == 3
        region = run("generate", {"region": [1600, 2400], "settings": {"keys": 6}})
        assert region["meter"] == 3
        exported(3)
        manual = run("generate", {"bpm": 150, "offset": 193, "timing_confirmed": True, "settings": {"keys": 6}})
        assert manual["meter"] == 4
        audition(4, 193)
        assert client.post(base + "/variant", headers=headers, json={"index": 0}).status_code == 200
        audition(3)
        # An old saved candidate has its original four-beat header even when current evidence says three.
        state = projects.load(pid)
        del state["variants"][0]["meter"]
        state["analysis"]["separation"] = analysis()["separation"]
        projects.save(state)
        audition(4)
        exported(4)
        audition(4, 193, bpm=150, offset=193)
        state["active"] = -1
        del state["analysis"]["tempo"]["offset"]
        projects.save(state)
        audition(4, bpm=150, offset=93)
    finally:
        app.extensions["executor"].shutdown()
