import io
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from manialab.audition import grid_events, imported_grid_events, render_audition
from manialab.audio_io import decode
from manialab.server import create_app


ROOT = Path(__file__).resolve().parents[1]
LINES = ["3000,400,2,0,0,70,1,0", "0,500,3", "1750,-50,4,0,0,70,0,0", "3000,250,5,0,0,70,1,0"]
EXPECTED = [(0, "accent"), (500, "beat"), (1000, "beat"), (1500, "accent"), (2000, "beat"), (2500, "beat"),
            (3000, "accent"), (3250, "beat"), (3500, "beat"), (3750, "beat"), (4000, "beat"),
            (4250, "accent"), (4500, "beat"), (4750, "beat")]


def test_red_lines_reset_meter_and_exclude_the_next_segment_start():
    assert imported_grid_events(LINES, 0, 5000) == EXPECTED
    assert imported_grid_events(LINES, 2500, 500) == [(2500, "beat")]
    assert imported_grid_events(LINES, 3000, 250) == [(3000, "accent")]
    assert imported_grid_events(["750,250"], 0, 1100) == [
        (0, "beat"), (250, "beat"), (500, "beat"), (750, "accent"), (1000, "beat")]


def test_unusable_or_unrenderable_timing_does_not_invent_an_audible_clock():
    with pytest.raises(ValueError, match="Timing"):
        imported_grid_events(["", "x,y", "nan,100", "0,inf", "1000,-50,3,2,0,70,0,0"], 0, 1000)
    with pytest.raises(ValueError, match="过密"):
        imported_grid_events(["0,1e-300"], 0, 1000)
    assert imported_grid_events(["0,250,0"], 900, 400) == [(1000, "accent"), (1250, "beat")]


def test_fresh_import_auditions_the_original_mix_on_the_cached_sample_clock(tmp_path):
    rate = 44100
    t = np.arange(rate * 4) / rate
    samples = np.stack([np.sin(2 * np.pi * t * hz) for hz in (220, 330)], axis=1)
    sound = io.BytesIO()
    wavfile.write(sound, rate, (samples * 10000).astype(np.int16))
    sound.seek(0)
    chart = "\n".join(["osu file format v14", "[General]", "Mode:3", "AudioFilename:mix.wav",
                       "[Difficulty]", "CircleSize:6", "[TimingPoints]", *LINES, "[HitObjects]",
                       "0,192,1000,1,0,0:0:0:0:"])
    app = create_app(ROOT, tmp_path)
    try:
        client = app.test_client()
        headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
        state = client.post("/api/import", headers=headers, data={"files": [
            (io.BytesIO(chart.encode()), "chart.osu"), (sound, "mix.wav")]}).json
        path = app.extensions["projects"].path(state["id"])
        url = f"/api/projects/{state['id']}/audition"
        payload = dict(mode="beats", start_ms=1250.25, duration_ms=3000)
        direct = client.post(url, headers=headers, json=payload)
        assert direct.status_code == 200 and int(direct.headers["X-ManiaLab-Audition-Events"]) > 0
        assert not (path / "preview.wav").exists()
        decode(path / "files" / "mix.wav", path)
        cached = client.post(url, headers=headers, json=payload)
        assert cached.status_code == 200 and cached.data == direct.data
    finally:
        app.extensions["executor"].shutdown()


@pytest.mark.parametrize("legacy", [False, True])
def test_remap_audition_renders_original_pcm_and_keeps_the_fixed_draft_selectable(tmp_path, legacy):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    chart = "\n".join(["osu file format v14", "[General]", "Mode:3", "AudioFilename:song.wav",
                       "[Difficulty]", "CircleSize:6", "[TimingPoints]", *LINES, "[HitObjects]",
                       "0,192,1000,1,0,0:0:0:0:"])
    try:
        project = client.post("/api/import", headers=headers, data={"files": [
            (io.BytesIO(chart.encode()), "chart.osu"), (io.BytesIO(b"audio placeholder"), "song.wav")]}).json
        projects = app.extensions["projects"]
        if legacy:
            del project["input_timing"]
            projects.save(project)
        preview = projects.path(project["id"]) / "preview.wav"
        wavfile.write(preview, 8000, np.zeros(40000, dtype=np.int16))
        url = "/api/projects/" + project["id"] + "/audition"
        payload = {"mode": "beats", "duration_ms": 5000, "bpm": 150, "offset": 75}
        for clock, expected in [({}, EXPECTED), ({"beat_clock": "original"}, EXPECTED),
                                ({"beat_clock": "fixed"}, grid_events(150, 75, 0, 5000))]:
            response = client.post(url, headers=headers, json=payload | clock)
            assert response.status_code == 200
            assert int(response.headers["X-ManiaLab-Audition-Events"]) == len(expected)
            reference, _ = render_audition(preview, 0, 5000, expected)
            assert response.data == reference.getvalue()
        assert client.post(url, headers=headers, json=payload | {"beat_clock": "unknown"}).status_code == 400
        assert ("input_timing" not in projects.load(project["id"])) == legacy
    finally:
        app.extensions["executor"].shutdown()
