import io
from pathlib import Path
import threading
import time

import numpy as np

from manialab import audio, audio_analysis
from manialab.audio_config import SR
from manialab.server import create_app


ROOT = Path(__file__).parents[1]


def test_http_cancel_during_rhythmic_stft_preserves_previous_analysis(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    entered_second_chunk = threading.Event()
    release_second_chunk = threading.Event()
    real_stft = audio_analysis.stft
    calls = []
    samples = np.zeros(SR * 8, dtype=np.float32)

    def fake_infer(*args, **kwargs):
        frames = np.zeros((1, 88), dtype=np.float32)
        return frames, frames.copy(), np.zeros(1, dtype=np.float64)

    def blocking_stft(values, *args, **kwargs):
        result = real_stft(values, *args, **kwargs)
        calls.append(len(values))
        if len(calls) == 2:
            entered_second_chunk.set()
            if not release_second_chunk.wait(5):
                raise AssertionError("test did not release the second STFT chunk")
        return result

    monkeypatch.setattr(audio, "decode", lambda *args, **kwargs: samples)
    monkeypatch.setattr(audio, "model_identity", lambda *args, **kwargs: "synthetic-model")
    monkeypatch.setattr(audio, "infer", fake_infer)
    monkeypatch.setattr(audio_analysis, "stft", blocking_stft)

    try:
        imported = client.post(
            "/api/import",
            data={"files": (io.BytesIO(b"synthetic audio"), "song.wav")},
            headers=headers,
        )
        assert imported.status_code == 200
        project_id = imported.json["id"]
        projects = app.extensions["projects"]
        state = projects.load(project_id)
        previous_analysis = {"sentinel": "previous analysis"}
        state["analysis"] = previous_analysis
        projects.save(state)

        started = client.post(
            f"/api/projects/{project_id}/analyze",
            json={"engine": "basic", "settings": {}},
            headers=headers,
        )
        assert started.status_code == 200
        job_id = started.json["id"]
        assert entered_second_chunk.wait(5), "analysis did not reach the second STFT chunk"

        cancelled = client.post(f"/api/jobs/{job_id}/cancel", headers=headers)
        assert cancelled.status_code == 200

        release_second_chunk.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job_id}", headers=headers).json
            if job["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(.01)

        assert job["status"] == "cancelled", job
        assert len(calls) == 2
        assert projects.load(project_id)["analysis"] == previous_analysis
    finally:
        release_second_chunk.set()
        app.extensions["executor"].shutdown(wait=True)
