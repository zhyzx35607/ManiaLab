from pathlib import Path
import hashlib
import io

import numpy as np
import pytest
from scipy.io import wavfile

from manialab.chart_workspace import audio_work
from manialab.server import create_app
from manialab.stems import archive_review_stems, separation_identity
from manialab.storage import Projects


ROOT = Path(__file__).resolve().parents[1]


def _write_pulse(path, at_ms):
    rate = 22050
    samples = np.zeros(rate * 2, dtype=np.int16)
    samples[round(at_ms * rate / 1000)] = 24000
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, samples)


def _peak_ms(data):
    index = int(np.argmax(data["high"]))
    return (data["first_sample"] + index * data["step_samples"]) * 1000 / data["sample_rate"]


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _audition_peak(client, url, context, headers, source_identity):
    response = client.post(url, json=dict(mode=context, voice="other",
        source_identity=source_identity, audio="song.wav",
        start_ms=0, duration_ms=1000), headers=headers)
    assert response.status_code == 200, response.get_json()
    rate, samples = wavfile.read(io.BytesIO(response.data))
    return int(np.argmax(np.abs(samples[:, 0] if samples.ndim == 2 else samples))), rate


def test_selected_candidate_voice_and_waveform_keep_their_source_set(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    projects = Projects(tmp_path / "projects")
    pid, folder = projects.create()
    old_set, current_set = "a" * 64, "b" * 64
    wavfile.write(folder / "files/song.wav", 22050, np.zeros(22050 * 2, dtype=np.int16))
    work = audio_work(projects.path(pid), {"audio": "song.wav"})
    old_path = work / "stems-provided" / old_set / "other.wav"
    current_path = work / "stems-provided" / current_set / "other.wav"
    _write_pulse(old_path, 100)
    _write_pulse(current_path, 600)
    old_separation = dict(source="provided", audio_sha256="c" * 64,
        set_sha256=old_set, voices=["other"],
        decoded_files={"other": {"wav_sha256": _sha256(old_path)}})
    current_separation = dict(source="provided", audio_sha256="c" * 64,
        set_sha256=current_set, voices=["other"],
        decoded_files={"other": {"wav_sha256": _sha256(current_path)}})
    old_identity = separation_identity(old_separation)
    current_identity = separation_identity(current_separation)
    state = dict(
        id=pid, title="Source identity", artist="Fixture", mode="audio", audio="song.wav",
        original=None, variants=[dict(mode="audio", notes=[], source_separation=dict(
            identity=old_identity,
            separation=old_separation,
        ))], active=0,
        analysis=dict(separation=current_separation),
    )
    projects.save(state)
    base = f"/api/projects/{pid}"

    chart_stem = client.get(base + f"/stem/other?context=chart&source_identity={old_identity}")
    evidence_stem = client.get(base + f"/stem/other?context=evidence&source_identity={current_identity}")
    assert chart_stem.status_code == evidence_stem.status_code == 200
    chart_rate, chart_samples = wavfile.read(io.BytesIO(chart_stem.data))
    evidence_rate, evidence_samples = wavfile.read(io.BytesIO(evidence_stem.data))
    assert chart_rate == evidence_rate == 22050
    assert np.argmax(np.abs(chart_samples)) == round(.1 * chart_rate)
    assert np.argmax(np.abs(evidence_samples)) == round(.6 * evidence_rate)

    chart_wave = client.get(base + f"/waveform?voice=other&context=chart&source_identity={old_identity}&audio=song.wav&start_ms=0&duration_ms=1000").json
    evidence_wave = client.get(base + f"/waveform?voice=other&context=evidence&source_identity={current_identity}&audio=song.wav&start_ms=0&duration_ms=1000").json
    assert abs(_peak_ms(chart_wave) - 100) < 5
    assert abs(_peak_ms(evidence_wave) - 600) < 5

    chart_peak, chart_rate = _audition_peak(client, base + "/audition", "chart", headers, old_identity)
    evidence_peak, evidence_rate = _audition_peak(client, base + "/audition", "evidence", headers, current_identity)
    assert chart_rate == evidence_rate == 22050
    assert chart_peak == round(.1 * chart_rate)
    assert evidence_peak == round(.6 * evidence_rate)


def test_model_stem_archives_keep_candidate_audio_after_cache_replacement(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    projects = Projects(tmp_path / "projects")
    pid, folder = projects.create()
    wavfile.write(folder / "files/song.wav", 22050, np.zeros(22050 * 2, dtype=np.int16))
    work = audio_work(projects.path(pid), {"audio": "song.wav"})
    model_a = dict(audio_sha256="a" * 64, model_sha256="b" * 64, profile="four",
        input_processing={"revision": 2}, algorithm_revision=1)
    model_b = model_a | {"model_sha256": "c" * 64}
    stem_root = work / "stems"
    _write_pulse(stem_root / "other.wav", 100)
    saved_a = archive_review_stems(work, stem_root, model_a, ["other"])
    _write_pulse(stem_root / "other.wav", 600)
    saved_b = archive_review_stems(work, stem_root, model_b, ["other"])
    state = dict(
        id=pid, title="Model source identity", artist="Fixture", mode="audio", audio="song.wav",
        original=None, variants=[dict(mode="audio", notes=[], source_separation=dict(
            identity=separation_identity(saved_a), separation=saved_a,
        ))], active=0,
        analysis=dict(separation=saved_b),
    )
    projects.save(state)
    base = f"/api/projects/{pid}"
    chart_identity = separation_identity(saved_a)
    evidence_identity = separation_identity(saved_b)
    chart_stem = client.get(base + f"/stem/other?context=chart&source_identity={chart_identity}")
    evidence_stem = client.get(base + f"/stem/other?context=evidence&source_identity={evidence_identity}")
    assert chart_stem.status_code == evidence_stem.status_code == 200
    assert np.argmax(np.abs(wavfile.read(io.BytesIO(chart_stem.data))[1])) == 2205
    assert np.argmax(np.abs(wavfile.read(io.BytesIO(evidence_stem.data))[1])) == 13230

    chart_peak, chart_rate = _audition_peak(client, base + "/audition", "chart", headers, chart_identity)
    evidence_peak, evidence_rate = _audition_peak(client, base + "/audition", "evidence", headers, evidence_identity)
    assert chart_rate == evidence_rate == 22050
    assert chart_peak == 2205
    assert evidence_peak == 13230


def test_legacy_candidate_without_source_identity_never_uses_current_stem(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    projects = Projects(tmp_path / "projects")
    pid, folder = projects.create()
    old_set = "a" * 64
    separation = dict(source="provided", audio_sha256="c" * 64,
        set_sha256=old_set, voices=["other"])
    wavfile.write(folder / "files/song.wav", 22050, np.zeros(22050, dtype=np.int16))
    state = dict(id=pid, title="Legacy source", artist="Fixture", mode="audio", audio="song.wav",
        variants=[dict(mode="audio", notes=[])], active=0, analysis=dict(separation=separation))
    projects.save(state)
    _write_pulse(audio_work(projects.path(pid), state) / "stems-provided" / old_set / "other.wav", 600)

    response = client.get(f"/api/projects/{pid}/stem/other?context=chart")
    assert response.status_code == 400
    assert "未保存声轨来源" in response.json["error"]


@pytest.mark.parametrize("source", ["provided", "model"])
def test_candidate_playback_rejects_stem_bytes_that_no_longer_match_identity(tmp_path, source):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    projects = Projects(tmp_path / "projects")
    pid, folder = projects.create()
    wavfile.write(folder / "files/song.wav", 22050, np.zeros(22050, dtype=np.int16))
    work = audio_work(projects.path(pid), {"audio": "song.wav"})
    if source == "provided":
        set_hash = "a" * 64
        separation = dict(source="provided", audio_sha256="b" * 64, set_sha256=set_hash,
            voices=["other"])
        stem = work / "stems-provided" / set_hash / "other.wav"
        _write_pulse(stem, 100)
        separation["decoded_files"] = {"other": {"wav_sha256": _sha256(stem)}}
    else:
        separation = dict(audio_sha256="a" * 64, model_sha256="b" * 64, profile="four",
            input_processing={"revision": 2}, algorithm_revision=1)
        root = work / "stems"
        stem = root / "other.wav"
        _write_pulse(stem, 100)
        separation = archive_review_stems(work, root, separation, ["other"])
        stem = work / "stems-review" / separation["review_audio_identity"] / "other.wav"
    state = dict(id=pid, title="Tampered source", artist="Fixture", mode="audio", audio="song.wav",
        variants=[dict(mode="audio", notes=[], source_separation=dict(
            identity=separation_identity(separation), separation=separation))],
        active=0, analysis=dict(separation=separation))
    projects.save(state)
    _write_pulse(stem, 600)

    response = client.get(f"/api/projects/{pid}/stem/other?context=chart&source_identity={separation_identity(separation)}")
    assert response.status_code == 400
    assert "文件内容已改变" in response.json["error"]


def test_stale_candidate_source_requests_are_rejected_after_another_tab_switches_candidate(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    projects = Projects(tmp_path / "projects")
    pid, folder = projects.create()
    wavfile.write(folder / "files/song.wav", 22050, np.zeros(22050, dtype=np.int16))
    work = audio_work(projects.path(pid), {"audio": "song.wav"})
    separations = []
    for set_hash, at_ms in (("a" * 64, 100), ("b" * 64, 600)):
        stem = work / "stems-provided" / set_hash / "other.wav"
        _write_pulse(stem, at_ms)
        separation = dict(source="provided", audio_sha256="c" * 64, set_sha256=set_hash,
            voices=["other"], decoded_files={"other": {"wav_sha256": _sha256(stem)}})
        separations.append(separation)
    variants = [dict(mode="audio", notes=[], source_separation=dict(
        identity=separation_identity(separation), separation=separation))
        for separation in separations]
    state = dict(id=pid, title="Stale source", artist="Fixture", mode="audio", audio="song.wav",
        variants=variants, active=0, analysis=dict(separation=separations[0]))
    projects.save(state)
    old_identity = variants[0]["source_separation"]["identity"]
    base = f"/api/projects/{pid}"
    old_url = base + f"/stem/other?context=chart&source_identity={old_identity}"
    assert client.get(old_url).status_code == 200

    state["active"] = 1
    state["analysis"]["separation"] = separations[1]
    projects.save(state)
    assert client.get(old_url).status_code == 400
    assert client.get(base + f"/waveform?voice=other&context=chart&source_identity={old_identity}&audio=song.wav").status_code == 400
    stale_audition = client.post(base + "/audition", json=dict(
        mode="chart", voice="other", context="chart", source_identity=old_identity,
        audio="song.wav", start_ms=0, duration_ms=1000), headers=headers)
    assert stale_audition.status_code == 400


def test_chart_audition_rejects_an_old_tabs_candidate_with_the_same_audio(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    projects = app.extensions["projects"]
    try:
        pid, folder = projects.create()
        wavfile.write(folder / "preview.wav", 22050, np.zeros(22050 * 2, dtype=np.int16))
        state = dict(id=pid, title="Stale chart audition", artist="Fixture", mode="audio",
            audio="song.wav", active=0, variants=[
                dict(mode="audio", notes=[dict(start=100, end=100)]),
                dict(mode="audio", notes=[dict(start=500, end=700)]),
            ])
        projects.save(state)
        old_id, new_id = [candidate["candidate_id"] for candidate in state["variants"]]
        url = f"/api/projects/{pid}/audition"
        payload = dict(mode="chart", voice="mix", audio="song.wav", duration_ms=1000)
        stale_request = payload | dict(candidate_id=old_id)
        first = client.post(url, json=stale_request, headers=headers)
        assert first.status_code == 200, first.json
        assert first.headers["X-ManiaLab-Audition-Events"] == "1"

        state["active"] = 1
        projects.save(state)
        stale = client.post(url, json=stale_request, headers=headers)
        assert stale.status_code == 400
        assert "刷新" in stale.json["error"]
        assert client.post(url, json=payload | dict(candidate_id=None), headers=headers).status_code == 400
        fresh = client.post(url, json=payload | dict(candidate_id=new_id), headers=headers)
        unbound = client.post(url, json=payload, headers=headers)
        assert fresh.status_code == unbound.status_code == 200
        assert fresh.headers["X-ManiaLab-Audition-Events"] == "2"
        assert fresh.data == unbound.data
    finally:
        app.extensions["executor"].shutdown()


def test_evidence_and_beat_audition_remain_available_without_a_candidate(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    projects = app.extensions["projects"]
    try:
        pid, folder = projects.create()
        wavfile.write(folder / "preview.wav", 22050, np.zeros(22050 * 2, dtype=np.int16))
        state = dict(id=pid, title="Analysis audition", artist="Fixture", mode="audio",
            audio="song.wav", active=-1, variants=[],
            analysis=dict(candidates=[dict(start=100, end=100)]))
        projects.save(state)
        url = f"/api/projects/{pid}/audition"
        payload = dict(voice="mix", audio="song.wav", duration_ms=1000)
        evidence = client.post(url, json=payload | dict(mode="evidence"), headers=headers)
        beats = client.post(url, json=payload | dict(mode="beats", beat_clock="fixed",
            bpm=120, offset=0, division=4, audio_shift_ms=0), headers=headers)
        assert evidence.status_code == beats.status_code == 200
        assert evidence.headers["X-ManiaLab-Audition-Events"] == "1"
        assert beats.headers["X-ManiaLab-Audition-Meter"] == "4"
        chart = client.post(url, json=payload | dict(mode="chart", candidate_id=None), headers=headers)
        assert chart.status_code == 400
        assert "先生成候选" in chart.json["error"]
    finally:
        app.extensions["executor"].shutdown()


def test_stale_audio_range_requests_are_rejected_after_chart_audio_changes(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    projects = Projects(tmp_path / "projects")
    pid, folder = projects.create()
    for name in ("first.wav", "second.wav"):
        wavfile.write(folder / f"files/{name}", 22050, np.zeros(22050, dtype=np.int16))
    state = dict(id=pid, title="Audio identity", artist="Fixture", mode="audio",
        audio="first.wav", variants=[], active=-1)
    projects.save(state)
    base = f"/api/projects/{pid}/audio"
    assert client.get(base + "?audio=first.wav", headers={"Range": "bytes=0-3"}).status_code == 206

    state["audio"] = "second.wav"
    projects.save(state)
    stale = client.get(base + "?audio=first.wav", headers={"Range": "bytes=0-3"})
    assert stale.status_code == 400
