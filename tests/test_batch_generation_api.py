"""Batch generation and bundle export keep independent project identities."""
import hashlib
import io
from pathlib import Path
import time
from urllib.parse import urlencode
import zipfile

import pytest

from manialab import candidate_routes
from manialab import export_routes
from manialab.arrange import parameters
from manialab.candidate_identity import new_candidate_id
from manialab.domain import Note, parse_osu
from manialab.server import create_app


ROOT = Path(__file__).resolve().parents[1]


def _wait_job(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(.01)
    pytest.fail(f"batch job {job_id} did not finish")


def _import_song(client, headers, name):
    content = f"synthetic local audio for {name}".encode()
    response = client.post(
        "/api/import", headers=headers,
        data={"files": (io.BytesIO(content), name)},
    )
    assert response.status_code == 200, response.json
    return response.json, content


def _set_analysis(projects, song, *, bpm=120, offset=115, stable=True, revision=3):
    state = projects.load(song["id"])
    state["audio_shift_ms"] = 22.5
    state["settings"] = parameters({"keys": 7})
    note = Note(1078, 1078, lane=0, pitch=60, source="pitch", id=f"{song['id']}-note")
    digest = hashlib.sha256(
        (projects.path(song["id"]) / "files" / state["audio"]).read_bytes()
    ).hexdigest()
    state["analysis"] = {
        "revision": revision,
        "duration": 8,
        "audio_sha256": digest,
        "candidates": [note.data()],
        "timing_attacks": [note.data()],
        "phrase_candidates": [],
        "tempo": {
            "bpm": bpm,
            "offset": offset,
            "division": 4,
            "stable": stable,
        },
        "waveform": [0] * 40,
    }
    projects.save(state)
    return digest


def _install_fake_generator(monkeypatch, calls, failures=()):
    failures = set(failures)

    def fake_generate(state, payload, prefs, tempo, progress, cancelled):
        calls.append({
            "project": state["id"],
            "settings": dict(prefs),
            "payload": dict(payload),
            "tempo": dict(tempo),
        })
        if state["title"] in failures:
            raise ValueError(f"synthetic generation failure: {state['title']}")
        note = Note(
            int(round(tempo["offset"] + 1000)),
            int(round(tempo["offset"] + 1000)),
            lane=0,
            pitch=60,
            source="pitch",
            id=f"{state['id']}-generated",
        )
        variant = {
            "candidate_id": new_candidate_id(),
            "mode": "audio",
            "timing_revision": 3,
            "settings": dict(prefs),
            "notes": [note.data()],
            "bpm": tempo["bpm"],
            "offset": tempo["offset"],
            "division": tempo["division"],
            "meter": tempo.get("meter", 4),
            "audio_shift_ms": tempo["audio_shift_ms"],
            "source_audio_sha256": state["analysis"]["audio_sha256"],
            "selection": {"structure": {"boundaries": []}},
            "report": {"valid": True, "notes": 1, "ln": 0},
        }
        state["variants"] = (state.get("variants", []) + [variant])[-8:]
        state["active"] = len(state["variants"]) - 1
        state["settings"] = dict(prefs)
        progress(90, "合成测试候选已准备")
        return variant["report"]

    monkeypatch.setattr(candidate_routes, "generate_variant", fake_generate)


def _submit_generation(client, headers, projects, *, preset="onset_only",
                       keys=7, key_layout="7k_3_1_3", density=8,
                       timing_confirmed=False):
    return client.post(
        "/api/batch/generate",
        headers=headers,
        json={
            "projects": projects,
            "preset": preset,
            "keys": keys,
            "key_layout": key_layout,
            "density": density,
            "timing_confirmed": timing_confirmed,
        },
    )


def test_batch_generation_uses_each_song_clock_and_returns_candidate_ids(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    calls = []
    _install_fake_generator(monkeypatch, calls)
    songs = [_import_song(client, headers, f"song-{i}.wav")[0] for i in range(2)]
    projects = app.extensions["projects"]
    _set_analysis(projects, songs[0], bpm=120, offset=115)
    _set_analysis(projects, songs[1], bpm=150, offset=207)

    try:
        response = _submit_generation(
            client, headers, [song["id"] for song in songs], density=17,
        )
        assert response.status_code == 200, response.json
        job = _wait_job(client, response.json["id"])

        assert job["status"] == "done"
        assert [item["status"] for item in job["items"]] == ["done", "done"]
        candidates = [item["result"]["candidate_id"] for item in job["items"]]
        assert all(len(candidate_id) == 32 for candidate_id in candidates)
        assert len(set(candidates)) == 2
        assert [call["tempo"]["bpm"] for call in calls] == [120, 150]
        # Each song's detected offset is shifted independently; no open-project clock leaks in.
        assert [call["tempo"]["offset"] for call in calls] == [93, 185]
        assert [call["tempo"]["audio_shift_ms"] for call in calls] == [22.5, 22.5]
        assert all(call["settings"]["keys"] == 7 for call in calls)
        assert all(call["settings"]["key_layout"] == "7k_3_1_3" for call in calls)
        assert all(call["settings"]["density"] == 17 for call in calls)
        assert all(call["settings"]["ln_ratio"] == 0 for call in calls)
        assert all(call["settings"]["phrase_ln"] == 0 for call in calls)
    finally:
        app.extensions["task_service"].shutdown()


def test_batch_generation_requires_explicit_timing_confirmation_per_song(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    calls = []
    _install_fake_generator(monkeypatch, calls)
    songs = [_import_song(client, headers, f"timing-{i}.wav")[0] for i in range(2)]
    projects = app.extensions["projects"]
    _set_analysis(projects, songs[0], stable=False)
    _set_analysis(projects, songs[1], stable=True)

    try:
        response = _submit_generation(client, headers, [song["id"] for song in songs])
        assert response.status_code == 200, response.json
        job = _wait_job(client, response.json["id"])
        assert job["status"] == "error"
        assert [item["status"] for item in job["items"]] == ["error", "done"]
        assert "Timing 不可靠" in job["items"][0]["message"]
        assert "timing-0" in job["items"][0]["message"]
        assert len(calls) == 1

        response = _submit_generation(
            client, headers, [song["id"] for song in songs], timing_confirmed=True,
        )
        assert response.status_code == 200, response.json
        confirmed = _wait_job(client, response.json["id"])
        assert [item["status"] for item in confirmed["items"]] == ["done", "done"]
        assert len(calls) == 3
        assert all(call["payload"]["timing_confirmed"] is True for call in calls[1:])
    finally:
        app.extensions["task_service"].shutdown()


def test_batch_generation_isolates_generation_errors_and_validates_request(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    calls = []
    _install_fake_generator(monkeypatch, calls, failures={"failure"})
    songs = [_import_song(client, headers, name)[0]
             for name in ("one.wav", "failure.wav", "three.wav")]
    projects = app.extensions["projects"]
    for song in songs:
        _set_analysis(projects, song)

    try:
        response = _submit_generation(
            client, headers, [song["id"] for song in songs], preset="rhythm",
        )
        assert response.status_code == 200, response.json
        job = _wait_job(client, response.json["id"])
        assert [item["status"] for item in job["items"]] == ["done", "error", "done"]
        assert job["result"] == {"done": 2, "error": 1, "cancelled": 0}
        assert "synthetic generation failure" in job["items"][1]["message"]
        assert len(job["items"][0]["result"]["candidate_id"]) == 32
        assert len(job["items"][2]["result"]["candidate_id"]) == 32
        assert 0 < job["max_parallel"] <= 1
        assert all(call["settings"]["ln_ratio"] == .25 for call in calls)

        for invalid in (
            {"projects": [songs[0]["id"], songs[0]["id"]], "preset": "onset_only", "keys": 7},
            {"projects": [songs[0]["id"], songs[2]["id"]], "preset": "unknown", "keys": 7},
            {"projects": [songs[0]["id"], songs[2]["id"]], "preset": [], "keys": 7},
            {"projects": [songs[0]["id"], songs[2]["id"]], "preset": {}, "keys": 7},
            {"projects": [songs[0]["id"], songs[2]["id"]], "preset": "onset_only", "keys": 11},
            {"projects": [f"{index:032x}" for index in range(11)],
             "preset": "onset_only", "keys": 7},
        ):
            result = client.post("/api/batch/generate", headers=headers, json=invalid)
            assert result.status_code == 400

        # An analysis batch can leave a single ready song; generation still applies
        # the shared preset and produces a one-member bundle without weakening the
        # analysis route's own minimum of two.
        one = _submit_generation(client, headers, [songs[0]["id"]], preset="onset_only")
        assert one.status_code == 200, one.json
        one_job = _wait_job(client, one.json["id"])
        assert one_job["status"] == "done"
        assert len(one_job["items"]) == 1
        assert one_job["items"][0]["result"]["candidate_id"]
        assert client.post(
            "/api/batch/analyze", headers=headers,
            json={"projects": [songs[0]["id"]]},
        ).status_code == 400
    finally:
        app.extensions["task_service"].shutdown()


def test_batch_bundle_contains_only_requested_current_candidates_and_audio(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    calls = []
    _install_fake_generator(monkeypatch, calls)
    songs_and_audio = [
        _import_song(client, headers, "same title.wav"),
        _import_song(client, headers, "same title copy.wav"),
    ]
    songs = [song for song, _ in songs_and_audio]
    projects = app.extensions["projects"]
    builder_inputs = []
    temp_dirs = []
    build_osz = export_routes._build_osz_file
    temporary_file = export_routes.tempfile.TemporaryFile

    def inspect_builder(*args, **kwargs):
        state, variant = args[3], args[4]
        builder_inputs.append((set(state), set(variant), kwargs.get("temp_dir")))
        return build_osz(*args, **kwargs)

    def inspect_temporary_file(*args, **kwargs):
        temp_dirs.append(kwargs.get("dir"))
        return temporary_file(*args, **kwargs)

    monkeypatch.setattr(export_routes, "_build_osz_file", inspect_builder)
    monkeypatch.setattr(export_routes.tempfile, "TemporaryFile", inspect_temporary_file)
    for song in songs:
        _set_analysis(projects, song)
        state = projects.load(song["id"])
        state["title"] = "same/title"
        projects.save(state)
    try:
        generated = _submit_generation(client, headers, [song["id"] for song in songs])
        assert generated.status_code == 200
        job = _wait_job(client, generated.json["id"])
        items = [
            {"project": row["project"], "candidate_id": row["result"]["candidate_id"]}
            for row in job["items"]
        ]
        response = client.post(
            "/api/batch/export-osz", headers=headers, json={"items": items},
        )
        assert response.status_code == 200, response.json
        assert len(builder_inputs) == 2
        for state_keys, variant_keys, temp_dir in builder_inputs:
            assert state_keys == {"id", "audio"}
            assert variant_keys == {"mode", "source_audio_sha256"}
            assert Path(temp_dir) == projects.root
        assert len(temp_dirs) == 3
        assert all(Path(temp_dir) == projects.root for temp_dir in temp_dirs)
        with zipfile.ZipFile(io.BytesIO(response.data)) as bundle:
            assert bundle.testzip() is None
            names = bundle.namelist()
            assert len(names) == 2
            assert len(set(name.casefold() for name in names)) == 2
            for name, (song, audio) in zip(names, songs_and_audio):
                assert name.endswith(".osz")
                assert "/" not in name and "\\" not in name
                with zipfile.ZipFile(io.BytesIO(bundle.read(name))) as osz:
                    assert osz.testzip() is None
                    chart_name = next(
                        member for member in osz.namelist() if member.endswith(".osu")
                    )
                    audio_name = parse_osu(osz.read(chart_name).decode("utf-8"))[
                        "general"
                    ]["AudioFilename"]
                    assert osz.read(audio_name) == audio
                    parsed = parse_osu(osz.read(chart_name).decode("utf-8"))
                    assert parsed["metadata"]["Title"] == "same/title"
                    assert len(parsed["notes"]) == 1

        # A previously valid candidate cannot silently resolve to the newly selected variant.
        first_project = items[0]["project"]
        state = projects.load(first_project)
        current = state["variants"][state["active"]]
        replacement = dict(current)
        replacement.pop("candidate_id")
        replacement["notes"] = [Note(2000, 2000, lane=1, source="pitch", id="replacement").data()]
        state["variants"].append(replacement)
        state["active"] = len(state["variants"]) - 1
        projects.save(state)
        stale = client.post(
            "/api/batch/export-osz", headers=headers, json={"items": items},
        )
        assert stale.status_code == 400
        assert "不是当前候选" in stale.json["error"]
    finally:
        app.extensions["task_service"].shutdown()


def test_batch_bundle_rejects_duplicate_and_missing_projects_before_writing(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    try:
        response = client.post(
            "/api/batch/export-osz", headers=headers,
            json={"items": [
                {"project": "a" * 32, "candidate_id": "b" * 32},
                {"project": "a" * 32, "candidate_id": "c" * 32},
            ]},
        )
        assert response.status_code == 400
        assert "重复" in response.json["error"]
    finally:
        app.extensions["task_service"].shutdown()


def test_batch_bundle_direct_get_download_is_candidate_bound_and_rejects_bad_list(
        tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {"X-ManiaLab-Token": app.config["TOKEN"]}
    calls = []
    _install_fake_generator(monkeypatch, calls)
    song, audio = _import_song(client, headers, "direct-download.wav")
    projects = app.extensions["projects"]
    _set_analysis(projects, song)
    try:
        submitted = _submit_generation(client, headers, [song["id"]])
        assert submitted.status_code == 200
        job = _wait_job(client, submitted.json["id"])
        row = job["items"][0]
        value = f'{row["project"]}:{row["result"]["candidate_id"]}'
        query = urlencode({"items": value})
        response = client.get(f"/api/batch/export-osz?{query}")
        assert response.status_code == 200
        assert response.headers["Content-Disposition"].startswith("attachment;")
        assert "filename*=UTF-8''ManiaLab-" in response.headers["Content-Disposition"]
        with zipfile.ZipFile(io.BytesIO(response.data)) as bundle:
            assert bundle.testzip() is None
            nested = zipfile.ZipFile(io.BytesIO(bundle.read(bundle.namelist()[0])))
            assert nested.testzip() is None
            chart_name = next(name for name in nested.namelist() if name.endswith(".osu"))
            audio_name = parse_osu(nested.read(chart_name).decode("utf-8"))[
                "general"
            ]["AudioFilename"]
            assert nested.read(audio_name) == audio
        assert client.get("/api/batch/export-osz?items=malformed").status_code == 400
        assert client.post(
            "/api/batch/export-osz", headers=headers,
            json={"items": [{"project": row["project"],
                             "candidate_id": "f" * 32}]},
        ).status_code == 400
    finally:
        app.extensions["task_service"].shutdown()
