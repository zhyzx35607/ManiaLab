import hashlib
import io
import json
from urllib.error import URLError

import pytest

from scripts import fetch_audio_models


def test_six_source_mirror_uses_a_pinned_revision():
    manifest = json.loads(
        (fetch_audio_models.ROOT / "assets" / "deep-models.json").read_text(
            encoding="utf-8"
        )
    )
    model = next(
        item for item in manifest["models"] if item["name"] == "HTDemucs six-source"
    )

    assert model["mirror_urls"] == [
        "https://huggingface.co/adepusricharan/Demucs-v4/resolve/"
        "d8da960b3245f2b0d8a44d737cbc66f56123f896/"
        "5c90dfd2-34c22ccb.th"
    ]
    assert model["bytes"] == 54_996_327
    assert model["sha256"] == "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"


def _model_manifest(root, payload, sources):
    assets = root / "assets"
    assets.mkdir()
    (assets / "deep-models.json").write_text(
        json.dumps({
            "models": [{
                "name": "test model",
                "filename": "model.bin",
                "url": sources[0],
                "mirror_urls": sources[1:],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }]
        }),
        encoding="utf-8",
    )


def test_fetch_audio_models_uses_verified_mirror_after_source_failure(tmp_path, monkeypatch):
    payload = b"verified model bytes"
    sources = ["https://primary.example/model", "https://mirror.example/model"]
    _model_manifest(tmp_path, payload, sources)
    monkeypatch.setattr(fetch_audio_models, "ROOT", tmp_path)
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        if url == sources[0]:
            raise URLError("certificate has expired")
        return io.BytesIO(payload)

    monkeypatch.setattr(fetch_audio_models.urllib.request, "urlopen", open_url)

    fetch_audio_models.fetch(tmp_path / "output")

    assert calls == sources
    assert (tmp_path / "output" / "model.bin").read_bytes() == payload
    assert not (tmp_path / "output" / "model.bin.pending").exists()


def test_fetch_audio_models_rejects_invalid_mirror_and_cleans_partial_file(
    tmp_path, monkeypatch
):
    payload = b"expected bytes"
    sources = ["https://primary.example/model", "https://mirror.example/model"]
    _model_manifest(tmp_path, payload, sources)
    monkeypatch.setattr(fetch_audio_models, "ROOT", tmp_path)
    responses = iter([
        io.BytesIO(b"partial"),
        io.BytesIO(b"x" * len(payload)),
    ])
    monkeypatch.setattr(
        fetch_audio_models.urllib.request,
        "urlopen",
        lambda url, timeout: next(responses),
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fetch_audio_models.fetch(tmp_path / "output")

    assert not (tmp_path / "output" / "model.bin").exists()
    assert not (tmp_path / "output" / "model.bin.pending").exists()


def test_fetch_audio_models_reuses_an_already_verified_model(tmp_path, monkeypatch):
    payload = b"verified model bytes"
    sources = ["https://primary.example/model", "https://mirror.example/model"]
    _model_manifest(tmp_path, payload, sources)
    monkeypatch.setattr(fetch_audio_models, "ROOT", tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "model.bin").write_bytes(payload)
    monkeypatch.setattr(
        fetch_audio_models.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("verified model should not be downloaded"),
    )

    fetch_audio_models.fetch(output)
