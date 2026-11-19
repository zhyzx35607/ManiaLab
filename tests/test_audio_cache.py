import hashlib
from pathlib import Path

import numpy as np
import pytest

from manialab.audio_cache import (
    BASIC_PITCH_INFERENCE_REVISION,
    STEM_ACTIVATION_CACHE_VERSION,
    load_stem_activations,
    save_stem_activations,
    sha256_file,
)


MODEL_IDENTITY = "sha256:model"


def activations():
    return (
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.arange(12, dtype=np.float32).reshape(3, 4) / 10,
        np.arange(3, dtype=np.float64) / 86,
    )


def test_sha256_file_streams_contents_without_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / "source-audio.bin"
    contents = bytes(range(256)) * 12289
    path.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()

    def reject_read_bytes(self):
        raise AssertionError(f"whole-file read attempted for {self}")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    assert sha256_file(path) == expected


def test_sha256_file_checks_cancellation_between_chunks(tmp_path, monkeypatch):
    path = tmp_path / "large-audio.bin"
    path.write_bytes(b"x" * (3 * 1024 * 1024))
    opened = []
    real_open = Path.open

    class TrackingReader:
        def __init__(self, stream):
            self.stream = stream
            self.bytes_read = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            chunk = self.stream.read(size)
            self.bytes_read += len(chunk)
            return chunk

    def tracking_open(target, *args, **kwargs):
        reader = TrackingReader(real_open(target, *args, **kwargs))
        opened.append(reader)
        return reader

    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks == 2

    monkeypatch.setattr(Path, "open", tracking_open)
    with pytest.raises(InterruptedError, match="已取消"):
        sha256_file(path, cancelled=cancelled)

    assert opened[0].bytes_read == 1024 * 1024


def test_stem_activation_cache_round_trips_the_current_format(tmp_path):
    cache = tmp_path / "vocals-activations.npz"
    expected = activations()

    save_stem_activations(
        cache, expected, "audio-hash", "separation-hash",
        MODEL_IDENTITY, BASIC_PITCH_INFERENCE_REVISION)

    loaded = load_stem_activations(
        cache, "audio-hash", "separation-hash",
        MODEL_IDENTITY, BASIC_PITCH_INFERENCE_REVISION)
    assert loaded is not None
    for actual, wanted in zip(loaded, expected):
        np.testing.assert_array_equal(actual, wanted)
    with np.load(cache, allow_pickle=False) as data:
        assert set(data.files) == {
            "frames", "onsets", "times", "version", "audio_sha256",
            "separation_sha256", "model_identity", "inference_revision"
        }
        assert int(data["version"]) == STEM_ACTIVATION_CACHE_VERSION


@pytest.mark.parametrize(
    "audio_hash,separation_hash,model_identity,inference_revision",
    [
        ("other-audio", "separation-hash", MODEL_IDENTITY, BASIC_PITCH_INFERENCE_REVISION),
        ("audio-hash", "other-separation", MODEL_IDENTITY, BASIC_PITCH_INFERENCE_REVISION),
        ("audio-hash", "separation-hash", "sha256:other-model", BASIC_PITCH_INFERENCE_REVISION),
        ("audio-hash", "separation-hash", MODEL_IDENTITY, BASIC_PITCH_INFERENCE_REVISION + 1),
    ],
)
def test_stem_activation_cache_requires_all_source_and_detector_identities(
    tmp_path, audio_hash, separation_hash, model_identity, inference_revision
):
    cache = tmp_path / "activations.npz"
    save_stem_activations(
        cache, activations(), "audio-hash", "separation-hash",
        MODEL_IDENTITY, BASIC_PITCH_INFERENCE_REVISION)

    assert load_stem_activations(
        cache, audio_hash, separation_hash, model_identity, inference_revision) is None


@pytest.mark.parametrize("kind", ["old-version", "missing-field", "corrupt", "missing-file"])
def test_stem_activation_cache_treats_invalid_entries_as_misses(tmp_path, kind):
    cache = tmp_path / "activations.npz"
    if kind == "old-version":
        np.savez_compressed(
            cache,
            frames=activations()[0],
            onsets=activations()[1],
            times=activations()[2],
            version=STEM_ACTIVATION_CACHE_VERSION - 1,
            audio_sha256="audio-hash",
            separation_sha256="separation-hash",
            model_identity=MODEL_IDENTITY,
            inference_revision=BASIC_PITCH_INFERENCE_REVISION,
        )
    elif kind == "missing-field":
        np.savez_compressed(
            cache, version=STEM_ACTIVATION_CACHE_VERSION,
            audio_sha256="audio-hash")
    elif kind == "corrupt":
        cache.write_bytes(b"not an npz archive")

    assert load_stem_activations(
        cache, "audio-hash", "separation-hash",
        MODEL_IDENTITY, BASIC_PITCH_INFERENCE_REVISION) is None
