import numpy as np
import pytest

from manialab.audio_cache import load_stem_activations, save_stem_activations


def activations():
    return (
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.arange(12, dtype=np.float32).reshape(3, 4) / 10,
        np.arange(3, dtype=np.float64) / 86,
    )


def test_stem_activation_cache_round_trips_the_existing_format(tmp_path):
    cache = tmp_path / "vocals-activations.npz"
    expected = activations()

    save_stem_activations(cache, expected, "audio-hash", "separation-hash")

    loaded = load_stem_activations(cache, "audio-hash", "separation-hash")
    assert loaded is not None
    for actual, wanted in zip(loaded, expected):
        np.testing.assert_array_equal(actual, wanted)
    with np.load(cache, allow_pickle=False) as data:
        assert set(data.files) == {
            "frames", "onsets", "times", "version", "audio_sha256", "separation_sha256"
        }
        assert int(data["version"]) == 3


@pytest.mark.parametrize(
    "audio_hash,separation_hash",
    [("other-audio", "separation-hash"), ("audio-hash", "other-separation")],
)
def test_stem_activation_cache_requires_both_source_identities(
    tmp_path, audio_hash, separation_hash
):
    cache = tmp_path / "activations.npz"
    save_stem_activations(cache, activations(), "audio-hash", "separation-hash")

    assert load_stem_activations(cache, audio_hash, separation_hash) is None


@pytest.mark.parametrize("kind", ["old-version", "missing-field", "corrupt", "missing-file"])
def test_stem_activation_cache_treats_invalid_entries_as_misses(tmp_path, kind):
    cache = tmp_path / "activations.npz"
    if kind == "old-version":
        np.savez_compressed(
            cache,
            frames=activations()[0],
            onsets=activations()[1],
            times=activations()[2],
            version=2,
            audio_sha256="audio-hash",
            separation_sha256="separation-hash",
        )
    elif kind == "missing-field":
        np.savez_compressed(cache, version=3, audio_sha256="audio-hash")
    elif kind == "corrupt":
        cache.write_bytes(b"not an npz archive")

    assert load_stem_activations(cache, "audio-hash", "separation-hash") is None
