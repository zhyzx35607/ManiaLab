"""Versioned Basic Pitch activation cache for separated audio stems."""
import hashlib
from pathlib import Path

import numpy as np


STEM_ACTIVATION_CACHE_VERSION = 3
_SHA256_CHUNK_SIZE = 1024 * 1024


def sha256_file(path, cancelled=None):
    """Hash a file incrementally without copying its full contents into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("已取消音频校验")
            chunk = stream.read(_SHA256_CHUNK_SIZE)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def load_stem_activations(path, audio_sha256, separation_sha256):
    """Load cached activations only when both source identities still match."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if (int(data["version"]) != STEM_ACTIVATION_CACHE_VERSION
                    or str(data["audio_sha256"]) != audio_sha256
                    or str(data["separation_sha256"]) != separation_sha256):
                return None
            return data["frames"], data["onsets"], data["times"]
    except (ValueError, KeyError, OSError, EOFError):
        return None


def save_stem_activations(path, activations, audio_sha256, separation_sha256):
    """Write the current cache format used by generic separated-stem analysis."""
    path = Path(path)
    frames, onsets, times = activations
    np.savez_compressed(
        path,
        frames=frames,
        onsets=onsets,
        times=times,
        version=STEM_ACTIVATION_CACHE_VERSION,
        audio_sha256=audio_sha256,
        separation_sha256=separation_sha256,
    )
