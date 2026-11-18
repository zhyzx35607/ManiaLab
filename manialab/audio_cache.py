"""Versioned Basic Pitch activation cache for separated audio stems."""
from pathlib import Path

import numpy as np


STEM_ACTIVATION_CACHE_VERSION = 3


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
