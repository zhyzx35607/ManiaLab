"""Versioned Basic Pitch activation caches without inference-runtime imports."""
import hashlib
from pathlib import Path
import tempfile
import zipfile

import numpy as np


WHOLE_AUDIO_ACTIVATION_CACHE_VERSION = 3
STEM_ACTIVATION_CACHE_VERSION = 4
# Bump when infer() changes its activation inputs, clock, or output semantics.
BASIC_PITCH_INFERENCE_REVISION = 1
_SHA256_CHUNK_SIZE = 1024 * 1024


def model_identity(path, cancelled=None):
    """Use model bytes when available; keep synthetic/cache-only paths stable."""
    path = Path(path)
    if path.is_file():
        return "sha256:" + sha256_file(path, cancelled)
    return "missing:" + str(path.resolve())


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


def load_stem_activations(path, audio_sha256, separation_sha256,
                          model_identity_value, inference_revision):
    """Load cached activations only when all detector and source identities match."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if (int(data["version"]) != STEM_ACTIVATION_CACHE_VERSION
                    or str(data["audio_sha256"]) != audio_sha256
                    or str(data["separation_sha256"]) != separation_sha256
                    or str(data["model_identity"]) != model_identity_value
                    or int(data["inference_revision"]) != inference_revision):
                return None
            return data["frames"], data["onsets"], data["times"]
    except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile):
        return None


def save_stem_activations(path, activations, audio_sha256, separation_sha256,
                          model_identity_value, inference_revision, *, cancelled=None):
    """Atomically publish the current cache format for generic stem analysis."""
    path = Path(path)
    frames, onsets, times = activations
    temporary = None
    try:
        if cancelled is not None and cancelled():
            raise InterruptedError("已取消音频缓存保存")
        with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.", suffix=".pending",
                dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream,
                frames=frames,
                onsets=onsets,
                times=times,
                version=STEM_ACTIVATION_CACHE_VERSION,
                audio_sha256=audio_sha256,
                separation_sha256=separation_sha256,
                model_identity=model_identity_value,
                inference_revision=inference_revision,
            )
        if cancelled is not None and cancelled():
            raise InterruptedError("已取消音频缓存保存")
        temporary.replace(path)
    except BaseException as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(f"激活缓存暂存文件无法删除（路径：{temporary}）：{cleanup_error!r}")
        raise
