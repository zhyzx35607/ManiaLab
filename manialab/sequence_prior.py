"""Model loading and scoring for the research-only sequence prior."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .pattern_prior import score_network
from .pattern_sequence import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_COUNTS,
    FEATURE_SCHEMA_NAMES,
    LN_INTERACTION_SCHEMA,
    LN_INTERACTION_ROW_FEATURES,
    LN_PHRASE_CONTEXT_SCHEMA,
    LANE_FEATURES,
    MAX_LANES,
    ROW_FEATURES,
    ROW_WIDTH,
    WINDOW_ROWS,
    mirror_window_features,
    window_features,
    window_features_with_ln_interactions,
    window_features_with_ln_phrase_context,
)
from .sequence_windows import (
    MAX_SEQUENCE_BEAT_MS,
    MIN_SEQUENCE_BEAT_MS,
    SEQUENCE_PRIOR_REVISION,
    SequenceRow,
    _sequence_windows_ln_interactions,
    sequence_beat_clock,
    sequence_beat_positions,
    sequence_window_features,
    sequence_windows_features,
)
from .timing import BeatClock


@dataclass(frozen=True, slots=True)
class SequencePrior:
    layers: tuple[dict[str, np.ndarray], ...]
    sha256: str
    training_manifest_sha256: str = ""
    feature_schema: str = "sequence-window-v1"


def load_sequence_prior(path):
    """Load and validate an explicitly selected research sequence model."""
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    feature_schema = value.get("revision")
    if (not isinstance(feature_schema, str)
            or feature_schema not in FEATURE_SCHEMA_NAMES):
        raise ValueError("Unsupported sequence prior revision")
    feature_names = FEATURE_SCHEMA_NAMES[feature_schema]
    if value.get("features") != list(feature_names):
        raise ValueError("Sequence prior feature schema mismatch")
    manifest_hash = value.get("training_manifest_sha256")
    if not isinstance(manifest_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
        raise ValueError("Sequence prior training manifest hash is invalid")
    exported = value.get("layers")
    if not isinstance(exported, list) or not exported:
        raise ValueError("Sequence prior must contain exported layers")

    layers = []
    width = len(feature_names)
    for index, layer in enumerate(exported):
        weights = np.asarray(layer["weight"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        if (
            weights.ndim != 2
            or weights.shape[1] != width
            or weights.shape[0] == 0
            or bias.shape != (weights.shape[0],)
            or not np.isfinite(weights).all()
            or not np.isfinite(bias).all()
        ):
            raise ValueError(f"Invalid sequence prior layer {index}")
        layers.append({"weight": weights, "bias": bias})
        width = weights.shape[0]
    if width != 1:
        raise ValueError("Sequence prior must produce one score per window")
    return SequencePrior(
        tuple(layers),
        hashlib.sha256(raw).hexdigest(),
        manifest_hash,
        feature_schema,
    )


def score_sequence_windows(windows, keys, prior):
    """Score candidate windows with the training-time left/right mirror average."""
    if not windows:
        return np.empty(0, dtype=np.float32)
    encoded = sequence_windows_features(windows, keys, prior.feature_schema)
    mirrored = mirror_window_features(encoded, keys)
    batch_size = len(encoded)
    scores = score_network(
        np.concatenate((encoded, mirrored), axis=0), prior.layers)
    return (scores[:batch_size] + scores[batch_size:]) / 2
