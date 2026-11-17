"""Lane reflection for all supported sequence feature schemas."""
from __future__ import annotations

import numpy as np

from .sequence_schema import (
    FEATURE_SCHEMA_COUNTS,
    LANE_FEATURES,
    MAX_LANES,
    ROW_WIDTH,
    WINDOW_ROWS,
)
from .sequence_features_common import _positive_keys


def mirror_window_features(x, keys):
    """Mirror lane-indexed features without changing row order or global data."""
    keys = _positive_keys(keys)
    values = np.asarray(x, dtype=np.float32)
    valid_counts = tuple(FEATURE_SCHEMA_COUNTS.values())
    if values.ndim not in (1, 2) or values.shape[-1] not in valid_counts:
        raise ValueError(
            "x must have shape (width,) or (batch, width) for a supported "
            f"feature schema with width in {valid_counts}"
        )
    mirrored = np.array(values, copy=True)
    row_data = mirrored[..., : WINDOW_ROWS * ROW_WIDTH].reshape(
        -1, WINDOW_ROWS, ROW_WIDTH
    )
    lanes = row_data[:, :, : MAX_LANES * len(LANE_FEATURES)].reshape(
        -1, WINDOW_ROWS, MAX_LANES, len(LANE_FEATURES)
    )
    lanes[:, :, :keys] = lanes[:, :, :keys][:, :, ::-1].copy()
    return mirrored
