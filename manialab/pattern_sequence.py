"""Versioned sequence feature API with stable re-exports for runtime callers."""
from __future__ import annotations

from .sequence_schema import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_COUNTS,
    FEATURE_SCHEMA_NAMES,
    GLOBAL_FEATURES,
    KEY_FEATURES,
    LANE_FEATURES,
    LN_INTERACTION_FEATURE_COUNT,
    LN_INTERACTION_FEATURE_NAMES,
    LN_INTERACTION_ROW_FEATURES,
    LN_INTERACTION_SCHEMA,
    LN_PHRASE_CONTEXT_BEATS,
    LN_PHRASE_CONTEXT_BINS,
    LN_PHRASE_CONTEXT_FEATURE_NAMES,
    LN_PHRASE_CONTEXT_FEATURES,
    LN_PHRASE_CONTEXT_SCHEMA,
    MAX_LANES,
    ROW_FEATURES,
    ROW_WIDTH,
    WINDOW_ROWS,
)
from .sequence_features_common import (
    _beat_duration,
    _normalize_rows,
    _positive_keys,
    _state_times,
    _time,
)
from .sequence_features_base import (
    _LANE_INDEX,
    _MAX_HISTORY_BEATS,
    _MAX_HOLD_BEATS,
    _MAX_REMAINING_BEATS,
    _ROW_INDEX,
    _UNKNOWN_TIME,
    window_features,
)
from .sequence_features_interactions import window_features_with_ln_interactions
from .sequence_features_phrase_context import (
    _ln_phrase_context_features,
    _normalize_phrase_rows,
    _phrase_hand_weights,
    window_features_with_ln_phrase_context,
)
from .sequence_features_mirror import mirror_window_features
