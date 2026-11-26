"""Compatibility imports for the V32 audio-to-chart utilities.

Inference, model discovery, and cache management live in :mod:`v32_engine`.
The old one-off four-way/no-vocal experiment runner was retired.
"""
from __future__ import annotations

from .v32_audio import (
    _audio_duration_ms,
    _decode_for_energy,
    choose_climax_window,
    extract_audio_excerpt,
)
from .v32_charts import (
    repair_native_10k_collisions,
    sanitize_zero_length_model_holds,
    shift_reference_timing,
    snap_chart_to_grid,
    trim_air_ln_tails,
    validate_map_grid,
)
from .v32_packaging import _CHART_LABELS, package_four_difficulties

__all__ = [
    "choose_climax_window",
    "extract_audio_excerpt",
    "package_four_difficulties",
    "repair_native_10k_collisions",
    "sanitize_zero_length_model_holds",
    "shift_reference_timing",
    "snap_chart_to_grid",
    "trim_air_ln_tails",
    "validate_map_grid",
]
