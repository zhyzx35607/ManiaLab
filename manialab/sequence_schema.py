"""Versioned feature names and dimensions for mania sequence windows."""

WINDOW_ROWS = 4
MAX_LANES = 10
LANE_FEATURES = (
    "attack",
    "hold_head",
    "hold_duration",
    "occupied_before",
    "hold_remaining",
    "since_attack",
    "since_release",
    "release_between_rows",
    "valid_lane",
)
ROW_FEATURES = ("event", "chord_share", "hold_share", "gap_beats")
KEY_FEATURES = tuple(f"keys_{keys}" for keys in range(6, 11))
GLOBAL_FEATURES = KEY_FEATURES + ("beat_seconds",)
ROW_WIDTH = MAX_LANES * len(LANE_FEATURES) + len(ROW_FEATURES)
FEATURE_COUNT = WINDOW_ROWS * ROW_WIDTH + len(GLOBAL_FEATURES)
FEATURE_NAMES = tuple(
    name
    for row in range(WINDOW_ROWS)
    for name in (
        *(f"row_{row}_lane_{lane}_{feature}"
          for lane in range(MAX_LANES) for feature in LANE_FEATURES),
        *(f"row_{row}_{feature}" for feature in ROW_FEATURES),
    )
) + GLOBAL_FEATURES
LN_INTERACTION_ROW_FEATURES = (
    "active_ln_density",
    "active_ln_same_hand_pairs",
    "active_ln_cross_hand_pairs",
    "ln_head_density",
    "tap_head_density",
    "release_density",
    "hybrid_same_hand_taps",
    "hybrid_cross_hand_taps",
    "hybrid_same_hand_ln_heads",
    "hybrid_cross_hand_ln_heads",
    "release_press_same_hand",
    "release_press_cross_hand",
    "split_release_hands",
    "inverse_ln_heads",
    "reverse_shield_taps",
    "shield_ln_heads",
    "near_release_press",
)
LN_INTERACTION_FEATURE_NAMES = tuple(
    f"interaction_row_{row}_{feature}"
    for row in range(WINDOW_ROWS)
    for feature in LN_INTERACTION_ROW_FEATURES
)
LN_INTERACTION_FEATURE_COUNT = len(LN_INTERACTION_FEATURE_NAMES)
LN_INTERACTION_SCHEMA = "sequence-window-ln-interactions-v1"
LN_PHRASE_CONTEXT_SCHEMA = "sequence-window-ln-phrase-context-v3"
LN_PHRASE_CONTEXT_BEATS = 8.0
LN_PHRASE_CONTEXT_BINS = 16
LN_PHRASE_CONTEXT_FEATURES = (
    "tap_heads",
    "ln_heads",
    "ln_releases",
    "active_ln_fraction",
    "active_ln_same_hand_pairs",
    "active_ln_cross_hand_pairs",
    "hybrid_same_hand_taps",
    "hybrid_cross_hand_taps",
    "hybrid_same_hand_ln_heads",
    "hybrid_cross_hand_ln_heads",
    "release_press_same_hand",
    "release_press_cross_hand",
    "near_release_press_same_hand",
    "near_release_press_cross_hand",
    "ln_duration_short",
    "ln_duration_medium",
    "ln_duration_long",
)
LN_PHRASE_CONTEXT_FEATURE_NAMES = tuple(
    f"phrase_bin_{row}_{feature}"
    for row in range(LN_PHRASE_CONTEXT_BINS)
    for feature in LN_PHRASE_CONTEXT_FEATURES
)
FEATURE_SCHEMA_NAMES = {
    "sequence-window-v1": FEATURE_NAMES,
    LN_INTERACTION_SCHEMA: FEATURE_NAMES + LN_INTERACTION_FEATURE_NAMES,
    LN_PHRASE_CONTEXT_SCHEMA: (
        FEATURE_NAMES
        + LN_INTERACTION_FEATURE_NAMES
        + LN_PHRASE_CONTEXT_FEATURE_NAMES
    ),
}
FEATURE_SCHEMA_COUNTS = {
    schema: len(names) for schema, names in FEATURE_SCHEMA_NAMES.items()
}
