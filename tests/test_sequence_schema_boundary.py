from manialab import pattern_sequence, sequence_schema, sequence_windows


def test_versioned_schema_values_have_one_owner_and_stable_reexports():
    exports = (
        "WINDOW_ROWS",
        "MAX_LANES",
        "LANE_FEATURES",
        "ROW_FEATURES",
        "KEY_FEATURES",
        "GLOBAL_FEATURES",
        "ROW_WIDTH",
        "FEATURE_COUNT",
        "FEATURE_NAMES",
        "LN_INTERACTION_ROW_FEATURES",
        "LN_INTERACTION_FEATURE_NAMES",
        "LN_INTERACTION_FEATURE_COUNT",
        "LN_INTERACTION_SCHEMA",
        "LN_PHRASE_CONTEXT_SCHEMA",
        "LN_PHRASE_CONTEXT_BEATS",
        "LN_PHRASE_CONTEXT_BINS",
        "LN_PHRASE_CONTEXT_FEATURES",
        "LN_PHRASE_CONTEXT_FEATURE_NAMES",
        "FEATURE_SCHEMA_NAMES",
        "FEATURE_SCHEMA_COUNTS",
    )
    for name in exports:
        value = getattr(sequence_schema, name)
        assert getattr(pattern_sequence, name) is value
        if hasattr(sequence_windows, name):
            assert getattr(sequence_windows, name) is value
