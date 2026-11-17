from manialab import pattern_sequence, sequence_prior, sequence_windows


def test_sequence_prior_facade_preserves_window_runtime_exports():
    pattern_exports = (
        "FEATURE_COUNT",
        "FEATURE_NAMES",
        "FEATURE_SCHEMA_COUNTS",
        "FEATURE_SCHEMA_NAMES",
        "LN_INTERACTION_SCHEMA",
        "LN_INTERACTION_ROW_FEATURES",
        "LN_PHRASE_CONTEXT_SCHEMA",
        "LANE_FEATURES",
        "MAX_LANES",
        "ROW_FEATURES",
        "ROW_WIDTH",
        "WINDOW_ROWS",
        "mirror_window_features",
        "window_features",
        "window_features_with_ln_interactions",
        "window_features_with_ln_phrase_context",
    )
    window_exports = (
        "MIN_SEQUENCE_BEAT_MS",
        "MAX_SEQUENCE_BEAT_MS",
        "SEQUENCE_PRIOR_REVISION",
        "SequenceRow",
        "_sequence_windows_ln_interactions",
        "sequence_beat_clock",
        "sequence_beat_positions",
        "sequence_window_features",
        "sequence_windows_features",
    )
    for name in pattern_exports:
        assert getattr(sequence_prior, name) is getattr(pattern_sequence, name)
    for name in window_exports:
        assert getattr(sequence_prior, name) is getattr(sequence_windows, name)


def test_sequence_window_module_does_not_own_prior_model_lifecycle():
    assert not hasattr(sequence_windows, "SequencePrior")
    assert not hasattr(sequence_windows, "load_sequence_prior")
    assert not hasattr(sequence_windows, "score_sequence_windows")
