from pathlib import Path
import subprocess
import sys

from manialab import (
    pattern_sequence,
    sequence_prior,
    sequence_window_interactions,
    sequence_windows,
)


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
    assert sequence_windows._sequence_windows_ln_interactions is (
        sequence_window_interactions._sequence_windows_ln_interactions
    )


def test_sequence_window_module_does_not_own_prior_model_lifecycle():
    assert not hasattr(sequence_windows, "SequencePrior")
    assert not hasattr(sequence_windows, "load_sequence_prior")
    assert not hasattr(sequence_windows, "score_sequence_windows")


def test_sequence_window_interactions_import_does_not_load_runtime_facades():
    code = (
        "import sys; import manialab.sequence_window_interactions; "
        "assert 'manialab.sequence_windows' not in sys.modules; "
        "assert 'manialab.sequence_prior' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
