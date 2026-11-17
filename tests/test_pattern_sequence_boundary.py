from pathlib import Path
import subprocess
import sys

from manialab import (
    pattern_sequence,
    sequence_features_base,
    sequence_features_common,
    sequence_features_interactions,
    sequence_features_mirror,
    sequence_features_phrase_context,
    sequence_schema,
)


def test_pattern_sequence_reexports_versioned_encoder_apis():
    exports = (
        ("_positive_keys", sequence_features_common),
        ("_time", sequence_features_common),
        ("_state_times", sequence_features_common),
        ("_beat_duration", sequence_features_common),
        ("_normalize_rows", sequence_features_common),
        ("window_features", sequence_features_base),
        ("window_features_with_ln_interactions", sequence_features_interactions),
        ("_normalize_phrase_rows", sequence_features_phrase_context),
        ("_phrase_hand_weights", sequence_features_phrase_context),
        ("_ln_phrase_context_features", sequence_features_phrase_context),
        ("window_features_with_ln_phrase_context", sequence_features_phrase_context),
        ("mirror_window_features", sequence_features_mirror),
    )
    for name, module in exports:
        assert getattr(pattern_sequence, name) is getattr(module, name)

    for name in ("FEATURE_NAMES", "FEATURE_SCHEMA_NAMES", "FEATURE_SCHEMA_COUNTS"):
        assert getattr(pattern_sequence, name) is getattr(sequence_schema, name)


def test_feature_modules_do_not_import_the_compatibility_facade():
    code = """import sys
from manialab import sequence_features_base
from manialab import sequence_features_common
from manialab import sequence_features_interactions
from manialab import sequence_features_mirror
from manialab import sequence_features_phrase_context
assert 'manialab.pattern_sequence' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
