from pathlib import Path
import subprocess
import sys

import pytest

from manialab import arrange, arrange_config


def test_arrange_reexports_config_objects_by_identity():
    for name in ("DEFAULTS", "BOUNDS", "INTEGER", "parameters"):
        assert getattr(arrange, name) is getattr(arrange_config, name)


def test_parameters_returns_an_independent_defaults_copy():
    defaults = arrange_config.DEFAULTS.copy()
    configured = arrange_config.parameters()

    configured["keys"] = 6

    assert arrange_config.DEFAULTS == defaults
    assert configured is not arrange_config.DEFAULTS


def test_density_ceiling_accepts_50_and_rejects_values_above_it():
    assert arrange_config.BOUNDS["density"] == (1, 50)
    assert arrange_config.parameters({"density": 50})["density"] == 50
    with pytest.raises(ValueError, match="1–50"):
        arrange_config.parameters({"density": 50.01})


@pytest.mark.parametrize("keys,expected", [
    (6, "6k_3_3"), (7, "7k_3_1_3"), (8, "8k_3_2_3"),
    (9, "9k_4_1_4"), (10, "10k_5_5"),
])
def test_key_layout_defaults_follow_selected_key_count(keys, expected):
    assert arrange_config.DEFAULTS["keys"] == 10
    assert arrange_config.DEFAULTS["key_layout"] == "10k_5_5"
    assert arrange_config.parameters({"keys": keys})["key_layout"] == expected


def test_legacy_settings_without_key_layout_use_standard_profile():
    legacy = {key: value for key, value in arrange_config.DEFAULTS.items()
              if key != "key_layout"}
    legacy["keys"] = 8

    assert arrange_config.parameters(legacy)["key_layout"] == "8k_3_2_3"


def test_key_layout_validation_rejects_unknown_and_wrong_key_count_pairs():
    for settings in (
        {"keys": 8, "key_layout": "unknown"},
        {"keys": 8, "key_layout": "10k_5_5"},
        {"keys": 6, "key_layout": "8k_4_4"},
    ):
        with pytest.raises(ValueError, match="键型"):
            arrange_config.parameters(settings)
    with pytest.raises(ValueError, match="6–10"):
        arrange_config.parameters({"keys": 11, "key_layout": "10k_5_5"})


def test_merging_new_key_count_resets_inherited_layout_unless_layout_is_explicit():
    current = arrange_config.parameters({"keys": 10, "key_layout": "10k_5_5"})

    changed = arrange_config.parameters(arrange_config.merge_parameters(current, {"keys": 8}))
    explicit = arrange_config.parameters(arrange_config.merge_parameters(
        current, {"keys": 8, "key_layout": "8k_4_4"},
    ))

    assert changed["key_layout"] == "8k_3_2_3"
    assert explicit["key_layout"] == "8k_4_4"


def test_merging_other_settings_preserves_current_key_layout():
    current = arrange_config.parameters({"keys": 8, "key_layout": "8k_4_4"})

    merged = arrange_config.parameters(arrange_config.merge_parameters(current, {"density": 12}))

    assert merged["key_layout"] == "8k_4_4"


def test_merging_the_same_key_count_preserves_current_key_layout():
    current = arrange_config.parameters({"keys": 8, "key_layout": "8k_4_4"})

    merged = arrange_config.parameters(arrange_config.merge_parameters(current, {"keys": 8}))

    assert merged["key_layout"] == "8k_4_4"


def test_arrange_config_import_does_not_load_solver_dependencies():
    code = (
        "import sys; import manialab.arrange_config; "
        "assert 'manialab.arrange' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert not any(name == 'scipy' or name.startswith('scipy.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_config_consumers_do_not_load_arrange_or_scipy():
    code = (
        "import sys; "
        "import manialab.planner, manialab.recommendations, manialab.project_routes; "
        "assert 'manialab.arrange' not in sys.modules; "
        "assert not any(name == 'scipy' or name.startswith('scipy.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
