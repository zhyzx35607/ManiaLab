import pytest

from manialab.arrange import parameters
from manialab.arrange_config import DEFAULT_KEY_LAYOUTS
from manialab.presets import (
    DEFAULT_PRESET,
    canonical_preset_id,
    get_preset,
    legacy_generation_settings,
    v32_preset_options,
)


@pytest.mark.parametrize(
    "preset_id,expected_keep_jacks",
    [
        ("rice_stream", False),
        ("rice_jack", True),
        ("hybrid_adaptive", True),
        ("ln_focus", True),
        ("torosu_remap", True),
    ],
)
def test_canonical_presets_have_real_torosu_preferences(preset_id, expected_keep_jacks):
    preset = get_preset(preset_id)

    assert preset["id"] == preset_id
    assert preset["torosu"]["keep_jacks"] is expected_keep_jacks
    assert preset["torosu"]["playstyle_by_layout"] == {
        "6k_3_3": "3+3",
        "7k_3_1_3": "3+1+3",
        "8k_3_2_3": "3+2+3",
        "9k_4_1_4": "4+1+4",
        "10k_5_5": "5+5",
    }


def test_torosu_remap_exposes_supported_ln_shape_modes():
    preset = get_preset("torosu_remap")
    assert preset["torosu"]["ln_mode"] == "keep"
    assert preset["torosu"]["ln_modes"] == ("keep", "fln", "short_ln")
    assert legacy_generation_settings("torosu_remap")["torosu_ln_mode"] == "keep"


@pytest.mark.parametrize(
    "preset_id,minimum_ratio",
    [("rice_stream", 0.01), ("rice_jack", 0.01), ("hybrid_adaptive", 0.01)],
)
def test_rice_and_hybrid_presets_preserve_natural_holds(preset_id, minimum_ratio):
    preset = get_preset(preset_id)

    assert preset["settings"]["ln_ratio"] >= minimum_ratio
    assert preset["v32"]["hold_note_ratio"] >= minimum_ratio


@pytest.mark.parametrize("keys", [6, 7, 8, 9, 10])
@pytest.mark.parametrize(
    "preset_id",
    ["rice_stream", "rice_jack", "hybrid_adaptive", "ln_focus", "torosu_remap"],
)
def test_legacy_settings_are_valid_for_every_supported_key_count(preset_id, keys):
    settings = legacy_generation_settings(preset_id, keys=keys)

    assert parameters(settings)["keys"] == keys
    assert settings["key_layout"] == DEFAULT_KEY_LAYOUTS[keys]
    assert settings["layout_engine"] == "torosu"


def test_named_density_override_is_independent_of_note_density():
    settings = legacy_generation_settings("rice_stream", keys=8, density=17)

    assert settings["density"] == 17
    assert settings["ln_ratio"] > 0
    assert settings["torosu_keep_jacks"] is False


@pytest.mark.parametrize(
    "preset_id,torosu_density",
    [
        ("rice_stream", "dense"),
        ("rice_jack", "more-dense"),
        ("hybrid_adaptive", "dense"),
        ("ln_focus", "dense"),
        ("torosu_remap", "none"),
    ],
)
def test_legacy_settings_forward_torosu_density_from_the_registry(preset_id, torosu_density):
    settings = legacy_generation_settings(preset_id, keys=10)
    assert settings["torosu_density"] == torosu_density


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("balanced", "hybrid_adaptive"),
        ("onsets", "rice_stream"),
        ("onset_only", "rice_stream"),
        ("sustain", "ln_focus"),
        ("rhythm", "hybrid_adaptive"),
        ("remap", "torosu_remap"),
    ],
)
def test_legacy_preset_aliases_resolve_to_canonical_records(alias, canonical):
    assert canonical_preset_id(alias) == canonical
    assert get_preset(alias)["id"] == canonical


def test_legacy_aliases_keep_the_previous_generation_parameters():
    expected = {
        "balanced": (7, 0.55, 0.65, "balanced"),
        "onsets": (8, 0, 0, "balanced"),
        "onset_only": (8, 0, 0, "balanced"),
        "sustain": (6, 0.85, 0.85, "pitched"),
        "rhythm": (8, 0.25, 0.25, "transient"),
    }
    for preset_id, (density, ln_ratio, phrase_ln, focus) in expected.items():
        settings = legacy_generation_settings(preset_id, keys=7)
        assert settings["density"] == density
        assert settings["ln_ratio"] == ln_ratio
        assert settings["phrase_ln"] == phrase_ln
        assert settings["focus"] == focus
        assert settings["layout_engine"] == "beam"


def test_default_preset_is_hybrid_adaptive():
    assert DEFAULT_PRESET == "hybrid_adaptive"
    assert get_preset()["id"] == DEFAULT_PRESET


def test_v32_options_resolve_preset_values_and_overrides():
    options = v32_preset_options(
        "rice_stream",
        keys=8,
        model_variant="v32",
        difficulty=6.2,
        hold_note_ratio=0.2,
    )

    assert options["preset"] == "rice_stream"
    assert options["keys"] == 8
    assert options["model_variant"] == "v32"
    assert options["difficulty"] == 6.2
    assert options["hold_note_ratio"] == 0.2
    assert options["generation_strategy"] == "7k_to_torosu"
    assert options["torosu_keep_jacks"] is False
    assert options["torosu_ln_mode"] == "keep"

    with pytest.raises(ValueError, match="不使用 V32"):
        v32_preset_options("torosu_remap")


def test_preset_returns_independent_records():
    first = get_preset("ln_focus")
    first["settings"]["ln_ratio"] = 0
    assert get_preset("ln_focus")["settings"]["ln_ratio"] == 0.55


def test_unknown_or_non_text_preset_is_rejected():
    with pytest.raises(ValueError, match="未知制谱预设"):
        get_preset("missing")
    with pytest.raises(ValueError, match="必须为文本"):
        get_preset(None)
