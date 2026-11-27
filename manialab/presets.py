"""Canonical 6-10K generation presets shared by single and batch workflows."""
from __future__ import annotations

from copy import deepcopy

from .arrange_config import DEFAULT_KEY_LAYOUTS, TOROSU_PLAYSTYLES


DEFAULT_PRESET = "hybrid_adaptive"

_PRESET_REGISTRY = {
    "rice_stream": {
        "id": "rice_stream",
        "label": "切指偏米",
        "mode": "audio",
        "settings": {
            "density": 8,
            "ln_ratio": 0.12,
            "phrase_ln": 0.2,
            "min_ln": 180,
            "max_held": 3,
            "chord_size": 3,
            "focus": "transient",
            "layout_engine": "torosu",
            "torosu_keep_jacks": False,
            "torosu_density": "none",
            "torosu_ln_mode": "keep",
        },
        "v32": {
            "model_variant": "v32-mini",
            "generation_strategy": "7k_to_torosu",
            "difficulty": 5.5,
            "hold_note_ratio": 0.12,
            "mania_column_temperature": 1.0,
        },
        "torosu": {"density": "dense", "keep_jacks": False, "ln_mode": "keep"},
    },
    "rice_jack": {
        "id": "rice_jack",
        "label": "叠键偏米",
        "mode": "audio",
        "settings": {
            "density": 8,
            "ln_ratio": 0.12,
            "phrase_ln": 0.2,
            "min_ln": 180,
            "max_held": 4,
            "chord_size": 5,
            "focus": "balanced",
            "layout_engine": "torosu",
            "torosu_keep_jacks": True,
            "torosu_density": "none",
            "torosu_ln_mode": "keep",
        },
        "v32": {
            "model_variant": "v32-mini",
            "generation_strategy": "7k_to_torosu",
            "difficulty": 5.5,
            "hold_note_ratio": 0.12,
            "mania_column_temperature": 1.0,
        },
        "torosu": {"density": "more-dense", "keep_jacks": True, "ln_mode": "keep"},
    },
    "hybrid_adaptive": {
        "id": "hybrid_adaptive",
        "label": "综合随歌",
        "mode": "audio",
        "settings": {
            "density": 7,
            "ln_ratio": 0.3,
            "phrase_ln": 0.45,
            "min_ln": 240,
            "max_held": 5,
            "chord_size": 4,
            "focus": "balanced",
            "layout_engine": "torosu",
            "torosu_keep_jacks": True,
            "torosu_density": "none",
            "torosu_ln_mode": "keep",
        },
        "v32": {
            "model_variant": "v32-mini",
            "generation_strategy": "7k_to_torosu",
            "difficulty": 5.5,
            "hold_note_ratio": 0.3,
            "mania_column_temperature": 1.0,
        },
        "torosu": {"density": "dense", "keep_jacks": True, "ln_mode": "keep"},
    },
    "ln_focus": {
        "id": "ln_focus",
        "label": "精致 LN",
        "mode": "audio",
        "settings": {
            "density": 6,
            "ln_ratio": 0.55,
            "phrase_ln": 0.75,
            "min_ln": 300,
            "max_held": 4,
            "chord_size": 3,
            "focus": "pitched",
            "layout_engine": "torosu",
            "torosu_keep_jacks": True,
            "torosu_density": "none",
            "torosu_ln_mode": "keep",
        },
        "v32": {
            "model_variant": "v32-mini",
            "generation_strategy": "7k_to_torosu",
            "difficulty": 5.5,
            "hold_note_ratio": 0.55,
            "mania_column_temperature": 0.8,
        },
        "torosu": {"density": "dense", "keep_jacks": True, "ln_mode": "keep"},
    },
    "torosu_remap": {
        "id": "torosu_remap",
        "label": "已有谱 Torosu 转键/改造",
        "mode": "remap",
        "settings": {
            "density": 7,
            "ln_ratio": 0.55,
            "phrase_ln": 0.65,
            "min_ln": 240,
            "max_held": 5,
            "chord_size": 4,
            "focus": "balanced",
            "layout_engine": "torosu",
            "torosu_keep_jacks": True,
            "torosu_density": "none",
            "torosu_ln_mode": "keep",
        },
        "v32": None,
        "torosu": {
            "density": "none",
            "keep_jacks": True,
            "ln_mode": "keep",
            "ln_modes": ("keep", "fln", "short_ln"),
        },
    },
}
PRESET_IDS = tuple(_PRESET_REGISTRY)

_ALIASES = {
    "balanced": "hybrid_adaptive",
    "onset": "rice_stream",
    "onsets": "rice_stream",
    "onset_only": "rice_stream",
    "sustain": "ln_focus",
    "rhythm": "hybrid_adaptive",
    "remap": "torosu_remap",
}

# Keep the pre-registry API's concrete values for clients that still send old
# preset ids. New UI paths use the canonical registry above.
_LEGACY_SETTINGS = {
    "balanced": {
        "density": 7,
        "ln_ratio": 0.55,
        "phrase_ln": 0.65,
        "min_ln": 240,
        "max_held": 5,
        "chord_size": 4,
        "focus": "balanced",
    },
    "onset": {
        "density": 8,
        "ln_ratio": 0,
        "phrase_ln": 0,
        "min_ln": 240,
        "max_held": 5,
        "chord_size": 4,
        "focus": "balanced",
    },
    "onsets": {
        "density": 8,
        "ln_ratio": 0,
        "phrase_ln": 0,
        "min_ln": 240,
        "max_held": 5,
        "chord_size": 4,
        "focus": "balanced",
    },
    "onset_only": {
        "density": 8,
        "ln_ratio": 0,
        "phrase_ln": 0,
        "min_ln": 240,
        "max_held": 5,
        "chord_size": 4,
        "focus": "balanced",
    },
    "sustain": {
        "density": 6,
        "ln_ratio": 0.85,
        "phrase_ln": 0.85,
        "min_ln": 300,
        "max_held": 4,
        "chord_size": 3,
        "focus": "pitched",
    },
    "rhythm": {
        "density": 8,
        "ln_ratio": 0.25,
        "phrase_ln": 0.25,
        "min_ln": 240,
        "max_held": 3,
        "chord_size": 4,
        "focus": "transient",
    },
}


def canonical_preset_id(preset_id: str) -> str:
    """Resolve the old preset names without making them separate presets."""
    if not isinstance(preset_id, str):
        raise ValueError("制谱预设必须为文本")
    key = preset_id.strip().casefold().replace("-", "_")
    canonical = _ALIASES.get(key, key)
    if canonical not in _PRESET_REGISTRY:
        raise ValueError(f"未知制谱预设：{preset_id}")
    return canonical


def get_preset(preset_id: str = DEFAULT_PRESET) -> dict:
    """Return a fresh registry record with key-layout playstyles attached."""
    requested = preset_id.strip().casefold().replace("-", "_") if isinstance(preset_id, str) else preset_id
    canonical = canonical_preset_id(preset_id)
    result = deepcopy(_PRESET_REGISTRY[canonical])
    result["requested_id"] = requested
    result["aliases"] = tuple(
        alias for alias, target in _ALIASES.items() if target == canonical
    )
    result["torosu"]["playstyle_by_layout"] = dict(TOROSU_PLAYSTYLES)
    return result


def legacy_generation_settings(
    preset_id: str = DEFAULT_PRESET,
    *,
    keys: int = 10,
    density: float | None = None,
) -> dict:
    """Translate a preset into the established ``parameters()`` settings."""
    if type(keys) is not int or keys not in DEFAULT_KEY_LAYOUTS:
        raise ValueError("目标键数必须为 6–10")
    preset = get_preset(preset_id)
    settings = deepcopy(preset["settings"])
    legacy_id = str(preset.get("requested_id") or "").casefold().replace("-", "_")
    if legacy_id in _LEGACY_SETTINGS:
        settings.update(_LEGACY_SETTINGS[legacy_id])
        settings["layout_engine"] = "beam"
    settings["keys"] = keys
    settings["key_layout"] = DEFAULT_KEY_LAYOUTS[keys]
    settings["torosu_density"] = preset["torosu"]["density"]
    settings["torosu_ln_mode"] = preset["torosu"]["ln_mode"]
    if density is not None:
        settings["density"] = density
    return settings


def v32_preset_options(
    preset_id: str = DEFAULT_PRESET,
    *,
    keys: int = 10,
    model_variant: str | None = None,
    difficulty: float | None = None,
    hold_note_ratio: float | None = None,
    mania_column_temperature: float | None = None,
    torosu_density: str | None = None,
    torosu_keep_jacks: bool | None = None,
    torosu_ln_mode: str | None = None,
) -> dict:
    """Build V32 request fields from the canonical preset and user overrides."""
    if type(keys) is not int or keys not in DEFAULT_KEY_LAYOUTS:
        raise ValueError("目标键数必须为 6–10")
    preset = get_preset(preset_id)
    if preset["mode"] != "audio" or not isinstance(preset["v32"], dict):
        raise ValueError("已有谱 Torosu 改造不使用 V32 音频写谱")
    values = dict(preset["v32"])
    if model_variant is not None:
        values["model_variant"] = model_variant
    if difficulty is not None:
        values["difficulty"] = difficulty
    if hold_note_ratio is not None:
        values["hold_note_ratio"] = hold_note_ratio
    if mania_column_temperature is not None:
        values["mania_column_temperature"] = mania_column_temperature
    values.update(
        keys=keys,
        preset=preset["id"],
        torosu_density=(
            torosu_density if torosu_density is not None else preset["torosu"]["density"]
        ),
        torosu_keep_jacks=(
            torosu_keep_jacks
            if torosu_keep_jacks is not None
            else preset["torosu"]["keep_jacks"]
        ),
        torosu_ln_mode=(
            torosu_ln_mode
            if torosu_ln_mode is not None
            else preset["torosu"]["ln_mode"]
        ),
    )
    return values
