"""Lightweight arrangement defaults and parameter validation."""
import math


KEY_LAYOUT_PROFILES = {
    6: {
        "6k_3_3": ((0, 4), (0, 3), (0, 2), (1, 2), (1, 3), (1, 4)),
    },
    7: {
        "7k_3_1_3": ((0, 4), (0, 3), (0, 2), (2, 1), (1, 2), (1, 3), (1, 4)),
    },
    8: {
        "8k_3_2_3": ((0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4)),
        "8k_4_4": ((0, 5), (0, 4), (0, 3), (0, 2), (1, 2), (1, 3), (1, 4), (1, 5)),
    },
    9: {
        "9k_4_1_4": ((0, 5), (0, 4), (0, 3), (0, 2), (2, 1), (1, 2), (1, 3), (1, 4), (1, 5)),
    },
    10: {
        "10k_5_5": ((0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)),
    },
}
DEFAULT_KEY_LAYOUTS = {
    6: "6k_3_3",
    7: "7k_3_1_3",
    8: "8k_3_2_3",
    9: "9k_4_1_4",
    10: "10k_5_5",
}
TOROSU_PLAYSTYLES = {
    "6k_3_3": "3+3",
    "7k_3_1_3": "3+1+3",
    "8k_3_2_3": "3+2+3",
    "9k_4_1_4": "4+1+4",
    "10k_5_5": "5+5",
}


def merge_parameters(base=None, updates=None):
    """Merge a partial settings update, resetting layout when key count changes."""
    merged = dict(base or {})
    updates = updates or {}
    previous_keys = merged.get("keys")
    merged.update(updates)
    if ("keys" in updates and "key_layout" not in updates
            and updates["keys"] != previous_keys):
        # A layout belongs to a specific key count. Do not carry the previous
        # count's layout forward; parameters() will select the new default.
        merged.pop("key_layout", None)
    return merged


def key_layout_profile(keys, key_layout=None):
    """Return one shared lane-to-hand/finger profile for a supported key mode."""
    choices = KEY_LAYOUT_PROFILES.get(keys)
    if choices is None:
        raise ValueError(f"不支持 {keys}K 键型")
    key_layout = DEFAULT_KEY_LAYOUTS[keys] if key_layout is None else key_layout
    profile = choices.get(key_layout) if isinstance(key_layout, str) else None
    if profile is None:
        raise ValueError(f"{keys}K 不支持键型 {key_layout!r}")
    return profile


DEFAULTS = dict(keys=10, key_layout="10k_5_5", density=7.0, ln_ratio=.55, min_ln=240, max_held=5, release_gap=55,
                chord_size=4, pitch_follow=.55, balance=.8, jack_avoid=.8, seed=42,
                onset_threshold=.45, frame_threshold=.3, focus="balanced", phrase_ln=.65, layout_engine="beam", pattern_prior=.35,
                drums_weight=1., bass_weight=1., vocals_weight=1., other_weight=1., guitar_weight=1., piano_weight=1.,
                ln_coordination=.35, ln_release=.65, ln_inverse=0.)
MAX_DENSITY = 50
BOUNDS = dict(keys=(6, 10), density=(1, MAX_DENSITY), ln_ratio=(0, 1), min_ln=(100, 2000), max_held=(1, 10),
              release_gap=(0, 250), chord_size=(1, 10), pitch_follow=(0, 1), balance=(0, 1),
              jack_avoid=(0, 1), seed=(0, 2147483647), onset_threshold=(.1, .9), frame_threshold=(.1, .8), phrase_ln=(0, 1), pattern_prior=(0,1),
              drums_weight=(0,2), bass_weight=(0,2), vocals_weight=(0,2), other_weight=(0,2), guitar_weight=(0,2), piano_weight=(0,2),
              ln_coordination=(0,1), ln_release=(0,1), ln_inverse=(0,1))
INTEGER = {"keys", "min_ln", "max_held", "release_gap", "chord_size", "seed"}


def parameters(raw=None):
    raw = raw or {}
    result = DEFAULTS.copy()
    for k, v in raw.items():
        if k in BOUNDS:
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError(f"参数 {k} 必须是有限数字")
            lo, hi = BOUNDS[k]
            if not lo <= v <= hi or (k in INTEGER and int(v) != v):
                raise ValueError(f"参数 {k} 必须在 {lo}–{hi} 范围内")
            result[k] = int(v) if k in INTEGER else float(v)
        elif k == "focus":
            if v not in ("balanced", "pitched", "transient"):
                raise ValueError("采音侧重无效")
            result[k] = v
        elif k == "layout_engine":
            if v not in ("beam", "classic", "torosu"):
                raise ValueError("定轨引擎无效")
            result[k] = v
        elif k == "key_layout":
            if v is not None and not isinstance(v, str):
                raise ValueError("键型无效")
            result[k] = v
        else:
            raise ValueError(f"未知参数：{k}")
    # Old saved settings have no key_layout field. Resolve them after reading
    # keys so each key count gets its canonical layout (including the new 8K default).
    key_layout = raw.get("key_layout") if "key_layout" in raw else None
    if key_layout is None:
        key_layout = DEFAULT_KEY_LAYOUTS[result["keys"]]
    key_layout_profile(result["keys"], key_layout)
    if result["layout_engine"] == "torosu" and key_layout not in TOROSU_PLAYSTYLES:
        raise ValueError("Torosu 不支持 8K 4+4 键型；请切换到 8K 3+2+3 键型")
    result["key_layout"] = key_layout
    result["max_held"] = min(result["keys"], result["max_held"])
    result["chord_size"] = min(result["keys"], result["chord_size"])
    return result
