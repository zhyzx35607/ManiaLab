"""Lightweight arrangement defaults and parameter validation."""
import math


DEFAULTS = dict(keys=10, density=7.0, ln_ratio=.55, min_ln=240, max_held=5, release_gap=55,
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
    result = DEFAULTS.copy()
    for k, v in (raw or {}).items():
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
            if v not in ("beam", "classic"):
                raise ValueError("定轨引擎无效")
            result[k] = v
        else:
            raise ValueError(f"未知参数：{k}")
    result["max_held"] = min(result["keys"], result["max_held"])
    result["chord_size"] = min(result["keys"], result["chord_size"])
    return result
