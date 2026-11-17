"""Robust beat-period and phase fitting primitives for audio timing."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d


def normalize_bpm(bpm):
    """Keep common musical fractions within 0.01 BPM; otherwise round half up.

    This is an inference policy, never applied to a manually supplied tempo.
    Diagnostics must be recomputed after changing the period.
    """
    whole = math.floor(bpm)
    fractions = (.25, .333, .5, .667, .75)
    nearest = min((whole + f for f in fractions), key=lambda x: abs(x - bpm))
    return round(nearest, 3) if abs(nearest - bpm) <= .01 else float(math.floor(bpm + .5))


def _phase(times, weights, period):
    bins = 192
    phases = np.mod(times, period)
    hist = np.bincount((phases / period * bins).astype(int) % bins, weights=weights, minlength=bins)
    smooth = gaussian_filter1d(hist, max(.8, 12 / period * bins), mode="wrap")
    return float(np.argmax(smooth) / bins * period)


def _fit(times, weights, bpm, division):
    period = 60000 / bpm
    offset = _phase(times, weights, period)
    # Fit integer tick numbers, rejecting attacks unrelated to this grid.
    for _ in range(5):
        tick = period / division
        index = np.round((times - offset) / tick)
        residual = times - (offset + index * tick)
        good = np.abs(residual) < min(35, tick * .24)
        if good.sum() < 6 or np.ptp(index[good]) < 12:
            break
        x, y, w = index[good], times[good], weights[good]
        w = w * np.maximum(.05, 1 - (residual[good] / min(35, tick * .24)) ** 2)
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        fitted = float(np.sum(w * (x - xm) * (y - ym)) / max(1e-9, np.sum(w * (x - xm) ** 2)))
        if abs(fitted / tick - 1) > .025:
            break
        period, offset = fitted * division, float(ym - fitted * xm)
    return _score(times, weights, 60000 / period, offset % period, division)


def _score(times, weights, bpm, offset, division, choose_phase=True):
    period = 60000 / bpm
    # Reconsider the beat phase, since regression operates on subdivisions.
    offsets = [(offset + k * period / division) % period for k in range(division)] if choose_phase else [offset]
    best = None
    for phase in offsets:
        pos = (times - phase) / period
        beat_dist = np.abs(pos - np.round(pos)) * period
        half_dist = np.abs(pos * 2 - np.round(pos * 2)) * period / 2
        fine_dist = np.abs(pos * division - np.round(pos * division)) * period / division
        proximity = np.maximum(np.exp(-.5 * (beat_dist / 14) ** 2),
                               .5 * np.exp(-.5 * (fine_dist / 14) ** 2))
        # A /3 grid cannot represent a half-beat. Rewarding those attacks
        # here made a triplet grid win even when its exported ticks missed
        # the straight rhythm. /6 and other even grids do include halves.
        if division % 2 == 0:
            proximity = np.maximum(proximity, .72 * np.exp(-.5 * (half_dist / 14) ** 2))
        # Reward regularly populated beats, without rewarding a grid with twice
        # as many empty beats. Half/double-time remains a user-visible choice.
        hit = beat_dist < 25
        slots = max(1, (times[-1] - times[0]) / period)
        occupied = len(np.unique(np.round(pos[hit]))) / slots
        score = float(np.average(proximity, weights=weights)) * (.65 + .35 * min(1, occupied))
        if best is None or score > best[0]:
            best = (score, phase, fine_dist)
    score, offset, dist = best
    tolerance = min(25, period / division * .2)
    return dict(bpm=round(60000 / period, 5), offset=round(offset, 3), division=division,
                score=round(float(score), 4), coverage=round(float(np.average(dist < tolerance, weights=weights)), 4),
                median_error_ms=round(float(np.median(dist)), 2))


def refit_offset(attacks, bpm, division=4):
    """Fit phase with the BPM fixed, so rounding tempo cannot retain stale phase."""
    if not attacks:
        raise ValueError("没有可用于校准偏移的起音，请先分析音频")
    times = np.asarray([n.start for n in attacks], float)
    weights = np.asarray([max(.05, n.strength) for n in attacks])
    period, tick = 60000 / bpm, 60000 / bpm / division
    offset = _phase(times, weights, tick)
    # A modal phase followed by a weighted median is less sensitive to late
    # soft attacks than a least-squares mean over unrelated voices.
    for _ in range(3):
        residual = (times - offset + tick / 2) % tick - tick / 2
        good = np.abs(residual) < min(30, tick * .24)
        if not np.any(good):
            break
        order = np.argsort(residual[good])
        r, w = residual[good][order], weights[good][order]
        offset += float(r[np.searchsorted(np.cumsum(w), w.sum() / 2)])
    scored = _score(times, weights, bpm, offset % period, division)
    # Evaluate the *integer* offset that will be displayed and exported.
    return _score(times, weights, bpm, round(scored["offset"]), division, choose_phase=False) | {"offset": round(scored["offset"])}


def _precision_candidate(attacks, raw, rounded, duration):
    """Expose a real fractional tempo instead of hiding it as another harmonic.

    Normal rounding remains the default UI policy. Only a substantial loss of
    whole-song fit warrants a separate, explicit precision suggestion. Tiny
    estimator decimals such as 210.00073 never create one.
    """
    span = max(0, attacks[-1].start - attacks[0].start)
    drift = span * abs(1 - raw["bpm"] / rounded["bpm"])
    if (span < 30000 or drift < 25 or raw["coverage"] < .7 or
            raw["score"] < rounded["score"] + .06):
        return None
    # Prefer the shortest decimal that changes the fitted grid by at most 8ms
    # over the observed span, then recompute offset and all quality measures.
    bpm = raw["bpm"]
    for digits in (1, 2, 3):
        simple = round(bpm, digits)
        if span * abs(1 - bpm / simple) <= 8:
            bpm = simple
            break
    candidate = refit_offset(attacks, bpm, raw["division"])
    if candidate["coverage"] < .7 or candidate["score"] < rounded["score"] + .05:
        return None
    candidate.update(raw_bpm=raw["bpm"], rounded_bpm=rounded["bpm"],
        rounding_drift_ms=round(duration * 1000 * abs(1 - bpm / rounded["bpm"])),
        score_gain=round(candidate["score"] - rounded["score"], 4), precision_suggestion=True)
    return candidate


def beat_meter_support(beats_ms, bpm):
    """Support for the *beat unit*, never use model timestamps as an offset.

    Beat This predicts at 50fps. Consecutive gaps may guide half/double-time
    selection but cannot establish a millisecond-accurate physical attack.
    """
    gaps = np.diff(np.asarray(beats_ms, dtype=float))
    gaps = gaps[np.isfinite(gaps) & (gaps >= 150) & (gaps <= 1800)]
    if len(gaps) < 16:
        return 0.
    period = 60000 / bpm
    return float(np.mean(np.exp(-.5 * ((gaps - period) / max(25, period * .04)) ** 2)))
