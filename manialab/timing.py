"""Audio timing estimates with explicit ambiguity and local consistency checks.

The FFT proposes periods; weighted phase folding and robust regression fit the
actual attacks over the whole song. A tempo is not considered safe for snapping
merely because it has a strong autocorrelation peak.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.fft import rfft, rfftfreq, next_fast_len
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


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
    offset %= period
    # Reconsider the beat phase, since regression operates on subdivisions.
    offsets = [(offset + k * period / division) % period for k in range(division)]
    best = None
    for phase in offsets:
        pos = (times - phase) / period
        beat_dist = np.abs(pos - np.round(pos)) * period
        half_dist = np.abs(pos * 2 - np.round(pos * 2)) * period / 2
        fine_dist = np.abs(pos * division - np.round(pos * division)) * period / division
        proximity = np.maximum(np.exp(-.5 * (beat_dist / 14) ** 2),
                    np.maximum(.72 * np.exp(-.5 * (half_dist / 14) ** 2),
                               .5 * np.exp(-.5 * (fine_dist / 14) ** 2)))
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


def grid_diagnostics(attacks, bpm, offset, division=4):
    """Describe local phase drift instead of pretending all songs are constant BPM."""
    if not attacks:
        return []
    times = np.asarray([n.start for n in attacks], float)
    weights = np.asarray([max(.05, n.strength) for n in attacks])
    tick = 60000 / bpm / division
    result = []
    for a in range(0, int(times[-1]) + 1, 12000):
        mask = (times >= a) & (times < a + 12000)
        if mask.sum() < 5:
            continue
        residual = (times[mask] - offset + tick / 2) % tick - tick / 2
        coverage = float(np.average(np.abs(residual) < min(25, tick * .2), weights=weights[mask]))
        phase = _phase(times[mask] - offset, weights[mask], tick)
        shift = (phase + tick / 2) % tick - tick / 2
        result.append(dict(start_ms=a, end_ms=a + 12000, attacks=int(mask.sum()),
                           coverage=round(coverage, 3), phase_error_ms=round(shift, 2),
                           suspect=coverage < .65 or abs(shift) > 25))
    return result


def estimate_timing(attacks, duration):
    empty = dict(bpm=120., offset=0., estimated=True, confidence=0., stable=False,
                 candidates=[], sections=[], division=4, reliability="节奏证据不足，请设置 BPM / Offset 后试听")
    if len(attacks) < 8 or attacks[-1].start - attacks[0].start < 5000:
        return empty
    times = np.asarray([n.start for n in attacks], float)
    weights = np.asarray([max(.05, n.strength) for n in attacks])
    # 2 ms impulse train; FFT padding improves the candidate frequency spacing,
    # and fitting below removes the remaining FFT-bin quantization.
    fs = 500
    pulse = np.zeros(int(duration * fs) + 2)
    np.add.at(pulse, np.minimum(len(pulse) - 1, np.round(times * fs / 1000).astype(int)), weights)
    nfft = next_fast_len(len(pulse) * 8)
    spectrum = np.abs(rfft(pulse, nfft))
    freqs = rfftfreq(nfft, 1 / fs)
    peaks, _ = find_peaks(spectrum)
    peaks = peaks[(freqs[peaks] >= .75) & (freqs[peaks] <= 20)]
    peaks = sorted(peaks, key=lambda i: spectrum[i], reverse=True)[:45]
    seeds = []
    for i in peaks:
        for harmonic in (1, 2, 3, 4):
            bpm = freqs[i] * 60 / harmonic
            if 45 <= bpm <= 300 and all(abs(bpm - s) > .12 for s in seeds):
                seeds.append(float(bpm))
    fitted = [_fit(times, weights, bpm, division) for bpm in seeds for division in (4, 3)]
    fitted.sort(key=lambda c: c["score"], reverse=True)
    candidates = []
    for c in fitted:
        if not 44 <= c["bpm"] <= 301 or any(abs(c["bpm"] - p["bpm"]) < .15 for p in candidates):
            continue
        candidates.append(c)
        if len(candidates) == 6:
            break
    if not candidates:
        return empty
    best = candidates[0]
    sections = grid_diagnostics(attacks, best["bpm"], best["offset"], best["division"])
    suspect = sum(s["suspect"] for s in sections)
    stable = best["coverage"] >= .8 and best["median_error_ms"] <= 18 and suspect == 0 and len(sections) >= 2
    confidence = round(best["coverage"] * (1 - suspect / max(1, len(sections))), 3)
    explanation = "全曲拍线一致；仍需试听确认倍速 / 半速和弱起" if stable else "拍线不稳定或证据不足：可能有变速、切分或误检；自动吸附已停用"
    return {**best, "estimated": True, "confidence": confidence, "stable": stable,
            "candidates": candidates, "sections": sections, "reliability": explanation}


def align_candidates(notes, tempo, enabled=True, tolerance=30, confirmed=False):
    """Move only nearby attacks on a verified grid; never extend a sustain tail."""
    if not enabled or not (confirmed or tempo.get("stable", False)):
        return list(notes), dict(enabled=False, moved=0, reason="拍线未确认或已关闭吸附")
    bpm, offset = float(tempo["bpm"]), float(tempo["offset"])
    division = int(tempo.get("division", 4))
    tick = 60000 / bpm / division
    limit = min(tolerance, tick * .23)
    aligned, deltas = [], []
    for n in notes:
        start = max(0, round(offset + round((n.start - offset) / tick) * tick))
        if abs(start - n.start) > limit:
            start = n.start
        if n.hold and start >= n.end:
            start = n.start
        end = max(start, n.end) if n.hold else start
        if n.hold:
            snapped_end = round(offset + math.floor((n.end - offset) / tick + 1e-9) * tick)
            if 0 <= n.end - snapped_end <= limit and snapped_end > start:
                end = snapped_end
        if start != n.start:
            deltas.append(abs(start - n.start))
        aligned.append(replace(n, start=start, end=end))
    return aligned, dict(enabled=True, moved=len(deltas), max_shift_ms=max(deltas, default=0),
                          tolerance_ms=limit, reason="仅吸附容差内的起音；LN 尾端仅允许向前收齐")
