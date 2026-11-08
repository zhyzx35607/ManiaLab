"""Audio timing estimates with explicit ambiguity and local consistency checks.

The FFT proposes periods; weighted phase folding and robust regression fit the
actual attacks over the whole song. A tempo is not considered safe for snapping
merely because it has a strong autocorrelation peak.
"""
from __future__ import annotations

from dataclasses import replace
from bisect import bisect_right
import math

import numpy as np
from scipy.fft import rfft, rfftfreq, next_fast_len
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

DIVISIONS = (3, 4, 6, 8, 12, 16)


def chart_clock_notes(notes, audio_shift_ms=0):
    """Convert measured audio times to an explicit mapping convention.

    A positive shift places chart events earlier. This is not a decoder fix or
    a change to the source audio; raw detection remains available in analysis.
    """
    if not math.isfinite(audio_shift_ms) or abs(audio_shift_ms) > 100:
        raise ValueError("制谱提前量必须在 -100–100 ms 内")
    return [replace(n, start=max(0, math.floor(n.start-audio_shift_ms+.5)),
                    end=max(0, math.floor(n.end-audio_shift_ms+.5))) for n in notes]


class BeatClock:
    """Read original red lines for arrangement without changing their timestamps."""

    def __init__(self, timing_points=(), fallback=500.):
        points = {}
        for line in timing_points:
            fields = line.split(',')
            try:
                offset, period = float(fields[0]), float(fields[1])
                red = len(fields) < 7 or fields[6].strip() == '1'
            except (ValueError, IndexError):
                continue
            if red and math.isfinite(offset) and math.isfinite(period) and period > 0:
                points[offset] = period
        self.offsets = sorted(points)
        self.periods = [points[t] for t in self.offsets]
        self.fallback = fallback

    def __call__(self, time):
        if not self.offsets:
            return self.fallback
        return self.periods[max(0, bisect_right(self.offsets, time) - 1)]


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


def cover_missing_timing_windows(tempo, primary, fallback, primary_name='drums'):
    """Do not let a selected source silently omit an intro or breakdown.

    The original source, tempo, phase and acceptance gates stay unchanged.
    Only 12s windows without the five attacks needed by grid_diagnostics use
    mixture attacks instead. Fallback is tested on the SAME proposed grid.
    A conflict can revoke automatic acceptance; this never turns a rejection
    into an acceptance or treats two correlated sources as independent votes.
    """
    supported = {s['start_ms'] for s in grid_diagnostics(primary,tempo['bpm'],tempo['offset'],tempo.get('division',4))}
    fill = {s['start_ms'] for s in grid_diagnostics(fallback,tempo['bpm'],tempo['offset'],tempo.get('division',4))} - supported
    combined = [n for n in primary if n.start//12000*12000 not in fill]
    combined.extend(n for n in fallback if n.start//12000*12000 in fill)
    combined.sort(key=lambda n:(n.start,n.id))
    sections = grid_diagnostics(combined,tempo['bpm'],tempo['offset'],tempo.get('division',4))
    for section in sections:
        section['evidence_source'] = 'mix' if section['start_ms'] in fill else primary_name
        section['fallback'] = section['start_ms'] in fill
    added = [s for s in sections if s['fallback']]
    conflict = any(s['suspect'] for s in added)
    revised = dict(tempo, sections=sections, stable=bool(tempo.get('stable') and not conflict),
                   uncovered_source_windows=len(added),timing_coverage_revision=1)
    if conflict:
        revised['reliability'] = '部分无鼓或起音稀少的段落与当前拍线不一致；请检查前奏、间奏或变速，再确认 Timing'
        suspect = sum(s['suspect'] for s in sections)
        revised['confidence'] = round(tempo.get('coverage',0)*(1-suspect/max(1,len(sections))),3)
    return revised, combined


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


def estimate_timing(attacks, duration, beats_ms=None):
    empty = dict(bpm=120., offset=0., estimated=True, confidence=0., stable=False,
                 candidates=[], precise_candidates=[], sections=[], division=4, reliability="节奏证据不足，请设置 BPM / Offset 后试听")
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
    if beats_ms is not None and len(beats_ms) >= 17:
        gaps = np.diff(np.asarray(beats_ms, dtype=float))
        gaps = gaps[np.isfinite(gaps) & (gaps >= 200) & (gaps <= 1400)]
        if len(gaps):
            for gap in np.percentile(gaps, [25, 50, 75]):
                for factor in (.5, 1, 2):
                    bpm = 60000 / gap * factor
                    if 45 <= bpm <= 300 and all(abs(bpm - s) > .12 for s in seeds):
                        seeds.append(float(bpm))
    fitted = [_fit(times, weights, bpm, division) for bpm in seeds for division in (4, 3)]
    fitted.sort(key=lambda c: c["score"], reverse=True)
    normalized, precise = {}, {}
    for raw in fitted:
        bpm = normalize_bpm(raw["bpm"])
        key = (bpm, raw["division"])
        if not 44 <= bpm <= 301 or key in normalized:
            continue
        c = refit_offset(attacks, bpm, raw["division"])
        c.update(raw_bpm=raw["bpm"], raw_offset=raw["offset"])
        exact = _precision_candidate(attacks, raw, c, duration)
        if exact:
            precise[key] = exact
            c.update(exact_bpm=exact["bpm"], rounding_drift_ms=exact["rounding_drift_ms"])
        normalized[key] = c
    candidates = []
    if beats_ms is not None:
        for c in normalized.values():
            c["beat_support"] = round(beat_meter_support(beats_ms, c.get("exact_bpm", c["bpm"])), 4)
    use_meter = max((c.get("beat_support", 0) for c in normalized.values()), default=0) >= .55
    def rank(c):
        exact = precise.get((c["bpm"], c["division"]))
        # Do not replace the strongest original tempo with a different harmonic
        # just because rounding damaged it. Keep it visible with a warning.
        acoustic = max(c["score"], exact["score"] * .98) if exact else c["score"]
        return acoustic + (c.get("beat_support", 0) * .24 if use_meter else 0)
    for c in sorted(normalized.values(), key=rank, reverse=True):
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
    stable = best["coverage"] >= .8 and best["median_error_ms"] <= 18 and suspect == 0 and len(sections) >= 2 and "exact_bpm" not in best
    confidence = round(best["coverage"] * (1 - suspect / max(1, len(sections))), 3)
    explanation = "起音与网格整体一致；仍需试听确认 Offset、倍速 / 半速和弱起" if stable else "拍线不稳定或证据不足：请校准 BPM / Offset 并试听确认后再生成"
    if "exact_bpm" in best:
        explanation = (f"取整为 {best['bpm']:g} BPM 可能造成约 {best['rounding_drift_ms']}ms 的整曲漂移。"
                       f"另有 {best['exact_bpm']:g} BPM 精确候选，请比较歌曲开头与结尾后采用。")
    strongest = max((c["score"] for c in normalized.values()), default=0)
    strongest = max(strongest, max((c["score"] for c in precise.values()), default=0))
    precise_candidates = sorted((c for c in precise.values() if c["score"] >= strongest * .85), key=lambda c: c["score"], reverse=True)[:4]
    return {**best, "estimated": True, "confidence": confidence, "stable": stable,
            "candidates": candidates, "precise_candidates": precise_candidates, "sections": sections, "reliability": explanation,
            "meter_model_used": use_meter}


def grid_report(notes, tempo):
    """Check every exported endpoint, allowing only integer-ms serialization."""
    tick = 60000 / tempo["bpm"] / tempo.get("division", 4)
    times = [t for n in notes for t in ([n.start, n.end] if n.hold else [n.start])]
    errors = [abs((t - tempo["offset"] + tick / 2) % tick - tick / 2) for t in times]
    return dict(endpoints=len(times), off_grid=sum(e > .501 for e in errors),
                max_error_ms=round(max(errors, default=0), 4), tolerance_ms=.501)


def align_candidates(notes, tempo, enabled=True, tolerance=30, confirmed=False):
    """Quantize both endpoints; keep original evidence in the analysis, not here.

    tolerance is a reporting threshold, never a bypass for off-grid objects.
    A short interval collapsing to one tick becomes a tap.
    """
    if not enabled:
        raise ValueError("从音频生成需要完整拍线量化；原始检测时间可在「采音证据」查看")
    if not (confirmed or tempo.get("stable", False)):
        raise ValueError("当前自动 Timing 不可靠。请校准 BPM / Offset，试听并勾选确认后再生成")
    bpm, offset = float(tempo["bpm"]), float(tempo["offset"])
    division = int(tempo.get("division", 4))
    tick = 60000 / bpm / division
    first = math.ceil(-offset / tick)
    def snap(time):
        return round(offset + max(first, math.floor((time - offset) / tick + .5)) * tick)
    aligned, deltas, tail_deltas = [], [], []
    collapsed = 0
    for n in notes:
        start = snap(n.start)
        end = max(start, snap(n.end)) if n.hold else start
        if n.hold:
            tail_deltas.append(abs(end - n.end))
            collapsed += end == start
        if start != n.start:
            deltas.append(abs(start - n.start))
        aligned.append(replace(n, start=start, end=end, raw="",
            evidence_start=n.evidence_start if n.evidence_start is not None else n.start,
            evidence_end=n.evidence_end if n.evidence_end is not None else n.end))
    return aligned, dict(enabled=True, moved=len(deltas), tails_moved=sum(d > 0 for d in tail_deltas),
                         max_shift_ms=max(deltas + tail_deltas, default=0),
                         large_attack_shifts=sum(d > tolerance for d in deltas),
                         collapsed_holds=collapsed, warning_threshold_ms=tolerance,
                         reason="短键与 LN 起止均已量化；超过 30ms 的起音移动需回听检查，落线不代表采音正确")
