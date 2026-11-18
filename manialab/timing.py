"""Audio timing estimates with explicit ambiguity and local consistency checks.

The FFT proposes periods; weighted phase folding and robust regression fit the
actual attacks over the whole song. A tempo is not considered safe for snapping
merely because it has a strong autocorrelation peak.
"""
from __future__ import annotations

from dataclasses import replace
from bisect import bisect_right
from collections import Counter
import math

import numpy as np
from scipy.fft import rfft, rfftfreq, next_fast_len
from scipy.signal import find_peaks

from .timing_fitting import (
    _fit,
    _phase,
    _precision_candidate,
    _score,
    beat_meter_support,
    normalize_bpm,
    refit_offset,
)

DIVISIONS = (3, 4, 6, 8, 12, 16)
TEMPO_AMBIGUITY_METER_SUPPORT_MARGIN = .2
PRECISE_BPM_DEDUP_TOLERANCE = .001


def chart_clock_notes(notes, audio_shift_ms=0):
    """Convert measured audio times to an explicit mapping convention.

    A positive shift places chart events earlier. This is not a decoder fix or
    a change to the source audio; raw detection remains available in analysis.
    """
    if not math.isfinite(audio_shift_ms) or abs(audio_shift_ms) > 100:
        raise ValueError("制谱提前量必须在 -100–100 ms 内")
    return [replace(n, audio_start=n.audio_start if n.audio_start is not None else n.start,
                    start=max(0, math.floor(n.start-audio_shift_ms+.5)),
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


def grid_diagnostics(attacks, bpm, offset, division=4):
    """Measure chart residuals in fixed original-audio windows.

    Shifting or clipping the chart must not move physical evidence into a
    different diagnostic window. Returned intervals always use audio time.
    """
    if not attacks:
        return []
    times = np.asarray([n.start for n in attacks], float)
    audio_times = np.asarray([n.audio_start if n.audio_start is not None else n.start for n in attacks], float)
    weights = np.asarray([max(.05, n.strength) for n in attacks])
    tick = 60000 / bpm / division
    result = []
    for a in range(0, int(audio_times.max()) + 1, 12000):
        mask = (audio_times >= a) & (audio_times < a + 12000)
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


def _insufficient_timing_windows(attacks, duration_ms):
    """List 12-second windows that lack the five attacks needed for a grid check."""
    times = [n.audio_start if n.audio_start is not None else n.start for n in attacks]
    counts = Counter(int(t // 12000) * 12000 for t in times)
    end = max(math.ceil(duration_ms), max((math.floor(t) + 1 for t in times), default=0))
    return [dict(start_ms=start, end_ms=min(start + 12000, end), attacks=counts[start])
            for start in range(0, end, 12000) if counts[start] < 5]


def analysis_timing_diagnostics(analysis, tempo):
    """Recheck frozen timing evidence without losing its window provenance.

    Source labels describe how the original evidence was assembled. Coverage,
    phase and suspect flags are always recalculated for the requested clock.
    Diagnostic availability never grants automatic Timing acceptance.
    """
    from .domain import note_from
    clock = {k: tempo[k] for k in ('bpm', 'offset')}
    clock.update(division=tempo.get('division', 4), audio_shift_ms=tempo.get('audio_shift_ms', 0))
    attacks = chart_clock_notes([note_from(n) for n in analysis.get('timing_attacks', [])], clock['audio_shift_ms'])
    sections = grid_diagnostics(attacks, clock['bpm'], clock['offset'], clock['division'])
    originals = {(s['start_ms'], s['end_ms']): s for s in analysis.get('tempo', {}).get('sections', [])}
    for section in sections:
        original = originals.get((section['start_ms'], section['end_ms']), {})
        for key in ('evidence_source', 'fallback'):
            if key in original:
                section[key] = original[key]
    insufficient = _insufficient_timing_windows(attacks, analysis.get('duration', 0) * 1000)
    return dict(sections=sections, suspect=sum(s['suspect'] for s in sections),
        insufficient_windows=insufficient,
        status='insufficient' if not sections else 'partial' if insufficient else 'available',
        clock=clock, window_clock='audio')


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


def estimate_timing(attacks, duration, beats_ms=None):
    empty = dict(bpm=120., offset=0., estimated=True, confidence=0., stable=False,
                 candidates=[], precise_candidates=[], sections=[], division=4, tempo_ambiguity=None,
                 reliability="节奏证据不足，请设置 BPM / Offset 后试听")
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
        for c in precise.values():
            c["beat_support"] = round(beat_meter_support(beats_ms, c["bpm"]), 4)
    use_meter = max((c.get("beat_support", 0) for c in
                     [*normalized.values(), *precise.values()]), default=0) >= .55
    def rank(c):
        exact = precise.get((c["bpm"], c["division"]))
        # Do not replace the strongest original tempo with a different harmonic
        # just because rounding damaged it. Keep it visible with a warning.
        acoustic = max(c["score"], exact["score"] * .98) if exact else c["score"]
        return acoustic + (c.get("beat_support", 0) * .24 if use_meter else 0)
    ranked = sorted(normalized.values(), key=rank, reverse=True)
    for c in ranked:
        if not 44 <= c["bpm"] <= 301 or any(abs(c["bpm"] - p["bpm"]) < .15 for p in candidates):
            continue
        candidates.append(c)
        if len(candidates) == 6:
            break
    if not candidates:
        return empty
    best = candidates[0]
    tempo_ambiguity = None
    octave_options = [(candidate, False) for candidate in ranked if candidate is not best]
    octave_options.extend((candidate, True) for candidate in precise.values())
    octave_options.sort(key=lambda row: rank(row[0]), reverse=True)
    octave_candidates = []
    for candidate, is_precise in octave_options:
        if (not 44 <= candidate["bpm"] <= 301 or
                abs(max(candidate["bpm"], best["bpm"]) / min(candidate["bpm"], best["bpm"]) - 2) > .04 or
                candidate["coverage"] < .8 or candidate["median_error_ms"] > 18 or
                any(abs(candidate["bpm"] - existing["bpm"]) <
                    (PRECISE_BPM_DEDUP_TOLERANCE if is_precise and existing_is_precise else .15)
                    for existing, existing_is_precise in octave_candidates)):
            continue
        octave_candidates.append((candidate, is_precise))
    ambiguous_octaves = []
    for candidate, is_precise in octave_candidates:
        if (use_meter and best.get("beat_support", 0) >= .55 and
                best.get("beat_support", 0) - candidate.get("beat_support", 0)
                >= TEMPO_AMBIGUITY_METER_SUPPORT_MARGIN):
            continue
        ambiguous_octaves.append((candidate, is_precise))
    if ambiguous_octaves:
        alternative, _ = ambiguous_octaves[0]
        rank_margin = max(0., rank(best) - rank(alternative))
        support_margin = best.get("beat_support", 0) - alternative.get("beat_support", 0)
        # Only pair-specific beat-model evidence can resolve half/double time;
        # an unrelated or equally supportive model must leave the choice open.
        tempo_ambiguity = dict(competitor_bpm=alternative["bpm"],
                               competitor_bpms=[c["bpm"] for c, _ in ambiguous_octaves],
                               rank_margin=round(rank_margin, 4),
                               beat_support_margin=round(support_margin, 4),
                               best_beat_support=round(best.get("beat_support", 0), 4),
                               competitor_beat_support=round(alternative.get("beat_support", 0), 4),
                               meter_model_used=use_meter)
        for candidate, is_precise in ambiguous_octaves:
            if is_precise:
                continue
            if not any(abs(c["bpm"] - candidate["bpm"]) < .15 for c in candidates):
                candidates.append(candidate)
        candidates.sort(key=rank, reverse=True)
    sections = grid_diagnostics(attacks, best["bpm"], best["offset"], best["division"])
    sparse_windows = [window for window in _insufficient_timing_windows(attacks, duration * 1000)
                      if window['attacks'] and window['end_ms'] - window['start_ms'] == 12000]
    suspect = sum(s["suspect"] for s in sections)
    stable = (best["coverage"] >= .8 and best["median_error_ms"] <= 18 and suspect == 0
              and len(sections) >= 2 and not sparse_windows and not tempo_ambiguity
              and "exact_bpm" not in best)
    confidence = round(best["coverage"] * (1 - suspect / max(1, len(sections))), 3)
    explanation = "起音与网格整体一致；仍需试听确认 Offset、倍速 / 半速和弱起" if stable else "拍线不稳定或证据不足：请校准 BPM / Offset 并试听确认后再生成"
    if sparse_windows:
        explanation = "存在起音稀少但非静音的段落，自动 Timing 无法确认；请检查这些段落后再生成"
    if tempo_ambiguity:
        bpm_options = " / ".join(f"{bpm:g}" for bpm in
                                 [best["bpm"], *tempo_ambiguity["competitor_bpms"]])
        evidence_note = ("拍点模型未能可靠消歧" if tempo_ambiguity["meter_model_used"]
                         else "拍点证据不足")
        explanation += (f"；半速 / 倍速候选 {bpm_options} BPM 均能解释起音，"
                        f"{evidence_note}，请试听确认拍速后再生成")
    if "exact_bpm" in best:
        precision_explanation = (f"取整为 {best['bpm']:g} BPM 可能造成约 {best['rounding_drift_ms']}ms 的整曲漂移。"
                                 f"另有 {best['exact_bpm']:g} BPM 精确候选，请比较歌曲开头与结尾后采用。")
        explanation = (f"{precision_explanation} {explanation}"
                       if sparse_windows or tempo_ambiguity else precision_explanation)
    strongest = max((c["score"] for c in normalized.values()), default=0)
    strongest = max(strongest, max((c["score"] for c in precise.values()), default=0))
    precise_candidates = []
    for candidate in sorted((c for c in precise.values() if c["score"] >= strongest * .85),
                            key=lambda c: c["score"], reverse=True):
        if any(abs(c["bpm"] - candidate["bpm"]) < PRECISE_BPM_DEDUP_TOLERANCE
               for c in precise_candidates):
            continue
        precise_candidates.append(candidate)
        if len(precise_candidates) == 4:
            break
    for candidate, is_precise in ambiguous_octaves:
        if (is_precise and not any(abs(c["bpm"] - candidate["bpm"])
                                   < PRECISE_BPM_DEDUP_TOLERANCE
                                   for c in precise_candidates)):
            precise_candidates.append(candidate)
    return {**best, "estimated": True, "confidence": confidence, "stable": stable,
            "candidates": candidates, "precise_candidates": precise_candidates, "sections": sections,
            "sparse_windows": sparse_windows, "tempo_ambiguity": tempo_ambiguity,
            "reliability": explanation,
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
        aligned.append(replace(n, start=start, end=end, phrase=n.phrase and end>start, raw="",
            evidence_start=n.evidence_start if n.evidence_start is not None else n.start,
            evidence_end=n.evidence_end if n.evidence_end is not None else n.end))
    return aligned, dict(enabled=True, moved=len(deltas), tails_moved=sum(d > 0 for d in tail_deltas),
                         max_shift_ms=max(deltas + tail_deltas, default=0),
                         large_attack_shifts=sum(d > tolerance for d in deltas),
                         collapsed_holds=collapsed, warning_threshold_ms=tolerance,
                         reason="短键与 LN 起止均已量化；超过 30ms 的起音移动需回听检查，落线不代表采音正确")
