"""Trace continuous vocal activity as optional phrase-LN evidence.

A sung syllable can move across several pitches while remaining one sustained
gesture. This module keeps its support explicit, and never applies this rule to
the mixed accompaniment (which can remain active for an entire song).
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .domain import Note, UNOBSERVED_RELEASES


def vocal_phrases(frames, times, samples, notes, sample_rate=22050, max_length_ms=6000):
    if not len(times) or frames.shape[0] != len(times):
        return []
    step = max(1, sample_rate // 100)
    count = len(samples) // step
    if count < 2:
        return []
    rms = np.sqrt(np.mean(samples[:count * step].reshape(count, step) ** 2, axis=1))
    level = np.interp(times, np.arange(count) * step / sample_rate, rms)
    # Both separated waveform energy and independent pitch activation must
    # support the voice. Low-energy reverb alone does not extend the phrase.
    strength = frames.max(axis=1)
    threshold = max(.002, float(np.percentile(rms, 80)) * .12)
    active = (strength >= .3) & (level >= threshold)
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    joined = []
    frame_ms = float(np.median(np.diff(times))) * 1000 if len(times) > 1 else 12.
    for a, b in runs:
        if joined and (times[a] - times[joined[-1][1] - 1]) * 1000 <= 85:
            joined[-1][1] = b
        else:
            joined.append([a, b])
    result = []
    for begin, finish in joined:
        # Very long continuous regions are divided at a local energy valley,
        # never at the next note in a lane or an arbitrary inverse boundary.
        while (times[finish - 1] - times[begin]) * 1000 > max_length_ms:
            lo = int(np.searchsorted(times, times[begin] + max_length_ms / 1000 * .6))
            hi = min(finish - 1, int(np.searchsorted(times, times[begin] + max_length_ms / 1000)))
            split = lo + int(np.argmin(gaussian_filter1d(level[lo:hi + 1], 2)))
            if split <= begin:
                break
            result.extend(_phrase(begin, split, times, active, strength, notes, frame_ms,
                                  release_reason='forced_segment_boundary'))
            begin = split
        result.extend(_phrase(begin, finish, times, active, strength, notes, frame_ms))
    return result


def _phrase(begin, finish, times, active, strength, notes, frame_ms, *, release_reason=None):
    start = round(times[begin] * 1000)
    end = round(times[finish - 1] * 1000 + frame_ms)
    if end - start < 650:
        return []
    support = [n for n in notes if n.source == "pitch" and n.end > start and n.start < end]
    if not support:
        return []
    coverage = float(active[begin:finish].mean())
    if coverage < .85:
        return []
    anchor = min(support, key=lambda n: (abs(n.start - start), -n.strength))
    if abs(anchor.start - start) > 160:
        return []
    start = anchor.start
    if end - start < 650:
        return []
    if release_reason is None:
        release_reason = 'input_boundary' if finish == len(times) else 'vocal_activity_decay'
    return [replace(anchor, end=end, voice="vocals", phrase=True,
                    support=tuple(n.id for n in support), support_coverage=round(coverage, 4),
                    release_reason=release_reason,
                    strength=round(float(np.mean(strength[begin:finish])), 4))]


def apply_phrase_evidence(candidates, phrases, amount, segment_preferences=()):
    """Substitute a bounded number of existing vocal anchors, no new attacks.

    Original short-pitch candidates remain in analysis. This function chooses
    an expression before quantization, so tails are then treated by the same
    timing and lane constraints as other holds.
    """
    # A phrase is useful only when it carries an existing attack beyond its
    # single pitch. A capped phrase must never truncate a longer original LN.
    original = {n.id: n for n in candidates}
    phrases = [n for n in phrases if n.id in original and n.end > original[n.id].end
               and n.release_reason not in UNOBSERVED_RELEASES]
    if (amount <= 0 and not segment_preferences) or not phrases:
        return list(candidates), dict(available=len(phrases), used=0)
    groups = {}
    for phrase in phrases:
        local_amount = amount
        segment_index = -1
        for index, segment in enumerate(segment_preferences):
            if segment["start_ms"] <= phrase.start < segment["end_ms"] and "phrase_ln" in segment["preferences"]:
                local_amount, segment_index = segment["preferences"]["phrase_ln"], index
        groups.setdefault((segment_index, local_amount), []).append(phrase)
    chosen = []
    for (_, local_amount), local in groups.items():
        chosen.extend(select_phrase_subset(local, local_amount))
    replacements = {n.id: n for n in chosen}
    result = [replacements.get(n.id, n) for n in candidates]
    return result, dict(available=len(phrases), used=sum(n.phrase for n in result),
                        explanation="跨音高人声持续段仅替换已有起音的音尾；保留其他起音，LN 仍需通过占轨约束")


def select_phrase_subset(phrases, amount):
    """Keep the strongest supported phrases within a local expression budget."""
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not np.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError("人声乐句比例必须在0到1之间")
    count = min(len(phrases), int(len(phrases) * amount + .5))
    ranked = sorted(phrases,
        key=lambda n: n.support_coverage * n.strength * np.log1p((n.end - n.start) / 500),
        reverse=True)
    return ranked[:count]
