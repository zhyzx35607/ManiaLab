"""Model-independent audio event extraction and correction."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, stft

from .audio_config import FRAME, SR
from .domain import Note
from .timing import estimate_timing


def pitched_notes(frames, onsets, times, duration, onset_threshold=.45, frame_threshold=.3, attacks=None, suppressed=()):
    # infer() retains frame starts strictly inside the audio. A final frame's
    # cell can extend past EOF, but observations after EOF are not valid release
    # evidence (for example when a caller pairs a cache with a shorter source).
    if np.any(times >= duration):
        raise ValueError("采音时间戳超出音频时长，请重新分析匹配的音源")
    notes = []
    suppressed = {(r['pitch'], r['time_ms']) for r in suppressed}
    attack_times = np.array([n.start / 1000 for n in attacks]) if attacks is not None else None
    for pitch in range(88):
        peaks, _ = find_peaks(onsets[:, pitch], height=onset_threshold, distance=3)
        if len(onsets) > 1 and onsets[0, pitch] >= onset_threshold and onsets[0, pitch] > onsets[1, pitch]:
            peaks = np.r_[0, peaks]
        accepted = []
        for i in peaks:
            if (pitch+21, round(float(times[i])*1000)) in suppressed:
                continue
            before = frames[max(0, i - 6):i, pitch]
            after = frames[i + 3:i + 12, pitch]
            supported = attack_times is None or bool(np.any(np.abs(attack_times - times[i]) <= .045))
            # The model can fire another onset during a release, splitting one
            # long sound into an LN plus a fake tap. Reject only this falling
            # tail shape when independent attack evidence is also absent.
            release_peak = (len(before) >= 3 and len(after) >= 4 and
                            np.median(before) >= frame_threshold and
                            np.mean(after) < np.median(before) * .8 and
                            np.min(after) < frame_threshold)
            if not supported and release_peak:
                continue
            accepted.append(i)
        peaks = np.asarray(accepted, dtype=int)
        for j, start in enumerate(peaks):
            stop = int(peaks[j + 1]) if j + 1 < len(peaks) else len(times)
            release_reason = 'same_pitch_reattack' if j + 1 < len(peaks) else 'input_boundary'
            last, gap = int(start), 0
            for k in range(int(start), stop):
                if frames[k, pitch] >= frame_threshold:
                    last, gap = k, 0
                else:
                    gap += 1
                    if gap > 4:
                        # This is a model activation decay, not a physical
                        # key release. EOF with only 1–4 low frames is unresolved.
                        release_reason = 'frame_decay'
                        break
            if last - start < 3:
                continue
            a = max(0, round(float(times[start]) * 1000))
            b = min(round(duration * 1000), round((float(times[last]) + FRAME / SR) * 1000))
            # Retained windows overlap by less than one model frame. The last
            # cell before that seam must not extend through the next accepted
            # same-pitch attack, whose actual sample clock takes precedence.
            if stop < len(times):
                b = min(b, round(float(times[stop]) * 1000))
            if b <= a:
                continue
            strength = float(np.mean(frames[start:last + 1, pitch]))
            notes.append(Note(a, b, -1, pitch + 21, round(strength, 4), "pitch", f"p{pitch}:{a}",
                              release_reason=release_reason))
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def rhythmic_evidence(samples, beats_ms=None, estimate=True):
    _, times, z = stft(samples, SR, nperseg=1024, noverlap=768)
    magnitude = np.abs(z)
    flux = np.r_[0, np.maximum(0, np.diff(magnitude, axis=1)).sum(axis=0)]
    energy = np.sqrt(np.sum(magnitude ** 2, axis=0))
    rise = energy - np.r_[np.zeros(min(3, len(energy))), energy[:-3]]
    # Spectral spreading at a release is not a new attack. Require a small energy rise.
    flux[rise <= np.maximum(np.max(energy) * .002, energy * .01)] = 0
    amplitude = np.max(magnitude, axis=0)
    flux[amplitude < max(.00003, float(np.max(amplitude)) * .004)] = 0
    if not np.any(flux):
        return [], {"bpm": 120.0, "offset": 0.0, "estimated": True, "reliability": "无可靠节奏证据"}
    base = median_filter(flux, size=43)
    peaks, _ = find_peaks(flux, height=base * 1.6 + np.max(flux) * .025, distance=3, prominence=np.max(flux) * .02)
    scale = max(float(np.percentile(flux[peaks], 90)) if len(peaks) else 1, 1e-9)
    notes = []
    # Refine each STFT attack on a 1 ms RMS envelope. The flux peak itself can
    # precede or follow the physical attack by a window-dependent amount.
    step = 22
    usable = len(samples) // step * step
    rms = np.sqrt(np.mean(samples[:usable].reshape(-1, step) ** 2, axis=1))
    smooth = np.convolve(rms, np.ones(3) / 3, mode="same")
    for i in peaks:
        center = int(times[i] * SR / step)
        lo, hi = max(0, center - 30), min(len(smooth), center + 25)
        if hi - lo < 3:
            continue
        region = smooth[lo:hi]
        peak = int(np.argmax(region))
        floor = float(np.percentile(region[:max(1, peak)], 15))
        threshold = floor + (float(region[peak]) - floor) * .2
        crossing = peak
        while crossing > 0 and region[crossing - 1] > threshold:
            crossing -= 1
        at = max(0, round((lo + crossing) * step / SR * 1000))
        if notes and at - notes[-1].start < 35:
            continue
        notes.append(Note(at, at, -1, 48, round(min(1, float(flux[i]) / scale), 4), "transient", f"t{i}"))
    return notes, estimate_timing(notes, len(samples) / SR, beats_ms) if estimate else None


def fuse_attacks(pitched, transients):
    """Anchor attacks while preserving retriggers and their release bounds."""
    times = np.array([n.start for n in transients])
    if not len(times):
        return pitched, transients, 0
    used, corrected = set(), []
    for n in pitched:
        i = int(np.searchsorted(times, n.start))
        nearby = [j for j in (i - 1, i) if 0 <= j < len(times)]
        j = min(nearby, key=lambda j: abs(int(times[j]) - n.start))
        # A small correction is allowed; remote transients are unrelated voices.
        if abs(int(times[j]) - n.start) <= 40 and times[j] < n.end:
            used.add(j)
            n = replace(n, start=int(times[j]))
        corrected.append(n)
    tracks = {}
    for index, note in enumerate(pitched):
        tracks.setdefault((note.voice, note.instrument_id, note.pitch), []).append(index)
    for indices in tracks.values():
        indices.sort(key=lambda i: pitched[i].start)
        # One broadband transient may support several pitches in a chord, but
        # must not collapse successive same-pitch attacks onto one timestamp.
        # Reverting one correction can expose a conflict with its neighbour.
        while True:
            reverted = False
            for left, right in zip(indices, indices[1:]):
                if (pitched[left].start < pitched[right].start and
                        corrected[left].start >= corrected[right].start):
                    for index in (left, right):
                        if corrected[index].start != pitched[index].start:
                            corrected[index] = replace(corrected[index], start=pitched[index].start)
                            reverted = True
            if not reverted:
                break
        for left, right in zip(indices, indices[1:]):
            note, following = corrected[left], corrected[right]
            if (note.release_reason == 'same_pitch_reattack' and
                    note.start < following.start < note.end):
                # This endpoint was defined by the next attack, not a measured
                # decay. Carry an earlier corroborated attack into that bound;
                # a later attack is not evidence to lengthen the old note.
                corrected[left] = replace(note, end=following.start)
    return corrected, [n for j, n in enumerate(transients) if j not in used], len(used)


def refine_releases(notes, samples, sample_rate=SR):
    """Remove model tails extending into a source's near-silence.

    A 5ms waveform envelope can shorten a detected interval, never lengthen
    one. Other simultaneous sounds may keep a tail ambiguous; in that case
    the pitch model's original endpoint is preserved.
    """
    step = max(1, round(sample_rate * .005))
    padded = np.pad(samples, (0, (-len(samples)) % step))
    level = np.sqrt(np.mean(padded.reshape(-1, step) ** 2, axis=1))
    floor = max(.00001, float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) * .002)
    result, shifts = [], []
    for note in notes:
        a = max(0, int(note.start * sample_rate / 1000 / step))
        b = min(len(level), int(np.ceil(note.end * sample_rate / 1000 / step)))
        region = level[a:b]
        if not note.hold or not len(region):
            result.append(note)
            continue
        threshold = max(floor, float(np.percentile(region, 85)) * .02)
        active = np.flatnonzero(region >= threshold)
        end = min(note.end, round((a + int(active[-1]) + 1) * step / sample_rate * 1000)) if len(active) else note.start
        end = max(note.start, end)
        if end < note.end:
            shifts.append(note.end - end)
        result.append(replace(note, end=end))
    return result, dict(shortened=len(shifts), max_trim_ms=max(shifts, default=0),
                        reason="只收回进入声部近静音的模型音尾，不延长持续音")
