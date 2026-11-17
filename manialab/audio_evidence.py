"""Lightweight waveform checks for source provenance and separation bleed."""
from __future__ import annotations

import numpy as np

from .audio_config import SR


def align_stem_samples(stem, length):
    """Reconcile only final-sample rounding from two equivalent resample paths.

    FFmpeg's direct 48k->22.05k and 48k->44.1k->22.05k paths can differ by
    one sample at EOF. Never move the start or conceal a meaningful mismatch.
    """
    if stem.ndim != 1 or not np.isfinite(stem).all() or abs(len(stem)-length) > 2:
        raise ValueError("分离声部长度或采样值无效，请重新分析")
    difference = length-len(stem)
    if difference > 0:
        stem = np.pad(stem, (0, difference))
    elif difference < 0:
        stem = stem[:length]
    return stem, difference


def source_evidence(notes, stem, mixture, sample_rate=SR):
    """Reject numerical/leakage residue before normalizing a source's attacks.

    Per-source peak normalization can turn a -60dB residual into a supposedly
    strong event. Require waveform energy relative to the original local
    mixture, with a small whole-track floor for silent gaps.
    """
    global_level = float(np.sqrt(np.mean(mixture.astype(np.float64) ** 2)))
    kept = []
    for note in notes:
        at = round(note.start * sample_rate / 1000)
        a, b = max(0, at - round(sample_rate * .012)), min(len(mixture), at + round(sample_rate * .06))
        if b <= a:
            continue
        level = float(np.sqrt(np.mean(stem[a:b] ** 2)))
        reference = float(np.sqrt(np.mean(mixture[a:b] ** 2)))
        if level >= max(.00001, global_level * .0015, reference * .01):
            kept.append(note)
    return kept, len(notes) - len(kept)


def timing_evidence_quality(tempo):
    """Compare sources before UI rounding can damage a real fractional beat."""
    return next((c for c in tempo.get("precise_candidates", []) if c["bpm"] == tempo.get("exact_bpm")), tempo)


def reject_source_bleed(notes, stems, mixture_attacks, sample_rate=SR):
    """Reject short weak cross-source duplicates and release-edge imitations.

    A weak 100ms pitch at a stronger source's same-pitch onset or release can
    be separation leakage, not a second instrument. Restrict this test to a
    narrowly contained, low-activation candidate; preserve independent
    attacks and different-pitch voices. Unison instruments remain inherently
    ambiguous.
    """
    by_pitch = {}
    for note in notes:
        if note.source == 'pitch' and note.strength >= .55:
            by_pitch.setdefault(note.pitch, []).append(note)
    attack_times = np.array([n.start for n in mixture_attacks])
    result, rejected = [], []
    for note in notes:
        reason = None
        if note.source == 'pitch' and note.strength < .5 and 0 < note.end-note.start < 200:
            parents = [p for pitch in range(note.pitch-1,note.pitch+2) for p in by_pitch.get(pitch,())
                       if p.voice != note.voice and p.start <= note.start+40 and p.end >= note.end-40
                       and p.end-p.start >= max(250,(note.end-note.start)*3)]
            for parent in parents:
                a = max(0, round(max(note.start,parent.start)*sample_rate/1000))
                b = min(len(stems[note.voice]), round(min(note.end,parent.end,note.start+150)*sample_rate/1000))
                if b-a < sample_rate*.04:
                    continue
                x,y = stems[note.voice][a:b],stems[parent.voice][a:b]
                ratio = float(np.linalg.norm(x)/max(1e-9,np.linalg.norm(y)))
                if abs(note.start-parent.start) <= 40 and ratio < .1:
                    reason='weak_duplicate_attack'
                elif (note.start > parent.start+80 and abs(note.end-parent.end) <= 80 and ratio < .5
                      and not np.any(abs(attack_times-note.start) <= 45)):
                    correlation=float(abs(np.dot(x,y))/max(1e-9,float(np.linalg.norm(x)*np.linalg.norm(y))))
                    if correlation > .85:
                        reason='release_edge_bleed'
                if reason:
                    rejected.append(dict(note=note.data(),parent_id=parent.id,reason=reason,energy_ratio=round(ratio,4)))
                    break
        if not reason:
            result.append(note)
    return result, rejected
