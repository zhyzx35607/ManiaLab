"""Local acoustic-change segmentation, with explicit editable boundaries.

These are acoustic sections, not learned verse/chorus labels. Time is always
the original audio clock; only the exporter applies the mapping clock shift.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.signal import find_peaks, stft
from scipy.ndimage import gaussian_filter1d

REVISION = 1
STEP_MS = 500


def reference_marks(text):
    from .domain import sections, values
    parts = sections(text)
    bookmarks = []
    for value in values(parts.get('Editor', [])).get('Bookmarks', '').split(','):
        try:
            time = int(value)
            if time >= 0:
                bookmarks.append(time)
        except ValueError:
            pass
    green = []
    for line in parts.get('TimingPoints', []):
        p = line.split(',')
        if len(p) >= 8 and p[6].strip() == '0':
            try:
                green.append(dict(time_ms=round(float(p[0])), beat_length=float(p[1]),
                                  volume=int(p[5]), kiai=bool(int(p[7]) & 1)))
            except ValueError:
                continue
    return dict(bookmarks=sorted(set(bookmarks)), green_lines=green)


def acoustic_features(samples, sample_rate, candidates, duration_ms):
    count = max(1, math.ceil(duration_ms / STEP_MS))
    features = []
    names = []
    if samples is not None and len(samples) >= 2048:
        # Aggregate before retaining features: memory grows linearly in audio
        # duration, not quadratically as a full song similarity matrix would.
        freq, times, spectrum = stft(samples, fs=sample_rate, nperseg=2048,
                                    noverlap=1024, boundary=None, padded=False)
        power = np.abs(spectrum) ** 2
        bins = np.clip((times * 1000 / STEP_MS).astype(int), 0, count - 1)
        counts = np.maximum(1, np.bincount(bins, minlength=count))
        bands = np.geomspace(35, sample_rate / 2, 17)
        energy = np.array([power[(freq >= a) & (freq < b)].sum(axis=0)
                           for a, b in zip(bands[:-1], bands[1:])])
        band_level = np.array([np.bincount(bins, weights=row, minlength=count) / counts for row in energy]).T
        # Spectral shape, separate from the total energy.
        shape = np.sqrt(band_level / np.maximum(band_level.sum(axis=1, keepdims=True), 1e-12))
        features.extend(shape.T)
        names.extend(f'band_{i}' for i in range(16))
        features.append(np.log1p(band_level.sum(axis=1) * 1e5))
        names.append('energy')
    voices = sorted({n.get('voice', 'mix') for n in candidates})
    for voice in voices:
        attacks = np.zeros(count)
        sustain = np.zeros(count)
        for n in candidates:
            if n.get('voice', 'mix') != voice or n.get('strength', 0) < .35:
                continue
            a = min(count - 1, max(0, n['start'] // STEP_MS))
            b = min(count, math.ceil(n['end'] / STEP_MS))
            attacks[a] += 1
            sustain[a:b] += 1
        features.extend([np.log1p(attacks), np.log1p(sustain)])
        names.extend([voice + '_attacks', voice + '_sustain'])
    if not features:
        return np.zeros((count, 1)), ['silence']
    x = np.asarray(features, dtype=float).T
    # Robust channel scaling, so one loud residual stem cannot dominate.
    scale = np.maximum(np.percentile(x, 90, axis=0) - np.percentile(x, 10, axis=0), .15)
    x = np.clip(x / scale, 0, 6)
    return gaussian_filter1d(x, 1, axis=0), names


def describe_sections(boundaries, analysis, features=None):
    notes = analysis.get('candidates', [])
    result, representatives = [], []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        local = [n for n in notes if a <= n['start'] < b]
        vector = None
        group = None
        if features is not None:
            lo, hi = a // STEP_MS, max(a // STEP_MS + 1, math.ceil(b / STEP_MS))
            vector = np.mean(features[lo:hi], axis=0)
            norm = np.linalg.norm(vector)
            for index, previous in enumerate(representatives):
                distance = np.linalg.norm(vector - previous) / max(.1, norm + np.linalg.norm(previous))
                if distance < .15:
                    group = index
                    break
        if group is None:
            group = len(representatives)
            representatives.append(vector if vector is not None else np.zeros(1))
        result.append(dict(start_ms=a, end_ms=b, label=f'段 {len(result)+1}',
            similarity_group=f'S{group+1}',
            pitched=sum(n.get('source') == 'pitch' for n in local),
            transients=sum(n.get('source') == 'transient' for n in local),
            sustains_over_800ms=sum(n['end'] - n['start'] >= 800 for n in local),
            voices={v: sum(n.get('voice', 'mix') == v for n in local) for v in sorted({n.get('voice', 'mix') for n in local})}))
    return result


def detect_structure(analysis, samples=None, sample_rate=22050):
    length = round(analysis['duration'] * 1000)
    if length <= 0:
        raise ValueError('歌曲长度无效')
    x, names = acoustic_features(samples, sample_rate, analysis.get('candidates', []), length)
    novelty = np.zeros(len(x))
    sums = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    for width in (8, 16):  # Compare 4- and 8-second contexts on either side.
        for i in range(1, len(x)):
            a, b = max(0, i-width), min(len(x), i+width)
            left = (sums[i]-sums[a])/(i-a)
            right = (sums[b]-sums[i])/(b-i)
            novelty[i] += np.mean((right-left)**2) / 2
    novelty = gaussian_filter1d(novelty, 1)
    middle = novelty[8:-8] if len(novelty) > 16 else novelty
    median = float(np.median(middle))
    mad = float(np.median(np.abs(middle-median)))
    peaks, _ = find_peaks(novelty, distance=16, prominence=max(.03, mad*.7),
                          height=max(.04, median + mad*.6))
    tempo = analysis.get('tempo', {})
    bpm = float(tempo.get('bpm', 120))
    beat = 60000 / bpm if 10 <= bpm <= 1000 else 500
    offset = float(tempo.get('offset', 0))
    boundaries, evidence = [0], []
    for index in peaks:
        raw = int(index * STEP_MS)
        snapped = round(offset + round((raw-offset)/beat)*beat)
        if snapped < 4000 or length-snapped < 4000 or snapped-boundaries[-1] < 6000:
            continue
        boundaries.append(snapped)
        evidence.append(dict(time_ms=snapped, raw_ms=raw, novelty=round(float(novelty[index]), 4)))
    boundaries.append(length)
    return dict(revision=REVISION, source='acoustic-change' if samples is not None else 'detected-activity',
        duration_ms=length, boundaries=boundaries,
        segments=describe_sections(boundaries, analysis, x), boundary_evidence=evidence,
        feature_names=names, notice='按音色、能量及声部活动变化建议分段；S 编号仅表示统计相似，不等于主歌／副歌识别。请试听并编辑边界。')


def edit_boundaries(structure, values, analysis, source='manual'):
    if not isinstance(values, list) or len(values) > 256 or any(type(v) is not int for v in values):
        raise ValueError('段落边界必须是整数毫秒列表，最多 256 个')
    length = round(analysis['duration'] * 1000)
    if any(v < 0 or v > length for v in values):
        raise ValueError('段落边界超出音频范围')
    boundaries = sorted({0, length, *values})
    if any(b-a < 250 for a, b in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError('相邻段落边界至少相隔 250 ms')
    return dict(structure, revision=REVISION, source=source, duration_ms=length,
                boundaries=boundaries, segments=describe_sections(boundaries, analysis), boundary_evidence=[])
