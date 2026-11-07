"""Sample-aligned music/click previews for timing and LN release inspection."""
from __future__ import annotations

import io
import math

import numpy as np
from scipy.io import wavfile


def render_audition(preview, start_ms, duration_ms, events, music_volume=.7, click_volume=.3):
    """Mix a short excerpt and events in one PCM stream, with no UI timer jitter.

    Events are (absolute milliseconds, kind), where kind is beat/accent/head/tail.
    No offset is added to the music. The output begins at an integer sample.
    """
    if not all(math.isfinite(v) for v in (start_ms, duration_ms, music_volume, click_volume)):
        raise ValueError("试听参数必须是有限数字")
    if start_ms < 0 or not 1000 <= duration_ms <= 30000:
        raise ValueError("试听范围必须为 1–30 秒，起点不能为负")
    if not 0 <= music_volume <= 1 or not 0 <= click_volume <= 1:
        raise ValueError("试听音量必须在 0–1 范围内")
    sample_rate, data = wavfile.read(preview, mmap=True)
    first = round(start_ms * sample_rate / 1000)
    last = min(len(data), first + round(duration_ms * sample_rate / 1000))
    if first >= last:
        raise ValueError("试听起点超过音频结尾")
    source = np.asarray(data[first:last], dtype=np.float32)
    if np.issubdtype(data.dtype, np.integer):
        source /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if source.ndim == 1:
        source = np.repeat(source[:, None], 2, axis=1)
    mixed = source * music_volume
    audible_events, seen = [], set()
    for time_ms, kind in sorted(events):
        if kind not in ("beat", "accent", "head", "tail") or not math.isfinite(time_ms):
            continue
        position = round(time_ms * sample_rate / 1000) - first
        if position < 0 or position >= len(mixed):
            continue
        # A chord is one click, not a louder click for every object in the row.
        if (position, kind) in seen:
            continue
        seen.add((position, kind))
        length = min(round(.04 * sample_rate), len(mixed) - position)
        t = np.arange(length) / sample_rate
        frequency = {"accent": 1600, "beat": 1150, "head": 900, "tail": 550}[kind]
        click = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 110)
        mixed[position:position + length] += (click * click_volume)[:, None]
        audible_events.append(dict(time_ms=time_ms, sample=position, kind=kind))
    # One constant gain preserves event phase and the music's dynamics.
    peak = float(np.max(np.abs(mixed)))
    gain = min(1., .98 / max(peak, .001))
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, np.round(mixed * gain * 32767).astype(np.int16))
    buffer.seek(0)
    return buffer, dict(sample_rate=sample_rate, first_sample=first, samples=len(mixed),
                        events=audible_events, gain=gain)


def grid_events(bpm, offset, start_ms, duration_ms, division=1):
    tick = 60000 / bpm / division
    first = math.ceil((start_ms - offset) / tick)
    last = math.ceil((start_ms + duration_ms - offset) / tick)
    return [(offset + i * tick, "accent" if i % (4 * division) == 0 else "beat")
            for i in range(first, last)]


def waveform_window(preview, start_ms, duration_ms, bins=1600):
    """Return min/max envelopes with original sample positions for close inspection."""
    if not all(math.isfinite(v) for v in (start_ms, duration_ms)) or start_ms < 0 or not 100 <= duration_ms <= 10000:
        raise ValueError("波形窗口必须为 0.1–10 秒，起点不能为负")
    if not isinstance(bins, int) or not 100 <= bins <= 4000:
        raise ValueError("波形分辨率必须为 100–4000")
    rate, data = wavfile.read(preview, mmap=True)
    first = round(start_ms * rate / 1000)
    last = min(len(data), first + round(duration_ms * rate / 1000))
    if first >= last:
        raise ValueError("波形起点超过音频结尾")
    source = np.asarray(data[first:last], np.float32)
    if np.issubdtype(data.dtype, np.integer):
        source /= max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
    if source.ndim == 2:
        source = source.mean(axis=1)
    step = max(1, math.ceil(len(source) / bins))
    positions = np.arange(0, len(source), step)
    low = np.minimum.reduceat(source, positions)
    high = np.maximum.reduceat(source, positions)
    return dict(sample_rate=rate, first_sample=first, samples=len(source), step_samples=step,
                start_ms=first * 1000 / rate, end_ms=last * 1000 / rate,
                low=np.round(low, 6).tolist(), high=np.round(high, 6).tolist())
