"""Section preferences with explicit audio and chart clock ownership."""
from __future__ import annotations

import math

from .domain import Note


class ChartPreferences(tuple):
    """Chart intervals for capacity sweeps, original intervals for note intent.

    Quantization can move notes across a section boundary, and clipping can
    merge early attacks at zero. Neither operation changes a note's voice or
    expression policy. Sustained occupancy is still checked on the chart clock.
    """

    def __new__(cls, segments, audio_shift_ms):
        original = tuple(dict(s, preferences=dict(s['preferences'])) for s in segments)
        def shifted(value):
            return max(0, math.floor(value - audio_shift_ms + .5))
        result = super().__new__(cls, (dict(s, start_ms=shifted(s['start_ms']),
            end_ms=shifted(s['end_ms'])) for s in original))
        result.audio_segments = original
        return result


def settings_at(settings, time, segments):
    """A Note selects original musical intent; a number selects chart capacity."""
    if isinstance(time, Note):
        if isinstance(segments, ChartPreferences) and time.audio_start is not None:
            time, segments = time.audio_start, segments.audio_segments
        else:
            time = time.start
    local = settings.copy()
    for segment in segments:
        if segment['start_ms'] <= time < segment['end_ms']:
            local.update(segment['preferences'])
    return local


def row_settings(settings, notes, segments):
    """Use the earliest original attack for shared layout preferences.

    If quantization merges sections into one row, enforce the tightest capacity
    among its notes. Voice weights and LN eligibility remain individual.
    """
    notes = list(notes)
    first = min(notes, key=lambda n: (n.audio_start if n.audio_start is not None else n.start, n.id))
    local = settings_at(settings, first, segments)
    for note in notes:
        pref = settings_at(settings, note, segments)
        for key in ('density', 'chord_size', 'max_held'):
            local[key] = min(local[key], pref[key])
    return local
