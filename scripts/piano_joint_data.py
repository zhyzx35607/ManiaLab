"""Physical sustain-frame labels and ownership for the bounded joint-head study.

The caller supplies already pedal-extended intervals. This module neither
infers pedal state nor turns MIDI sustain into a chart LN or an audible tail.
"""
from math import isfinite
from numbers import Integral, Real

import numpy as np

from scripts.piano_adaptation_data import window_slices


def _time(value):
    return (not isinstance(value, bool) and isinstance(value, Real)
            and isfinite(value) and value >= 0)


def frame_targets(notes, start_seconds, frames=1001):
    """Sample the per-pitch union of half-open [onset, offset) intervals.

    ``offset`` is the supplied sustain endpoint, not ``key_offset``. Times are
    compared directly on the 100fps physical clock without rounding or clipping
    intervals to either a local window or the 180-second experiment boundary.
    Thus a right-censored note stays active through the observable window.
    """
    if (not _time(start_seconds) or isinstance(frames, bool)
            or not isinstance(frames, Integral) or frames < 1):
        raise ValueError('Invalid physical frame clock')
    times = start_seconds + np.arange(frames, dtype=np.float64) / 100
    if not np.isfinite(times).all():
        raise ValueError('Nonfinite physical frame clock')
    target = np.zeros((frames, 88), dtype=np.float32)
    for note in notes:
        try:
            pitch, onset, offset = note['pitch'], note['onset'], note['offset']
        except (KeyError, TypeError) as error:
            raise ValueError('Expected pitch, onset and sustain offset') from error
        if (isinstance(pitch, bool) or not isinstance(pitch, Integral)
                or not 21 <= pitch <= 108 or not _time(onset)
                or not _time(offset) or onset >= offset):
            raise ValueError('Invalid MIDI pitch or sustain interval')
        target[(times >= onset) & (times < offset), pitch - 21] = 1
    return target


def owned_mask(index):
    """Return the unique loss frames for window index 0..34, with edge guards.

    Full 1001-frame windows still enter the model; only the loss excludes the
    global first/last 50ms. Internal overlapping windows keep their context.
    """
    if (isinstance(index, bool) or not isinstance(index, Integral)
            or not 0 <= index < 35):
        raise ValueError('Expected a window index from 0 through 34')
    local = np.arange(1001)
    left, right = window_slices(35)[index]
    global_frame = 500 * index + local
    return ((local >= left) & (local < right)
            & (global_frame >= 5) & (global_frame < 17995))
