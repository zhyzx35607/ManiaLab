"""Preserve observed outgoing sustain when an incoming window lacks its attack.

Never averages note attacks or extrapolates a tail
beyond the previous window's observed decline and offset-regression context.
"""
from __future__ import annotations

import numpy as np

from manialab.piano import OUTPUTS, decode, regression_peaks


def continue_overlap(previous, incoming, *, owned_left=250, require_release_regression=False):
    if owned_left not in (0, 250):
        raise ValueError('Unexpected preceding ownership interval')
    for window in (previous, incoming):
        if any(np.asarray(window[k]).shape != (1001, 88)
               or not np.isfinite(window[k]).all() for k in OUTPUTS):
            raise ValueError('Expected complete finite piano windows')
    result = {k: np.array(incoming[k], copy=True) for k in OUTPUTS}
    old_frame, new_frame = previous['frame_output'], incoming['frame_output']
    old_onsets, old_shifts = regression_peaks(previous['reg_onset_output'], .3, 2)
    new_onsets, _ = regression_peaks(incoming['reg_onset_output'], .3, 2)
    changed = []
    previous_notes = baseline_notes = None
    # Outgoing 7.5 s and incoming 2.5 s describe the same PCM frame.
    for pitch in np.flatnonzero((old_frame[749] > .1) & (old_frame[750] > .1)
                               & (new_frame[250] <= .1)):
        attacks = np.flatnonzero(old_onsets[:750, pitch])
        if not len(attacks):
            continue
        anchor = int(attacks[-1])
        # The attack must be owned by the prior window, and absent from the
        # new window's input. No continuation is inferred from a decay alone.
        if not owned_left <= anchor < 500 or new_onsets[:251, pitch].any():
            continue
        if not np.all(old_frame[anchor:751, pitch] > .1):
            continue
        fades = np.flatnonzero(old_frame[750:1000, pitch] <= .1)
        if not len(fades):
            continue
        fade = 750 + int(fades[0])
        # Four frames on each side are used by the offset peak decoder. Keep
        # that complete context, without the reflected last input frame.
        if fade + 4 >= 1000:
            continue
        local_end = fade - 500
        if old_onsets[750:fade + 5, pitch].any() or new_onsets[251:local_end + 5, pitch].any():
            continue
        if require_release_regression:
            if previous_notes is None:
                previous_notes = {(n['pitch'],n['onset']):n for n in decode(previous)}
                joined = {k:np.concatenate([previous[k][:750],incoming[k][250:1000]]) for k in OUTPUTS}
                baseline_notes = {(n['pitch'],n['onset']):n for n in decode(joined)}
            onset = float(np.float32((anchor+old_shifts[anchor,pitch])/100))
            key = (int(pitch+21),onset)
            old_note, baseline_note = previous_notes.get(key), baseline_notes.get(key)
            if (old_note is None or baseline_note is None
                    or old_note['release_reason'] != 'release_regression'
                    or old_note['offset'] <= 7.5
                    or baseline_note['release_reason'] != 'frame_decay'
                    or baseline_note['offset'] != 7.5):
                continue
        result['frame_output'][250:local_end + 1, pitch] = old_frame[750:fade + 1, pitch]
        result['reg_offset_output'][250:local_end + 5, pitch] = previous['reg_offset_output'][750:fade + 5, pitch]
        changed.append(dict(pitch=int(pitch + 21), anchor_frame=anchor,
            outgoing_fade_frame=fade, incoming_fade_frame=local_end,
            frame_slice=[250, local_end + 1], offset_slice=[250, local_end + 5],
            outgoing_boundary_activity=float(old_frame[750, pitch]),
            incoming_boundary_activity=float(new_frame[250, pitch])))
    return result, changed
