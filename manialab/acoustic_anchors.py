"""Optional frozen transcription evidence for the experimental LN planner.

Acoustic durations are model observations, never hard LN endpoints. Their
forced-window flags remain visible. This module accepts no authored events.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from .anchor_gestures import AnchorContext, FEATURES

ROLES = ('piano', 'other')
ATTRIBUTES = ('present', 'head_pitches', 'duration_median', 'duration_max', 'forced_fraction',
              'piano_family', 'string_family', 'pad_family', 'bass_family',
              'survives_to_tail', 'explicit_near_tail', 'forced_near_tail',
              'same_pitch_coverage', 'same_pitch_restarts', 'end_vs_observed_max',
              'head_energy', 'tail_before_energy', 'tail_after_energy', 'energy_drop', 'body_energy')
ACOUSTIC_FEATURES = FEATURES+[role+'_'+name for role in ROLES for name in ATTRIBUTES]


class AcousticAnchorContext(AnchorContext):
    feature_names = ACOUSTIC_FEATURES

    def __init__(self, notes, keys, timing, ln_condition, evidence):
        super().__init__(notes, keys, timing, ln_condition)
        self.evidence = {}; self.heads = {}
        for role in ROLES:
            data = evidence[role]
            starts = np.asarray([n['onset']*1000-22.5 for n in data['notes']], float)
            ends = np.asarray([n['offset']*1000-22.5 for n in data['notes']], float)
            if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(ends)) or np.any(ends < starts):
                raise ValueError('Invalid acoustic interval observations')
            order = np.argsort(starts, kind='stable')
            pitch = np.asarray([n['pitch'] for n in data['notes']])[order]
            programs = np.asarray([n['program'] for n in data['notes']])[order]
            flags = np.asarray(data['forced'], bool)[order]
            starts, ends = starts[order], ends[order]
            by_pitch = defaultdict(list)
            for i, p in enumerate(pitch): by_pitch[int(p)].append(i)
            energy = np.asarray(data['energy'], float)
            if np.any(~np.isfinite(energy)) or np.any(energy < 0): raise ValueError('Invalid acoustic energy')
            self.evidence[role] = dict(starts=starts, ends=ends, pitch=pitch, programs=programs, forced=flags,
                by_pitch=by_pitch, energy=energy, hop_ms=float(data['hop_ms']))
            self.heads[role] = [np.flatnonzero(np.abs(starts-t) <= 60) for t in self.times]

    def acoustic_row(self, role, i, j):
        return self.acoustic_at_time(role, i, float(self.times[j]) if j > i else float(self.times[i]))

    def acoustic_at_time(self, role, i, end):
        data = self.evidence[role]; head = self.heads[role][i]
        t = float(self.times[i]); holding = end > t
        beat = self.periods[i]
        def rms(at, window=80):
            lo = max(0, int((at+22.5-window/2)/data['hop_ms']))
            hi = min(len(data['energy']), max(lo+1, int((at+22.5+window/2)/data['hop_ms'])))
            return float(data['energy'][lo:hi].mean()) if hi > lo else 0.
        head_energy = rms(t)
        before, after = (rms(end-40), rms(end+40)) if holding else (0., 0.)
        body = np.mean([rms(at) for at in np.linspace(t, end, 9)]) if holding else 0.
        if not len(head):
            return [0.]*15+[head_energy, before, after, before-after, body]
        observed_ends = np.minimum(data['ends'][head], t+8*beat)
        duration = (observed_ends-data['starts'][head])/beat
        program = data['programs'][head]
        forced = data['forced'][head] & (data['ends'][head] <= t+8*beat)
        coverage = []; restarts = []
        if holding:
            # Union observed intervals for a starting pitch. Rearticulations
            # remain a separate feature; union coverage is not proof of legato.
            for p in np.unique(data['pitch'][head]):
                total = 0.; right = t; count = 0
                for k in data['by_pitch'][int(p)]:
                    lo, hi = max(t, data['starts'][k]), min(end, data['ends'][k])
                    if lo < end and hi > t:
                        total += max(0., hi-max(right, lo)); right = max(right, hi)
                        count += data['starts'][k] > t+60
                coverage.append(min(1., total/(end-t))); restarts.append(count)
        tail_error = np.abs(data['ends'][head]-end)
        return [1., len(np.unique(data['pitch'][head]))/8, np.median(duration)/8, np.max(duration)/8,
            float(forced.mean()), *[float(((program >= low)&(program < high)).mean())
                for low, high in ((0, 8), (40, 56), (88, 96), (32, 40))],
            float((data['ends'][head] >= end-30).mean()) if holding else 0.,
            float(np.max(np.where(~forced, np.exp(-.5*(tail_error/80)**2), 0.))) if holding else 0.,
            float(np.max(np.where(forced, np.exp(-.5*(tail_error/80)**2), 0.))) if holding else 0.,
            max(coverage, default=0.), min(16., max(restarts, default=0))/8,
            min(8., (end-t)/max(100., float(np.max(observed_ends)-t))) if holding else 0.,
            head_energy, before, after, before-after, body]

    def features(self, i, options=None):
        options = self.options(i) if options is None else options
        base = super().features(i, options)
        acoustic = np.asarray([[value for role in ROLES for value in self.acoustic_row(role, i, int(j))]
                               for j in options], dtype=np.float32)
        return np.c_[base, acoustic]
