"""Research observations from unnormalised, synchronous separated audio.

These features describe energy and spectral changes. They neither identify
individual notes nor declare physical decay endpoints to be LN ground truth.
"""
import numpy as np
from scipy.signal import stft

from .release_candidates import ReleaseContext, RELEASE_FEATURES

EVIDENCE_REVISION = 'four-stem-release-observations-v1'
ROLES = ('drums', 'bass', 'other', 'vocals')
ATTRIBUTES = ('head_level', 'head_rise', 'head_flux', 'head_low', 'head_high',
              'body_level', 'body_min', 'body_variation', 'body_active',
              'tail_before', 'tail_after', 'tail_drop', 'tail_flux', 'tail_low', 'tail_high')
STEM_FEATURES = RELEASE_FEATURES+[role+'_'+name for role in ROLES for name in ATTRIBUTES]


def observe(signal, sample_rate):
    """20ms spectral frames in fixed PCM scale, never song-peak normalised."""
    signal = np.asarray(signal, np.float32)
    if (signal.ndim != 1 or len(signal) < 1024 or not np.isfinite(signal).all()
            or sample_rate != 22050):
        raise ValueError('Expected finite mono 22050Hz stem samples')
    frequency, times, spectrum = stft(signal, fs=sample_rate, nperseg=1024, noverlap=583,
                                     boundary='zeros', padded=True)
    magnitude = np.abs(spectrum); power = magnitude*magnitude
    level = np.sqrt(power.sum(axis=0))
    bands = np.stack([power[frequency < 200].sum(axis=0),
                      power[frequency >= 3000].sum(axis=0)])/(power.sum(axis=0)+1e-12)
    flux = np.r_[0., np.maximum(0., np.diff(magnitude, axis=1)).sum(axis=0)]
    return dict(times_ms=times*1000, level=np.log1p(10*level),
                flux=np.log1p(flux), low=bands[0], high=bands[1])


class StemReleaseContext(ReleaseContext):
    feature_names = STEM_FEATURES

    def __init__(self, notes, keys, timing, ln_condition, evidence, audio_start_ms=0):
        super().__init__(notes, keys, timing, ln_condition)
        self.stem_evidence = evidence; self.audio_start_ms = audio_start_ms; self.stem_cache = {}
        for role in ROLES:
            row = evidence[role]; times = np.asarray(row['times_ms'])
            if (len(times) < 2 or np.any(np.diff(times) <= 0)
                    or any(np.asarray(row[k]).shape != times.shape or not np.isfinite(row[k]).all()
                           for k in ('times_ms', 'level', 'flux', 'low', 'high'))):
                raise ValueError('Invalid synchronous stem observations')

    def stem_features(self, role, head, tail):
        row = self.stem_evidence[role]; times = row['times_ms']
        head = head+22.5-self.audio_start_ms; tail = tail+22.5-self.audio_start_ms
        def values(name, start, end):
            lo = int(np.searchsorted(times, start)); hi = int(np.searchsorted(times, end, side='right'))
            return row[name][lo:hi]
        def mean(name, at, radius=40):
            data = values(name, at-radius, at+radius)
            return float(np.mean(data)) if len(data) else 0.
        level = mean('level', head)
        common = [level, mean('level', head+40)-mean('level', head-40), mean('flux', head),
                  mean('low', head), mean('high', head)]
        if tail <= head: return common+[0.]*10
        body = values('level', head, tail)
        before, after = mean('level', tail-40), mean('level', tail+40)
        return common+([float(np.mean(body)), float(np.min(body)), float(np.std(body)),
                        float(np.mean(body > max(.01, .3*level)))] if len(body) else [0.]*4)+[
            before, after, before-after, mean('flux', tail), mean('low', tail), mean('high', tail)]

    def features(self, i, options=None):
        base = super().features(i, options)
        if i not in self.stem_cache:
            head = int(self.times[i])
            self.stem_cache[i] = np.asarray([[v for role in ROLES
                for v in self.stem_features(role, head, max(head, int(t)))] for t in self.options(i)], np.float32)
        return np.c_[base, self.stem_cache[i]]
