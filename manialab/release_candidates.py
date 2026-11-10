"""Research release events can occur without an attack at the same instant."""
import math

import numpy as np

from .anchor_gestures import AnchorContext, FEATURES, ROW_FEATURES, DURATIONS
from .timing import BeatClock

REVISION = 'independent-release-v2'
EXTRA = ['release_at_attack', 'release_distance_to_attack', 'duration_12', 'duration_16',
         'grid_quarter', 'grid_eighth', 'grid_triplet', 'grid_twelfth']
RELEASE_FEATURES = FEATURES+EXTRA
RELATIVE = np.asarray([1/12, 1/8, 1/6, 1/4, 1/3, .5, 2/3, .75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16])


class ReleaseContext(AnchorContext):
    feature_names = RELEASE_FEATURES
    revision = REVISION
    endpoint_source = 'supplied_red_grid_and_onset_anchors'

    def __init__(self, notes, keys, timing, ln_condition=.35):
        notes = list(notes); timing = list(timing)
        clock = BeatClock(timing)
        self.clock_times = np.asarray(clock.offsets, float)
        self.clock_periods = np.asarray(clock.periods, float)
        if not len(self.clock_times):
            raise ValueError('Explicit valid red TimingPoints are required')
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            self.clock_beats = np.r_[0., np.cumsum(np.diff(self.clock_times)/self.clock_periods[:-1])]
            resolution = np.spacing(np.abs(self.clock_beats))*self.clock_periods
        # Some SV-effect charts use extreme positive red periods. Their
        # accumulated beat coordinates may stop distinguishing real time.
        # They remain usable for layout but cannot supervise this clock.
        if (not np.isfinite(self.clock_beats).all() or np.any(np.diff(self.clock_beats) <= 0)
                or not np.isfinite(resolution).all() or np.any(resolution > .1)):
            raise ValueError('Red TimingPoints lose sub-millisecond beat-clock precision')
        super().__init__(notes, keys, timing, ln_condition)
        if (np.any(np.diff(self.beats) <= 0) or not np.isfinite(self.beats).all()
                or np.any(np.abs(self.time_at(self.beats)-self.times) > .1)):
            raise ValueError('Attack times do not round-trip through the beat clock')
        with np.errstate(over='ignore', invalid='ignore'):
            horizon = self.time_at(self.beats+20)
        if (not np.isfinite(horizon).all() or np.any(horizon >= np.iinfo(np.int64).max)
                or np.any(horizon <= self.times)):
            raise ValueError('Beat-clock horizon is not representable in milliseconds')
        self.prefix_counts = np.r_[0, np.cumsum(self.counts)]
        self.options_cache = {}; self.feature_cache = {}

    def beat_at(self, times):
        times = np.asarray(times, float)
        at = np.maximum(0, np.searchsorted(self.clock_times, times, side='right')-1)
        return self.clock_beats[at]+(times-self.clock_times[at])/self.clock_periods[at]

    def time_at(self, beats):
        beats = np.asarray(beats, float)
        at = np.maximum(0, np.searchsorted(self.clock_beats, beats, side='right')-1)
        return self.clock_times[at]+(beats-self.clock_beats[at])*self.clock_periods[at]

    def options(self, i):
        if i in self.options_cache: return self.options_cache[i].copy()
        head = int(self.times[i]); beat = self.beats[i]
        minimum = max(1, int(math.floor(self.periods[i]/12-.5)))
        last = int(math.floor(float(self.time_at(beat+16))+.5))
        future = self.times[(self.times >= head+minimum)&(self.times <= last)]
        if len(future) > 32:
            future = np.r_[future[:16], future[16:][np.linspace(0, len(future)-17, 16).round().astype(int)]]
        grid = set()
        # Each tier uses the active red line's local musical phase, including
        # phase resets at red lines that keep the same tempo.
        for horizon, divisor in ((1, 12), (2, 8), (4, 4), (8, 2), (16, 1)):
            boundary = float(self.time_at(beat+horizon))
            first_segment = max(0, int(np.searchsorted(self.clock_times, head, side='right'))-1)
            for seg in range(first_segment, len(self.clock_times)):
                left = max(head+minimum, self.clock_times[seg]) if seg > first_segment else head+minimum
                right = min(boundary, self.clock_times[seg+1] if seg+1 < len(self.clock_times) else boundary)
                if left > boundary: break
                step = self.clock_periods[seg]/divisor
                first = int(math.ceil((left-self.clock_times[seg])/step-1e-8))
                final = int(math.floor((right-self.clock_times[seg])/step+1e-8))
                grid.update(int(math.floor(self.clock_times[seg]+j*step+.5)) for j in range(first, final+1))
        grid.update(int(math.floor(t+.5)) for t in self.time_at(beat+RELATIVE))
        candidates = sorted(int(t) for t in future)
        # Canonicalize near-equal grid/attack milliseconds before fitting.
        # Never shift a real attack to an invented grid position.
        for time in sorted(grid):
            if head+minimum <= time <= last and all(abs(time-t) > 3 for t in candidates): candidates.append(time)
        result = np.asarray([head, *sorted(candidates), -1], np.int64)
        self.options_cache[i] = result
        return result.copy()

    def endpoint_times(self, i, options):
        return np.asarray(options, dtype=np.int64)

    def endpoint_rows(self, endpoints):
        endpoints = np.asarray(endpoints)
        beats = self.beat_at(endpoints)
        loc = np.searchsorted(self.times, endpoints)
        previous = np.maximum(0, loc-1); following = np.minimum(len(self.times)-1, loc)
        nearest = np.where(np.abs(self.times[previous]-endpoints) < np.abs(self.times[following]-endpoints), previous, following)
        distance = np.abs(self.times[nearest]-endpoints)
        at_attack = distance <= 3
        rows = np.zeros((len(endpoints), len(ROW_FEATURES)), np.float32)
        before = np.where(loc > 0, np.maximum(0., beats-self.beats[previous]), 8.)
        after = np.where(loc < len(self.times), np.maximum(0., self.beats[following]-beats), 8.)
        lo = np.searchsorted(self.beats, beats-4); hi = np.searchsorted(self.beats, beats+4, side='right')
        rows[:, 1] = np.minimum(8., before)/4; rows[:, 2] = np.minimum(8., after)/4
        rows[:, 3] = (self.prefix_counts[loc]-self.prefix_counts[lo])/(4*self.keys)
        rows[:, 4] = (self.prefix_counts[hi]-self.prefix_counts[loc])/(4*self.keys)
        seg = np.maximum(0, np.searchsorted(self.clock_times, endpoints, side='right')-1)
        phase = (endpoints-self.clock_times[seg])/self.clock_periods[seg]
        rows[:, 5:9] = np.c_[np.sin(2*np.pi*phase), np.cos(2*np.pi*phase), np.sin(np.pi*phase/2), np.cos(np.pi*phase/2)]
        rows[at_attack] = self.rows[nearest[at_attack]]
        return rows, at_attack, distance/self.clock_periods[seg], phase

    def features(self, i, options=None):
        canonical = self.options(i)
        if options is not None and not np.array_equal(options, canonical): raise ValueError('Unknown release candidates')
        if i in self.feature_cache: return self.feature_cache[i].copy()
        options = canonical; head = int(self.times[i]); tail_times = options[1:-1]
        duration = np.r_[0., self.beat_at(tail_times)-self.beats[i], 0.]
        if not np.isfinite(duration).all() or np.any(duration[1:-1] <= 0):
            raise ValueError('Release duration is not positive in the beat clock')
        tail_rows, at_attack, distance, phase = self.endpoint_rows(tail_times)
        tails = np.zeros((len(options), len(ROW_FEATURES)), np.float32); tails[1:-1] = tail_rows
        x = np.zeros((len(options), len(RELEASE_FEATURES)), np.float32)
        x[0, 0] = 1; x[-1, 1] = 1; x[:, 2] = self.keys/10; x[:, 3] = self.ln_condition
        x[:, 4] = np.log1p(np.maximum(0, duration))
        x[1:-1, 5] = np.maximum(0, np.searchsorted(self.times, tail_times)-i-1)/np.maximum(1., duration[1:-1])/8
        x[:, 6] = np.minimum(self.rows[i, 9:], tails[:, 9:]).sum(axis=1)
        x[1:-1, 7:18] = np.exp(-.5*((np.log2(np.maximum(duration[1:-1, None], .03125))-np.log2(DURATIONS))/.2)**2)
        x[:, 18:36] = self.rows[i]; x[:, 36:54] = tails
        rhythm = []
        for direction in (-1, 1):
            for k in range(1, 5):
                j = i+direction*k; prev = i+direction*(k-1)
                rhythm.append(min(8., abs(self.beats[j]-self.beats[prev]))/4 if 0 <= j < len(self.times) else 2.)
        for direction in (-1, 1):
            rhythm.extend(self.counts[i+direction*k]/self.keys if 0 <= i+direction*k < len(self.times) else 0. for k in range(1, 5))
        x[:, 54:70] = rhythm
        x[1:-1, 70] = at_attack; x[1:-1, 71] = np.minimum(8., distance)
        x[1:-1, 72:74] = np.exp(-.5*((np.log2(duration[1:-1, None])-np.log2([12, 16]))/.2)**2)
        for at, div in enumerate((4, 8, 3, 12)):
            error = np.abs(phase*div-np.rint(phase*div))/div
            x[1:-1, 74+at] = np.exp(-.5*(error/.02)**2)
        self.feature_cache[i] = x
        return x.copy()

    def contained(self, i, start, end):
        # Four previous rows and +/-4 beat row context can cross long rests;
        # check the entire consumed range, not just the head's time.
        lo = max(0, np.searchsorted(self.beats, self.beats[i]-4)-4)
        hi = min(len(self.times)-1, np.searchsorted(self.beats, self.beats[i]+20, side='right')+4)
        return self.times[lo] >= start and self.times[hi] < end and self.time_at(self.beats[i]+20) < end


from .acoustic_anchors import AcousticAnchorContext, ATTRIBUTES, ROLES


class AcousticReleaseContext(ReleaseContext):
    feature_names = RELEASE_FEATURES+[role+'_'+name for role in ROLES for name in ATTRIBUTES]

    def __init__(self, notes, keys, timing, ln_condition, evidence):
        notes = list(notes)
        super().__init__(notes, keys, timing, ln_condition)
        self.acoustic = AcousticAnchorContext(notes, keys, timing, ln_condition, evidence)
        self.acoustic_cache = {}

    def features(self, i, options=None):
        base = super().features(i, options)
        if i not in self.acoustic_cache:
            candidates = self.options(i)
            self.acoustic_cache[i] = np.asarray([[value for role in ROLES
                for value in self.acoustic.acoustic_at_time(role, i, float(t) if t > self.times[i] else float(self.times[i]))]
                for t in candidates], np.float32)
        return np.c_[base, self.acoustic_cache[i]]
