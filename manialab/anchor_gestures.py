"""Experimental LN planning on fixed attacks; no acoustic-tail assumptions.

This module is intentionally not called by the application. An endpoint must
be another supplied attack. Source labels describe input evidence, not an
inferred instrument identity for any authored finished note.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csc_matrix

from .arrange import _solve_binary
from .timing import BeatClock

REVISION = 'anchor-gestures-v1'
SOURCES = ('mix', 'kick', 'snare', 'cymbal', 'piano', 'other', 'bass', 'guitar', 'vocals')
DURATIONS = np.asarray([.125, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8])
ROW_FEATURES = ['count', 'gap_before', 'gap_after', 'past_density', 'future_density',
                'beat_sin', 'beat_cos', 'bar_sin', 'bar_cos', *SOURCES]
FEATURES = ['tap', 'unresolved', 'keys', 'ln_condition', 'log_duration',
            'interior_density', 'source_overlap', *['duration_'+str(v) for v in DURATIONS],
            *['head_'+v for v in ROW_FEATURES], *['tail_'+v for v in ROW_FEATURES],
            *['before_gap_'+str(i) for i in range(4)],
            *['after_gap_'+str(i) for i in range(4)],
            *['before_count_'+str(i) for i in range(4)],
            *['after_count_'+str(i) for i in range(4)]]


class AnchorContext:
    feature_names = FEATURES
    def __init__(self, notes, keys, timing, ln_condition=.35):
        if not 6 <= keys <= 10 or not math.isfinite(ln_condition) or not 0 <= ln_condition <= 1:
            raise ValueError('Invalid keycount or LN condition')
        self.keys, self.ln_condition = keys, ln_condition
        grouped = defaultdict(list)
        for n in notes:
            if not isinstance(n.start, (int, np.integer)) or n.start < 0:
                raise ValueError('Anchors require nonnegative integer chart timestamps')
            grouped[n.start].append(n)
        self.times = np.asarray(sorted(grouped), dtype=np.int64)
        self.notes = [tuple(grouped[int(t)]) for t in self.times]
        self.counts = np.asarray([len(row) for row in self.notes], dtype=np.int64)
        if np.any(self.counts > keys):
            raise ValueError('An input attack already exceeds key capacity')
        clock = BeatClock(timing)
        self.periods = np.asarray([clock(t) for t in self.times])
        if not clock.offsets or np.any(~np.isfinite(self.periods)) or np.any(self.periods <= 0):
            raise ValueError('Explicit valid red TimingPoints are required')
        # Integrate tempo changes for distance; use the active red line's phase
        # for beat strength. Inherited timing and kiai never enter features.
        cumulative = [0.]
        for i in range(1, len(clock.offsets)):
            cumulative.append(cumulative[-1]+(clock.offsets[i]-clock.offsets[i-1])/clock.periods[i-1])
        at = np.maximum(0, np.searchsorted(clock.offsets, self.times, side='right')-1)
        phase = (self.times-np.asarray(clock.offsets)[at])/self.periods
        self.beats = np.asarray(cumulative)[at]+phase
        self.rows = np.zeros((len(self.times), len(ROW_FEATURES)), dtype=np.float32)
        for i, (t, row) in enumerate(zip(self.times, self.notes)):
            before = self.beats[i]-self.beats[i-1] if i else 8.
            after = self.beats[i+1]-self.beats[i] if i+1 < len(self.times) else 8.
            lo = np.searchsorted(self.beats, self.beats[i]-4)
            hi = np.searchsorted(self.beats, self.beats[i]+4, side='right')
            voice = Counter(n.voice if n.voice in SOURCES else 'mix' for n in row)
            self.rows[i] = [len(row)/keys, min(8., before)/4, min(8., after)/4,
                float(self.counts[lo:i].sum())/(4*keys), float(self.counts[i+1:hi].sum())/(4*keys),
                math.sin(2*math.pi*phase[i]), math.cos(2*math.pi*phase[i]),
                math.sin(math.pi*phase[i]/2), math.cos(math.pi*phase[i]/2),
                *[voice[v]/len(row) for v in SOURCES]]

    def options(self, i):
        stop = np.searchsorted(self.beats, self.beats[i]+8., side='right')
        future = np.arange(i+1, stop)
        future = future[self.times[future]-self.times[i] >= 100]
        if len(future) > 32:
            future = np.concatenate((future[:16], future[16:][np.linspace(0, len(future)-17, 16).round().astype(int)]))
        # -1 is a supervised unresolved endpoint, never a fabricated time.
        return np.asarray([i, *future, -1], dtype=np.int64)

    def features(self, i, options=None):
        options = self.options(i) if options is None else np.asarray(options)
        x = np.zeros((len(options), len(FEATURES)), dtype=np.float32)
        rhythm = []
        for direction in (-1, 1):
            for k in range(1, 5):
                j = i+direction*k
                previous = i+direction*(k-1)
                rhythm.append(min(8., abs(self.beats[j]-self.beats[previous]))/4 if 0 <= j < len(self.times) else 2.)
        for direction in (-1, 1):
            rhythm.extend(self.counts[i+direction*k]/self.keys if 0 <= i+direction*k < len(self.times) else 0. for k in range(1, 5))
        for k, j in enumerate(options):
            duration = self.beats[j]-self.beats[i] if j > i else 0.
            tail = self.rows[j] if j > i else np.zeros(len(ROW_FEATURES))
            basis = np.exp(-.5*((np.log2(max(duration, .03125))-np.log2(DURATIONS))/.2)**2) if j > i else np.zeros(len(DURATIONS))
            x[k] = [j == i, j == -1, self.keys/10, self.ln_condition,
                np.log1p(duration), max(0, j-i-1)/max(1., duration)/8,
                float(np.minimum(self.rows[i, 9:], tail[9:]).sum()), *basis,
                *self.rows[i], *tail, *rhythm]
        return x

    def contained(self, i, start, end):
        # Feature context includes nearby rhythm and endpoint neighbourhoods.
        lo = max(0, np.searchsorted(self.beats, self.beats[i]-4)-4)
        hi = min(len(self.times)-1, np.searchsorted(self.beats, self.beats[i]+12, side='right')+4)
        return self.times[lo] >= start and self.times[hi] < end


def labels(context, i, authored, tolerance=30):
    """Supervise a whole attack distribution; no claimed source-note matching."""
    options = context.options(i)
    values = np.zeros(len(options), dtype=np.float32)
    details = Counter()
    local = [n for n in authored if abs(n.start-int(context.times[i])) <= tolerance]
    if not local:
        # A detected onset absent from this author interpretation is retained
        # as a tap by this deliberately attack-preserving experiment.
        values[0] = 1.
        return values, dict(unrepresented_attack=1)
    ends = context.times[options[1:-1]]
    for n in local:
        if not n.hold:
            values[0] += 1
            details['taps'] += 1
        elif len(ends) and np.min(np.abs(ends-n.end)) <= 60:
            values[1+int(np.argmin(np.abs(ends-n.end)))] += 1
            details['represented_holds'] += 1
        else:
            values[-1] += 1
            details['unrepresented_holds'] += 1
    return values/values.sum(), dict(details)


def probabilities(context, model, ln_bias=0.):
    if model['revision'] != REVISION or model['features'] != context.feature_names:
        raise ValueError('Anchor model feature identity changed')
    result = []
    weights = [np.asarray(w, np.float32) for w in model['weights']]
    biases = [np.asarray(b, np.float32) for b in model['biases']]
    for i in range(len(context.times)):
        options = context.options(i)
        x = np.clip((context.features(i, options)-model['mean'])/model['scale'], -10, 10).astype(np.float32)
        for j, (w, b) in enumerate(zip(weights, biases)):
            x = x@w.T+b
            if j+1 < len(weights):
                x = np.maximum(x, 0.)
        score = x[:, 0].astype(np.float64)
        score[1:-1] += ln_bias
        p = np.exp(score-score.max()); p /= p.sum()
        result.append(dict(options=options, probabilities=p))
    return result


def plan(context, predictions, coordination=0., time_limit=60, max_ln_ratio=None):
    """Integer histogram projection with global interval capacity constraints.

    Every slot remains present, including uncertain or capacity-limited taps.
    A release-opening cost encourages sharing an existing release event. It is
    a preference, not a claim that every pair should end together.
    """
    if len(predictions) != len(context.times) or not math.isfinite(coordination) or coordination < 0:
        raise ValueError('Invalid anchor predictions or coordination')
    if max_ln_ratio is not None and (not math.isfinite(max_ln_ratio) or not 0 <= max_ln_ratio <= 1):
        raise ValueError('LN ceiling must be a finite ratio between zero and one')
    if not len(context.times):
        return [], dict(input_objects=0, objects=0, holds=0, unresolved=0, optimal=True)
    costs, variables, release_vars = [], [], defaultdict(list)
    row_variables = defaultdict(list)
    for i, pred in enumerate(predictions):
        options, p = np.asarray(pred['options']), np.asarray(pred['probabilities'], float)
        if (not np.array_equal(options, context.options(i)) or p.shape != options.shape or
                not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1)):
            raise ValueError('Predictions must refer to the frozen legal candidate list')
        count = int(context.counts[i])
        for k, end_index in enumerate(options):
            for occurrence in range(count):
                var = len(costs)
                # Incremental squared count error. Increasing marginals make
                # selected copies a prefix without extra order constraints.
                costs.append(2*occurrence+1-2*count*p[k])
                variables.append((i, int(end_index), occurrence))
                row_variables[i].append(var)
                if end_index > i:
                    release_vars[int(end_index)].append(var)
    note_variables = len(costs)
    openings = {}
    if coordination:
        for end_index in release_vars:
            openings[end_index] = len(costs)
            costs.append(coordination)
    rows, cols, data, lower, upper = [], [], [], [], []
    def add(items, low, high):
        r = len(upper); lower.append(low); upper.append(high)
        for col, value in items:
            rows.append(r); cols.append(col); data.append(value)
    for i, cols_i in row_variables.items():
        add(((j, 1) for j in cols_i), int(context.counts[i]), int(context.counts[i]))
    ln_limit = math.floor(int(context.counts.sum())*max_ln_ratio) if max_ln_ratio is not None else None
    if ln_limit is not None:
        # A maximum is not a quota: the planner may use fewer LNs. It chooses
        # which supported gestures survive, without trimming a chosen tail.
        add(((j, 1) for j, (start, end, _) in enumerate(variables) if end > start), -np.inf, ln_limit)
    active = defaultdict(list)
    for j, (start, end, _) in enumerate(variables):
        for at in range(start+1, end):
            active[at].append(j)
    for at, variables_at in active.items():
        add(((j, 1) for j in variables_at), -np.inf, context.keys-int(context.counts[at]))
    if coordination:
        for end_index, cols_i in release_vars.items():
            add([*[(j, 1) for j in cols_i], (openings[end_index], -len(cols_i))], -np.inf, 0)
    matrix = csc_matrix((data, (rows, cols)), shape=(len(upper), len(costs)))
    solved = _solve_binary(np.asarray(costs), integrality=np.ones(len(costs)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options=dict(time_limit=time_limit, mip_rel_gap=.001))
    if solved.x is None:
        raise ValueError('Anchor planning returned no feasible assignment')
    chosen = solved.x > .5
    actual = matrix@chosen.astype(float)
    if np.any(actual < np.asarray(lower)-1e-6) or np.any(actual > np.asarray(upper)+1e-6):
        raise ValueError('Anchor plan failed independent capacity validation')
    ends = defaultdict(list); unresolved = 0
    for j in np.flatnonzero(chosen[:note_variables]):
        start, end, _ = variables[j]
        unresolved += end == -1
        ends[start].append(int(context.times[end if end > start else start]))
    output = []
    for i, original in enumerate(context.notes):
        # This association is deterministic bookkeeping, not supervised source
        # attribution. Layout is a later operation and may move each object.
        for n, end in zip(original, sorted(ends[i])):
            # Original import text contains the old tap/hold type and tail.
            # Keeping it would silently export the pre-planning event again.
            output.append(replace(n, end=end, raw=''))
    if Counter(n.start for n in output) != Counter(n.start for row in context.notes for n in row):
        raise ValueError('Anchor planner changed input attacks')
    releases = Counter(n.end for n in output if n.hold)
    return output, dict(input_objects=int(context.counts.sum()), objects=len(output),
        holds=sum(n.hold for n in output), unresolved=unresolved, optimal=solved.status == 0,
        mip_gap=float(solved.mip_gap), ln_limit=ln_limit, shared_release_objects=sum(v for v in releases.values() if v >= 2),
        endpoint_source='supplied_onset_anchors', source_assignment='ambiguous_within_simultaneous_group')


def arrange_gestures(notes, keys, timing, settings=None):
    """Source-sketch columns are identities, not prior performance positions.

    Import-jack preservation and the original-column positional preference are
    appropriate for relayout of a finished chart, but not for a stem sketch.
    The existing search still uses source continuity, hand load and LN costs.
    """
    from .arrange import assign_lanes, parameters
    from .domain import invariant
    prepared = [replace(n, source='anchor:'+n.voice, raw='') for n in notes]
    preferences = parameters(dict(settings or {}, keys=keys, pitch_follow=0.))
    result = assign_lanes(prepared, preferences, timing_points=timing)
    if invariant(result) != invariant(prepared):
        raise ValueError('Source-aware layout changed fixed gesture intervals')
    return result
