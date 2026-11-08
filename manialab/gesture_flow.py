"""Research-only global interval planning as an integral minimum-cost flow.

Each key is a unit of available capacity moving forward through onset rows.
A fixed-flow gate at every row preserves its attack count. A hold skips the
intervening rows and returns capacity at the first attack at/after its release.
No authored tails or physical source-column positions enter this decoder.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import math
import time
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import csc_matrix

from .arrange import _solver_lock

REVISION = 'gesture-flow-v1'


def plan_flow(context, predictions, *, hierarchical=False, endpoint_weight=.25, time_limit=60):
    """Solve the flat histogram cost or a separate tap/hold/unresolved cost.

    Hierarchical mode adds a shared hold-count gate so uncertainty distributed
    over many plausible tails does not itself turn a likely hold into a tap.
    It does not force a requested LN quota. Shared-release opening costs and a
    global LN ceiling are deliberately absent from this network experiment.
    """
    if (len(predictions) != len(context.times) or not math.isfinite(endpoint_weight)
            or not 0 < endpoint_weight <= 1 or not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError('Invalid flow predictions or parameters')
    nrows = len(context.times)
    if not nrows:
        return [], dict(input_objects=0, objects=0, holds=0, unresolved=0, optimal=True,
                        planner=REVISION, hierarchical=hierarchical)
    costs, bounds, matrix_rows, matrix_cols, entries, choices = [], [], [], [], [], []
    def edge(start, end, cost=0., low=0, high=1, choice=None):
        col = len(costs)
        costs.append(cost); bounds.append((low, high)); choices.append(choice)
        matrix_rows.extend((start, end)); matrix_cols.extend((col, col)); entries.extend((1., -1.))
    for i, pred in enumerate(predictions):
        options, p = np.asarray(pred['options']), np.asarray(pred['probabilities'], float)
        if (not np.array_equal(options, context.options(i)) or p.shape != options.shape or
                not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1)):
            raise ValueError('Predictions must refer to the frozen legal candidate list')
        count = int(context.counts[i]); decision = nrows+1+i; hold_gate = 2*nrows+1+i
        edge(i, i+1, high=context.keys)
        edge(i, decision, low=count, high=count)
        if hierarchical:
            for occurrence in range(count):
                edge(decision, hold_gate, 2*occurrence+1-2*count*p[1:-1].sum())
        endpoints = context.endpoint_times(i, options)
        for k, end_time in enumerate(endpoints):
            hold = end_time > context.times[i]
            target = int(np.searchsorted(context.times, end_time, side='left')) if hold else i+1
            if target <= i:
                raise ValueError('A release must advance beyond its own attack')
            for occurrence in range(count):
                cost = 2*occurrence+1-2*count*p[k]
                if hierarchical and hold: cost *= endpoint_weight
                edge(hold_gate if hierarchical and hold else decision, target, cost,
                     choice=(i, int(end_time)))
    nodes = (3 if hierarchical else 2)*nrows+1
    matrix = csc_matrix((entries, (matrix_rows, matrix_cols)), shape=(nodes, len(costs)))
    supply = np.zeros(nodes); supply[0] = context.keys; supply[nrows] = -context.keys
    started = time.perf_counter()
    # A network incidence matrix with integral bounds has integral vertices;
    # dual simplex returns such a vertex without binary branch-and-bound.
    with _solver_lock, warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected:.*', category=OptimizeWarning)
        solved = linprog(costs, A_eq=matrix, b_eq=supply, bounds=bounds, method='highs-ds',
                         options=dict(time_limit=time_limit, threads=1, parallel=False))
    if not solved.success:
        raise ValueError('Flow planning did not reach an optimum: '+solved.message)
    integer = np.rint(solved.x)
    if (np.max(np.abs(integer-solved.x)) > 1e-6 or np.any(np.abs(matrix@integer-supply) > 1e-6)
            or np.any(integer < np.asarray(bounds)[:, 0]) or np.any(integer > np.asarray(bounds)[:, 1])):
        raise ValueError('Flow plan failed independent integer conservation validation')
    ends = defaultdict(list); unresolved = 0
    for j in np.flatnonzero(integer):
        if choices[j] is None: continue
        start, end = choices[j]
        unresolved += end == -1
        ends[start].append(max(end, int(context.times[start])))
    output = [replace(note, end=end, raw='') for i, original in enumerate(context.notes)
              for note, end in zip(original, sorted(ends[i]))]
    if Counter(n.start for n in output) != Counter(n.start for row in context.notes for n in row):
        raise ValueError('Flow plan changed attack counts')
    # Verify capacity directly in milliseconds, independent of graph indexing.
    held = []
    for at in context.times:
        held = [end for end in held if end > at]
        row = ends[int(np.searchsorted(context.times, at))]
        if len(held)+len(row) > context.keys:
            raise ValueError('Flow plan exceeded interval capacity')
        held.extend(end for end in row if end > at)
    releases = Counter(n.end for n in output if n.hold)
    return output, dict(input_objects=int(context.counts.sum()), objects=len(output),
        holds=sum(n.hold for n in output), unresolved=unresolved, optimal=True,
        mip_gap=0., objective=float(solved.fun), solve_seconds=time.perf_counter()-started,
        planner=REVISION, hierarchical=hierarchical, endpoint_weight=endpoint_weight if hierarchical else 1.,
        expected_holds=sum(float(context.counts[i]*np.sum(p['probabilities'][1:-1])) for i, p in enumerate(predictions)),
        expected_unresolved=sum(float(context.counts[i]*p['probabilities'][-1]) for i, p in enumerate(predictions)),
        shared_release_objects=sum(v for v in releases.values() if v >= 2),
        endpoint_source=context.endpoint_source, source_assignment='ambiguous_within_simultaneous_group')
