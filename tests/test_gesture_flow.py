from collections import Counter
import numpy as np
import pytest

from manialab.anchor_gestures import AnchorContext, plan
from manialab.domain import Note
from manialab.gesture_flow import plan_flow
from manialab.release_candidates import ReleaseContext
from scripts.evaluate_anchor_gestures import greedy_lanes

TIMING = ['0,500,4,2,1,60,1,0']


@pytest.mark.parametrize('seed', range(5))
def test_flat_flow_agrees_with_integer_histogram_optimum(seed):
    rng = np.random.default_rng(seed)
    notes = [Note(t, t, j) for t in (0, 250, 500, 750) for j in range(4)]
    context = AnchorContext(notes, 6, TIMING)
    predictions = [dict(options=context.options(i), probabilities=rng.dirichlet(np.ones(len(context.options(i)))))
                   for i in range(len(context.times))]
    expected, _ = plan(context, predictions)
    actual, report = plan_flow(context, predictions)
    assert Counter((n.start, n.end) for n in expected) == Counter((n.start, n.end) for n in actual)
    greedy_lanes(actual, 6)
    assert report['optimal']


def test_diffuse_release_distribution_keeps_confident_hold():
    context = ReleaseContext([Note(0, 0, 0)], 6, TIMING)
    options = context.options(0); p = np.zeros(len(options))
    p[0] = .2
    for endpoint in (500, 1000, 1500, 2000, 2500):
        p[list(options).index(endpoint)] = .16
    predictions = [dict(options=options, probabilities=p)]
    flat, _ = plan_flow(context, predictions)
    hierarchical, _ = plan_flow(context, predictions, hierarchical=True)
    assert not flat[0].hold and hierarchical[0].hold
    p[:] = 0; p[0] = 1
    taps, _ = plan_flow(context, predictions, hierarchical=True)
    assert not taps[0].hold


@pytest.mark.parametrize('hierarchical', [False, True])
def test_independent_release_capacity_and_same_time_reuse(hierarchical):
    notes = [Note(t, t, j) for t in (0, 500, 1000) for j in range(6)]
    context = ReleaseContext(notes, 6, TIMING)
    predictions = []
    for i, t in enumerate(context.times):
        options = context.options(i); p = np.zeros(len(options))
        p[list(options).index(750 if i == 0 else int(t)+500)] = 1
        predictions.append(dict(options=options, probabilities=p))
    generated, _ = plan_flow(context, predictions, hierarchical=hierarchical)
    assert Counter(n.start for n in generated) == Counter(n.start for n in notes)
    assert all(n.end <= 500 for n in generated if n.start == 0)
    assert all(n.end == 1000 for n in generated if n.start == 500)
    greedy_lanes(generated, 6)


def test_unknown_endpoint_preserved_as_uncertainty_not_a_fabricated_hold():
    context = AnchorContext([Note(0, 0, 0)], 6, TIMING)
    generated, report = plan_flow(context, [dict(options=context.options(0), probabilities=[0., 1.])], hierarchical=True)
    assert report['unresolved'] == 1 and not generated[0].hold


def test_invalid_probability_rejected():
    context = AnchorContext([Note(0, 0, 0)], 6, TIMING)
    with pytest.raises(ValueError, match='candidate'):
        plan_flow(context, [dict(options=context.options(0), probabilities=[np.nan, 0.])])


def test_unfinished_solver_incumbent_is_not_exported(monkeypatch):
    from types import SimpleNamespace
    import manialab.gesture_flow as module
    context = AnchorContext([Note(0, 0, 0)], 6, TIMING)
    monkeypatch.setattr(module, 'linprog', lambda *a, **k:
                        SimpleNamespace(success=False, message='Time limit reached', x=np.ones(4)))
    with pytest.raises(ValueError, match='did not reach an optimum'):
        plan_flow(context, [dict(options=context.options(0), probabilities=[1., 0.])])
