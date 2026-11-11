"""Replay fixed expression objectives with zero relative-gap early stopping."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import manialab.expression as expression
from manialab import expression_sustain
from manialab.arrange import assign_lanes, parameters
from manialab.domain import export_osu, parse_osu, validate
from manialab.timing import align_candidates, chart_clock_notes, grid_report
from scripts.check_expression_current_replay import (
    ANALYSIS, ANALYSIS_SHA, endpoints, read, save, sha, summary, whole_capacity)
from scripts.check_expression_duplicates import coordinates
from scripts.check_expression_coherence import BASE, BASE_SHA

STUDY = ROOT/'test-data/sustain-coverage-v1'
STUDY_SHA = 'eadd62a18e779d9c1d7cf843025eaed7eeaddbc63b339eb49216fe90a312587b'


def reconstruct(selection, units, cost, matrix, upper):
    """Recover original primary variables, excluding the later drum fallback."""
    decisions = {d['id']: d for d in selection['decisions']}
    selected = {d['id'] for d in decisions.values()
                if d['decision'] == 'selected' and d['reason'] != 'drum_group_fallback'}
    groups = defaultdict(list)
    for index, notes in enumerate(units):
        families = {decisions[n.id]['group'] for n in notes}
        assert len(families) == 1
        groups[next(iter(families))].append(index)
    chosen = np.zeros(len(cost), dtype=np.int8)
    for indices in groups.values():
        eligible = [i for i in indices if {n.id for n in units[i]} <= selected]
        if eligible:
            chosen[max(eligible, key=lambda i: len(units[i]))] = 1
    assert {n.id for i, notes in enumerate(units) if chosen[i] for n in notes} == selected
    # All enabled auxiliary variables here are positive threshold rewards;
    # coherence is deliberately disabled. Each has a single upper-bound row.
    csc = matrix.tocsc()
    csr = matrix.tocsr()
    for i in range(len(units), len(cost)):
        column = csc.getcol(i)
        assert cost[i] < 0 and column.nnz == 1 and column.data[0] == 1
        row = int(column.indices[0])
        assert upper[row] == 0
        weights = csr.getrow(row)
        others = [(j, v) for j, v in zip(weights.indices, weights.data) if j != i]
        assert others and all(j < len(units) and v == -1 for j, v in others)
        chosen[i] = any(chosen[j] for j, _ in others)
    assert np.all(matrix @ chosen <= upper+1e-7)
    value = float(-cost @ chosen)
    assert abs(value-selection['selection_solver']['objective']) < 1e-7
    return chosen, value


def run(output):
    assert not output.exists(), 'Preserve earlier and partial runs'
    assert sha(ANALYSIS) == ANALYSIS_SHA and sha(BASE/'report.json') == BASE_SHA
    assert sha(STUDY/'report.json') == STUDY_SHA
    inputs = {ANALYSIS: ANALYSIS_SHA}
    for folder in (BASE, STUDY):
        report = read(folder/'report.json')
        assert sha(folder/'execution-plan.json') == report['execution_plan_sha256']
        inputs.update({folder/name: sha(folder/name) for name in ('report.json', 'execution-plan.json')})
        for case in report['cases']:
            for name, digest in case['artifacts'].items():
                path = folder/f"{case['keys']}k"/name
                assert sha(path) == digest
                inputs[path] = digest
    sources = ('manialab/expression.py', 'manialab/expression_sustain.py', 'manialab/expression_coherence.py',
        'manialab/expression_articulation.py', 'manialab/expression_activity.py', 'manialab/arrange.py',
        'manialab/layout.py', 'manialab/domain.py', 'manialab/timing.py',
        'scripts/check_expression_current_replay.py', 'scripts/check_expression_duplicates.py',
        'scripts/check_expression_coherence.py', 'scripts/check_expression_precision.py',
        'docs/expression-precision-protocol.md')
    source_hashes = {name: sha(ROOT/name) for name in sources}
    frozen = read(STUDY/'execution-plan.json')
    save(output/'execution-plan.json', dict(sources=source_hashes,
        inputs={p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
        settings=frozen['settings'], generation_clock=frozen['generation_clock'],
        weights=[0., .25], keys=list(range(6, 11)), mip_rel_gap=0., unchanged_time_limits_seconds=[15., 5.],
        neural_inference=False, reference_notes_read=False, musical_acceptance=False))
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    analysis, clock = read(ANALYSIS), frozen['generation_clock']
    original = copy.deepcopy(analysis)
    cases = []
    solve, intervals = expression._solve_binary, expression_sustain.coverage_intervals
    for mode, weight, baseline in [('default', 0., BASE), ('sustain', .25, STUDY)]:
        for keys in range(6, 11):
            folder = output/mode/f'{keys}k'
            settings = parameters(frozen['settings'] | dict(keys=keys))
            plan = expression.propose_expression(analysis, settings)
            assert plan == read(baseline/f'{keys}k/plan.json')
            evidence, metadata, trace = expression.expression_evidence(analysis, plan, settings)
            candidates, _ = align_candidates(chart_clock_notes(evidence, clock['audio_shift_ms']), clock, confirmed=True)
            assert [n.data() for n in candidates] == read(baseline/f'{keys}k/candidates.json')
            units, calls, primary = [], [], {}
            def capture_units(options, *args, **kwargs):
                units.extend(options)
                return intervals(options, *args, **kwargs)
            def strict_solve(cost, **kwargs):
                options = dict(kwargs['options'], mip_rel_gap=0.)
                kwargs = dict(kwargs, options=options)
                started = time.perf_counter()
                result = solve(cost, **kwargs)
                calls.append(dict(options=options, elapsed_seconds=time.perf_counter()-started,
                    status=int(result.status), objective=float(-result.fun) if result.fun is not None else None,
                    gap=float(result.mip_gap) if result.x is not None else None))
                if len(calls) == 1:
                    primary.update(cost=cost, matrix=kwargs['constraints'].A,
                        upper=kwargs['constraints'].ub, x=result.x)
                return result
            with patch.object(expression, '_solve_binary', strict_solve), \
                    patch.object(expression_sustain, 'coverage_intervals', capture_units):
                selected, selection = expression.select_expression(candidates, metadata, copy.deepcopy(trace),
                    settings, 60000/clock['bpm'], sustain_coverage=weight)
            old = read(baseline/f'{keys}k/selected.json')
            old_selection = read(baseline/f'{keys}k/selection.json')
            old_x, old_value = reconstruct(old_selection, units, primary['cost'], primary['matrix'], primary['upper'])
            np.testing.assert_allclose(primary['x'], np.round(primary['x']), atol=1e-6)
            assert np.all(primary['matrix'] @ primary['x'] <= primary['upper']+1e-6)
            witness = None
            if keys == 10:
                old9 = read(baseline/'9k/selection.json')
                witness_x, witness_value = reconstruct(old9, units, primary['cost'], primary['matrix'], primary['upper'])
                witness = dict(source=f'{baseline.name}/9k/selection.json', feasible=True,
                    objective=witness_value, improvement_over_old_10k=witness_value-old_value)
                save(folder/'9k-witness.json', dict(report=witness, chosen=np.flatnonzero(witness_x).tolist()))
            lookup = {n.id: n for n in candidates}
            assert all(n == lookup[n.id] for n in selected)
            placed = assign_lanes(selected, settings, beat_ms=60000/clock['bpm'])
            assert endpoints(placed) == endpoints(selected)
            capacity, structural, grid = whole_capacity(selected, settings), validate(placed, keys), grid_report(placed, clock)
            assert capacity['feasible'] and structural['valid'] and grid['off_grid'] == 0
            exported = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu',
                title='Sakura expression precision', audio='audio.mp3', bpm=clock['bpm'], offset=clock['offset'],
                version=f'{keys}K {mode} precision research', grid_division=clock['division'],
                bookmarks=analysis['structure']['boundaries'][1:-1])
            assert coordinates(parse_osu(exported)['notes']) == coordinates(placed)
            old_attacks = {(n['voice'], n['start']) for n in old}
            new_attacks = {(n.voice, n.start) for n in selected}
            artifacts = dict(selected=[n.data() for n in selected], selection=selection,
                placed=[n.data() for n in placed],
                model=dict(cost=primary['cost'].tolist(), upper=primary['upper'].tolist(),
                    units=[[n.id for n in group] for group in units], old_chosen=np.flatnonzero(old_x).tolist(),
                    chosen=np.flatnonzero(np.round(primary['x'])).tolist()),
                changes=dict(removed=sorted({n['id'] for n in old}-{n.id for n in selected}),
                    added=sorted({n.id for n in selected}-{n['id'] for n in old}),
                    removed_attacks=sorted(old_attacks-new_attacks), added_attacks=sorted(new_attacks-old_attacks)))
            for name, value in artifacts.items():
                save(folder/f'{name}.json', value)
            save_npz(folder/'constraints.npz', primary['matrix'])
            with (folder/'chart.osu').open('x', encoding='utf-8') as stream:
                stream.write(exported)
            paths = [folder/f'{name}.json' for name in artifacts] + [folder/'constraints.npz', folder/'chart.osu']
            if witness:
                paths.append(folder/'9k-witness.json')
            cases.append(dict(mode=mode, keys=keys, old_solver=old_selection['selection_solver'],
                old_notes=len(old), old_ln=sum(n['end'] > n['start'] for n in old),
                new=summary(selected, selection), solver_calls=calls, old_primary_reconstructed=True,
                witness_9k=witness, capacity=capacity, structural=structural, grid=grid,
                removed_attacks=dict(Counter(v for v, _ in old_attacks-new_attacks)),
                added_attacks=dict(Counter(v for v, _ in new_attacks-old_attacks)),
                artifacts={p.name: sha(p) for p in paths}))
            print(f'{mode} {keys}K: {old_value:.6f} -> {selection["selection_solver"]["objective"]:.6f}; '
                  f'{len(old)} -> {len(selected)} notes; calls={calls}', flush=True)
    assert analysis == original and all(sha(p) == digest for p, digest in inputs.items())
    assert source_hashes == {name: sha(ROOT/name) for name in sources}
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    save(output/'report.json', dict(cases=cases, execution_plan_sha256=sha(output/'execution-plan.json'),
        inputs_unchanged=True, neural_inference=False, reference_notes_read=False,
        musical_acceptance=False, automatic_timing_acceptance=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args().output)
