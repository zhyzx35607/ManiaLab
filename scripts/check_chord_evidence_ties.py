"""Replay equal-weight chord ranking against the last complete Sakura baseline.

No new model inference, reference-chart reading or automatic timing claim.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import manialab.expression as expression
from manialab.arrange import assign_lanes, parameters
from manialab.domain import export_osu, parse_osu, validate
from manialab.timing import align_candidates, chart_clock_notes, grid_report
from scripts.check_expression_current_replay import (
    ANALYSIS, ANALYSIS_SHA, endpoints, read, save, sha, summary, whole_capacity)
from scripts.check_expression_duplicates import coordinates

BASE = ROOT/'test-data/articulation-interval-gaps-v3'
BASE_SHA = '1fcf7a60dbf820c21a49efc866a56a035821706dc7a6ba609e02eb4d2fd8d279'
WINDOWS = ((16000, 24000), (96000, 104000), (200000, 208000))


def former_layers():
    path = BASE/'source/manialab/expression.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'chord_layer_options')
    namespace = dict(vars(expression))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['chord_layer_options']


def attack_sets(notes):
    return {(n.voice, n.start) for n in notes}


def run(output):
    assert not output.exists(), 'Preserve previous and partial runs'
    assert sha(ANALYSIS) == ANALYSIS_SHA and sha(BASE/'report.json') == BASE_SHA
    baseline = read(BASE/'report.json')
    assert sha(BASE/'execution-plan.json') == baseline['execution_plan_sha256']
    frozen = read(BASE/'execution-plan.json')
    legacy = BASE/'source/manialab/expression.py'
    assert sha(legacy) == frozen['sources']['manialab/expression.py']
    inputs = {path: sha(path) for path in (ANALYSIS, BASE/'report.json', BASE/'execution-plan.json', legacy)}
    for case in baseline['cases']:
        for name, digest in case['artifacts'].items():
            path = BASE/f"{case['keys']}k"/name
            assert sha(path) == digest
            inputs[path] = digest
    sources = ('manialab/expression.py', 'manialab/expression_articulation.py',
        'manialab/expression_activity.py', 'manialab/arrange.py', 'manialab/layout.py',
        'manialab/domain.py', 'manialab/timing.py', 'scripts/check_expression_current_replay.py',
        'scripts/check_expression_duplicates.py', 'scripts/check_chord_evidence_ties.py')
    hashes = {name: sha(ROOT/name) for name in sources}
    save(output/'execution-plan.json', dict(sources=hashes,
        inputs={path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        settings=frozen['settings'], generation_clock=frozen['generation_clock'], keys=list(range(6, 11)),
        windows_ms=WINDOWS, neural_inference=False, reference_notes_read=False,
        automatic_timing_acceptance=False, musical_acceptance=False))
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    analysis, clock = read(ANALYSIS), frozen['generation_clock']
    original = copy.deepcopy(analysis)
    former, current = former_layers(), expression.chord_layer_options
    cases = []
    for keys in range(6, 11):
        folder = output/f'{keys}k'
        settings = parameters(frozen['settings'] | dict(keys=keys))
        plan = expression.propose_expression(analysis, settings)
        assert plan == read(BASE/f'{keys}k/plan.json')
        evidence, metadata, trace = expression.expression_evidence(analysis, plan, settings)
        assert trace == read(BASE/f'{keys}k/expression_trace.json')
        aligned, _ = align_candidates(chart_clock_notes(evidence, clock['audio_shift_ms']), clock, confirmed=True)
        candidates = [n.data() for n in aligned]
        assert candidates == read(BASE/f'{keys}k/candidates.json')
        with patch.object(expression, 'chord_layer_options', former):
            old, control = expression.select_expression(aligned, metadata, copy.deepcopy(trace), settings, 60000/clock['bpm'])
        assert [n.data() for n in old] == read(BASE/f'{keys}k/selected.json')
        assert control['decisions'] == read(BASE/f'{keys}k/selection.json')['decisions']
        ranks = {}

        def record(notes, prefs, segments):
            before, after = former(notes, prefs, segments), current(notes, prefs, segments)
            assert len(before) == len(after)
            for a, b in zip(before, after):
                assert Counter(n.start for n in a) == Counter(n.start for n in b)
                assert sorted((n.start, n.strength) for n in a) == sorted((n.start, n.strength) for n in b)
                assert sum(expression._attack_value(n, expression.settings_at(prefs, n.start, segments)) for n in b) >= sum(
                    expression._attack_value(n, expression.settings_at(prefs, n.start, segments)) for n in a)-1e-9
            if before != after:
                ranks[tuple(sorted(n.id for n in notes))] = dict(
                    input_ids=sorted(n.id for n in notes),
                    old=[[n.id for n in option] for option in before],
                    new=[[n.id for n in option] for option in after])
            return after

        with patch.object(expression, 'chord_layer_options', record):
            selected, selection = expression.select_expression(aligned, metadata, copy.deepcopy(trace), settings, 60000/clock['bpm'])
        assert [n.data() for n in aligned] == candidates
        lookup = {n.id: n for n in aligned}
        assert len({n.id for n in selected}) == len(selected)
        assert all(n == lookup[n.id] for n in selected)
        assert all(not n.hold or n.release_reason not in expression.UNOBSERVED_RELEASES for n in selected)
        selected_by_id = {n.id: n for n in selected}
        for decision in selection['decisions']:
            if decision['reason'] == 'represented_by_chord':
                assert decision['represented_by'] and all(identity in selected_by_id
                    and selected_by_id[identity].start == decision['represented_at']
                    for identity in decision['represented_by'])
        placed = assign_lanes(selected, settings, beat_ms=60000/clock['bpm'])
        assert endpoints(placed) == endpoints(selected)
        capacity, structural, grid = whole_capacity(selected, settings), validate(placed, keys), grid_report(placed, clock)
        assert capacity['feasible'] and structural['valid'] and grid['off_grid'] == 0
        exported = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu',
            title='Sakura chord evidence comparison', audio='audio.mp3', bpm=clock['bpm'], offset=clock['offset'],
            version=f'{keys}K chord evidence research', grid_division=clock['division'],
            bookmarks=analysis['structure']['boundaries'][1:-1])
        parsed = parse_osu(exported)
        assert parsed['keys'] == keys and coordinates(parsed['notes']) == coordinates(placed)
        old_ids, new_ids = {n.id for n in old}, {n.id for n in selected}
        old_attacks, new_attacks = attack_sets(old), attack_sets(selected)
        decisions = {side: {d['id']: d for d in report['decisions']}
                     for side, report in [('old', control), ('new', selection)]}
        windows = [dict(start_ms=start, end_ms=end, notes=[dict(
            raw=n, aligned=lookup[n['id']].data() if n['id'] in lookup else None,
            old=decisions['old'][n['id']], new=decisions['new'][n['id']])
            for n in analysis['candidates'] if n['voice']=='other' and start <= n['start'] < end])
            for start, end in WINDOWS]
        artifacts = dict(plan=plan, candidates=candidates, selection=selection,
            selected=[n.data() for n in selected], placed=[n.data() for n in placed],
            changed_ranks=list(ranks.values()), windows=windows,
            changes=dict(removed=sorted(old_ids-new_ids), added=sorted(new_ids-old_ids),
                removed_attacks=sorted(old_attacks-new_attacks), added_attacks=sorted(new_attacks-old_attacks)))
        for name, value in artifacts.items():
            save(folder/f'{name}.json', value)
        with (folder/'chart.osu').open('x', encoding='utf-8') as stream:
            stream.write(exported)
        cases.append(dict(keys=keys, old=summary(old, control), new=summary(selected, selection),
            baseline_reproduced=True, changed_rank_inputs=len(ranks),
            removed_ids=len(old_ids-new_ids), added_ids=len(new_ids-old_ids),
            removed_attacks=dict(Counter(v for v, _ in old_attacks-new_attacks)),
            added_attacks=dict(Counter(v for v, _ in new_attacks-old_attacks)),
            original_candidates_unchanged=True, retained_notes_unchanged=True,
            capacity=capacity, structural=structural, grid=grid,
            artifacts={name: sha(folder/name) for name in [*(f'{name}.json' for name in artifacts), 'chart.osu']}))
        print(f'{keys}K: {len(old)} -> {len(selected)} objects; {sum(n.hold for n in old)} -> '
              f'{sum(n.hold for n in selected)} LNs; {len(old_ids-new_ids)} removed, {len(new_ids-old_ids)} added', flush=True)
    assert analysis == original
    assert hashes == {name: sha(ROOT/name) for name in sources}
    assert inputs == {path: sha(path) for path in inputs}
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    save(output/'report.json', dict(cases=cases, execution_plan_sha256=sha(output/'execution-plan.json'),
        inputs_unchanged=True, neural_inference=False, reference_notes_read=False,
        automatic_timing_acceptance=False, musical_acceptance=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args().output)
