"""Replay explicit piano rhythm subsets without inferring accompaniment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab import expression
from manialab.arrange import assign_lanes, parameters
from manialab.domain import export_osu, parse_osu, validate
from manialab.timing import align_candidates, chart_clock_notes, grid_report
from scripts.check_expression_current_replay import (
    ANALYSIS, ANALYSIS_SHA, endpoints, read, save, sha, summary, whole_capacity)
from scripts.check_expression_duplicates import coordinates
from scripts.check_sustain_coverage import duration_union

BASE = ROOT/'test-data/expression-precision-v1'
BASE_SHA = '6e249755768c8d4bf70064f5b6cbf3b8077932a85254050860a8f34a0115fe8e'


def scope_rows(candidates, selected, metadata, selection):
    """Count actual attack omissions; no LN can stand in for a missing row."""
    streams = defaultdict(lambda: defaultdict(list))
    decisions = {d['id']: d for d in selection['decisions']}
    selected_ids = {n.id for n in selected}
    for n in candidates:
        streams[(metadata[n.id]['section'], n.voice, n.instrument_id)][n.start].append(n)
    result = []
    for (section, voice, identity), attacks in sorted(streams.items()):
        rows, runs, run, chosen_times = [], [], [], []
        for t, notes in sorted(attacks.items()):
            picked = sorted(n.id for n in notes if n.id in selected_ids)
            allowed = any(decisions[n.id].get('rhythm_policy', {}).get('keep', True) for n in notes)
            rows.append(dict(start=t, candidate_ids=sorted(n.id for n in notes), selected_ids=picked,
                policy_allowed=allowed, reasons=dict(Counter(decisions[n.id]['reason']
                    for n in notes if n.id not in selected_ids))))
            if picked:
                chosen_times.append(t)
                if run:
                    runs.append(run)
                    run = []
            else:
                run.append(t)
        if run:
            runs.append(run)
        longest = max(runs, key=lambda r: (len(r), r[-1]-r[0]), default=[])
        result.append(dict(section=section, voice=voice, instrument_id=identity,
            candidate_attacks=len(rows), policy_allowed_attacks=sum(r['policy_allowed'] for r in rows),
            policy_omitted_attacks=sum(not r['policy_allowed'] for r in rows),
            selected_attacks=len(chosen_times),
            allowed_but_unselected_attacks=sum(r['policy_allowed'] and not r['selected_ids'] for r in rows),
            longest_unselected_run=dict(attacks=len(longest), first=longest[0] if longest else None,
                last=longest[-1] if longest else None),
            max_selected_gap_ms=max((b-a for a,b in zip(chosen_times, chosen_times[1:])), default=None),
            rows=rows))
    return result


def run(output):
    assert not output.exists(), 'Preserve previous and partial runs'
    assert sha(ANALYSIS) == ANALYSIS_SHA and sha(BASE/'report.json') == BASE_SHA
    baseline, frozen = read(BASE/'report.json'), read(BASE/'execution-plan.json')
    assert sha(BASE/'execution-plan.json') == baseline['execution_plan_sha256']
    inputs = {ROOT/name:digest for name,digest in frozen['inputs'].items()}
    inputs.update({BASE/'report.json':BASE_SHA, BASE/'execution-plan.json':baseline['execution_plan_sha256']})
    for case in baseline['cases']:
        inputs.update({BASE/case['mode']/f"{case['keys']}k"/name:digest
                       for name,digest in case['artifacts'].items()})
    assert all(sha(path) == digest for path,digest in inputs.items())
    sources = ('manialab/expression.py', 'manialab/expression_rhythm.py',
        'manialab/expression_sustain.py', 'manialab/expression_coherence.py',
        'manialab/expression_articulation.py', 'manialab/expression_activity.py',
        'manialab/arrange.py', 'manialab/layout.py', 'manialab/domain.py', 'manialab/timing.py',
        'scripts/check_expression_current_replay.py', 'scripts/check_expression_duplicates.py',
        'scripts/check_sustain_coverage.py', 'scripts/check_expression_rhythm.py',
        'docs/expression-rhythm-protocol.md')
    hashes = {name:sha(ROOT/name) for name in sources}
    clock = frozen['generation_clock']
    save(output/'execution-plan.json', dict(sources=hashes,
        inputs={p.relative_to(ROOT).as_posix():digest for p,digest in inputs.items()},
        settings=frozen['settings'], generation_clock=clock, weights=[0., .25], keys=list(range(6,11)),
        masks=[dict(name='full', stride=1, phase=0), dict(name='phase0', stride=2, phase=0),
               dict(name='phase1', stride=2, phase=1)], scope='every_existing_piano_section_and_class',
        automatic_role_inference=False, automatic_timing_acceptance=False, musical_acceptance=False))
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    analysis, cases = read(ANALYSIS), []
    original = copy.deepcopy(analysis)
    solve = expression._solve_binary
    for keys in range(6,11):
        settings = parameters(frozen['settings'] | dict(keys=keys))
        plan = expression.propose_expression(analysis, settings)
        evidence, metadata, trace = expression.expression_evidence(analysis, plan, settings)
        candidates, _ = align_candidates(chart_clock_notes(evidence, clock['audio_shift_ms']), clock, confirmed=True)
        candidate_data = [n.data() for n in candidates]
        assert candidate_data == read(ROOT/f'test-data/chord-evidence-ties-v1/{keys}k/candidates.json')
        lookup = {n.id:n for n in candidates}
        scopes = sorted({(metadata[n.id]['section'], n.voice, n.instrument_id) for n in candidates if n.voice=='piano'})
        for mode, weight in [('default', 0.), ('sustain', .25)]:
            old_folder = BASE/mode/f'{keys}k'
            old = read(old_folder/'selected.json')
            old_selection = read(old_folder/'selection.json')
            for mask, phase in [('full', None), ('phase0', 0), ('phase1', 1)]:
                folder = output/mode/mask/f'{keys}k'
                policies = [dict(section=s,voice=v,instrument_id=i,stride=2,phase=phase) for s,v,i in scopes]
                policy = None if phase is None else dict(clock=clock, confirmed=True, policies=policies)
                calls = []
                def capture(cost, **kwargs):
                    start = time.perf_counter()
                    solved = solve(cost, **kwargs)
                    calls.append(dict(options=copy.deepcopy(kwargs['options']), elapsed_seconds=time.perf_counter()-start,
                        status=int(solved.status), gap=float(solved.mip_gap) if solved.x is not None else None))
                    return solved
                with patch.object(expression, '_solve_binary', capture):
                    selected, selection = expression.select_expression(candidates, metadata, copy.deepcopy(trace),
                        settings, 60000/clock['bpm'], sustain_coverage=weight, rhythm_policy=policy)
                assert candidate_data == [n.data() for n in candidates]
                assert all(n == lookup[n.id] for n in selected)
                selected_ids = {n.id for n in selected}
                for d in selection['decisions']:
                    if d['reason'] == 'rhythm_policy_omitted':
                        assert not d['rhythm_policy']['keep'] and d['id'] not in selected_ids
                        assert not d.get('represented_by')
                    if d.get('rhythm_policy', {}).get('keep') is False:
                        assert d['id'] not in selected_ids
                placed = assign_lanes(selected, settings, beat_ms=60000/clock['bpm'])
                assert endpoints(placed) == endpoints(selected)
                if phase is None:
                    assert [n.data() for n in selected] == old
                    assert selection['decisions'] == old_selection['decisions']
                    assert [n.data() for n in placed] == read(old_folder/'placed.json')
                capacity, structural, grid = whole_capacity(selected, settings), validate(placed, keys), grid_report(placed, clock)
                assert capacity['feasible'] and structural['valid'] and grid['off_grid']==0
                exported = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu',
                    title='Sakura fixed rhythm comparison', audio='audio.mp3', bpm=clock['bpm'], offset=clock['offset'],
                    version=f'{keys}K {mode} {mask} research', grid_division=clock['division'],
                    bookmarks=analysis['structure']['boundaries'][1:-1])
                assert coordinates(parse_osu(exported)['notes']) == coordinates(placed)
                old_attacks = {(n['voice'],n['start']) for n in old}
                new_attacks = {(n.voice,n.start) for n in selected}
                rows = scope_rows(candidates, selected, metadata, selection)
                artifacts = dict(selection=selection, selected=[n.data() for n in selected],
                    placed=[n.data() for n in placed], scopes=rows,
                    changes=dict(removed=sorted({n['id'] for n in old}-selected_ids),
                        added=sorted(selected_ids-{n['id'] for n in old}),
                        removed_attacks=sorted(old_attacks-new_attacks), added_attacks=sorted(new_attacks-old_attacks)))
                for name,value in artifacts.items():
                    save(folder/f'{name}.json',value)
                with (folder/'chart.osu').open('x',encoding='utf-8') as stream:
                    stream.write(exported)
                paths = [folder/f'{name}.json' for name in artifacts]+[folder/'chart.osu']
                cases.append(dict(keys=keys, mode=mode, mask=mask, summary=summary(selected,selection),
                    solver_calls=calls, full_reproduced=phase is None, sustain_ms=duration_union(selected),
                    removed_attacks=dict(Counter(v for v,_ in old_attacks-new_attacks)),
                    added_attacks=dict(Counter(v for v,_ in new_attacks-old_attacks)),
                    piano_scopes=[{k:v for k,v in row.items() if k!='rows'} for row in rows if row['voice']=='piano'],
                    capacity=capacity, structural=structural, grid=grid,
                    artifacts={p.name:sha(p) for p in paths}))
                print(f'{keys}K {mode} {mask}: {len(selected)} objects, {sum(n.hold for n in selected)} LN; '
                      f'{selection["selection_solver"]}; calls={calls}', flush=True)
    assert analysis == original and all(sha(p)==digest for p,digest in inputs.items())
    assert hashes == {name:sha(ROOT/name) for name in sources}
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    save(output/'report.json',dict(cases=cases, execution_plan_sha256=sha(output/'execution-plan.json'),
        inputs_unchanged=True, neural_inference=False, reference_notes_read=False,
        automatic_timing_acceptance=False, musical_acceptance=False, production_default_changed=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args().output)
