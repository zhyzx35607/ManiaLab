"""Replay the fixed Sakura candidates with and without drum group reduction."""
from __future__ import annotations

import argparse
from collections import Counter
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

BASE = ROOT / 'test-data/expression-current-replay-v1'
BASE_SHA = '94ee190b32aafd035a3bc9fb5946b1f21089566ca2aed41f1e6275e47bd26005'


def run(output):
    assert not output.exists(), 'Preserve earlier or partial comparisons'
    assert sha(ANALYSIS) == ANALYSIS_SHA and sha(BASE/'report.json') == BASE_SHA
    baseline = read(BASE/'report.json')
    assert sha(BASE/'execution-plan.json') == baseline['execution_plan_sha256']
    frozen = read(BASE/'execution-plan.json')
    analysis = read(ANALYSIS)
    before = copy.deepcopy(analysis)
    sources = ('manialab/expression.py', 'manialab/expression_articulation.py',
               'manialab/expression_activity.py', 'manialab/arrange.py', 'manialab/layout.py',
               'manialab/domain.py', 'manialab/timing.py',
               'scripts/check_expression_current_replay.py', 'scripts/check_drum_group_fallback.py')
    source_hashes = {name: sha(ROOT/name) for name in sources}
    inputs = {str(ANALYSIS.relative_to(ROOT)): ANALYSIS_SHA,
              str((BASE/'report.json').relative_to(ROOT)): BASE_SHA,
              str((BASE/'execution-plan.json').relative_to(ROOT)): baseline['execution_plan_sha256']}
    for case in baseline['cases']:
        for name, digest in case['artifacts'].items():
            path = BASE/f"{case['keys']}k"/name
            assert sha(path) == digest
            inputs[path.relative_to(ROOT).as_posix()] = digest
    save(output/'execution-plan.json', dict(sources=source_hashes, inputs=inputs,
        settings=frozen['settings'], generation_clock=frozen['generation_clock'],
        keys=list(range(6, 11)), neural_inference=False, reference_chart_read=False,
        automatic_timing_acceptance=False, musical_acceptance=False,
        baseline_control='Disable only the new drum_group_anchor inside a scoped patch.'))
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    rows = []
    clock = frozen['generation_clock']
    for keys in range(6, 11):
        folder = output/f'{keys}k'
        settings = parameters(frozen['settings'] | dict(keys=keys))
        plan = expression.propose_expression(analysis, settings)
        assert plan == read(BASE/f'{keys}k/current_plan.json')
        candidates, metadata, trace = expression.expression_evidence(analysis, plan, settings)
        assert [n.data() for n in candidates] == read(BASE/f'{keys}k/current_expression.json')
        aligned, _ = align_candidates(chart_clock_notes(candidates, clock['audio_shift_ms']), clock, confirmed=True)
        with patch.object(expression, 'drum_group_anchor', return_value=None):
            old, control = expression.select_expression(aligned, metadata, copy.deepcopy(trace),
                settings, 60000/clock['bpm'])
        prior = read(BASE/f'{keys}k/current_selected.json')
        assert endpoints(old) == sorted((n['id'], n['start'], n['end'], n['voice'], n['pitch'],
                                        n['instrument_id']) for n in prior)
        assert control['decisions'] == read(BASE/f'{keys}k/current_selection.json')['decisions']
        chosen, selection = expression.select_expression(aligned, metadata, copy.deepcopy(trace),
            settings, 60000/clock['bpm'])
        all_candidates = {n.id: n for n in aligned}
        assert len({n.id for n in chosen}) == len(chosen)
        assert all(n.data() == all_candidates[n.id].data() for n in chosen)
        assert all(not n.hold or n.release_reason not in expression.UNOBSERVED_RELEASES for n in chosen)
        capacity = whole_capacity(chosen, settings)
        assert capacity['feasible']
        placed = assign_lanes(chosen, settings, beat_ms=60000/clock['bpm'])
        assert endpoints(placed) == endpoints(chosen)
        structure, grid = validate(placed, keys), grid_report(placed, clock)
        assert structure['valid'] and grid['off_grid'] == 0
        text = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu',
            title='Sakura drum group comparison', audio='audio.mp3', bpm=clock['bpm'],
            offset=clock['offset'], version=f'{keys}K drum fallback research',
            grid_division=clock['division'], bookmarks=analysis['structure']['boundaries'][1:-1])
        parsed = parse_osu(text)
        assert parsed['keys'] == keys
        coordinates = lambda notes: sorted((n.start, n.end, n.lane) for n in notes)
        assert coordinates(parsed['notes']) == coordinates(placed)
        old_ids, new_ids = {n.id for n in old}, {n.id for n in chosen}
        assert old_ids <= new_ids
        assert all(all_candidates[i].voice == 'drums' and not all_candidates[i].hold for i in new_ids-old_ids)
        assert sum(n.hold for n in old) == sum(n.hold for n in chosen)
        simplified = {d['group'] for d in selection['decisions'] if d.get('group_simplification')}
        drum_times = lambda notes: sorted({n.start for n in notes if n.voice == 'drums'})
        max_gap = lambda times: max((b-a for a, b in zip(times, times[1:])), default=0)
        row = dict(keys=keys, old=summary(old, control), new=summary(chosen, selection),
            added=dict(Counter(all_candidates[i].voice for i in new_ids-old_ids)),
            removed=dict(Counter(all_candidates[i].voice for i in old_ids-new_ids)),
            simplified_drum_groups=len(simplified), capacity=capacity,
            drum_fallback_solver=selection['drum_fallback_solver'], original_selected_ids_preserved=True,
            drum_max_gap_ms=dict(old=max_gap(drum_times(old)), new=max_gap(drum_times(chosen))),
            baseline_control_reproduced=True, candidate_times_unchanged=True,
            selected_times_preserved_after_layout=True, structural=structure, grid=grid)
        artifacts = dict(plan=plan, old_selection=control, new_selection=selection,
            old_selected=[n.data() for n in old], new_selected=[n.data() for n in chosen],
            placed=[n.data() for n in placed],
            changes=dict(added=sorted(new_ids-old_ids), removed=sorted(old_ids-new_ids)))
        for name, data in artifacts.items():
            save(folder/(name+'.json'), data)
        with (folder/'chart.osu').open('x', encoding='utf-8') as stream:
            stream.write(text)
        row['artifacts'] = {name: sha(folder/name) for name in
                            [*(name+'.json' for name in artifacts), 'chart.osu']}
        rows.append(row)
        print(f"{keys}K: drums {row['old']['voices'].get('drums', 0)} -> "
              f"{row['new']['voices'].get('drums', 0)}, simplified groups {len(simplified)}", flush=True)
    assert before == analysis
    assert source_hashes == {name: sha(ROOT/name) for name in sources}
    assert inputs == {name: sha(ROOT/name) for name in inputs}
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    save(output/'report.json', dict(cases=rows, execution_plan_sha256=sha(output/'execution-plan.json'),
        neural_inference=False, reference_chart_read=False, inputs_unchanged=True,
        layout_executed=True, exports_read_back=True,
        automatic_timing_acceptance=False, musical_acceptance=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args().output)
