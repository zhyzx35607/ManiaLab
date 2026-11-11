"""Compare bounded sustain expression coverage with frozen 6–10K selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import manialab.expression as expression
from manialab.arrange import assign_lanes, parameters
from manialab.domain import export_osu, parse_osu, validate
from manialab.timing import align_candidates, chart_clock_notes, grid_report
from scripts.check_expression_current_replay import (
    ANALYSIS, ANALYSIS_SHA, endpoints, read, save, sha, summary, whole_capacity)
from scripts.check_expression_duplicates import coordinates
from scripts.check_expression_coherence import BASE, BASE_SHA, WINDOWS

WEIGHT = .25


def duration_union(notes):
    """Full selected durations per source, separate from bounded solver reward."""
    voices = defaultdict(list)
    for n in notes:
        if n.hold:
            voices[n.voice].append((n.start, n.end))
    result = {}
    for voice, intervals in voices.items():
        total, right = 0, -1
        for begin, end in sorted(intervals):
            total += max(0, end-max(begin, right))
            right = max(right, end)
        result[voice] = total
    return result


def run(output):
    assert not output.exists(), 'Preserve previous and partial runs'
    assert sha(ANALYSIS) == ANALYSIS_SHA and sha(BASE/'report.json') == BASE_SHA
    baseline = read(BASE/'report.json')
    assert sha(BASE/'execution-plan.json') == baseline['execution_plan_sha256']
    frozen = read(BASE/'execution-plan.json')
    inputs = {path:sha(path) for path in (ANALYSIS, BASE/'report.json', BASE/'execution-plan.json')}
    for case in baseline['cases']:
        for name, digest in case['artifacts'].items():
            path = BASE/f"{case['keys']}k"/name
            assert sha(path) == digest
            inputs[path] = digest
    sources = ('manialab/expression.py', 'manialab/expression_sustain.py', 'manialab/expression_coherence.py',
        'manialab/expression_articulation.py', 'manialab/expression_activity.py', 'manialab/arrange.py',
        'manialab/layout.py', 'manialab/domain.py', 'manialab/timing.py',
        'scripts/check_expression_current_replay.py', 'scripts/check_expression_duplicates.py',
        'scripts/check_expression_coherence.py', 'scripts/check_sustain_coverage.py',
        'docs/sustain-coverage-protocol.md')
    hashes = {name:sha(ROOT/name) for name in sources}
    save(output/'execution-plan.json', dict(sources=hashes,
        inputs={path.relative_to(ROOT).as_posix():digest for path,digest in inputs.items()},
        settings=frozen['settings'], generation_clock=frozen['generation_clock'], keys=list(range(6,11)),
        sustain_coverage_weight=WEIGHT, horizon_beats=2, coherence=0,
        windows_ms=(*WINDOWS,(216000,224000)), neural_inference=False, reference_notes_read=False,
        automatic_timing_acceptance=False, musical_acceptance=False))
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    analysis, clock = read(ANALYSIS), frozen['generation_clock']
    original = copy.deepcopy(analysis)
    cases = []
    for keys in range(6,11):
        folder = output/f'{keys}k'
        settings = parameters(frozen['settings'] | dict(keys=keys))
        plan = expression.propose_expression(analysis, settings)
        assert plan == read(BASE/f'{keys}k/plan.json')
        evidence, metadata, trace = expression.expression_evidence(analysis, plan, settings)
        aligned, _ = align_candidates(chart_clock_notes(evidence, clock['audio_shift_ms']), clock, confirmed=True)
        candidates = [n.data() for n in aligned]
        assert candidates == read(BASE/f'{keys}k/candidates.json')
        old, control = expression.select_expression(aligned, metadata, copy.deepcopy(trace), settings, 60000/clock['bpm'])
        assert [n.data() for n in old] == read(BASE/f'{keys}k/selected.json')
        assert control['decisions'] == read(BASE/f'{keys}k/selection.json')['decisions']
        selected, selection = expression.select_expression(aligned, metadata, copy.deepcopy(trace), settings,
            60000/clock['bpm'], sustain_coverage=WEIGHT)
        assert [n.data() for n in aligned] == candidates
        lookup = {n.id:n for n in aligned}
        assert len({n.id for n in selected}) == len(selected) and all(n == lookup[n.id] for n in selected)
        assert all(not n.hold or n.release_reason not in expression.UNOBSERVED_RELEASES for n in selected)
        old_decisions = {d['id']:d for d in control['decisions']}
        selected_lookup = {n.id:n for n in selected}
        for decision in selection['decisions']:
            assert decision.get('group') == old_decisions[decision['id']].get('group')
            if decision['reason'] == 'represented_by_chord':
                assert decision['represented_by'] and all(identity in selected_lookup
                    and selected_lookup[identity].start == decision['represented_at'] for identity in decision['represented_by'])
        # Recompute every reward cell directly from selected notes and the
        # original policy; do not trust the auxiliary solver variables.
        reward = 0.
        for cell in selection['sustain_coverage']['intervals']:
            supported = [n for n in selected if n.voice == cell['voice']
                and metadata[n.id]['section'] == cell['section'] and trace[n.id].get('mode') == 'sustain'
                and n.source == 'pitch' and n.hold and not n.phrase and not n.support
                and n.release_reason and n.release_reason not in expression.UNOBSERVED_RELEASES
                and n.start <= cell['start'] < cell['end'] <= min(n.end,n.start+120000/clock['bpm'])]
            assert sorted(n.id for n in supported) == cell['selected_notes']
            value = max((expression._attack_value(n, settings)*(2 if metadata[n.id]['role']=='primary' else 1)
                         for n in supported), default=0.)
            assert abs(value-cell['selected_value']) < 1e-10
            reward += WEIGHT*(cell['end']-cell['start'])/(60000/clock['bpm'])*value
        assert abs(reward-selection['sustain_coverage']['reward']) < 1e-8
        placed = assign_lanes(selected, settings, beat_ms=60000/clock['bpm'])
        assert endpoints(placed) == endpoints(selected)
        capacity, structural, grid = whole_capacity(selected, settings), validate(placed, keys), grid_report(placed, clock)
        assert capacity['feasible'] and structural['valid'] and grid['off_grid'] == 0
        exported = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu',
            title='Sakura sustain coverage comparison', audio='audio.mp3', bpm=clock['bpm'], offset=clock['offset'],
            version=f'{keys}K sustain research', grid_division=clock['division'],
            bookmarks=analysis['structure']['boundaries'][1:-1])
        parsed = parse_osu(exported)
        assert parsed['keys'] == keys and coordinates(parsed['notes']) == coordinates(placed)
        old_ids, new_ids = {n.id for n in old}, {n.id for n in selected}
        old_attacks, new_attacks = ({(n.voice,n.start) for n in notes} for notes in (old,selected))
        artifacts = dict(plan=plan, candidates=candidates, selection=selection, selected=[n.data() for n in selected],
            placed=[n.data() for n in placed], changes=dict(removed=sorted(old_ids-new_ids),added=sorted(new_ids-old_ids),
                removed_attacks=sorted(old_attacks-new_attacks),added_attacks=sorted(new_attacks-old_attacks)))
        for name, value in artifacts.items():
            save(folder/f'{name}.json', value)
        with (folder/'chart.osu').open('x', encoding='utf-8') as stream:
            stream.write(exported)
        cases.append(dict(keys=keys, old=summary(old,control), new=summary(selected,selection),
            baseline_reproduced=True, old_sustain_ms=duration_union(old), new_sustain_ms=duration_union(selected),
            coverage_reward=reward, removed_ids=len(old_ids-new_ids),added_ids=len(new_ids-old_ids),
            removed_attacks=dict(Counter(v for v,_ in old_attacks-new_attacks)),
            added_attacks=dict(Counter(v for v,_ in new_attacks-old_attacks)),
            capacity=capacity, structural=structural, grid=grid,
            artifacts={name:sha(folder/name) for name in [*(f'{name}.json' for name in artifacts),'chart.osu']}))
        print(f'{keys}K: {len(old)} -> {len(selected)} objects; '
              f'{sum(n.hold for n in old)} -> {sum(n.hold for n in selected)} LNs', flush=True)
    assert analysis == original and hashes == {name:sha(ROOT/name) for name in sources}
    assert inputs == {path:sha(path) for path in inputs}
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    save(output/'report.json', dict(cases=cases,execution_plan_sha256=sha(output/'execution-plan.json'),
        inputs_unchanged=True, neural_inference=False, reference_notes_read=False,
        automatic_timing_acceptance=False, musical_acceptance=False, production_default_changed=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args().output)
