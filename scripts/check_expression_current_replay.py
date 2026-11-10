"""Replay frozen section selection with current and former tail eligibility.

No neural inference, reference chart, production edit, lane assignment or audio
export. The old release set is restored only inside a scoped in-memory control.
"""
from __future__ import annotations

import argparse
import ast
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import manialab.expression as expression
import manialab.expression_articulation as articulation
from manialab.arrange import parameters
from manialab.timing import align_candidates, chart_clock_notes

BASE = ROOT / 'test-data/balanced-expression-default-v1'
ANALYSIS = ROOT / 'test-data/sakura-tsumugi-expression-onset-v3/analysis.json'
BASE_PLAN_SHA = 'e750c0f188aa76bdc2b7d3f94f9a0c0f280082537d90dfabb10ddb07b16b81c9'
ANALYSIS_SHA = '26a0250568b2ce149022c4aa1549af2d9dd49b045372959a6b26b5342ba5313a'
SOURCES = ('manialab/expression.py', 'manialab/expression_articulation.py',
           'manialab/expression_activity.py', 'manialab/domain.py',
           'manialab/arrange.py', 'manialab/timing.py',
           'scripts/check_expression_current_replay.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def literal_release_set(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id == 'UNOBSERVED_RELEASES' for t in node.targets):
            assert isinstance(node.value, ast.Call) and node.value.func.id == 'frozenset'
            return frozenset(ast.literal_eval(node.value.args[0]))
    raise AssertionError('Frozen release eligibility is missing')


def function_ast(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


def policy_modes(plan):
    return [(s['main_voice'], {v: (p['mode'], p['role']) for v, p in s['voices'].items()})
            for s in plan['sections']]


def endpoints(notes):
    return sorted((n.id, n.start, n.end, n.voice, n.pitch, n.instrument_id) for n in notes)


def summary(notes, selection):
    return dict(notes=len(notes), ln=sum(n.hold for n in notes),
        voices=dict(Counter(n.voice for n in notes)),
        holds=dict(Counter(n.voice for n in notes if n.hold)),
        attacks=dict(Counter(v for v, _ in {(n.voice, n.start) for n in notes})),
        large_shifts=dict(Counter(n.voice for n in notes if abs(n.start-n.evidence_start) > 30)),
        solver=selection['selection_solver'])


def tap_pressure(note, selected, settings):
    assert not note.hold
    starts = sorted(n.start for n in selected)
    t = note.start
    chord = bisect_right(starts, t)-bisect_left(starts, t)
    occupied = chord+sum(n.hold and n.start < t < n.end+settings['release_gap'] for n in selected)
    windows = [t]+starts[bisect_left(starts, t):bisect_left(starts, t+1000)]
    density = max(bisect_right(starts, at)-bisect_right(starts, at-1000)+1 for at in windows)
    return dict(constraints=[name for name, failed in (
        ('chord', chord+1 > settings['chord_size']),
        ('lanes', occupied+1 > settings['keys']),
        ('density', density > math.ceil(settings['density']))) if failed],
        added_peak_density=density, added_chord=chord+1, added_occupied=occupied+1)


def whole_capacity(notes, settings):
    starts = sorted(n.start for n in notes)
    times = sorted(set(starts))
    density = max((bisect_right(starts, t)-bisect_right(starts, t-1000) for t in times), default=0)
    chords = Counter(starts)
    chord = max(chords.values(), default=0)
    occupied = max((chords[t]+sum(n.hold and n.start < t < n.end+settings['release_gap']
                                for n in notes) for t in times), default=0)
    held = max((sum(n.hold and n.start <= t < n.end for n in notes) for t in times), default=0)
    return dict(max_density=density, max_chord=chord, max_occupied=occupied, max_held=held,
        feasible=density <= math.ceil(settings['density']) and chord <= settings['chord_size']
                 and occupied <= settings['keys'] and held <= settings['max_held'])


def drum_pressure(aligned, selected, selection, settings):
    by_id = {n.id: n for n in aligned}
    groups = defaultdict(list)
    for d in selection['decisions']:
        if 'group' in d:
            groups[d['group']].append(d)
    rows = []
    for d in selection['decisions']:
        if d['voice'] != 'drums' or d['reason'] != 'group_capacity_or_density':
            continue
        n = by_id[d['id']]
        rows.append(dict(id=n.id, start=n.start, section=d['section'], group=d['group'],
            strength=n.strength, shift_ms=abs(n.start-n.evidence_start),
            group_size=len(groups[d['group']]), **tap_pressure(n, selected, settings)))
    omitted_groups = defaultdict(list)
    for row in rows:
        omitted_groups[row['group']].append(row)
    feasible = [r for r in rows if not r['constraints']]
    anchors = []
    drum_times = sorted(n.start for n in selected if n.voice == 'drums')
    def utility(n):
        shift = abs(n.start-n.evidence_start)
        return (.25+n.strength)*(.4+.6*math.exp(-.5*(shift/20)**2))
    for group, members in omitted_groups.items():
        anchor = min((by_id[r['id']] for r in members), key=lambda n: (-utility(n), n.start, n.id))
        pressure = tap_pressure(anchor, selected, settings)
        if not pressure['constraints']:
            index = bisect_left(drum_times, anchor.start)
            anchors.append(dict(group=group, id=anchor.id, start=anchor.start,
                group_starts=[r['start'] for r in members], strength=anchor.strength,
                shift_ms=abs(anchor.start-anchor.evidence_start), utility=utility(anchor),
                previous_selected_drum=drum_times[index-1] if index else None,
                next_selected_drum=drum_times[index] if index < len(drum_times) else None,
                **pressure))
    diagnostic_union = selected+[by_id[r['id']] for r in anchors]
    stats = dict(selected_drums=sum(n.voice == 'drums' for n in selected),
        omitted_drums=len(rows), omitted_groups=len(omitted_groups),
        individually_feasible=len(feasible), feasible_within_30ms=sum(r['shift_ms'] <= 30 for r in feasible),
        feasible_in_multi_note_group=sum(r['group_size'] > 1 for r in feasible),
        constraint_counts=dict(Counter(c for r in rows for c in r['constraints'])),
        mixed_feasibility_groups=sum(any(not r['constraints'] for r in g) and
            any(r['constraints'] for r in g) for g in omitted_groups.values()),
        whole_group_status={voice: dict(total=sum(g[0]['voice'] == voice for g in groups.values()),
            fully_omitted=sum(g[0]['voice'] == voice and all(d['decision'] == 'omitted' for d in g)
                              for g in groups.values()),
            partially_selected=sum(g[0]['voice'] == voice and any(d['decision'] == 'omitted' for d in g)
                and any(d['decision'] == 'selected' for d in g) for g in groups.values()))
            for voice in ('drums', 'piano', 'bass', 'other')},
        candidate_strongest_anchors=len(anchors), diagnostic_anchor_union=whole_capacity(diagnostic_union, settings),
        generated_chart_modified=False, musical_acceptance=False)
    return dict(summary=stats, omitted_events=rows, strongest_anchors=anchors,
        notice='Individual feasibility is not joint feasibility. Anchors are diagnostic options, not accepted notes.')


def expression_diff(old, new, old_trace, new_trace, frozen_ids, current_ids):
    before, after = {n.id: n for n in old}, {n.id: n for n in new}
    assert set(before) == set(after)
    assert all(before[i].start == after[i].start for i in before)
    rows = []
    runs = defaultdict(list)
    for identity in before:
        a, b = before[identity], after[identity]
        if a.end == b.end:
            continue
        old_rule, new_rule = old_trace[identity]['articulation'], new_trace[identity]['articulation']
        row = dict(id=identity, voice=b.voice, start=b.start, old_end=a.end, current_end=b.end,
            release_reason=b.release_reason, selected_in_frozen=identity in frozen_ids,
            selected_current=identity in current_ids, old_articulation=old_rule, current_articulation=new_rule)
        rows.append(row)
        key = (old_rule['run'], old_rule['run_start'], old_rule['run_end_attack'],
               old_rule['attack_groups'], old_rule['overlap_pairs'], new_rule['overlap_pairs'])
        runs[key].append(row)
    return dict(summary=dict(changed_candidates=len(rows),
        changed_previously_selected=sum(r['selected_in_frozen'] for r in rows),
        changed_still_selected=sum(r['selected_in_frozen'] and r['selected_current'] for r in rows),
        release_reasons=dict(Counter(r['release_reason'] for r in rows)),
        old_piano_candidates=sum(n.voice == 'piano' for n in old),
        current_piano_candidates=sum(n.voice == 'piano' for n in new),
        old_piano_holds=sum(n.voice == 'piano' and n.hold for n in old),
        current_piano_holds=sum(n.voice == 'piano' and n.hold for n in new),
        current_reattack_holds=sum(n.release_reason == 'reattack' and n.hold for n in new),
        candidate_attack_identity_equal=True),
        runs=[dict(run=k[0], start_ms=k[1], end_attack_ms=k[2], attack_groups=k[3],
            old_overlap_pairs=k[4], current_overlap_pairs=k[5], required_pairs=math.ceil((k[3]-1)/2),
            changed=len(v), previously_selected=sum(r['selected_in_frozen'] for r in v),
            still_selected=sum(r['selected_in_frozen'] and r['selected_current'] for r in v))
            for k, v in runs.items()], differences=rows)


def selection_diff(old, new):
    a, b = {n.id: n for n in old}, {n.id: n for n in new}
    added, removed = sorted(set(b)-set(a)), sorted(set(a)-set(b))
    changed = [dict(id=i, voice=b[i].voice, old_start=a[i].start, current_start=b[i].start,
                    old_end=a[i].end, current_end=b[i].end, release_reason=b[i].release_reason)
               for i in sorted(set(a) & set(b)) if (a[i].start, a[i].end) != (b[i].start, b[i].end)]
    return dict(summary=dict(added=len(added), removed=len(removed), changed_retained_endpoints=len(changed),
        added_voices=dict(Counter(b[i].voice for i in added)),
        removed_voices=dict(Counter(a[i].voice for i in removed))),
        added=[b[i].data() for i in added], removed=[a[i].data() for i in removed], changed_retained=changed)


def run(output):
    assert not output.exists(), 'Preserve previous diagnostic results'
    assert sha(BASE/'execution-plan.json') == BASE_PLAN_SHA and sha(ANALYSIS) == ANALYSIS_SHA
    base_plan, analysis = read(BASE/'execution-plan.json'), read(ANALYSIS)
    base_report = read(BASE/'report.json')
    for relative, digest in base_plan['source_files'].items():
        assert sha(BASE/'source'/Path(relative).name) == digest
    for name in ('select_expression', 'chord_layer_options'):
        assert function_ast(ROOT/'manialab/expression.py', name) == function_ast(BASE/'source/expression.py', name)
    old_releases = literal_release_set(BASE/'source/domain.py')
    assert expression.UNOBSERVED_RELEASES-old_releases == {'reattack'}
    assert old_releases <= expression.UNOBSERVED_RELEASES
    source_hashes = {name: sha(ROOT/name) for name in SOURCES}
    input_paths = [ANALYSIS, BASE/'report.json', BASE/'execution-plan.json']
    for keys in range(6, 11):
        input_paths.extend(BASE/f'independent-articulation-{keys}k-{suffix}.json'
                           for suffix in ('plan', 'notes', 'selection', 'metrics'))
    input_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in input_paths}
    plan = dict(source_files=source_hashes, inputs=input_hashes,
        baseline_source_files=base_plan['source_files'], settings=base_plan['settings'],
        generation_clock=base_plan['generation_clock'], keys=list(range(6, 11)),
        old_unobserved_releases=sorted(old_releases), current_unobserved_releases=sorted(expression.UNOBSERVED_RELEASES),
        old_control='Scoped in-memory eligibility restoration; no production or historical file changes.',
        neural_inference=False, reference_chart_read=False, layout_executed=False, exports_written=False,
        musical_acceptance=False, automatic_timing_acceptance=False)
    save(output/'execution-plan.json', plan)
    for relative in SOURCES:
        target = output/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    rows = []
    clock = base_plan['generation_clock']
    for keys in range(6, 11):
        settings = parameters(base_plan['settings'] | {'keys': keys})
        frozen_plan = read(BASE/f'independent-articulation-{keys}k-plan.json')
        current_plan = expression.propose_expression(analysis, settings)
        assert policy_modes(frozen_plan) == policy_modes(current_plan)
        with patch.object(expression, 'UNOBSERVED_RELEASES', old_releases), \
                patch.object(articulation, 'UNOBSERVED_RELEASES', old_releases):
            old_evidence, old_metadata, old_trace = expression.expression_evidence(analysis, frozen_plan, settings)
        current_evidence, current_metadata, current_trace = expression.expression_evidence(analysis, current_plan, settings)
        old_trace_before, current_trace_before = copy.deepcopy(old_trace), copy.deepcopy(current_trace)
        old_aligned, _ = align_candidates(chart_clock_notes(old_evidence, clock['audio_shift_ms']), clock, confirmed=True)
        current_aligned, _ = align_candidates(chart_clock_notes(current_evidence, clock['audio_shift_ms']), clock, confirmed=True)
        old, old_selection = expression.select_expression(old_aligned, old_metadata, old_trace, settings, 60000/clock['bpm'])
        current, current_selection = expression.select_expression(current_aligned, current_metadata, current_trace, settings, 60000/clock['bpm'])
        frozen_notes_path = BASE/f'independent-articulation-{keys}k-notes.json'
        frozen_notes, frozen_selection = read(frozen_notes_path), read(BASE/f'independent-articulation-{keys}k-selection.json')
        frozen_case = next(c for c in base_report['cases'] if c['keys'] == keys)
        assert sha(frozen_notes_path) == frozen_case['notes_sha256']
        assert endpoints(old) == sorted((n['id'], n['start'], n['end'], n['voice'], n['pitch'], n['instrument_id']) for n in frozen_notes)
        assert old_selection['decisions'] == frozen_selection['decisions']
        assert old_selection['selection_solver']['status'] == current_selection['selection_solver']['status'] == 0
        assert all(not n.hold or n.release_reason not in expression.UNOBSERVED_RELEASES for n in current)
        old_pressure = drum_pressure(old_aligned, old, old_selection, settings)
        current_pressure = drum_pressure(current_aligned, current, current_selection, settings)
        changes = expression_diff(old_evidence, current_evidence, old_trace_before, current_trace_before,
                                  {n.id for n in old}, {n.id for n in current})
        selected_changes = selection_diff(old, current)
        folder = output/f'{keys}k'
        artifacts = dict(frozen_selected=frozen_notes, old_tail_control_selected=[n.data() for n in old],
            current_selected=[n.data() for n in current], current_plan=current_plan,
            old_tail_control_selection=old_selection, current_selection=current_selection,
            old_tail_control_expression=[n.data() for n in old_evidence],
            current_expression=[n.data() for n in current_evidence],
            expression_differences=changes, selection_differences=selected_changes,
            frozen_drum_pressure=old_pressure, current_drum_pressure=current_pressure)
        for name, data in artifacts.items():
            save(folder/(name+'.json'), data)
        row = dict(keys=keys, frozen_matches_old_tail_control=True, frozen_decisions_reproduced=True,
            plan_modes_roles_equal=True, frozen=summary(old, old_selection), current=summary(current, current_selection),
            expression_changes=changes['summary'], selection_changes=selected_changes['summary'],
            frozen_drum_pressure=old_pressure['summary'], current_drum_pressure=current_pressure['summary'],
            artifacts={name+'.json': sha(folder/(name+'.json')) for name in artifacts})
        rows.append(row)
        print(json.dumps(dict(keys=keys, frozen=row['frozen'], current=row['current'],
                              changes=row['selection_changes']), ensure_ascii=True), flush=True)
    assert source_hashes == {name: sha(ROOT/name) for name in SOURCES}
    assert input_hashes == {name: sha(ROOT/name) for name in input_hashes}
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    report = dict(cases=rows, execution_plan_sha256=sha(output/'execution-plan.json'),
        input_files_unchanged=True, production_files_unchanged=True, former_tail_rule_restored_after_control=True,
        neural_inference=False, reference_chart_read=False, layout_executed=False, exports_written=False,
        automatic_timing_acceptance=False, musical_acceptance=False,
        limitations=['Frozen candidates and a manually supplied clock; no acoustic or timing accuracy test.',
            'Selected-note differences occur before lane assignment; no new chart was exported.',
            'Former release eligibility is a diagnostic control, not a proposal to restore unsafe LN tails.',
            'Feasible omitted attacks and anchor options do not prove they belong in a musical arrangement.'])
    save(output/'report.json', report)
    print(json.dumps(dict(completed=True, report=str(output/'report.json'),
                          report_sha256=sha(output/'report.json')), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args().output)
