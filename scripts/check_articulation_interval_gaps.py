"""Replay observed-interval gaps against frozen Sakura and SMD detections.

No network inference or reference-note scoring. Refuse existing output folders.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import manialab.expression as expression
from manialab.arrange import assign_lanes, parameters
from manialab.domain import Note, UNOBSERVED_RELEASES, export_osu, note_from, parse_osu
from manialab.domain import validate
from manialab.piano import OUTPUTS, decode
from manialab.timing import align_candidates, chart_clock_notes, grid_report
from scripts.check_expression_current_replay import (
    ANALYSIS, ANALYSIS_SHA, endpoints, read, save, sha, summary, whole_capacity)
from scripts.check_expression_duplicates import coordinates

BASE = ROOT/'test-data/expression-duplicates-v1'
BASE_SHA = '5908d5afe3661d29a9b65d4d80ac91b16feee8002d42487c7cb652ae635684e6'
LEGACY = BASE/'source/manialab/expression_articulation.py'
SMD = ROOT/'test-data/smd/selected-v1/benchmark'


def expression_pair(analysis, settings, former):
    original = copy.deepcopy(analysis)
    plan = expression.propose_expression(analysis, settings)
    with patch.object(expression, 'articulation_policy', former):
        old, old_meta, old_trace = expression.expression_evidence(analysis, plan, settings)
    new, metadata, trace = expression.expression_evidence(analysis, plan, settings)
    assert analysis == original and old_meta == metadata
    raw = {n['id']: n for n in analysis['candidates']}
    assert {n.id for n in old} == {n.id for n in new}
    for notes in (old, new):
        for n in notes:
            assert n.start == raw[n.id]['start'] and n.pitch == raw[n.id]['pitch']
            assert n.end in (n.start, raw[n.id]['end'])
            assert not n.hold or n.release_reason not in UNOBSERVED_RELEASES
            assert n == replace(note_from(raw[n.id]), end=n.end, phrase=n.phrase)
    by_id = {n.id: n for n in old}
    assert all(n.end >= by_id[n.id].end for n in new), 'Gap protection must not create new taps'
    changes = [dict(id=n.id, voice=n.voice, start=n.start,
                    old_end=by_id[n.id].end, new_end=n.end) for n in new if n != by_id[n.id]]
    return plan, old, new, metadata, old_trace, trace, changes


def attack_sets(notes, metadata):
    return {(metadata[n.id]['section'], n.voice, n.start) for n in notes}


def run(output):
    assert not output.exists(), 'Preserve all previous and partial comparisons'
    assert sha(ANALYSIS) == ANALYSIS_SHA and sha(BASE/'report.json') == BASE_SHA
    baseline = read(BASE/'report.json')
    assert sha(BASE/'execution-plan.json') == baseline['execution_plan_sha256']
    frozen = read(BASE/'execution-plan.json')
    assert sha(LEGACY) == frozen['sources']['manialab/expression_articulation.py']
    inputs = {path: sha(path) for path in (ANALYSIS, BASE/'report.json', BASE/'execution-plan.json', LEGACY)}
    for case in baseline['cases']:
        for name, digest in case['artifacts'].items():
            path = BASE/f"{case['keys']}k"/name
            assert sha(path) == digest
            inputs[path] = digest
    manifest, smd_report = read(SMD/'manifest.json'), read(SMD/'specialist-v2/report.json')
    assert sha(SMD/'manifest.json') == smd_report['identity']['manifest_sha256']
    for path in (SMD/'manifest.json', SMD/'specialist-v2/report.json'):
        inputs[path] = sha(path)
    for case in manifest['cases']:
        name = case['name']
        expected = next(row for row in smd_report['cases'] if row['name'] == name)
        path = SMD/'specialist-v2'/f'{name}-notes.json'
        assert sha(path) == expected['notes_sha256']
        inputs[path] = sha(path)
        path = SMD/'specialist-v2'/f'{name}-activations.npz'
        inputs[path] = sha(path)
    sources = ('manialab/expression_articulation.py', 'manialab/expression.py',
        'manialab/expression_activity.py', 'manialab/domain.py', 'manialab/arrange.py',
        'manialab/layout.py', 'manialab/timing.py', 'manialab/piano.py',
        'scripts/check_expression_current_replay.py', 'scripts/check_expression_duplicates.py',
        'scripts/check_articulation_interval_gaps.py')
    source_hashes = {name: sha(ROOT/name) for name in sources}
    save(output/'execution-plan.json', dict(sources=source_hashes,
        inputs={p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()},
        settings=frozen['settings'], generation_clock=frozen['generation_clock'],
        keys=list(range(6, 11)), interval_gap_ms=80, preserve_existing_sustains=True, neural_inference=False,
        reference_notes_read=False, musical_acceptance=False,
        smd_context='Use all decoded input events for context; report changes inside original score window.'))
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    spec = importlib.util.spec_from_file_location('manialab._interval_gap_baseline', LEGACY)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    analysis, clock = read(ANALYSIS), frozen['generation_clock']
    rows = []
    for keys in range(6, 11):
        folder = output/f'{keys}k'
        settings = parameters(frozen['settings'] | dict(keys=keys))
        plan, old, new, metadata, old_trace, trace, changes = expression_pair(
            analysis, settings, legacy.articulation_policy)
        assert plan == read(BASE/f'{keys}k/plan.json')
        old_aligned, _ = align_candidates(chart_clock_notes(old, clock['audio_shift_ms']), clock, confirmed=True)
        assert [n.data() for n in old_aligned] == read(BASE/f'{keys}k/candidates.json')
        previous, prior_selection = expression.select_expression(
            old_aligned, metadata, old_trace, settings, 60000/clock['bpm'])
        assert [n.data() for n in previous] == read(BASE/f'{keys}k/new_selected.json')
        assert prior_selection['decisions'] == read(BASE/f'{keys}k/new_selection.json')['decisions']
        old_placed = assign_lanes(previous, settings, beat_ms=60000/clock['bpm'])
        assert [n.data() for n in old_placed] == read(BASE/f'{keys}k/placed.json')
        aligned, _ = align_candidates(chart_clock_notes(new, clock['audio_shift_ms']), clock, confirmed=True)
        saved_candidates = [n.data() for n in aligned]
        before_selection = copy.deepcopy(trace)
        selected, selection = expression.select_expression(aligned, metadata, trace, settings, 60000/clock['bpm'])
        assert [n.data() for n in aligned] == saved_candidates
        lookup = {n.id: n for n in aligned}
        assert all(n == lookup[n.id] for n in selected)
        placed = assign_lanes(selected, settings, beat_ms=60000/clock['bpm'])
        assert endpoints(placed) == endpoints(selected)
        capacity, structural, grid = whole_capacity(selected, settings), validate(placed, keys), grid_report(placed, clock)
        assert capacity['feasible'] and structural['valid'] and grid['off_grid'] == 0
        exported = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu',
            title='Observed interval gap comparison', audio='audio.mp3', bpm=clock['bpm'], offset=clock['offset'],
            version=f'{keys}K interval gap comparison', grid_division=clock['division'],
            bookmarks=analysis['structure']['boundaries'][1:-1])
        parsed = parse_osu(exported)
        assert parsed['keys'] == keys and coordinates(parsed['notes']) == coordinates(placed)
        before_attacks, after_attacks = attack_sets(previous, metadata), attack_sets(selected, metadata)
        data = dict(plan=plan, expression_changes=changes, expression_trace=before_selection,
            candidates=saved_candidates, selection=selection, selected=[n.data() for n in selected],
            placed=[n.data() for n in placed])
        for name, value in data.items():
            save(folder/f'{name}.json', value)
        with (folder/'chart.osu').open('x', encoding='utf-8') as stream:
            stream.write(exported)
        row = dict(keys=keys, old=summary(previous, prior_selection), new=summary(selected, selection),
            changed_expression=len(changes), removed_attacks=dict(Counter(v for _, v, _ in before_attacks-after_attacks)),
            added_attacks=dict(Counter(v for _, v, _ in after_attacks-before_attacks)),
            baseline_reproduced=True, selected_exactly_unchanged=selected == previous,
            export_coordinates_unchanged=coordinates(placed) == coordinates(old_placed),
            capacity=capacity, structural=structural, grid=grid,
            artifacts={name: sha(folder/name) for name in [*(f'{name}.json' for name in data), 'chart.osu']})
        rows.append(row)
        print(f'{keys}K: {len(changes)} expression changes; {len(selected)} objects; {sum(n.hold for n in selected)} LNs', flush=True)
    smd_rows = []
    for case in manifest['cases']:
        name = case['name']
        with np.load(SMD/'specialist-v2'/f'{name}-activations.npz', allow_pickle=False) as data:
            events = decode({key: data[key] for key in OUTPUTS})
        archived = read(SMD/'specialist-v2'/f'{name}-notes.json')
        key = lambda n: (n['pitch'], n['onset'], n['offset'])
        assert sorted(map(key, events)) == sorted(map(key, archived))
        notes = [Note(round(n['onset']*1000), round(n['offset']*1000), pitch=n['pitch'],
                      strength=n['onset_score'], voice='piano', source='pitch', id=f'p:{i}',
                      release_reason=n['release_reason']) for i, n in enumerate(events)]
        duration = case['duration_seconds']
        source = dict(duration=duration, candidates=[n.data() for n in notes],
            structure=dict(segments=[dict(start_ms=0, end_ms=round(duration*1000))]))
        _, old, new, _, _, trace, changes = expression_pair(source, parameters(), legacy.articulation_policy)
        assert {n.id for n in new} == {n.id for n in notes if n.strength >= .35}
        stop = case['score_seconds']*1000
        scored_changes = [c for c in changes if c['start'] < stop]
        artifacts = dict(candidates=source['candidates'], old=[n.data() for n in old],
            new=[n.data() for n in new], trace=trace, changes=changes)
        for field, value in artifacts.items():
            save(output/'smd'/name/f'{field}.json', value)
        smd_rows.append(dict(name=name, input_events=len(notes), expressed_events=len(new),
            score_seconds=case['score_seconds'], observed_seconds=duration,
            old_ln=sum(n.hold and n.start < stop for n in old), new_ln=sum(n.hold and n.start < stop for n in new),
            full_context_changes=len(changes), scored_changes=len(scored_changes),
            promoted=sum(c['new_end'] > c['old_end'] for c in scored_changes),
            demoted=sum(c['new_end'] < c['old_end'] for c in scored_changes),
            artifacts={f'{field}.json': sha(output/'smd'/name/f'{field}.json') for field in artifacts}))
        print(f'SMD {name}: {len(scored_changes)} scored expression changes', flush=True)
    assert source_hashes == {name: sha(ROOT/name) for name in sources}
    assert inputs == {path: sha(path) for path in inputs}
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    save(output/'report.json', dict(execution_plan_sha256=sha(output/'execution-plan.json'),
        cases=rows, smd=smd_rows, neural_inference=False, reference_notes_read=False,
        inputs_unchanged=True, original_intervals_or_taps_only=True,
        automatic_timing_acceptance=False, musical_acceptance=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args().output)
