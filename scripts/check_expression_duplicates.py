"""Compare equivalent-pitch selection against the saved Sakura baseline.

Uses fixed audio detections and timing; no neural inference or reference notes.
"""
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

BASE = ROOT / 'test-data/drum-group-fallback-v3'
BASE_SHA = 'b3c822d35b91f87d9be1f51709c08b5c4dc361d4de01092c496c09961437c7e5'


def former_deduplication(notes, trace, settings, segment_preferences):
    """The former loop, after its unchanged start/strength/pitch sort."""
    seen, kept = set(), []
    for note in notes:
        key = (note.start, note.pitch, note.source)
        if key in seen:
            trace[note.id].update(decision='omitted', reason='duplicate_pitch')
        else:
            seen.add(key)
            kept.append(note)
    return kept


def coordinates(notes):
    return sorted((n.start, n.end, n.lane) for n in notes)


def semantic_coordinates(notes):
    return Counter((n.start, n.end, n.pitch, n.voice, n.instrument_id, n.source) for n in notes)


def run(output):
    assert not output.exists(), 'Preserve previous and partial comparisons'
    assert sha(ANALYSIS) == ANALYSIS_SHA and sha(BASE/'report.json') == BASE_SHA
    baseline = read(BASE/'report.json')
    assert sha(BASE/'execution-plan.json') == baseline['execution_plan_sha256']
    frozen = read(BASE/'execution-plan.json')
    analysis = read(ANALYSIS)
    original = copy.deepcopy(analysis)
    sources = ('manialab/expression.py', 'manialab/expression_articulation.py',
               'manialab/expression_activity.py', 'manialab/arrange.py', 'manialab/layout.py',
               'manialab/domain.py', 'manialab/timing.py',
               'scripts/check_expression_current_replay.py', 'scripts/check_expression_duplicates.py')
    source_hashes = {name: sha(ROOT/name) for name in sources}
    inputs = {ANALYSIS.relative_to(ROOT).as_posix(): ANALYSIS_SHA,
              (BASE/'report.json').relative_to(ROOT).as_posix(): BASE_SHA,
              (BASE/'execution-plan.json').relative_to(ROOT).as_posix(): baseline['execution_plan_sha256']}
    for case in baseline['cases']:
        for name, digest in case['artifacts'].items():
            path = BASE/f"{case['keys']}k"/name
            assert sha(path) == digest
            inputs[path.relative_to(ROOT).as_posix()] = digest
    save(output/'execution-plan.json', dict(sources=source_hashes, inputs=inputs,
        settings=frozen['settings'], generation_clock=frozen['generation_clock'],
        keys=list(range(6, 11)), neural_inference=False, reference_chart_read=False,
        automatic_timing_acceptance=False, musical_acceptance=False,
        baseline_control='Restore former first-duplicate loop in a scoped in-memory patch.'))
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    rows, clock = [], frozen['generation_clock']
    current_deduplication = expression._deduplicate_pitches
    for keys in range(6, 11):
        folder = output/f'{keys}k'
        settings = parameters(frozen['settings'] | dict(keys=keys))
        plan = expression.propose_expression(analysis, settings)
        assert plan == read(BASE/f'{keys}k/plan.json')
        evidence, metadata, trace = expression.expression_evidence(analysis, plan, settings)
        aligned, _ = align_candidates(chart_clock_notes(evidence, clock['audio_shift_ms']), clock, confirmed=True)
        candidates_before = [n.data() for n in aligned]
        with patch.object(expression, '_deduplicate_pitches', former_deduplication):
            old, control = expression.select_expression(aligned, metadata, copy.deepcopy(trace), settings, 60000/clock['bpm'])
        assert [n.data() for n in old] == read(BASE/f'{keys}k/new_selected.json')
        assert control['decisions'] == read(BASE/f'{keys}k/new_selection.json')['decisions']
        substitutions = []

        def record_duplicates(notes, decision_trace, prefs, segments):
            prior = former_deduplication(notes, copy.deepcopy(decision_trace), prefs, segments)
            after = current_deduplication(notes, decision_trace, prefs, segments)
            assert len(prior) == len(after)
            for a, b in zip(prior, after):
                if a.id == b.id:
                    continue
                assert semantic_coordinates([a]) == semantic_coordinates([b])
                assert all(getattr(a, key) == getattr(b, key) for key in
                    ('strength', 'lane', 'raw', 'release_reason', 'phrase', 'support', 'support_coverage'))
                pref = expression.settings_at(prefs, a.start, segments)
                old_score, new_score = expression._attack_value(a, pref), expression._attack_value(b, pref)
                assert new_score >= old_score
                substitutions.append(dict(old=a.data(), new=b.data(), old_score=old_score, new_score=new_score))
            return after

        with patch.object(expression, '_deduplicate_pitches', record_duplicates):
            chosen, selection = expression.select_expression(aligned, metadata, copy.deepcopy(trace), settings, 60000/clock['bpm'])
        assert [n.data() for n in aligned] == candidates_before
        all_candidates = {n.id: n for n in aligned}
        assert len({n.id for n in chosen}) == len(chosen)
        assert all(n.data() == all_candidates[n.id].data() for n in chosen)
        assert all(not n.hold or n.release_reason not in expression.UNOBSERVED_RELEASES for n in chosen)
        capacity = whole_capacity(chosen, settings)
        assert capacity['feasible']
        placed = assign_lanes(chosen, settings, beat_ms=60000/clock['bpm'])
        assert endpoints(placed) == endpoints(chosen)
        structural, grid = validate(placed, keys), grid_report(placed, clock)
        assert structural['valid'] and grid['off_grid'] == 0
        exported = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu',
            title='Sakura duplicate evidence comparison', audio='audio.mp3', bpm=clock['bpm'],
            offset=clock['offset'], version=f'{keys}K duplicate evidence research',
            grid_division=clock['division'], bookmarks=analysis['structure']['boundaries'][1:-1])
        parsed = parse_osu(exported)
        assert parsed['keys'] == keys and coordinates(parsed['notes']) == coordinates(placed)
        old_ids, new_ids = {n.id for n in old}, {n.id for n in chosen}
        for change in substitutions:
            change['old_selected'] = change['old']['id'] in old_ids
            change['new_selected'] = change['new']['id'] in new_ids
        old_semantics, new_semantics = semantic_coordinates(old), semantic_coordinates(chosen)
        row = dict(keys=keys, old=summary(old, control), new=summary(chosen, selection),
            substitutions=len(substitutions), strictly_higher_score=sum(s['new_score'] > s['old_score'] for s in substitutions),
            selected_substitutions=sum(s['old_selected'] and s['new_selected'] for s in substitutions),
            removed_ids=len(old_ids-new_ids), added_ids=len(new_ids-old_ids),
            removed_semantic_objects=sum((old_semantics-new_semantics).values()),
            added_semantic_objects=sum((new_semantics-old_semantics).values()),
            baseline_control_reproduced=True, original_candidates_unchanged=True,
            selected_endpoints_preserved_after_layout=True, capacity=capacity, structural=structural, grid=grid)
        artifacts = dict(plan=plan, candidates=candidates_before, old_selection=control, new_selection=selection,
            old_selected=[n.data() for n in old], new_selected=[n.data() for n in chosen],
            placed=[n.data() for n in placed], substitutions=substitutions,
            changes=dict(added=sorted(new_ids-old_ids), removed=sorted(old_ids-new_ids)))
        for name, data in artifacts.items():
            save(folder/(name+'.json'), data)
        with (folder/'chart.osu').open('x', encoding='utf-8') as stream:
            stream.write(exported)
        row['artifacts'] = {name: sha(folder/name) for name in [*(name+'.json' for name in artifacts), 'chart.osu']}
        rows.append(row)
        print(f"{keys}K: {len(substitutions)} evidence substitutions; "
              f"{len(old)} -> {len(chosen)} notes; "
              f"{row['removed_semantic_objects']} removed/{row['added_semantic_objects']} added chart objects", flush=True)
    assert original == analysis
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
