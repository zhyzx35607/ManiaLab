"""Replay saved analyses through production generation without audio or models.

The plan explicitly freezes each input, clock, key count and parameter set.
Confirmed-clock runs are diagnostic controls, never automatic Timing acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.arrange import parameters
from manialab.domain import UNOBSERVED_RELEASES, export_osu, note_from, parse_osu
from manialab.generation import generate_variant
from manialab.preferences import ChartPreferences, row_settings, settings_at
from manialab.recommendations import adjusted_segments
from manialab.timing import align_candidates, chart_clock_notes


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sources():
    files = [*sorted((ROOT/'manialab').glob('*.py')), Path(__file__),
        ROOT/'assets/pattern-prior.json', ROOT/'assets/arrangement-profiles.json',
        ROOT/'vendor/mapperatorinator/template.osu', ROOT/'vendor/mapperatorinator/event.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def capacity_report(notes, prefs, segments=()):
    """Independent event sweep, including capacity changes without an onset."""
    times = sorted({n.start for n in notes} | {n.end for n in notes if n.hold}
        | {n.end + settings_at(prefs,n,segments)['release_gap'] for n in notes if n.hold}
        | {s[k] for s in segments for k in ('start_ms', 'end_ms')})
    failures, boundary_density = [], []
    peaks = dict(density=0, chord_size=0, max_held=0, keys=0)
    for t in times:
        row = [n for n in notes if n.start == t]
        pref = settings_at(prefs, t, segments)
        if row:
            original = row_settings(prefs, row, segments)
            pref['max_held'] = min(pref['max_held'], original['max_held'])
            for key in ('density', 'chord_size'):
                pref[key] = original[key]
        values = dict(density=sum(t-1000 < n.start <= t for n in notes),
            chord_size=len(row), max_held=sum(n.start <= t < n.end for n in notes if n.hold),
            keys=len(row)+sum(n.start < t < n.end+settings_at(prefs,n,segments)['release_gap'] for n in notes if n.hold))
        for key, count in values.items():
            peaks[key] = max(peaks[key], count)
            limit = math.ceil(pref[key])
            if count > limit:
                issue = dict(time_ms=t, constraint=key, actual=count, limit=limit)
                if key == 'density' and not row:
                    boundary_density.append(issue)
                else:
                    failures.append(issue)
    return dict(valid=not failures, peaks=peaks, violations=failures,
        boundary_density_observations=boundary_density,
        scope='selected output only; omitted offered rows and their tighter original preferences are not reconstructed',
        density_contract='rolling (t-1000,t] checked at selected output attacks; non-attack events reported separately')


def union_ms(notes):
    end = total = 0
    for n in sorted((n for n in notes if n.hold), key=lambda n: (n.start, n.end)):
        total += max(0, n.end-max(end, n.start))
        end = max(end, n.end)
    return total


def summarize(analysis, variant, segments=()):
    notes = [note_from(n) for n in variant['notes']]
    selected = {n.id for n in notes}
    raw = {n['id']: n for n in analysis['candidates']}
    decisions = variant['selection'].get('decisions', [])
    voices = {}
    for voice in sorted({n.get('voice', 'mix') for n in analysis['candidates']} | {n.voice for n in notes}):
        output = [n for n in notes if n.voice == voice]
        input_notes = [n for n in analysis['candidates'] if n.get('voice', 'mix') == voice]
        omitted = [d for d in decisions if d.get('voice') == voice and d.get('decision') == 'omitted']
        shifts = sorted(abs(n.start-n.evidence_start) for n in output if n.evidence_start is not None)
        voices[voice] = dict(candidates=len(input_notes), selected=len(output),
            retained_raw_ids=sum(n['id'] in selected for n in input_notes),
            ln=sum(n.hold for n in output), union_ln_ms=union_ms(output),
            unknown_release_ln=sum(n.hold and not n.release_reason for n in output),
            large_attack_shifts=sum(s > 30 for s in shifts), max_attack_shift_ms=max(shifts, default=0),
            omitted_reasons=dict(Counter(d['reason'] for d in omitted)))
    sections = []
    for i, section in enumerate(analysis.get('structure', {}).get('segments', [])):
        start, end = section['start_ms'], section['end_ms']
        local = [n for n in notes if start <= (n.audio_start if n.audio_start is not None else raw.get(n.id, {}).get('start', n.start)) < end]
        sections.append(dict(index=i, start_ms=start, end_ms=end,
            selected_voices=dict(Counter(n.voice for n in local)), ln_voices=dict(Counter(n.voice for n in local if n.hold))))
    forbidden = [n.id for n in notes if n.hold and n.release_reason in UNOBSERVED_RELEASES]
    diagnostics = variant.get('timing_diagnostics') or {}
    return dict(notes=len(notes), ln=sum(n.hold for n in notes), voices=voices, sections=sections,
        capacity=capacity_report(notes, variant['settings'], segments), unsupported_release_ln=forbidden,
        structural=variant['report']['valid'], off_grid=variant['report']['grid']['off_grid'],
        timing_suspect_windows=sum(s.get('suspect', False) for s in variant['timing_sections']),
        timing_diagnostics=dict(status=diagnostics.get('status', 'legacy'),
            window_clock=diagnostics.get('window_clock', 'unspecified'),
            assessed_windows=len(variant['timing_sections']),
            insufficient_windows=len(diagnostics.get('insufficient_windows', [])),
            fallback_windows=sum(s.get('fallback', False) for s in variant['timing_sections'])),
        omission_reasons=variant['selection'].get('omission_reasons', {}),
        solver=variant['selection'].get('selection_solver'))


def compare_variants(before, after):
    old, new = ({n['id']: n for n in v['notes']} for v in (before, after))
    shared = set(old) & set(new)
    changed = [dict(id=i, before={k: old[i].get(k) for k in ('start','end','lane')},
        after={k: new[i].get(k) for k in ('start','end','lane')}) for i in sorted(shared)
        if any(old[i].get(k) != new[i].get(k) for k in ('start','end','lane'))]
    evidence_fields = ('pitch','voice','source','strength','instrument_id','release_reason',
        'evidence_start','evidence_end','audio_start','phrase','support','support_coverage')
    evidence_changed = [dict(id=i, fields=[k for k in evidence_fields if old[i].get(k) != new[i].get(k)])
        for i in sorted(shared) if any(old[i].get(k) != new[i].get(k) for k in evidence_fields)]
    return dict(added=sorted(set(new)-set(old)), removed=sorted(set(old)-set(new)), changed=changed,
        timing_changed=[c['id'] for c in changed if any(c['before'][k] != c['after'][k] for k in ('start','end'))],
        lane_only_changed=[c['id'] for c in changed if all(c['before'][k] == c['after'][k] for k in ('start','end'))],
        evidence_changed=evidence_changed,
        added_voices=dict(Counter(new[i].get('voice','mix') for i in set(new)-set(old))),
        removed_voices=dict(Counter(old[i].get('voice','mix') for i in set(old)-set(new))))


def replay_case(analysis, case):
    prefs = parameters(case['settings'])
    tempo = copy.deepcopy(case['tempo'])
    plan = copy.deepcopy(case.get('plan', {}))
    state = dict(analysis=copy.deepcopy(analysis), plan=plan, variants=[], active=-1)
    # Probe the actual automatic snapping gate without pretending confirmation.
    try:
        align_candidates(chart_clock_notes([note_from(n) for n in analysis['candidates']], tempo['audio_shift_ms']),
            tempo, confirmed=False)
        gate = dict(accepted=True)
    except ValueError as error:
        gate = dict(accepted=False, reason=str(error))
    generate_variant(state, dict(mode='audio', snap=True, timing_confirmed=True), prefs, tempo,
        lambda *_: None, lambda: False)
    if state['analysis'] != analysis:
        raise ValueError('Generation changed the frozen analysis')
    variant = state['variants'][-1]
    timeline = ChartPreferences(adjusted_segments(plan, prefs), tempo['audio_shift_ms'])
    summary = summarize(analysis, variant, timeline)
    summary['automatic_timing_gate'] = gate
    notes = [note_from(n) for n in variant['notes']]
    exported = export_osu(notes, prefs['keys'], ROOT/'vendor/mapperatorinator/template.osu',
        bpm=tempo['bpm'], offset=tempo['offset'], grid_division=tempo['division'],
        title='Frozen generation diagnostic', audio='diagnostic-audio-not-bundled.wav')
    parsed = parse_osu(exported)
    coordinates = lambda seq: sorted((n.start, n.end, n.lane) for n in seq)
    summary['export_roundtrip'] = parsed['keys'] == prefs['keys'] and coordinates(parsed['notes']) == coordinates(notes)
    summary['passed'] = (summary['structural'] and not summary['off_grid'] and summary['export_roundtrip']
        and summary['capacity']['valid'] and not summary['unsupported_release_ln'])
    return variant, summary, exported


def run(plan_path, output, baseline=None):
    plan_path, output = Path(plan_path), Path(output)
    plan_hash, snapshot = sha(plan_path), sources()
    plan = read(plan_path)
    if plan.get('schema') != 'manialab-frozen-generation-v1' or not plan.get('cases'):
        raise ValueError('Expected a nonempty frozen generation plan')
    identifiers = [c['id'] for c in plan['cases']]
    if len(set(identifiers)) != len(identifiers) or any(not i or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in i) for i in identifiers):
        raise ValueError('Case IDs must be unique simple directory names')
    for case in plan['cases']:
        if sha(ROOT/case['analysis']) != case['analysis_sha256']:
            raise ValueError('Frozen analysis changed: ' + case['id'])
    baseline_identity = None
    old_report = read(Path(baseline)/'report.json') if baseline else None
    if old_report and old_report['plan_sha256'] != plan_hash:
        raise ValueError('Baseline uses a different plan; compare identical inputs and settings')
    if old_report:
        baseline_identity = dict(report_sha256=sha(Path(baseline)/'report.json'),
            source_files=old_report['source_files'], plan_sha256=old_report['plan_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    write(output/'plan.json', plan)
    cases = []
    for case in plan['cases']:
        folder = output/case['id']
        folder.mkdir()
        try:
            variant, summary, exported = replay_case(read(ROOT/case['analysis']), case)
            write(folder/'variant.json', variant)
            (folder/'chart.osu').write_text(exported, encoding='utf-8')
            summary.update(id=case['id'], keys=case['settings']['keys'])
            if old_report:
                previous = next(r for r in old_report['cases'] if r['id'] == case['id'])
                old_path = Path(baseline)/case['id']/'variant.json'
                if sha(old_path) != previous['variant_sha256']:
                    raise ValueError('Baseline variant changed: ' + case['id'])
                comparison = compare_variants(read(old_path), variant)
                write(folder/'changes.json', comparison)
                summary['changes'] = {k: len(v) if isinstance(v, list) else v for k, v in comparison.items()}
            summary['variant_sha256'] = sha(folder/'variant.json')
        except Exception as error:
            summary = dict(id=case['id'], keys=case['settings']['keys'], passed=False,
                error=type(error).__name__ + ': ' + str(error))
        cases.append(summary)
        write(folder/'summary.json', summary)
        print(json.dumps(dict(id=case['id'], passed=summary['passed'], notes=summary.get('notes'),
            violations=len(summary.get('capacity', {}).get('violations', [])), error=summary.get('error')), ensure_ascii=True), flush=True)
    if sources() != snapshot or sha(plan_path) != plan_hash or any(sha(ROOT/c['analysis']) != c['analysis_sha256'] for c in plan['cases']):
        raise ValueError('Source or frozen input changed during replay; results are not comparable')
    result = dict(schema='manialab-frozen-generation-results-v1', plan_sha256=plan_hash,
        source_files=snapshot, cases=cases, passed=all(c['passed'] for c in cases),
        baseline=baseline_identity,
        runtime=dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
            threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}),
        limitations=dict(neural_inference=False, reference_chart_read=False, musical_acceptance=False,
            timing_acceptance=False, confirmation='forced only for diagnostic generation',
            scope=plan.get('scope'), edge_exclusion='none; full frozen clip retained'))
    write(output/'report.json', result)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--baseline', type=Path)
    args = cli.parse_args()
    report = run(args.plan, args.output, args.baseline)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
