"""Audit omitted drum attacks against frozen chart budgets, without changing charts.

An individually feasible attack is not a promise that all such attacks can be
added together. This diagnoses group selection, not physical transcription or
the musical value of adding an attack.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters
from manialab.expression import expression_evidence
from manialab.timing import align_candidates, chart_clock_notes
from scripts.build_provenance import sha256


def tap_pressure(note, selected, settings):
    """Exact additive feasibility for a tap and fixed, whole-song settings."""
    if note.hold:
        raise ValueError('This audit only handles percussion taps')
    starts = sorted(n.start for n in selected)
    t = note.start
    chord = bisect_right(starts, t)-bisect_left(starts, t)
    occupied = chord+sum(n.hold and n.start < t < n.end+settings['release_gap'] for n in selected)
    # The new onset affects its own trailing one-second window and all later
    # onset windows until t+1000 (exclusive), including windows after its group.
    windows = [t]+starts[bisect_left(starts, t):bisect_left(starts, t+1000)]
    density = max(bisect_right(starts, at)-bisect_right(starts, at-1000)+1 for at in windows)
    return dict(constraints=[name for name, failed in (
        ('chord', chord+1 > settings['chord_size']),
        ('lanes', occupied+1 > settings['keys']),
        ('density', density > math.ceil(settings['density']))) if failed],
        added_peak_density=density, added_chord=chord+1, added_occupied=occupied+1)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', type=Path, required=True)
    cli.add_argument('--analysis', type=Path, required=True)
    cli.add_argument('--baseline-report', type=Path, required=True)
    cli.add_argument('--source', choices=('basic-other', 'instrument-other'), required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier audits; choose a new output directory')
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    report, analysis = read(args.run/'report.json'), read(args.analysis)
    if sha256(args.analysis) != report['identity']['analyses_sha256'][args.source]:
        raise ValueError('Frozen analysis identity changed')
    settings = read(args.baseline_report)['identity']['settings']
    clock = report['identity']['generation_clock']
    rows, details = [], []
    for case in report['cases']:
        if case['source'] != args.source:
            continue
        name = case['name']; p = parameters(settings | {'keys': case['keys']})
        plan_path = args.run/(name+'-plan.json')
        selection_path = args.run/(name+'-selection.json')
        notes_path = args.run/(name+'-notes.json')
        evidence, _, _ = expression_evidence(analysis, read(plan_path), p)
        aligned, _ = align_candidates(chart_clock_notes(evidence, clock['audio_shift_ms']), clock, confirmed=True)
        by_id = {n.id: n for n in aligned}
        selected_data = read(notes_path)
        selected = [by_id[n['id']] for n in selected_data]
        for n in selected_data:
            original = by_id[n['id']]
            assert (original.start, original.end, original.voice) == (n['start'], n['end'], n['voice'])
        decisions = read(selection_path)['decisions']; events = []
        for decision in decisions:
            if decision['voice'] != 'drums' or decision['reason'] != 'group_capacity_or_density':
                continue
            note = by_id[decision['id']]
            events.append(dict(id=note.id, start=note.start, section=decision['section'], group=decision['group'],
                attack_shift_ms=abs(note.start-note.evidence_start), **tap_pressure(note, selected, p)))
        row = dict(name=name, keys=case['keys'], selected_drums=sum(n.voice=='drums' for n in selected),
            omitted_drums=len(events), individually_feasible=sum(not e['constraints'] for e in events),
            constraints=dict(Counter(c for e in events for c in e['constraints'])),
            feasible_within_30ms=sum(not e['constraints'] and e['attack_shift_ms']<=30 for e in events),
            input_hashes={p.name: sha256(p) for p in (plan_path, selection_path, notes_path)})
        rows.append(row); details.append(dict(name=name, events=events))
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    summary = dict(source=args.source, runner_sha256=sha256(__file__),
        analysis_sha256=sha256(args.analysis), run_report_sha256=sha256(args.run/'report.json'),
        baseline_report_sha256=sha256(args.baseline_report), runtime={
            name:sha256(name) for name in ('manialab/expression.py','manialab/expression_articulation.py','manialab/timing.py')},
        generation_clock=clock, settings=settings, cases=rows,
        charts_modified=False, reference_chart_used=False, musical_acceptance=False,
        notice='Individual additive feasibility only. Feasible attacks may compete with each other; neither feasibility nor counts prove musical value.')
    (args.output/'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.output/'events.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
