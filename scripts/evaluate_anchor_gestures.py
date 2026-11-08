"""Calibrate an anchor planner, freeze its decisions, then score later sections."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.anchor_gestures import AnchorContext, plan, probabilities, arrange_gestures
from manialab.arrange import assign_lanes, parameters
from manialab.domain import parse_osu, export_osu, validate
from scripts.probe_mapper_adapter import sha, timing_comparison, write_review, event_similarity
from scripts.train_anchor_gestures import source_chart, target_groups


def describe(notes):
    holds = [n for n in notes if n.hold]
    tails = Counter(n.end for n in holds)
    return dict(objects=len(notes), holds=len(holds),
        duration_quantiles_ms=np.quantile([n.end-n.start for n in holds], [.1, .5, .9]).tolist() if holds else [],
        longest_hold_ms=max((n.end-n.start for n in holds), default=0),
        holds_at_least_1000ms=sum(n.end-n.start >= 1000 for n in holds),
        holds_at_least_2000ms=sum(n.end-n.start >= 2000 for n in holds),
        shared_release_objects=sum(v for v in tails.values() if v >= 2),
        distinct_releases=len(tails))


def long_comparison(reference, predicted, start, end):
    spans = lambda notes: [(n.start, n.end) for n in notes
        if n.end-n.start >= 2000 and start <= n.start and n.end < end]
    return event_similarity(spans(reference), spans(predicted), [30, 60])


def candidate_ceiling(context, human, start, end):
    truth = [n for n in human if n.hold and start <= n.start and n.end < end]
    grouped = target_groups(context, truth)
    heads = tails = capacity = 0
    for i, local in grouped.items():
        heads += len(local)
        candidates = context.endpoint_times(i, context.options(i))[1:-1]
        count = sum(bool(len(candidates)) and np.min(np.abs(candidates-n.end)) <= 60 for n in local)
        tails += int(count); capacity += min(len(context.notes[i]), int(count))
    return dict(reference_holds=len(truth), supported_heads=heads, both_endpoints_available=tails,
        row_capacity_upper_bound=capacity,
        notice='Necessary candidate conditions only; not a quality score or simultaneous-occupancy proof.')


def nearest_baseline(context):
    predictions = []
    for i in range(len(context.times)):
        options = context.options(i); p = np.zeros(len(options))
        future = context.times[i+1:]
        candidates = context.endpoint_times(i, options)[1:-1]
        shared = [j+1 for j,t in enumerate(candidates) if t in future]
        if shared:
            p[0] = 1-context.ln_condition; p[shared[0]] = context.ln_condition
        else: p[0] = 1
        predictions.append(dict(options=options, probabilities=p))
    return predictions


def head_scoped_comparison(reference, predicted, start, end):
    spans = lambda notes:[(n.start, n.end) for n in notes if n.hold and start <= n.start < end]
    a, b = spans(reference), spans(predicted)
    return dict(complete_holds=event_similarity(a, b, [30, 60]),
        strict_holds=event_similarity(a, b, [10, 20]),
        short_holds=event_similarity([p for p in a if p[1]-p[0] < 100], [p for p in b if p[1]-p[0] < 100], [10, 15]),
        long_holds=event_similarity([p for p in a if p[1]-p[0] >= 2000], [p for p in b if p[1]-p[0] >= 2000], [30, 60]),
        notice='All holds with heads in the interval remain scored, including tails beyond the interval.')


def greedy_lanes(notes, keys):
    """Only for structural feasibility checks; not a performance layout."""
    busy = [-1]*keys; last = [-1]*keys; result = []
    for n in sorted(notes, key=lambda n: (n.start, -n.end, n.id)):
        free = [i for i in range(keys) if busy[i] <= n.start and last[i] < n.start]
        if not free: raise ValueError('Interval capacity was insufficient despite planner validation')
        lane = min(free, key=lambda i: (last[i], i))
        result.append(replace(n, lane=lane, raw=''))
        busy[lane] = n.end; last[lane] = n.start
    if not validate(result, keys)['valid']: raise ValueError('Independent interval coloring failed')
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--model', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--calibration', type=Path, help='Reuse previously frozen parameters, without recalibration')
    cli.add_argument('--review', action='store_true')
    cli.add_argument('--ln-ceiling', action='store_true', help='Cap total holds at the explicitly supplied LN condition; not a minimum quota')
    cli.add_argument('--resume-report', action='store_true', help='Score already frozen decisions after a reporting failure')
    cli.add_argument('--legacy-layout', action='store_true', help='Reproduce the earlier incorrect interpretation of source columns as performance positions')
    cli.add_argument('--planner', choices=('histogram', 'flow', 'flow-hierarchical'), default='histogram')
    args = cli.parse_args()
    if args.planner != 'histogram' and args.ln_ceiling:
        raise ValueError('Flow experiment does not implement a global LN ceiling')
    def solve(context, predictions, coordination=0., **kwargs):
        if args.planner == 'histogram':
            return plan(context, predictions, coordination, **kwargs)
        if coordination: raise ValueError('Flow experiment does not implement release-opening costs')
        from manialab.gesture_flow import plan_flow
        return plan_flow(context, predictions, hierarchical=args.planner == 'flow-hierarchical')
    if args.output.exists() and not args.resume_report: raise ValueError('Preserve earlier generated decisions')
    root = Path('test-data/sakura-benchmark'); args.output.mkdir(parents=True, exist_ok=args.resume_report)
    model = json.loads(args.model.read_text(encoding='utf-8'))
    sketch_path = root/'detected-onset-sketch-v1/sketch.osu'
    evidence = source_chart(sketch_path)
    acoustic_identity = None
    independent = model['revision'].startswith('independent-release-')
    from manialab.release_candidates import ReleaseContext, AcousticReleaseContext
    plain_context = ReleaseContext if independent else AnchorContext
    if independent and model['release_feature_sha256'] != sha('manialab/release_candidates.py'):
        raise ValueError('Independent release context changed since fitting')
    if model.get('acoustic_feature_sha256'):
        from manialab.acoustic_anchors import AcousticAnchorContext
        from scripts.anchor_acoustic_inputs import load_evidence
        if model['acoustic_feature_sha256'] != sha('manialab/acoustic_anchors.py'):
            raise ValueError('Acoustic features changed since fitting')
        acoustic, acoustic_identity = load_evidence(root)
        acoustic_context = AcousticReleaseContext if independent else AcousticAnchorContext
        make_context = lambda notes, ratio: acoustic_context(notes, 10, evidence['timing'], ratio, acoustic)
    else:
        make_context = lambda notes, ratio: plain_context(notes, 10, evidence['timing'], ratio)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if sha(root/'finished.osu') != manifest['charts']['finished']['chart_sha256']:
        raise ValueError('Known author target changed')
    # Whole-chart ratio was supplied to the earlier Mapper diagnostic, too.
    ratio = manifest['charts']['finished']['validation']['ln']/manifest['charts']['finished']['validation']['notes']
    if args.calibration or args.resume_report:
        calibration = json.loads((args.calibration or args.output/'calibration.json').read_text(encoding='utf-8'))
        if calibration.get('planner', 'histogram') != args.planner:
            raise ValueError('Frozen calibration belongs to a different planner')
        if calibration['model_sha256'] != sha(args.model):
            raise ValueError('Frozen calibration belongs to a different model')
        chosen = calibration['chosen']
    else:
        local = [n for n in evidence['notes'] if 162340 <= n.start < 188286]
        context = make_context(local, ratio)
        # Evaluation support is defined from supplied context before labels.
        # Artificial crop edges have insufficient feature context; exclude them.
        full = make_context(evidence['notes'], ratio)
        interior = [int(full.times[i]) for i in range(len(full.times)) if full.contained(i, 162340, 188286)]
        start, end = min(interior), max(interior)+1
        parsed = parse_osu((root/'finished.osu').read_text(encoding='utf-8-sig'))['notes']
        truth = [n for n in parsed if 162340 <= n.start and n.end < 188286]
        del parsed
        rows = []
        for bias in (0., .5, 1., 1.5, 2.):
            predictions = probabilities(context, model, bias)
            for coordination in ((0., .25, .5) if args.planner == 'histogram' else (0.,)):
                notes, solved = solve(context, predictions, coordination)
                comparison = timing_comparison(truth, notes, start, end)
                if independent:
                    comparison['head_scoped'] = head_scoped_comparison(truth, notes, start, end)
                row = dict(ln_bias=bias, coordination=coordination, comparison=comparison, planner=solved)
                rows.append(row)
                print(json.dumps(dict(calibrating=[bias, coordination],
                    f1=comparison['complete_hold_spans']['f1'])), flush=True)
        best = max(rows, key=lambda r: ((r['comparison']['head_scoped']['complete_holds']['f1'] if independent else r['comparison']['complete_hold_spans']['f1']),
            r['comparison']['release_groups']['f1'], -r['ln_bias'], -r['coordination']))
        chosen = {k: best[k] for k in ('ln_bias', 'coordination')}
        calibration = dict(rows=rows, chosen=chosen, bounds=[start, end], planner=args.planner,
            model_sha256=sha(args.model), target_scope='Calibration interval only; comparison labels not used.')
    if not args.resume_report:
        (args.output/'calibration.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')
    context = make_context(evidence['notes'], ratio)
    variants = {}
    if args.resume_report:
        from manialab.domain import note_from
        frozen = json.loads((args.output/'frozen-decisions.json').read_text(encoding='utf-8'))
        if (frozen['model_sha256'] != sha(args.model) or frozen['sketch_sha256'] != sha(sketch_path) or
                frozen['generated_sha256'] != sha(args.output/'learned-events.json') or
                frozen['calibration_sha256'] != sha(args.output/'calibration.json')):
            raise ValueError('Frozen generation changed; refuse recovery')
        for name in ('learned', 'next_anchor', 'all_taps'):
            variants[name] = [note_from(n) for n in json.loads((args.output/(name+'-events.json')).read_text(encoding='utf-8'))]
        origins = {n.id:n for n in evidence['notes']}
        generated = [replace(n, lane=origins[n.id].lane, voice=origins[n.id].voice) for n in variants['learned']]
        if (args.output/'solver.json').exists():
            solvers = json.loads((args.output/'solver.json').read_text(encoding='utf-8'))
            solver, baseline_solver = solvers['learned'], solvers['baseline']
        else:
            solver = baseline_solver = dict(optimal=None, notice='Original solver metadata not saved before reporting failure; recovered frozen events, no rerun.')
    else:
        predictions = probabilities(context, model, chosen['ln_bias'])
        generated, solver = solve(context, predictions, chosen['coordination'], max_ln_ratio=ratio if args.ln_ceiling else None)
        baseline, baseline_solver = solve(context, nearest_baseline(context))
        for name, notes in [('learned', generated), ('next_anchor', baseline), ('all_taps', evidence['notes'])]:
            notes = greedy_lanes(notes, 10)
            variants[name] = notes
            (args.output/(name+'-events.json')).write_text(json.dumps([n.data() for n in notes]), encoding='utf-8')
        frozen = dict(model_sha256=sha(args.model), sketch_sha256=sha(sketch_path), ln_ceiling=ratio if args.ln_ceiling else None,
            acoustic_identity=acoustic_identity,
            layout_mode='legacy_import_positions' if args.legacy_layout else 'source_identity',
            planner=args.planner, flow_sha256=sha('manialab/gesture_flow.py') if args.planner != 'histogram' else None,
            feature_sha256=sha('manialab/anchor_gestures.py'), runner_sha256=sha(__file__),
            calibration_sha256=sha(args.output/'calibration.json'),
            generated_sha256=sha(args.output/'learned-events.json'))
        (args.output/'frozen-decisions.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
        (args.output/'solver.json').write_text(json.dumps(dict(learned=solver, baseline=baseline_solver), indent=2), encoding='utf-8')
    # Later-section target events enter only after model, calibration and
    # whole-song generated endpoint decisions have been saved.
    human = parse_osu((root/'finished.osu').read_text(encoding='utf-8-sig'))['notes']
    ranges = [('whole_song_development', 0, 283840), ('later_sections', 188286, 283840),
              ('probe_215853', 215853, 227853), ('probe_241798', 241798, 253798)]
    scores = {}
    for name, start, end in ranges:
        scores[name] = dict(candidate_coverage=candidate_ceiling(context, human, start, end),
            author=describe([n for n in human if start <= n.start < end]),
            variants={k: dict(comparison=timing_comparison(human, notes, start, end),
                head_scoped=head_scoped_comparison(human, notes, start, end),
                holds_at_least_2000ms=long_comparison(human, notes, start, end),
                summary=describe([n for n in notes if start <= n.start < end])) for k, notes in variants.items()})
    # Torosu-inspired existing layout operates on fixed planned intervals.
    settings = parameters(dict(keys=10, release_gap=0, chord_size=10, max_held=10, ln_ratio=ratio))
    if args.resume_report and (args.output/'planned.osu').exists():
        output = (args.output/'planned.osu').read_text(encoding='utf-8-sig')
        layout = parse_osu(output)['notes']
    else:
        layout = (assign_lanes(generated, settings, timing_points=evidence['timing']) if args.legacy_layout
            else arrange_gestures(generated, 10, evidence['timing'], settings))
        invariant = lambda notes: sorted((n.id, n.start, n.end) for n in notes)
        if invariant(layout) != invariant(generated): raise ValueError('Layout altered the endpoint plan')
        output = export_osu(layout, 10, Path('vendor/mapperatorinator/template.osu'),
            original=sketch_path.read_text(encoding='utf-8-sig'), version='Experimental anchor gestures')
        (args.output/'planned.osu').write_text(output, encoding='utf-8')
    reloaded = parse_osu(output)['notes']
    if sorted((n.start, n.end) for n in reloaded) != sorted((n.start, n.end) for n in generated):
        raise ValueError('Export changed the LN plan')
    report = dict(identity=frozen, reporting_script_sha256=sha(__file__), calibration=chosen, scores=scores, planner=solver,
        baseline_planner=baseline_solver, validation=validate(reloaded, 10),
        input_attack_counts_preserved=Counter(n.start for n in reloaded) == Counter(n.start for n in evidence['notes']),
        enabled_in_application=False, musical_acceptance=False,
        notice='Fixed detected attacks and known author Timing; single-song development. Source attribution inside a chord remains ambiguous.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.review:
        for label, start, end in ranges[2:]:
            selected = lambda notes: dict(notes=[n.data() for n in notes if n.start < end and n.end >= start])
            write_review(args.output/label, root/'mapper-6664-26/accompaniment.wav', start, end,
                dict(reference=selected(evidence['notes']), finished=selected(human), arrangement=selected(layout)),
                labels=dict(reference='检测起音草稿', finished='作者成品', arrangement='实验：起音锚点 LN 规划'),
                sources={v:dict(label='Nova '+v, path=root/f'bundled/{v}/preview.wav') for v in ('drums', 'piano', 'other')},
                notice=('实验固定全部检测起音，独立拍位及其他起音为放手候选；已知作者 Timing，同曲前段训练，未通过音乐验收。'
                    if independent else '实验固定全部检测起音，以其他起音为放手候选，随后使用现有定轨。已知作者 Timing；同曲前段训练，未通过音乐验收。'))
            # Shared review template's old generic sentence would misdescribe
            # this diagnostic, whose inputs were snapped and layout is explicit.
            page = args.output/label/'review.html'
            page.write_text(page.read_text(encoding='utf-8').replace(
                '保留原始事件，未量化、未重新定轨。',
                '固定已对齐的输入起音；实验学习放手端点，再由定轨器分配轨道。'), encoding='utf-8')
    print(json.dumps(dict(chosen=chosen, scores=scores, validation=report['validation'])), flush=True)


if __name__ == '__main__': main()
