"""Freeze and evaluate song-group-separated 6–10K endpoint comparisons.

The inputs are authored onsets stripped of tails and columns. These comparisons
do not measure acoustic transcription, nor use the target holds as candidates.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.anchor_gestures import AnchorContext, plan, probabilities
from manialab.domain import parse_osu, validate
from scripts.evaluate_anchor_gestures import candidate_ceiling, describe, greedy_lanes, nearest_baseline, head_scoped_comparison
from scripts.probe_mapper_adapter import sha, timing_comparison
from scripts.train_layout_network import records_and_groups
from scripts.train_pattern_prior import split_for


def select(args):
    if args.output.exists(): raise ValueError('Do not overwrite a frozen selection')
    records, excluded = records_and_groups(args.corpus)
    records = [r for r in records if r.get('reference_role') == 'expression_and_layout' and r['report']['ln'] >= 20]
    selected = []; used = set()
    for split in ('validation', 'test'):
        for keys in range(6, 11):
            local = [r for r in records if r['keys'] == keys and split_for(r['group']) == split]
            for preference in ('ln_majority', 'hybrid'):
                options = [r for r in local if r['group'] not in used]
                options.sort(key=lambda r: (r['ln_class'] != preference, r['sha256']))
                for row in options:
                    path = args.corpus/'maps'/f"{row['beatmap_id']}.osu"
                    parsed = parse_osu(path.read_text(encoding='utf-8-sig'))
                    try:
                        AnchorContext(parsed['notes'], keys, parsed['timing'])
                    except ValueError:
                        continue
                    start = max(0, (min(n.start for n in parsed['notes'])+max(n.start for n in parsed['notes'])-32000)//2)
                    selected.append(dict(id=row['beatmap_id'], sha256=row['sha256'], group=row['group'],
                        keys=keys, split=split, ln_class=row['ln_class'], start_ms=start, end_ms=start+32000))
                    used.add(row['group']); break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    value = dict(rows=selected, corpus=str(args.corpus), selection_script_sha256=sha(__file__),
        notice='Two distinct available song groups per key/split, preferring LN-majority then hybrid; fixed central 32s; known author onsets and LN condition.')
    args.output.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value), flush=True)


def evaluate(args):
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    model = json.loads(args.model.read_text(encoding='utf-8'))
    from manialab.release_candidates import ReleaseContext
    context_type = ReleaseContext if model['revision'].startswith('independent-release-') else AnchorContext
    if context_type is ReleaseContext and model['release_feature_sha256'] != sha('manialab/release_candidates.py'):
        raise ValueError('Independent release context changed since fitting')
    def solve(context, predictions):
        if args.planner == 'histogram': return plan(context, predictions, coordination=0.)
        from manialab.gesture_flow import plan_flow
        return plan_flow(context, predictions, hierarchical=args.planner == 'flow-hierarchical')
    if args.output.exists(): raise ValueError('Preserve previous comparisons')
    args.output.mkdir(parents=True); results = []
    corpus = Path(selection['corpus'])
    for row in selection['rows']:
        if row['split'] != args.split: continue
        path = corpus/'maps'/f"{row['id']}.osu"
        if sha(path) != row['sha256']: raise ValueError('Frozen comparison chart changed')
        chart = parse_osu(path.read_text(encoding='utf-8-sig'))
        ratio = sum(n.hold for n in chart['notes'])/len(chart['notes'])
        # No source labels or finished columns enter the candidate features.
        notes = [replace(n, end=n.start, lane=0, voice='mix', raw='') for n in chart['notes']]
        full = context_type(notes, row['keys'], chart['timing'], ratio)
        lo = max(0, int(full.times.searchsorted(row['start_ms']))-1)
        hi = min(len(full.times)-1, int(full.times.searchsorted(row['end_ms'])))
        history_beats = 24 if context_type is ReleaseContext else 12
        first = max(0, int(full.beats.searchsorted(full.beats[lo]-history_beats))-4)
        last = min(len(full.times)-1, int(full.beats.searchsorted(full.beats[hi]+24))+4)
        context = context_type([n for n in notes if full.times[first] <= n.start <= full.times[last]],
            row['keys'], chart['timing'], ratio)
        predicted, solved = solve(context, probabilities(context, model))
        base, base_solve = solve(context, nearest_baseline(context))
        output = args.output/str(row['id']); output.mkdir()
        generated = greedy_lanes(predicted, row['keys'])
        (output/'events.json').write_text(json.dumps([n.data() for n in generated]), encoding='utf-8')
        # Labels were parsed to obtain declared LN condition, but do not enter
        # any endpoint prediction. Persist those predictions before scoring.
        result = dict(**row, model_sha256=sha(args.model), events_sha256=sha(output/'events.json'),
            scores={name:timing_comparison(chart['notes'], values, row['start_ms'], row['end_ms'])
                for name, values in [('learned', predicted), ('next_anchor', base)]},
            head_scoped={name:head_scoped_comparison(chart['notes'], values, row['start_ms'], row['end_ms'])
                for name, values in [('learned', predicted), ('next_anchor', base)]},
            candidate_coverage=candidate_ceiling(context, chart['notes'], row['start_ms'], row['end_ms']),
            predicted=describe([n for n in predicted if row['start_ms'] <= n.start < row['end_ms']]),
            context_history_beats=history_beats,
            validation=validate(generated, row['keys']), solver=solved,
            attacks_preserved=Counter(n.start for n in predicted)==Counter(n.start for group in context.notes for n in group))
        results.append(result)
        (output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(dict(id=row['id'], keys=row['keys'], split=row['split'],
            f1={k:v['complete_hold_spans']['f1'] for k, v in result['scores'].items()})), flush=True)
    report = dict(model_sha256=sha(args.model), selection_sha256=sha(args.selection),
        planner=args.planner, flow_sha256=sha('manialab/gesture_flow.py') if args.planner != 'histogram' else None,
        feature_sha256=sha('manialab/anchor_gestures.py'), runner_sha256=sha(__file__), rows=results,
        ln_bias=0., coordination=0., enabled_in_application=False,
        notice='Conditional expression from authored attacks, no audio recognition; existing development corpus, not untouched whole-project blind testing.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    selection = sub.add_parser('select'); selection.add_argument('--corpus', type=Path, default=Path('test-data/corpus-expanded/snapshot-verified'))
    selection.add_argument('--output', type=Path, required=True)
    evaluation = sub.add_parser('evaluate'); evaluation.add_argument('--selection', type=Path, required=True)
    evaluation.add_argument('--model', type=Path, required=True); evaluation.add_argument('--output', type=Path, required=True)
    evaluation.add_argument('--split', choices=('validation', 'test'), required=True)
    evaluation.add_argument('--planner', choices=('histogram', 'flow', 'flow-hierarchical'), default='histogram')
    args = parser.parse_args(); (select if args.command == 'select' else evaluate)(args)


if __name__ == '__main__': main()
