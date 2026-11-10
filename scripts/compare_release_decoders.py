"""Freeze equal-score decoder comparisons before reading author LN targets."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.anchor_gestures import probabilities
from manialab.gesture_flow import plan_flow
from manialab.release_candidates import ReleaseContext
from manialab.domain import parse_osu
from scripts.train_anchor_gestures import source_chart
from scripts.probe_mapper_adapter import sha
from scripts.evaluate_anchor_gestures import head_scoped_comparison, greedy_lanes


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--model', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--ln-bias', type=float, default=.5)
    args = cli.parse_args()
    if args.output.exists(): raise ValueError('Preserve earlier decisions')
    root = Path('test-data/sakura-benchmark')
    source = root/'detected-onset-sketch-v1/sketch.osu'
    chart = source_chart(source); model = json.loads(args.model.read_text(encoding='utf-8'))
    if model['release_feature_sha256'] != sha('manialab/release_candidates.py'):
        raise ValueError('Release context changed')
    context = ReleaseContext(chart['notes'], 10, chart['timing'], 961/2806)
    prediction = probabilities(context, model, args.ln_bias)
    args.output.mkdir(parents=True)
    result = dict(model_sha256=sha(args.model), source_sha256=sha(source), ln_bias=args.ln_bias,
                  flow_sha256=sha('manialab/gesture_flow.py'), runner_sha256=sha(__file__), runs={})
    generated = {}
    for name, hierarchy in [('flat', False), ('hierarchical', True)]:
        notes, report = plan_flow(context, prediction, hierarchical=hierarchy)
        generated[name] = greedy_lanes(notes, 10)
        path = args.output/(name+'-events.json')
        path.write_text(json.dumps([n.data() for n in generated[name]]), encoding='utf-8')
        result['runs'][name] = dict(planner=report, events_sha256=sha(path))
    (args.output/'frozen-decisions.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    human = parse_osu((root/'finished.osu').read_text(encoding='utf-8-sig'))['notes']
    for name, notes in generated.items():
        result['runs'][name]['later'] = head_scoped_comparison(human, notes, 188286, 283840)
    result['notice'] = 'Same frozen model probabilities; known Timing and detected attacks. Development diagnostic only.'
    (args.output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
