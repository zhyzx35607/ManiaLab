"""Separate observed-edge context from GraphNorm context at fixed weights."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch_geometric.nn import GraphNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.adapt_piano_streams import augment, f1, load_model, make_graph, verify_source
from scripts.build_provenance import sha256
from scripts.check_piano_stream_context import full_items, read, windows
from scripts.check_piano_stream_rest import expression_delta, prediction_delta, repair_rest, verify_repair
from scripts.evaluate_piano_stream_adaptation import component_pair_counts, graph_for
from scripts.research_piano_streams import expression_choices, predict
from manialab.domain import note_from

VARIANTS = ('full_global', 'cut_global', 'full_window', 'cut_window')
OBSERVED = ('onset', 'consecutive', 'during', 'rest')


def window_batch(measures):
    batch = torch.full((len(measures),), -1, dtype=torch.long)
    for i, indices in enumerate(windows(measures)):
        assert np.all(np.diff(indices) == 1)
        assert torch.all(batch[indices] == -1)
        batch[indices] = i
    assert torch.all(batch >= 0)
    return batch


def cut_observed(graph, batch):
    result = graph.clone()
    counts = {}
    for kind in OBSERVED:
        edges = graph['note', kind, 'note'].edge_index
        keep = batch[edges[0]] == batch[edges[1]]
        result['note', kind, 'note'].edge_index = edges[:, keep].clone()
        counts[kind] = int((~keep).sum())
    for kind in ('potential', 'chord_potential'):
        edges = graph['note', kind, 'note'].edge_index
        assert torch.all(batch[edges[0]] == batch[edges[1]])
        assert torch.equal(edges, result['note', kind, 'note'].edge_index)
    for key, value in graph['note'].items():
        assert torch.equal(value, result['note'][key])
    return result, counts


def norm_prediction(model, graph, batch=None):
    modules = {name: module for name, module in model.named_modules() if isinstance(module, GraphNorm)}
    assert modules, 'Expected GraphNorm modules'
    calls = Counter()
    handles = []
    if batch is not None:
        def hook(name):
            def before(module, args, kwargs):
                assert len(args) == 1 and not kwargs
                assert len(args[0]) == len(batch)
                calls[name] += 1
                return (args[0], batch.to(args[0].device)), kwargs
            return before
        handles = [module.register_forward_pre_hook(hook(name), with_kwargs=True)
                   for name, module in modules.items()]
    try:
        result = predict(model, graph)
    finally:
        for handle in handles:
            handle.remove()
    if batch is not None:
        assert set(calls) == set(modules)
    return result, dict(calls)


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key != 'observed_edge_audit_json'}


def aligned_heads(a, b):
    result = {}
    for kind in ('voice', 'chord'):
        values = {tuple(edge): value for edge, value in zip(b['potential_'+kind].T, b[kind+'_logits'])}
        aligned = np.asarray([values[tuple(edge)] for edge in a['potential_'+kind].T])
        result[kind] = bool(np.allclose(a[kind+'_logits'], aligned, atol=1e-4, rtol=1e-5))
    result['staff'] = bool(np.allclose(a['staff_scores'], b['staff_scores'], atol=1e-4, rtol=1e-5))
    return result


def factor_predictions(model, graph, measures, baseline, prefix, output):
    fixed = repair_rest(graph)
    repair_audit = verify_repair(graph, fixed)
    batch = window_batch(measures)
    cut, removed = cut_observed(fixed, batch)
    result, calls = {}, {}
    for variant in VARIANTS:
        result[variant], calls[variant] = norm_prediction(model,
            cut if variant.startswith('cut') else fixed,
            batch if variant.endswith('window') else None)
        np.savez_compressed(output/f'{prefix}-{variant}.npz', **result[variant])
    saved_full = load_predictions(baseline/f'{prefix}-full.npz')
    for key in saved_full:
        np.testing.assert_array_equal(saved_full[key], result['full_global'][key])
    actual_windows = load_predictions(baseline/f'{prefix}-four_measures.npz')
    comparisons = (('full_global', 'cut_global'), ('full_global', 'full_window'),
                   ('cut_global', 'cut_window'), ('full_window', 'cut_window'))
    audit = dict(repair=repair_audit, removed_cross_window=removed, norm_calls=calls,
        full_prediction_reproduced=True,
        comparisons={a+'-to-'+b: prediction_delta(result[a], result[b], measures) for a, b in comparisons},
        actual_window_residual=prediction_delta(actual_windows, result['cut_window'], measures),
        heads_close_to_actual_windows=aligned_heads(result['cut_window'], actual_windows))
    return result, actual_windows, audit


def totals(cases):
    result = {}
    for model in ('original', 'adapted'):
        result[model] = {}
        for kind in ('clean', 'whole'):
            subset = [c for c in cases if c['model'] == model and c['input'] == kind]
            result[model][kind] = {}
            for variant in VARIANTS:
                counts = Counter(tp=0, fp=0, fn=0)
                for case in subset:
                    counts.update(case['scores'][variant]['counts'])
                result[model][kind][variant] = dict(counts=dict(counts), f1=f1(counts),
                    piece_macro_f1=float(np.mean([c['scores'][variant]['f1'] for c in subset])))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--dataset', type=Path, default=ROOT/'test-data/piano-svsep/dataset-v2')
    cli.add_argument('--run', type=Path, default=ROOT/'test-data/piano-svsep/adaptation-v1')
    cli.add_argument('--baseline', type=Path, default=ROOT/'test-data/piano-svsep/rest-v1')
    cli.add_argument('--frozen', type=Path, default=ROOT/'test-data/piano-svsep/probe-v3')
    cli.add_argument('--analysis', type=Path, default=ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior factor experiments')
    revision = verify_source(args.source)
    sys.path.insert(0, str(args.source.resolve()))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    baseline = read(args.baseline/'report.json')
    assert sha256(args.baseline/'runner-snapshot.py') == baseline['identity']['runner_sha256']
    for field, path in (('dataset', args.dataset/'manifest.json'), ('selected', args.run/'selected.pt'),
            ('analysis', args.analysis), ('graph_rows', args.frozen/'sakura-graph-input.json')):
        assert sha256(path) == baseline['identity'][field+'_sha256']
    for name, digest in baseline['artifacts_sha256'].items():
        assert sha256(args.baseline/name) == digest
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    save = lambda name, value: (args.output/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    snapshots = {}
    for name in ('adapt_piano_streams', 'check_piano_stream_context', 'check_piano_stream_rest',
                 'evaluate_piano_stream_adaptation', 'research_piano_streams'):
        path = ROOT/'scripts'/f'{name}.py'
        snapshots[name] = sha256(path)
        shutil.copyfile(path, args.output/f'{name}-snapshot.py')
    identity = dict(source_revision=revision, baseline_sha256=sha256(args.baseline/'report.json'),
        runner_sha256=sha256(__file__), protocol_sha256=sha256(ROOT/'docs/piano-stream-context-factors-protocol.md'),
        snapshots=snapshots, **{key: value for key, value in baseline['identity'].items()
                              if key in ('dataset_sha256', 'selected_sha256', 'analysis_sha256', 'graph_rows_sha256')})
    save('identity.json', identity)
    model = load_model(args.source/'pretrained_models/model.ckpt').eval()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    adapted = torch.load(args.run/'selected.pt', map_location='cpu', weights_only=True)['state_dict']
    items = full_items(args.dataset, read(args.dataset/'manifest.json'))
    analysis = read(args.analysis)
    notes = [note_from(n) for n in analysis['candidates'] if n['voice'] == 'piano']
    before_notes = [n.data() for n in notes]
    rows = read(args.frozen/'sakura-graph-input.json')
    measures = np.asarray([row['tick']//1920 for row in rows])
    cases, sakura = [], []
    started = time.monotonic()
    for name, weights, expanded in (('original', original, False), ('adapted', adapted, True)):
        model.load_state_dict(weights, strict=True)
        model.eval()
        for item in items:
            before = item['notes'].copy()
            for kind in ('clean', 'whole'):
                frozen = dict(item, notes=augment(before, item['bounds'], kind))
                graph = make_graph(frozen, 'clean', expanded)[0]
                predictions, actual, audit = factor_predictions(model, graph, item['measures'], args.baseline,
                    f'{name}-{item["name"]}-{kind}', args.output)
                scores = {}
                for variant, pred in predictions.items():
                    counts = component_pair_counts(item, pred['components'])
                    scores[variant] = dict(counts=dict(counts), f1=f1(counts),
                        staff_accuracy=float(np.mean(pred['staff'] == before['staff']-1)))
                cases.append(dict(model=name, name=item['name'], input=kind, scores=scores, audit=audit))
                save('validation-progress.json', cases)
            assert np.array_equal(before, item['notes'])
            print(json.dumps(dict(model=name, completed=item['name'], cases=len(cases))), flush=True)
        predictions, actual, audit = factor_predictions(model, graph_for(rows, expanded), measures,
            args.baseline, name+'-sakura', args.output)
        policies = {}
        for variant, pred in dict(predictions, actual_windows=actual).items():
            assignments = {row['id']: dict(staff=int(pred['staff'][i]), components=int(pred['components'][i]))
                           for i, row in enumerate(rows)}
            policies[variant] = {mode: expression_choices(notes, analysis['structure']['segments'], assignments, mode)
                                 for mode in ('staff', 'components')}
            for mode, choices in policies[variant].items():
                save(f'{name}-sakura-{variant}-{mode}.json', choices)
        sakura.append(dict(model=name, audit=audit,
            expression_changes={variant: {mode: expression_delta(policies['full_global'][mode], policies[variant][mode])
                for mode in ('staff', 'components')} for variant in VARIANTS},
            window_expression_residual={mode: expression_delta(policies['actual_windows'][mode], policies['cut_window'][mode])
                for mode in ('staff', 'components')}))
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    assert [n.data() for n in notes] == before_notes
    report = dict(identity=identity, cases=cases, validation=totals(cases), sakura=sakura,
        elapsed_seconds=time.monotonic()-started, test_data_used=False, model_selection_changed=False,
        weights_unchanged=True, raw_candidates_unchanged=True, enabled_in_application=False, musical_acceptance=False)
    report['artifacts_sha256'] = {path.name: sha256(path) for path in args.output.iterdir()
        if path.suffix in ('.json', '.npz') and path.name not in ('validation-progress.json', 'report.json')}
    save('report.json', report)
    print(json.dumps(report['validation']), flush=True)


if __name__ == '__main__':
    main()
