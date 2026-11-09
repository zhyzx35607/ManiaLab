"""Audit saved factor predictions and summarize interactions without inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256
from scripts.summarize_piano_stream_context import assert_close, f1, read, verify

VARIANTS = ('full_global', 'cut_global', 'full_window', 'cut_window')


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key != 'observed_edge_audit_json'}


def edge_keys(edges):
    return [tuple(map(int, edge)) for edge in edges.T]


def aligned_heads(reference, other):
    result = {'staff': np.asarray(other['staff_scores'], dtype=np.float64)}
    if result['staff'].shape != reference['staff_scores'].shape:
        raise ValueError('Staff head shape differs')
    for kind in ('voice', 'chord'):
        reference_keys = edge_keys(reference['potential_'+kind])
        keys = edge_keys(other['potential_'+kind])
        values = np.asarray(other[kind+'_logits'], dtype=np.float64)
        if len(set(keys)) != len(keys) or len(set(reference_keys)) != len(reference_keys):
            raise ValueError('Duplicate candidate edge keys')
        if set(keys) != set(reference_keys) or len(keys) != len(values):
            raise ValueError('Candidate edge sets or logits differ in size')
        mapping = dict(zip(keys, values))
        result[kind] = np.asarray([mapping[key] for key in reference_keys], dtype=np.float64)
    for values in result.values():
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite prediction head')
    return result


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(values=int(values.size), max_abs=float(np.abs(values).max()) if values.size else 0.,
        mean_abs=float(np.abs(values).mean()) if values.size else 0.,
        mean_signed=float(values.mean()) if values.size else 0.)


def component_difference(a, b, measures):
    changed = 0
    for measure in np.unique(measures):
        indices = np.flatnonzero(measures == measure)
        left, right = a['components'][indices], b['components'][indices]
        mask = np.triu(np.ones((len(indices), len(indices)), dtype=bool), 1)
        changed += int(((left[:, None] == left[None, :]) !=
                        (right[:, None] == right[None, :]))[mask].sum())
    return changed


def summarize_case(folder, rest, prefix, measures, supplement):
    predictions = {variant: load(folder/f'{prefix}-{variant}.npz') for variant in VARIANTS}
    reference = predictions['full_global']
    baseline = load(rest/f'{prefix}-full.npz')
    if set(reference) != set(baseline):
        raise ValueError('Full prediction schema differs from rest baseline')
    for key in reference:
        np.testing.assert_array_equal(reference[key], baseline[key])
    heads = {variant: aligned_heads(reference, pred) for variant, pred in predictions.items()}
    interactions = {kind: heads['full_global'][kind] - heads['cut_global'][kind] -
                    heads['full_window'][kind] + heads['cut_window'][kind]
                    for kind in ('voice', 'chord', 'staff')}
    # Only this ignored supplement contains values per candidate edge/node.
    np.savez_compressed(supplement/f'{prefix}-interaction.npz',
        potential_voice=reference['potential_voice'], potential_chord=reference['potential_chord'],
        **{kind+'_interaction': value for kind, value in interactions.items()})
    actual = load(rest/f'{prefix}-four_measures.npz')
    actual_heads = aligned_heads(reference, actual)
    residual = {kind: describe(heads['cut_window'][kind] - actual_heads[kind])
                for kind in interactions}
    close = {kind: bool(np.allclose(heads['cut_window'][kind], actual_heads[kind], atol=1e-4, rtol=1e-5))
             for kind in interactions}
    residual['staff_label_flips'] = int((predictions['cut_window']['staff'] != actual['staff']).sum())
    residual['component_pair_disagreements'] = component_difference(predictions['cut_window'], actual, measures)
    for kind in ('voice', 'chord'):
        residual[kind]['threshold_flips'] = int(((heads['cut_window'][kind] > 0) != (actual_heads[kind] > 0)).sum())
        residual[kind]['predicted_edges_changed'] = len(set(edge_keys(predictions['cut_window'][kind+'_edges'])) ^
                                                       set(edge_keys(actual[kind+'_edges'])))
    return dict(interaction={kind: describe(value) for kind, value in interactions.items()},
                actual_window_residual=residual, heads_close_to_actual_windows=close)


def expression_summary(folder, model, supplement):
    summaries, private = {}, {}
    for mode in ('staff', 'components'):
        policies = {variant: read(folder/f'{model}-sakura-{variant}-{mode}.json')
                    for variant in (*VARIANTS, 'actual_windows')}
        keys = sorted(policies['full_global'])
        if not all(set(p) == set(keys) for p in policies.values()):
            raise ValueError('Expression candidate identities differ')
        holds = {variant: np.asarray([int(policies[variant][key]['hold']) for key in keys])
                 for variant in policies}
        baseline = holds['full_global']
        changed = {variant: {key for key, a, b in zip(keys, baseline, values) if a != b}
                   for variant, values in holds.items()}
        directions = {}
        for variant, values in holds.items():
            directions[variant] = dict(before_holds=int(baseline.sum()), after_holds=int(values.sum()),
                hold_to_tap=int(((baseline == 1) & (values == 0)).sum()),
                tap_to_hold=int(((baseline == 0) & (values == 1)).sum()), flips=len(changed[variant]))
        edge, norm, both = (changed[v] for v in ('cut_global', 'full_window', 'cut_window'))
        groups = dict(edge=edge, normalization=norm, combined=both,
            edge_and_normalization=edge & norm, edge_or_normalization=edge | norm,
            normalization_and_combined=norm & both, normalization_only_vs_combined=norm - both,
            combined_only_vs_normalization=both - norm, normalization_combined_symmetric_difference=norm ^ both,
            edge_and_combined=edge & both, emergent_combined=both - (edge | norm),
            single_effects_absent_when_combined=(edge | norm) - both,
            combined_differs_from_xor=both ^ (edge ^ norm))
        interaction = holds['full_global'] - holds['cut_global'] - holds['full_window'] + holds['cut_window']
        interaction_histogram = {str(int(k)): int(v) for k, v in Counter(interaction).items()}
        residual = int((holds['actual_windows'] != holds['cut_window']).sum())
        summaries[mode] = dict(candidates=len(keys), directions=directions,
            flip_set_counts={name: len(ids) for name, ids in groups.items()},
            binary_hold_interaction=describe(interaction), interaction_histogram=interaction_histogram,
            window_expression_residual=residual)
        private[mode] = dict(flip_sets={name: sorted(ids) for name, ids in groups.items()},
            directions={variant: {direction: [key for key, a, b in zip(keys, baseline, values)
                if a == old and b != old] for direction, old in (('hold_to_tap', 1), ('tap_to_hold', 0))}
                for variant, values in holds.items()},
            hold_interaction={key: int(value) for key, value in zip(keys, interaction) if value})
    (supplement/f'{model}-expression-interaction-ids.json').write_text(
        json.dumps(private, ensure_ascii=False, indent=2), encoding='utf-8')
    return summaries


def aggregate_interactions(cases):
    result = {}
    for model in ('original', 'adapted'):
        result[model] = {}
        for kind in ('clean', 'whole'):
            subset = [row for row in cases if row['model'] == model and row['input'] == kind]
            result[model][kind] = {}
            for head in ('voice', 'chord', 'staff'):
                stats = [row['interaction'][head] for row in subset]
                count = sum(row['values'] for row in stats)
                result[model][kind][head] = dict(values=count,
                    max_abs=max(row['max_abs'] for row in stats),
                    mean_abs=sum(row['mean_abs'] * row['values'] for row in stats) / max(1, count),
                    mean_signed=sum(row['mean_signed'] * row['values'] for row in stats) / max(1, count))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT/'test-data/piano-svsep')
    cli.add_argument('--analysis', type=Path, default=ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json')
    cli.add_argument('--supplement', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists() or args.supplement.exists():
        raise ValueError('Use new supplement and public output paths')
    root, folder, rest_folder = args.root, args.root/'context-factors-v1', args.root/'rest-v1'
    report, rest = read(folder/'report.json'), read(rest_folder/'report.json')
    for data, location, protocol in ((report, folder, 'piano-stream-context-factors-protocol.md'),
                                     (rest, rest_folder, 'piano-stream-rest-protocol.md')):
        verify(location/'runner-snapshot.py', data['identity']['runner_sha256'])
        verify(ROOT/'docs'/protocol, data['identity']['protocol_sha256'])
        for name, digest in data['identity']['snapshots'].items():
            verify(location/f'{name}-snapshot.py', digest)
        for name, digest in data['artifacts_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Artifact escaped its output folder')
            verify(location/name, digest)
        for flag in ('test_data_used', 'model_selection_changed', 'enabled_in_application', 'musical_acceptance'):
            if data[flag] is not False:
                raise ValueError('Unexpected experiment flag')
        if not data['weights_unchanged'] or not data['raw_candidates_unchanged']:
            raise ValueError('Fixed weights/inputs were not established')
    verify(rest_folder/'report.json', report['identity']['baseline_sha256'])
    for field in ('source_revision', 'dataset_sha256', 'selected_sha256', 'analysis_sha256', 'graph_rows_sha256'):
        if report['identity'][field] != rest['identity'][field]:
            raise ValueError('Factor/rest input identity mismatch')
    for name, digest in rest['identity']['snapshots'].items():
        if report['identity']['snapshots'][name] != digest:
            raise ValueError('Shared helper snapshot differs from rest experiment')
    verify(root/'adaptation-v1/selected.pt', report['identity']['selected_sha256'])
    trained = read(root/'adaptation-v1/report.json')
    selection = read(root/'adaptation-v1/selection.json')
    if trained['selected_sha256'] != selection['sha256'] or selection['sha256'] != report['identity']['selected_sha256']:
        raise ValueError('Training selection differs')
    verify(root/'source/pretrained_models/model.ckpt', trained['identity']['model_sha256'])
    verify(args.analysis, report['identity']['analysis_sha256'])
    verify(root/'probe-v3/sakura-graph-input.json', report['identity']['graph_rows_sha256'])
    verify(root/'dataset-v2/manifest.json', report['identity']['dataset_sha256'])
    revision = subprocess.check_output(['git', '-C', str(root/'source'), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(root/'source'), 'diff', 'HEAD', '--name-only'], text=True).strip()
    if changes or revision != report['identity']['source_revision']:
        raise ValueError('Upstream source identity differs')
    manifest = read(root/'dataset-v2/manifest.json')
    measures = {}
    for row in manifest['cases']:
        if row['status'] != 'ready' or row['split'] != 'validation':
            continue
        path = root/'dataset-v2'/row['cache']
        verify(path, row['cache_sha256'])
        with np.load(path, allow_pickle=False) as data:
            measures[row['name']] = data['measures']
    if len(measures) != 16 or len(report['cases']) != 64:
        raise ValueError('Incomplete validation factor experiment')
    args.supplement.mkdir(parents=True)
    shutil.copyfile(__file__, args.supplement/'runner-snapshot.py')
    cases = []
    for case in report['cases']:
        prefix = f'{case["model"]}-{case["name"]}-{case["input"]}'
        summary = summarize_case(folder, rest_folder, prefix, measures[case['name']], args.supplement)
        if summary['heads_close_to_actual_windows'] != case['audit']['heads_close_to_actual_windows']:
            raise ValueError('Head residual tolerance result differs')
        if summary['actual_window_residual']['component_pair_disagreements'] != case['audit']['actual_window_residual']['component_pair_disagreements']:
            raise ValueError('Component residual differs')
        cases.append(dict(model=case['model'], name=case['name'], input=case['input'],
            scores=case['scores'], comparisons=case['audit']['comparisons'], **summary))
    for model in ('original', 'adapted'):
        for kind in ('clean', 'whole'):
            subset = [c for c in cases if c['model'] == model and c['input'] == kind]
            if len(subset) != 16:
                raise ValueError('Missing per-model validation pieces')
            for variant in VARIANTS:
                counts = Counter(tp=0, fp=0, fn=0)
                for case in subset:
                    counts.update(case['scores'][variant]['counts'])
                saved = report['validation'][model][kind][variant]
                if dict(counts) != saved['counts']:
                    raise ValueError('Factor aggregate count differs')
                assert_close(f1(counts), saved['f1'])
                assert_close(sum(c['scores'][variant]['f1'] for c in subset)/16, saved['piece_macro_f1'])
    rows = read(root/'probe-v3/sakura-graph-input.json')
    sakura_measures = np.asarray([r['tick']//1920 for r in rows])
    sakura = []
    for case in report['sakura']:
        model = case['model']
        summary = summarize_case(folder, rest_folder, model+'-sakura', sakura_measures, args.supplement)
        expressions = expression_summary(folder, model, args.supplement)
        for mode, stats in expressions.items():
            for variant in VARIANTS:
                before = case['expression_changes'][variant][mode]
                after = stats['directions'][variant]
                for field in ('before_holds', 'after_holds'):
                    if before[field] != after[field]:
                        raise ValueError('Expression hold counts differ')
                if before['hold_tap_flips'] != after['flips']:
                    raise ValueError('Expression flips differ')
            if stats['window_expression_residual'] != case['window_expression_residual'][mode]['hold_tap_flips']:
                raise ValueError('Expression residual differs')
        sakura.append(dict(model=model, comparisons=case['audit']['comparisons'],
            expressions=expressions, **summary))
    local_identity = dict(runner_sha256=sha256(__file__), factors_report_sha256=sha256(folder/'report.json'),
        rest_report_sha256=sha256(rest_folder/'report.json'), computed_after_inference=True,
        inference_run=False, tuning_performed=False,
        formula='full_global - cut_global - full_window + cut_window',
        artifacts_sha256={p.name: sha256(p) for p in args.supplement.iterdir()})
    (args.supplement/'identity.json').write_text(json.dumps(local_identity, indent=2), encoding='utf-8')
    residuals = [c for c in (*cases, *sakura)]
    public = dict(date='2026-11-09', runner_sha256=sha256(__file__), factors_identity=report['identity'],
        factors_report_sha256=sha256(folder/'report.json'), rest_report_sha256=sha256(rest_folder/'report.json'),
        supplement_identity_sha256=sha256(args.supplement/'identity.json'),
        verification=dict(factors_artifacts=len(report['artifacts_sha256']), rest_artifacts=len(rest['artifacts_sha256']),
            helper_snapshots=True, executed_snapshots=True, validation_caches=len(measures),
            full_baseline_reproduced=True, all_heads_close=all(all(c['heads_close_to_actual_windows'].values()) for c in residuals),
            all_pair_residuals_zero=all(c['actual_window_residual']['component_pair_disagreements'] == 0 for c in residuals),
            inference_run=False, test_arrays_read=False),
        supplement=dict(computed_after_inference=True,tuning_performed=False,formula=local_identity['formula'],
            head_interaction_is_continuous_difference_of_differences=True,
            expression_interaction_is_binary_hold_difference_of_differences=True),
        validation=report['validation'], interaction_totals=aggregate_interactions(cases),
        cases=cases,sakura=sakura,elapsed_inference_seconds=report['elapsed_seconds'],
        enabled_in_application=False,musical_acceptance=False,
        limitations=['The fixed validation/sample comparison identifies context sensitivity, not musical correctness.',
            'GraphNorm and observed-edge changes interact; flip counts cannot be added or identified by cardinality alone.',
            'Interaction values and flip intersections were computed after inference from frozen outputs without tuning.',
            'Tolerance equality of raw heads does not imply exact floating-point equality; final relationships were checked separately.',
            'No audio, symbolic events, expression IDs or research weights are distributed.'])
    args.output.write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),sha256=sha256(args.output),cases=len(cases),
        all_heads_close=public['verification']['all_heads_close'],
        all_pair_residuals_zero=public['verification']['all_pair_residuals_zero'])),flush=True)


if __name__ == '__main__':
    main()
