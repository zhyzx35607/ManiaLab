"""Verify and publish context/rest aggregates without symbolic or audio events."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'Identity mismatch: {path.name}')
    return actual


def f1(counts):
    denominator = 2 * counts['tp'] + counts['fp'] + counts['fn']
    return 2 * counts['tp'] / denominator if denominator else 1.


def assert_close(left, right):
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'Inconsistent aggregate: {left} != {right}')


def safe_difference(row):
    keys = ('notes', 'staff_label_flips', 'staff_scores_max_abs', 'staff_scores_mean_abs',
            'voice_logits', 'chord_logits', 'predicted_voice_edges_changed',
            'predicted_chord_edges_changed', 'same_measure_pairs', 'component_pair_disagreements')
    result = {key: row[key] for key in keys}
    if 'observed_edges' in row:
        result['observed_edges'] = row['observed_edges']
    return result


def graph_summary(graphs):
    # The first graph is the full input, followed by each local window.
    result = {}
    for name, subset in (('full', graphs[:1]), ('four_measures', graphs[1:])):
        result[name] = dict(graphs=len(subset), notes=sum(g['notes'] for g in subset),
            graphs_with_removed_edges=sum(g['removed'] > 0 for g in subset),
            before=sum(g['before'] for g in subset), after=sum(g['after'] for g in subset),
            removed=sum(g['removed'] for g in subset))
    return result


def validation_summary(cases, context_report, rest_report):
    result = {}
    for model in ('original', 'adapted'):
        result[model] = {}
        for kind in ('clean', 'whole'):
            subset = [r for r in cases if r['model'] == model and r['input'] == kind]
            if len(subset) != 16 or len({r['name'] for r in subset}) != 16:
                raise ValueError('Expected 16 distinct validation pieces per model/input')
            result[model][kind] = {}
            for context in ('full', 'four_measures'):
                row = {}
                for variant in ('raw', 'fixed'):
                    counts = Counter(tp=0, fp=0, fn=0)
                    for case in subset:
                        counts.update(case['scores'][variant][context]['counts'])
                    macro = sum(c['scores'][variant][context]['f1'] for c in subset) / len(subset)
                    notes = sum(c['repair_changes'][context]['notes'] for c in subset)
                    staff = sum(c['scores'][variant][context]['staff_accuracy'] *
                                c['repair_changes'][context]['notes'] for c in subset) / notes
                    row[variant] = dict(counts=dict(counts), f1=f1(counts), piece_macro_f1=macro,
                                        staff_accuracy=staff, notes=notes)
                    saved = rest_report['validation'][model][kind][context][variant]
                    if row[variant]['counts'] != saved['counts']:
                        raise ValueError('Rest aggregate counts differ from piece results')
                    for metric in ('f1', 'piece_macro_f1'):
                        assert_close(row[variant][metric], saved[metric])
                    if variant == 'raw':
                        prior = context_report['validation'][model][kind][context]
                        if counts != prior['counts']:
                            raise ValueError('Raw counts differ from context experiment')
                        assert_close(macro, prior['piece_macro_f1'])
                deltas = [c['scores']['fixed'][context]['f1'] - c['scores']['raw'][context]['f1']
                          for c in subset]
                row['piece_changes'] = dict(better=sum(d > 1e-9 for d in deltas),
                    equal=sum(abs(d) <= 1e-9 for d in deltas), worse=sum(d < -1e-9 for d in deltas))
                if row['piece_changes'] != rest_report['validation'][model][kind][context]['piece_changes']:
                    raise ValueError('Piece change counts differ')
                row['rest_edges_removed'] = sum(c['graphs'][context]['removed'] for c in subset)
                row['graphs_with_removed_edges'] = sum(c['graphs'][context]['graphs_with_removed_edges'] for c in subset)
                result[model][kind][context] = row
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT/'test-data/piano-svsep')
    cli.add_argument('--analysis', type=Path, default=ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previously published reports')
    root = args.root
    context_folder, rest_folder = root/'context-v1', root/'rest-v1'
    context, rest = read(context_folder/'report.json'), read(rest_folder/'report.json')
    identities = {}
    for name, folder, report in (('context', context_folder, context), ('rest', rest_folder, rest)):
        identities[name] = dict(report_sha256=sha256(folder/'report.json'), **report['identity'])
        verify(folder/'runner-snapshot.py', report['identity']['runner_sha256'])
        verify(ROOT/f'docs/piano-stream-{name}-protocol.md', report['identity']['protocol_sha256'])
        for flag in ('test_data_used', 'model_selection_changed', 'enabled_in_application', 'musical_acceptance'):
            if report[flag] is not False:
                raise ValueError(f'Unexpected experiment flag: {name}.{flag}')
        if report['raw_candidates_unchanged'] is not True:
            raise ValueError('Raw candidate preservation was not established')
    verify(context_folder/'report.json', rest['identity']['baseline_sha256'])
    helper_hashes = rest['identity']['snapshots']
    for name, digest in helper_hashes.items():
        verify(rest_folder/f'{name}-snapshot.py', digest)
    if helper_hashes['check_piano_stream_context'] != context['identity']['runner_sha256']:
        raise ValueError('Rest experiment did not use the frozen context runner')
    for name, folder in (('adapt_piano_streams', 'adaptation-v1'),
                         ('evaluate_piano_stream_adaptation', 'adaptation-evaluation-v1'),
                         ('research_piano_streams', 'probe-v3')):
        anchored = read(root/folder/'report.json')
        verify(root/folder/'runner-snapshot.py', helper_hashes[name])
        if anchored['identity']['runner_sha256'] != helper_hashes[name]:
            raise ValueError('Helper differs from anchored experiment')
    for filename, digest in rest['artifacts_sha256'].items():
        if Path(filename).name != filename:
            raise ValueError('Expected only local artifact filenames')
        verify(rest_folder/filename, digest)
    selected = sha256(root/'adaptation-v1/selected.pt')
    selection = read(root/'adaptation-v1/selection.json')
    trained = read(root/'adaptation-v1/report.json')
    if selected != selection['sha256'] or selected != trained['selected_sha256'] or selection['epoch'] != 5:
        raise ValueError('Selected checkpoint differs from frozen training selection')
    verify(root/'source/pretrained_models/model.ckpt', trained['identity']['model_sha256'])
    revision = subprocess.check_output(['git', '-C', str(root/'source'), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(root/'source'), 'diff', 'HEAD', '--name-only'], text=True).strip()
    if changes:
        raise ValueError('Tracked upstream source changed')
    manifest_path = root/'dataset-v2/manifest.json'
    manifest = read(manifest_path)
    verify(root/'dataset-v2/runner-snapshot.py', manifest['runner_sha256'])
    validation_rows = [r for r in manifest['cases'] if r['status'] == 'ready' and r['split'] == 'validation']
    if len(validation_rows) != 16:
        raise ValueError('Unexpected validation partition')
    for row in validation_rows:
        verify(root/'dataset-v2'/row['cache'], row['cache_sha256'])
    for report in (context, rest):
        identity = report['identity']
        if identity['source_revision'] != revision or identity['selected_sha256'] != selected:
            raise ValueError('Source or selected weight identity differs')
        verify(manifest_path, identity['dataset_sha256'])
        verify(args.analysis, identity['analysis_sha256'])
        verify(root/'probe-v3/sakura-graph-input.json', identity['graph_rows_sha256'])
    verify(root/'adaptation-evaluation-v1/report.json', context['identity']['evaluation_sha256'])
    prior = {(c['model'], c['name'], c['input']): c for c in context['cases']}
    if len(prior) != 64 or len(rest['cases']) != 64:
        raise ValueError('Incomplete validation context/rest comparison')
    cases = []
    for case in rest['cases']:
        old = prior[(case['model'], case['name'], case['input'])]
        if case['scores']['raw'] != old['scores'] or not case['audit']['raw_predictions_reproduced']:
            raise ValueError('Raw statistics do not reproduce the context experiment')
        for relation in case['audit']['observed_edges'].values():
            if relation['window_only'] or relation['full_internal_missing_in_windows']:
                raise ValueError('Repaired windows are not induced subgraphs')
        if case['audit']['observed_edges']['rest']['full_nonforward'] or case['audit']['observed_edges']['rest']['window_nonforward']:
            raise ValueError('Nonforward rest edges remain')
        cases.append(dict(model=case['model'], name=case['name'], input=case['input'],
            scores=case['scores'], graphs=graph_summary(case['audit']['graphs']),
            raw_context_difference=safe_difference(old['differences']),
            fixed_context_difference=safe_difference(case['audit']['fixed_context_difference']),
            repair_changes={key: safe_difference(value) for key, value in case['audit']['changes'].items()}))
    prior_sakura = {row['model']: row for row in context['sakura']}
    sakura = []
    for case in rest['sakura']:
        model = case['model']
        local_ids = read(rest_folder/f'{model}-sakura-expression-change-ids.json')
        direction_counts = {context_name: {mode: {direction: len(ids) for direction, ids in directions.items()}
            for mode, directions in modes.items()} for context_name, modes in local_ids.items()}
        for context_name, modes in direction_counts.items():
            for mode, counts in modes.items():
                change = case['repair_expression_changes'][context_name][mode]
                if sum(counts.values()) != change['hold_tap_flips']:
                    raise ValueError('Expression direction counts differ from aggregate')
                if change['before_holds'] - counts['hold_to_tap'] + counts['tap_to_hold'] != change['after_holds']:
                    raise ValueError('Expression net hold count differs')
        sakura.append(dict(model=model, graphs=graph_summary(case['audit']['graphs']),
            raw_context_difference=safe_difference(prior_sakura[model]['differences']),
            fixed_context_difference=safe_difference(case['audit']['fixed_context_difference']),
            repair_changes={key: safe_difference(value) for key, value in case['audit']['changes'].items()},
            repair_expression_changes=case['repair_expression_changes'],
            repair_expression_direction_counts=direction_counts,
            context_expression_changes=case['context_expression_changes']))
    report = dict(date='2026-11-09',runner_sha256=sha256(__file__), identities=identities,
        verification=dict(context_and_rest_snapshots=True,helper_snapshots=True,rest_artifacts=len(rest['artifacts_sha256']),
            selected_epoch=selection['epoch'],original_model_sha256=trained['identity']['model_sha256'],
            validation_caches=len(validation_rows),validation_case_count=len(cases),raw_statistics_reproduced=True,
            source_clean=True,upstream_revision=revision,test_arrays_read=False),
        validation=validation_summary(cases,context,rest),cases=cases,sakura=sakura,
        controls=rest['controls'],elapsed_seconds=rest['elapsed_seconds'],
        actual_rest_runner_sha256=rest['identity']['runner_sha256'],
        current_rest_runner_sha256=sha256(ROOT/'scripts/check_piano_stream_rest.py'),
        enabled_in_application=False,musical_acceptance=False,
        limitations=['Validation score-voice labels are not perceptual or physical-release ground truth.',
            'Only impossible rest edges were repaired; normal cross-window observations and normalization still differ.',
            'Fixed weights had encountered uncorrected graphs; repair need not improve every piece.',
            'The frozen Sakura analysis was not re-transcribed, retimed, selected, laid out or exported in this experiment.',
            'Expression counts are pre-selection decisions and are not final-chart LN quality.',
            'No score events, audio, expression IDs or research weights are included in this public report.',
            'Results belong to the recorded executed snapshot, not later assertion-only source edits.'])
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),sha256=sha256(args.output),cases=len(cases),
                         verified_rest_artifacts=report['verification']['rest_artifacts'])),flush=True)


if __name__ == '__main__':
    main()
