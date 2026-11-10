"""Independently check saved refinement queries, decisions and matching counts.

No model imports or inference. Matching uses maximum bipartite cardinality,
separately from the comparison's minimum-cost assignment implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def count_matches(reference, predicted, ref_classes, pred_classes, *, tolerance=.05,
                  offsets=False, fixed_offset=False, families=False):
    valid = np.zeros((len(reference), len(predicted)), dtype=bool)
    for i, ref in enumerate(reference):
        for j, pred in enumerate(predicted):
            valid[i, j] = (not pred['is_drum'] and ref['pitch'] == pred['pitch']
                and abs(ref['onset']-pred['onset']) <= tolerance
                and (not families or ref_classes[i] == pred_classes[j])
                and (not offsets or abs(ref['offset']-pred['offset']) <=
                     (.08 if fixed_offset else max(.05, .2*(ref['offset']-ref['onset'])))))
    if not valid.size:
        return 0
    assigned = maximum_bipartite_matching(csr_matrix(valid.astype(np.int8)), perm_type='column')
    return int(np.count_nonzero(assigned >= 0))


def controls():
    a = dict(onset=1., offset=1.5, pitch=60, is_drum=False)
    b = dict(a, offset=3.)
    assert count_matches([a, a], [a], [0, 1], [0], families=True) == 1
    assert count_matches([a, a], [a, a], [0, 1], [1, 0], families=True) == 2
    assert count_matches([a], [b], [0], [0], offsets=True) == 0
    assert count_matches([a], [dict(a, is_drum=True)], [0], [0]) == 0
    assert count_matches([], [a], [], [0]) == 0
    return 5


def part_grouping(reference, predicted, groupings):
    """Inspect grouping only where onset/pitch correspondence is unambiguous.

    This deliberately does not resolve unisons by predicted class or part.
    Coverage and unmatched/ambiguous detections remain explicit.
    """
    neighbors = [[j for j,pred in enumerate(predicted) if not pred['is_drum']
        and pred['pitch'] == ref['pitch'] and abs(pred['onset']-ref['onset']) <= .05]
        for ref in reference]
    degrees = Counter(j for row in neighbors for j in row)
    pairs = [(i, row[0]) for i,row in enumerate(neighbors) if len(row) == 1 and degrees[row[0]] == 1]
    choose2 = lambda n:n*(n-1)//2
    truth = [reference[i]['part'] for i,_ in pairs]
    true_pairs = sum(choose2(n) for n in Counter(truth).values())
    results = {}
    for name, labels in groupings.items():
        assert len(labels) == len(predicted)
        groups = [labels[j] for _,j in pairs]
        proposed_pairs = sum(choose2(n) for n in Counter(groups).values())
        correct_pairs = sum(choose2(n) for n in Counter(zip(truth, groups)).values())
        results[name] = dict(reference_pairs=true_pairs, grouped_pairs=proposed_pairs,
            correct_pairs=correct_pairs,
            precision=correct_pairs/proposed_pairs if proposed_pairs else None,
            recall=correct_pairs/true_pairs if true_pairs else None,
            f1=2*correct_pairs/(true_pairs+proposed_pairs) if true_pairs+proposed_pairs else None)
    return dict(reference=len(reference), predicted=len(predicted), unambiguous_notes=len(pairs),
        matched_reference_fraction=len(pairs)/len(reference) if reference else None,
        matched_prediction_fraction=len(pairs)/len(predicted) if predicted else None,
        no_compatible_prediction=sum(not row for row in neighbors),
        no_compatible_reference=sum(j not in degrees for j in range(len(predicted))),
        ambiguous_reference=sum(bool(row) and (len(row)>1 or degrees[row[0]]>1) for row in neighbors),
        matched_pairs=pairs, grouping_metrics=results,
        scope='Only uniquely compatible 50ms/pitch pairs; not overall transcription or clustering accuracy.')


def run(folder, output):
    assert not output.exists(), 'Preserve earlier audits'
    files = {}
    def verify(path, expected=None):
        path = Path(path)
        value = digest(path)
        if expected is not None:
            assert value == expected, str(path)
        files[str(path)] = value
        return value
    report, plan = read(folder/'report.json'), read(folder/'plan.json')
    verify(folder/'report.json')
    verify(folder/'plan.json', report['plan_sha256'])
    verify(folder/'inference-completed.json', report['inference_completed_sha256'])
    completed = read(folder/'inference-completed.json')
    assert completed['newly_inferred'] == 34 and completed['reused'] == 1
    assert completed['labels_read'] is False and len(completed['reports']) == 35
    for path, expected in plan['inputs'].items():
        verify(path, expected)
    for name, key in [('check_instrument_refinement.py', 'runner_sha256'),
                      ('research_instrument_refinement.py', 'adapter_sha256'),
                      ('evaluate_instrument_benchmark.py', 'matcher_sha256')]:
        verify(folder/'source'/name, plan[key])
    manifest_path = ROOT/'test-data/urmp/comparison-extension/manifest.json'
    verify(manifest_path, plan['manifest_sha256'])
    manifest = read(manifest_path)
    classes = plan['classes']
    mapping = {int(k):v for k,v in plan['program_to_family'].items()}
    taxonomy_path = ROOT/'test-data/other-instrument-model-survey-2026-11-10/tsumugi/instrument_agnostic_amt/taxonomy/instrument_merge.json'
    verify(taxonomy_path, plan['taxonomy_sha256'])
    taxonomy = read(taxonomy_path)
    actual_mapping = {int(member['program_number']):classes.index(name)
        for name, data in taxonomy['labels'].items() for member in data['members']
        if member.get('program_number') is not None}
    assert mapping == actual_mapping
    metrics_checked, decisions_checked, clusters_checked = 0, 0, 0
    summaries, grouping_rows = [], []
    groups = defaultdict(lambda: defaultdict(lambda: Counter(reference=0, predicted=0, matched=0)))
    for case, planned, scored in zip(manifest['cases'], plan['cases'], report['rows']):
        assert case['name'] == planned['name'] == scored['name']
        result = Path(planned['result_folder'])
        verify(result/'report.json', completed['reports'][case['name']])
        item = read(result/'report.json')
        for name, expected in item['evidence_sha256'].items():
            verify(result/name, expected)
        source = Path(planned['original_folder'])
        verify(source/'notes.json', scored['original_notes_sha256'])
        events, decisions, clusters = read(source/'notes.json'), read(result/'decisions.json'), read(result/'clusters.json')
        assert len(decisions) == len(events) == item['original_notes']
        with np.load(result/'queries.npz', allow_pickle=False) as stored:
            query = {key:stored[key] for key in stored.files}
        with np.load(result/'model-evidence.npz', allow_pickle=False) as stored:
            evidence = {key:stored[key] for key in stored.files}
        order = sorted((i for i,n in enumerate(events) if not n['is_drum']),
                       key=lambda i:(events[i]['onset'], events[i]['pitch'], events[i]['offset'], i))
        assert query['original_indices'].tolist() == order
        for name, key, dtype in [('start_seconds','onset',np.float32), ('end_seconds','offset',np.float32),
                                  ('pitch','pitch',np.int64)]:
            assert np.array_equal(query[name], np.asarray([events[i][key] for i in order], dtype=dtype))
        rate = item['identity']['model_config']['sample_rate']
        windows = evidence['window_starts']/rate
        observed = np.asarray([sum(begin <= t < begin+8. for begin in windows) for t in query['start_seconds']])
        assert np.array_equal(observed, evidence['observation_counts']) and np.all(observed > 0)
        assert np.isfinite(evidence['note_logits']).all() and np.isfinite(evidence['note_embeddings']).all()
        allowed = item['identity']['settings']['allowed_classes']
        assert allowed == [i for i,c in enumerate(classes) if c != 'drums']
        assigned_classes = {}
        for cluster in clusters:
            indices = np.flatnonzero(evidence['cluster_ids'] == cluster['cluster_id'])
            assert cluster['note_indices'] == indices.tolist() and cluster['note_count'] == len(indices)
            logits = evidence['note_logits'][indices].mean(axis=0)[allowed]
            probabilities = np.exp(logits-logits.max())
            probabilities /= probabilities.sum()
            best = np.argsort(-probabilities)[:3]
            assert [c['class_id'] for c in cluster['candidates']] == [allowed[i] for i in best]
            assert np.allclose([c['probability'] for c in cluster['candidates']], probabilities[best], atol=1e-12, rtol=0)
            for i in indices:
                assigned_classes[int(i)] = allowed[int(best[0])]
            clusters_checked += 1
        query_indices = {original:i for i,original in enumerate(order)}
        variant_labels = {key:[] for key in ('original', 'ungrouped', 'grouped')}
        for index, (event, decision) in enumerate(zip(events, decisions)):
            assert decision['original_index'] == index and decision['original_event'] == event
            variant_labels['original'].append(event['instrument_class_id'])
            if event['is_drum']:
                assert decision['decision'] == 'unsupported_drum_prediction'
                single = grouped = event['instrument_class_id']
            else:
                query_index = query_indices[index]
                single = int(np.argmax(evidence['note_logits'][query_index]))
                grouped = assigned_classes[query_index]
                assert decision['query_index'] == query_index and decision['observations'] == observed[query_index]
                assert decision['ungrouped_class'] == classes[single]
                assert decision['refined_class_id'] == grouped and decision['refined_class'] == classes[grouped]
            variant_labels['ungrouped'].append(single)
            variant_labels['grouped'].append(grouped)
            decisions_checked += 1
        verify(case['labels_path'], case['labels_sha256'])
        reference = [n for n in read(case['labels_path']) if 0 <= n['onset'] < 30]
        assert all(n['offset'] < planned['context_end_seconds'] for n in reference)
        predicted_indices = [i for i,n in enumerate(events) if 0 <= n['onset'] < 30]
        predicted = [events[i] for i in predicted_indices]
        ref_classes = [mapping[n['program']] for n in reference]
        grouping_labels = {mode:[values[i] for i in predicted_indices] for mode,values in variant_labels.items()}
        grouping_labels['timbre_clusters'] = [decisions[i].get('cluster_id', 'unsupported_drum')
                                             for i in predicted_indices]
        grouping_rows.append(dict(name=case['name'], kind=case['kind'],
            **part_grouping(reference, predicted, grouping_labels)))
        for mode, all_labels in variant_labels.items():
            pred_classes = [all_labels[i] for i in predicted_indices]
            for protocol, tolerance in [('standard', .05), ('strict', .04)]:
                for metric in ('pitched_onsets', 'complete_intervals', 'instrument_onsets', 'complete_instrument_intervals'):
                    amount = count_matches(reference, predicted, ref_classes, pred_classes,
                        tolerance=tolerance, offsets=metric.startswith('complete'),
                        fixed_offset=protocol == 'strict', families='instrument' in metric)
                    expected = scored['metrics'][mode][protocol][metric]
                    assert (len(reference),len(predicted),amount) == tuple(expected[k] for k in ('reference','predicted','matched'))
                    assert abs(expected['f1']-2*amount/max(1,len(reference)+len(predicted))) < 1e-12
                    target = groups[case['kind']]['/'.join((mode, protocol, metric))]
                    target.update(reference=len(reference), predicted=len(predicted), matched=amount)
                    metrics_checked += 1
        summaries.append(dict(name=case['name'], decisions=len(events), queries=len(order), clusters=len(clusters)))
    for kind, values in groups.items():
        for key, counts in values.items():
            expected = report['groups'][kind][key]
            assert all(counts[name] == expected[name] for name in ('reference','predicted','matched'))
            assert abs(expected['f1']-2*counts['matched']/max(1,counts['reference']+counts['predicted'])) < 1e-12
    assert len(summaries) == len(manifest['cases']) == len(report['rows']) == 35
    assert all(digest(path) == expected for path,expected in files.items())
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    output.mkdir(parents=True)
    (output/'auditor.py').write_bytes(Path(__file__).read_bytes())
    result = dict(source_report_sha256=files[str(folder/'report.json')], verified_files=files,
        auditor_sha256=digest(__file__), cases=summaries, controls=controls(),
        part_grouping=grouping_rows,
        decisions_checked=decisions_checked, clusters_checked=clusters_checked, metrics_checked=metrics_checked,
        matched_counts_independently_recomputed=True, mean_error_tie_break_recomputed=False,
        clustering_partition_recomputed=False, saved_cluster_classification_recomputed=True,
        neural_inference=False, source_or_outputs_modified=False, musical_acceptance=False)
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('verified_files','cases','part_grouping')}), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--controls-only', action='store_true')
    args = cli.parse_args()
    if args.controls_only:
        print(controls())
    else:
        if args.input is None or args.output is None:
            cli.error('--input and --output are required')
        run(args.input.resolve(), args.output.resolve())
