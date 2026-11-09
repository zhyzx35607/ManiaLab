"""Count proved later competitors separately from the frozen scored metrics."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_provenance import sha256
from scripts.piano_successor_decode import select_successor_groups
from scripts.prepare_piano_later_negatives import dataset_folders, MANIFESTS
from scripts.summarize_piano_successor_ranking import read, load, require, verify_artifacts, sum_fields


def competition_diagnostics(data, logits, selected, proved, auxiliary_eligible):
    """Count maxima inside the winning tick, without relabelling scored edges.

    Other ticks can tie at the same maximum. These are observed attribution
    counts, not counterfactual recovery counts after deleting proved edges.
    """
    size = len(data['source_ids'])
    for value in (logits, selected, proved, auxiliary_eligible, data['actual'], data['score_mask']):
        require(value.shape == (size,), 'Changed diagnostic shape')
    require(all(value.dtype == bool for value in (selected, proved, auxiliary_eligible, data['actual'], data['score_mask'])),
            'Expected Boolean masks and choices')
    require(np.isfinite(logits).all(), 'Nonfinite diagnostic scores')
    require(not np.any(proved & data['score_mask']) and not np.any(auxiliary_eligible & ~proved), 'Proof must remain outside scored labels')
    expected = select_successor_groups(data['source_ids'], data['target_ticks'], logits)
    require(np.array_equal(selected, expected), 'Changed frozen decoding decision')
    counts = Counter(proved_edges=int(proved.sum()), same_window_proved_edges=int(auxiliary_eligible.sum()),
        raw_proved_positive=int((proved & (logits >= 0)).sum()),
        decoded_proved_positive=int((proved & selected).sum()),
        raw_same_window_proved_positive=int((auxiliary_eligible & (logits >= 0)).sum()),
        decoded_same_window_proved_positive=int((auxiliary_eligible & selected).sum()))
    for key in ('removed_true_with_proved_maximum_any', 'removed_true_with_proved_maximum_only',
                'removed_true_with_same_window_proved_maximum_only', 'true_sources_with_proved_maximum_only'):
        counts[key] = 0
    groups = defaultdict(list)
    for index, source in enumerate(data['source_ids']):
        groups[str(source)].append(index)
    truth = data['actual'] & data['score_mask']
    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        removed = truth[indices] & (logits[indices] >= 0) & ~selected[indices]
        if not removed.any():
            continue
        winning = indices[selected[indices]]
        require(len(winning) > 0, 'A nonnegative true edge cannot lose without a winner')
        maxima = winning[logits[winning] == logits[winning].max()]
        lost = int(removed.sum())
        counts['removed_true_with_proved_maximum_any'] += lost * bool(proved[maxima].any())
        if proved[maxima].all():
            counts['removed_true_with_proved_maximum_only'] += lost
            counts['true_sources_with_proved_maximum_only'] += 1
            if auxiliary_eligible[maxima].all():
                counts['removed_true_with_same_window_proved_maximum_only'] += lost
    return dict(counts)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison', 'sidecars', 'run', 'output'):
        cli.add_argument('--'+name, type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve prior diagnostics')
    require(args.output.resolve().is_relative_to((ROOT/'test-data').resolve()), 'Keep event diagnostics local')
    paired = read(args.comparison/'report.json'); proofs = read(args.sidecars/'report.json')
    run = read(args.run/'report.json')
    require(paired['identity']['after_run'] == sha256(args.run/'report.json') and
            paired['selected_weights']['after'] == run['selected_sha256'] and
            run['identity']['proof_report_sha256'] == sha256(args.sidecars/'report.json'), 'Different fitted model/proof report')
    require(run['identity']['dataset_sha256'] == proofs['identity']['dataset_sha256'], 'Different annotation dataset')
    for current, snapshot in (('proof_loader','runner'), ('proof','proof'), ('source_context','source_context'), ('protocol','protocol')):
        require(run['identity']['code_sha256'][current] == proofs['identity']['code_sha256'][snapshot], 'Different fitted proof specification')
    verify_artifacts(args.comparison, paired); verify_artifacts(args.sidecars, proofs)
    rows = {(r['input'], r['name'], r['chunk']):r for r in proofs['items'] if r['split'] == 'validation'}
    owned, eligible_pairs = {}, defaultdict(set)
    for kind, folder in dataset_folders().items():
        require(sha256(folder/'manifest.json') == MANIFESTS[kind] == proofs['identity']['dataset_sha256'][kind],
                'Changed validation dataset')
        for row in read(folder/'manifest.json')['items']:
            if row['split'] != 'validation':
                continue
            key = (kind, row['name']); annotation = rows[(kind, row['name'], row['chunk'])]
            require(annotation['cache_sha256'] == row['cache_sha256'] == sha256(folder/row['cache']), 'Changed original cache')
            graph = load(folder/row['cache']); side = load(args.sidecars/annotation['file'])
            require(np.array_equal(side['event_ids'], graph['event_ids']) and
                    np.array_equal(side['potential_edges'], graph['potential_edges']), 'Changed candidate identity')
            a, b = graph['potential_edges']; own = graph['core_mask'][b]
            bucket = owned.setdefault(key, {})
            for index in np.flatnonzero(own):
                pair = (str(graph['event_ids'][a[index]]), str(graph['event_ids'][b[index]]))
                require(pair not in bucket, 'Duplicate prediction owner')
                bucket[pair] = bool(side['proof_later_mask'][index])
            for index in np.flatnonzero(side['derived_later_mask']):
                pair = (str(graph['event_ids'][a[index]]), str(graph['event_ids'][b[index]]))
                require(pair not in eligible_pairs[key], 'Duplicate auxiliary owner')
                eligible_pairs[key].add(pair)
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    variants = []
    for variant in paired['variants']:
        kind = variant['input']; cases = []
        for index, row in enumerate(variant['cases']):
            key = (kind, row['name']); data = load(args.comparison/row['paired_file'])
            pairs = list(zip(map(str, data['source_ids']), map(str, data['target_ids'])))
            require(len(set(pairs)) == len(pairs) and set(pairs) == set(owned[key]), 'Different paired candidate population')
            proved = np.asarray([owned[key][pair] for pair in pairs], dtype=bool)
            auxiliary_eligible = np.asarray([pair in eligible_pairs[key] for pair in pairs], dtype=bool)
            before = competition_diagnostics(data, data['before_logits'], data['before_selected'], proved, auxiliary_eligible)
            after = competition_diagnostics(data, data['after_logits'], data['after_selected'], proved, auxiliary_eligible)
            filename = f'{kind}-{index:02d}.npz'
            np.savez_compressed(args.output/filename, source_ids=data['source_ids'], target_ids=data['target_ids'],
                                proved=proved, auxiliary_eligible=auxiliary_eligible)
            cases.append(dict(name=row['name'], before=before, after=after, proof_file=filename))
        aggregate = {side:sum_fields([row[side] for row in cases]) for side in ('before', 'after')}
        require(aggregate['after']['proved_edges'] == proofs['coverage']['validation'][kind]['global_owned_provable_edges'],
                'Diagnostic proof population changed')
        variants.append(dict(input=kind, cases=cases, aggregate=aggregate))
    report = dict(identity=dict(comparison_sha256=sha256(args.comparison/'report.json'),
        proof_report_sha256=sha256(args.sidecars/'report.json'), training_report_sha256=sha256(args.run/'report.json'), runner_sha256=sha256(__file__)),
        variants=variants, original_scoring_changed=False, model_loaded=False, test_sources_read=False,
        auxiliary_eligible_means='Validation relationships fitting the same-window proof rule; never used for parameter updates.',
        maximum_scope='Maximum members within the selected winning tick; other ticks may tie. Counts do not imply recovery after deleting these edges.',
        musical_acceptance=False, application_enabled=False,
        artifacts_sha256={path.name:sha256(path) for path in args.output.iterdir()})
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps([dict(input=v['input'], **v['aggregate']) for v in variants]), flush=True)


if __name__ == '__main__':
    main()
