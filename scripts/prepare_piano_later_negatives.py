"""Create local-only proof annotations without changing frozen event graphs."""
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
from scripts.piano_event_training import load_event_items
from scripts.piano_later_negative_proof import PROOF_VERSION, derive_piece_sidecars
from scripts.summarize_piano_same_tick_errors import SOURCE_REPORT_SHA256, source_context
from scripts.summarize_piano_successor_ranking import read, require

KINDS = ('clean', 'plus1', 'plus2')
MANIFESTS = dict(clean='bfd8873a195bf0e4085d72c77cd6a409fbcc01cb3db2a2a9437d0933612b7999',
    plus1='cd81db7696445602fd98adee3304d2a8d0ec65af59a86a0f15f7f60336d02e20',
    plus2='8859f14593d884426117a695973ed6b485328fa442da0a02a0a8c10d8c900beb')
GLOBAL_EXPECTED = dict(train=(581745, 603381, 629157), validation=(30914, 32385, 34439))


def dataset_folders():
    base = ROOT/'test-data/piano-svsep'
    return dict(clean=base/'source-event-dataset-v1', plus1=base/'source-event-overlap-v1/plus1',
                plus2=base/'source-event-overlap-v1/plus2')


def attach_later_sidecars(folder, groups, split):
    """Attach only this split's verified annotations outside the graph payload."""
    require(split in ('train', 'validation'), 'Only train/validation annotations may be loaded')
    folder = Path(folder); report = read(folder/'report.json')
    require(report['identity']['proof_version'] == PROOF_VERSION and
            report['identity']['source_report_sha256'] == SOURCE_REPORT_SHA256,
            'Different proof specification or source corpus')
    originals = {}
    for kind, dataset in dataset_folders().items():
        require(sha256(dataset/'manifest.json') == report['identity']['dataset_sha256'][kind] == MANIFESTS[kind],
                'Changed frozen dataset')
        originals[kind] = {(r['name'], r['chunk']): r['cache_sha256']
                           for r in read(dataset/'manifest.json')['items'] if r['split'] == split}
    rows = {(r['input'], r['name'], r['chunk']): r for r in report['items'] if r['split'] == split}
    expected = {(kind, item['name'], item['chunk']) for kind, items in groups.items() for item in items}
    require(len(rows) == len(expected) and set(rows) == expected, 'Changed sidecar coverage')
    for kind, items in groups.items():
        for item in items:
            row = rows[(kind, item['name'], item['chunk'])]; p = item['payload']
            path = folder/row['file']
            require(Path(row['file']).name == row['file'] and sha256(path) == row['sha256'], 'Changed annotation')
            require(row['cache_sha256'] == originals[kind][(item['name'], item['chunk'])], 'Different original cache')
            with np.load(path, allow_pickle=False) as saved:
                require(np.array_equal(saved['event_ids'], p['event_ids']) and
                        np.array_equal(saved['potential_edges'], p['potential_edges']), 'Changed annotation edge identity')
                mask = saved['derived_later_mask'].copy()
                require(mask.dtype == bool and mask.shape == p['voice_mask'].shape and
                        not np.any(mask & p['voice_mask']) and not np.any(mask & (p['voice'] != 0)) and
                        np.all(~mask | saved['proof_later_mask']), 'Invalid derived mask')
            require('derived_later_mask' not in item and 'derived_later_mask' not in p,
                    'Annotations must remain separate from graph payload')
            item['derived_later_mask'] = mask
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve previous annotation runs')
    require(args.output.resolve().is_relative_to((ROOT/'test-data').resolve()), 'Research artifacts must remain local')
    source_path = ROOT/'test-data/piano-svsep/musicxml-events-v2/report.json'
    require(sha256(source_path) == SOURCE_REPORT_SHA256, 'Changed source report')
    source_report = read(source_path)
    folders = dataset_folders()
    for kind, folder in folders.items():
        require(sha256(folder/'manifest.json') == MANIFESTS[kind], 'Changed frozen manifest')
    args.output.mkdir(parents=True)
    code = dict(runner=Path(__file__), proof=ROOT/'scripts/piano_later_negative_proof.py',
                source_context=ROOT/'scripts/summarize_piano_same_tick_errors.py',
                graph_loader=ROOT/'scripts/piano_event_training.py',
                protocol=ROOT/'docs/piano-later-negative-protocol.md')
    for name, path in code.items():
        shutil.copyfile(path, args.output/(name+'-snapshot'+path.suffix))
    identity = dict(proof_version=PROOF_VERSION, source_report_sha256=SOURCE_REPORT_SHA256,
                    dataset_sha256=MANIFESTS, code_sha256={k:sha256(v) for k,v in code.items()})
    rows, cases, details, aggregate = [], [], {}, {}
    for split in ('train', 'validation'):
        groups = {kind:load_event_items(folder, split) for kind, folder in folders.items()}
        indexed = {}
        original_rows = {}
        for kind, items in groups.items():
            indexed[kind] = defaultdict(list)
            for item in items:
                indexed[kind][item['name']].append(item)
            original_rows[kind] = {(row['name'], row['chunk']):row for row in read(folders[kind]/'manifest.json')['items']
                                   if row['split'] == split}
        names = set(indexed['clean'])
        require(names and all(set(indexed[k]) == names for k in KINDS), 'Different input work coverage')
        source_rows = {row['name']:row for row in source_report['cases']
                       if row['split'] == split and row['status'] == 'complete'}
        totals = {kind:Counter() for kind in KINDS}; histograms = {kind:Counter() for kind in KINDS}
        for name in sorted(names):
            require(name in source_rows, 'Only completed train/validation sources may be read')
            source_row = source_rows[name]; detail = source_path.parent/source_row['detail_file']
            digest = sha256(detail)
            require(digest == source_report['artifacts_sha256'][source_row['detail_file']], 'Changed source detail')
            details[name] = digest
            context = source_context(read(detail)['source'])
            piece_id = source_row['input_sha256']; shared_proofs = {}
            for kind in KINDS:
                items = indexed[kind][name]
                sidecars, coverage, canonical = derive_piece_sidecars(items, context, piece_id)
                for pair, value in canonical.items():
                    if pair in shared_proofs:
                        require(value == shared_proofs[pair], 'Input tail augmentation changed proof truth')
                    shared_proofs[pair] = value
                histograms[kind].update(coverage['negative_tick_group_histogram'])
                totals[kind].update({k:v for k,v in coverage.items() if k != 'negative_tick_group_histogram'})
                cases.append(dict(split=split, input=kind, name=name, **coverage))
                for item in items:
                    chunk = item['chunk']; annotation = sidecars[chunk]; p = item['payload']
                    filename = f'{split}-{kind}-{len(rows):04d}.npz'
                    np.savez_compressed(args.output/filename, derived_later_mask=annotation['mask'],
                        proof_later_mask=annotation['proof_mask'], event_ids=p['event_ids'],
                        potential_edges=p['potential_edges'], positive_groups_json=np.asarray(json.dumps(annotation['positive_groups'])))
                    rows.append(dict(split=split, input=kind, name=name, chunk=chunk, file=filename,
                        sha256=sha256(args.output/filename), cache_sha256=original_rows[kind][(name,chunk)]['cache_sha256'],
                        source_detail_sha256=digest, derived_edges=int(annotation['mask'].sum()),
                        derived_sources=len(annotation['positive_groups'])))
        aggregate[split] = {}
        for index, kind in enumerate(KINDS):
            require(totals[kind]['global_owned_provable_edges'] == GLOBAL_EXPECTED[split][index],
                    'Global proof coverage differs from prior independent label audit')
            aggregate[split][kind] = dict(totals[kind], negative_tick_group_histogram=dict(sorted(histograms[kind].items())))
        print(json.dumps(dict(split=split, coverage=aggregate[split])), flush=True)
    require(all(sha256(path) == identity['code_sha256'][name] for name,path in code.items()), 'Annotation code changed')
    report = dict(identity=identity, source_detail_sha256=details, items=rows, cases=cases, coverage=aggregate,
        original_graphs_changed=False, original_supervision_changed=False, model_loaded=False, test_sources_read=False,
        application_enabled=False, musical_acceptance=False,
        artifacts_sha256={p.name:sha256(p) for p in args.output.iterdir()})
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(windows=len(rows), report_sha256=sha256(args.output/'report.json'))), flush=True)


if __name__ == '__main__':
    main()
