"""Prepare local-notation event graphs from frozen train/validation source checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.piano_event_training import (SCHEMA, build_piece_items, file_sha256,
                                         read_meter_map)

SOURCE_REPORT_SHA256 = '280ccd7a9e47a5b18b7f54a301b69924e7b267c75024796399bdfd2091bc1ecb'


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source-report', type=Path,
                     default=ROOT/'test-data/piano-svsep/musicxml-events-v2/report.json')
    cli.add_argument('--dataset-manifest', type=Path,
                     default=ROOT/'test-data/piano-svsep/dataset-v2/manifest.json')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous event-training caches')
    if file_sha256(args.source_report) != SOURCE_REPORT_SHA256:
        raise ValueError('Unexpected frozen source report')
    report = json.loads(args.source_report.read_text(encoding='utf-8'))
    if file_sha256(args.dataset_manifest) != report['identity']['dataset_manifest_sha256']:
        raise ValueError('Frozen dataset split changed')
    old = json.loads(args.dataset_manifest.read_text(encoding='utf-8'))
    original = {row['name']: row for row in old['cases']}
    args.output.mkdir(parents=True)
    builder = ROOT/'scripts/piano_event_training.py'
    identity = dict(source_report_sha256=SOURCE_REPORT_SHA256,
        dataset_manifest_sha256=file_sha256(args.dataset_manifest),
        runner_sha256=file_sha256(__file__), builder_sha256=file_sha256(builder),
        max_sounding_measures=16, core_bars=4, halo_bars=1,
        input_scope='completed train/validation source-event files only; original XML meter only',
        test_read_by_runner=False, labels='local notation successor; stable two-staff notation',
        chord_supervision=False, acoustic_release_supervision=False, melody_identity_supervision=False)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    shutil.copyfile(builder, args.output/'builder-snapshot.py')
    items, cases = [], []
    for record in report['cases']:
        if record['status'] != 'complete' or record['split'] not in ('train', 'validation'):
            continue
        row = original[record['name']]
        if row['split'] != record['split'] or row['sha256'] != record['input_sha256']:
            raise ValueError('Split/source identity disagrees')
        detail_path = args.source_report.parent/record['detail_file']
        if file_sha256(detail_path) != report['artifacts_sha256'][record['detail_file']]:
            raise ValueError('Frozen source detail changed')
        path = Path(row['path'])
        if file_sha256(path) != row['sha256']:
            raise ValueError('Original XML changed')
        detail = json.loads(detail_path.read_text(encoding='utf-8'))
        case = dict(name=row['name'], split=row['split'], source_sha256=row['sha256'],
                    detail_sha256=file_sha256(detail_path), status='error')
        try:
            meter = read_meter_map(path.read_bytes(), detail['source'])
            prepared = build_piece_items(detail['source'], meter, piece_id=row['sha256'])
            counts = Counter()
            for entry in prepared:
                metadata, payload = entry['metadata'], entry['payload']
                filename = f"{row['name']}-chunk-{metadata['chunk']:03d}.npz"
                np.savez_compressed(args.output/filename, **payload,
                                    metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
                items.append(dict(name=row['name'], split=row['split'], chunk=metadata['chunk'],
                    cache=filename, cache_sha256=file_sha256(args.output/filename),
                    counts=metadata['counts']))
                counts.update(metadata['counts'])
            case.update(status='ready', chunks=len(prepared), counts=dict(counts))
        except ValueError as error:
            case['error'] = str(error)
        cases.append(case)
        print(json.dumps(case, ensure_ascii=False), flush=True)
    if file_sha256(__file__) != identity['runner_sha256'] or file_sha256(builder) != identity['builder_sha256']:
        raise ValueError('Preparation code changed while running')
    manifest = dict(schema=SCHEMA, identity=identity, cases=cases, items=items,
                    source_files=len(cases), ready_files=sum(row['status']=='ready' for row in cases),
                    enabled_in_application=False, musical_acceptance=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(ready_files=manifest['ready_files'], chunks=len(items),
                         manifest_sha256=file_sha256(args.output/'manifest.json'))), flush=True)


if __name__ == '__main__':
    main()
