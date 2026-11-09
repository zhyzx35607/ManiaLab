"""Prepare +1/+2-quarter input tails from frozen source events and clean caches.

No fitting or inference is performed. Source supervision is held fixed, and
every clean payload is reproduced in memory against its frozen cache first.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.piano_event_training import SCHEMA, build_piece_items, file_sha256, read_meter_map
from scripts.piano_event_overlap import (EXTRA_QUARTERS, OVERLAP_SCHEMA, augment_events,
                                        summarize_augmentation)
from scripts.prepare_piano_event_data import SOURCE_REPORT_SHA256

CLEAN_MANIFEST_SHA256 = 'bfd8873a195bf0e4085d72c77cd6a409fbcc01cb3db2a2a9437d0933612b7999'


def assert_clean_cache(prepared: list[dict], cache_rows: list[dict], folder: Path):
    by_chunk = {row['chunk']: row for row in cache_rows}
    if len(by_chunk) != len(cache_rows) or set(by_chunk) != {item['metadata']['chunk'] for item in prepared}:
        raise ValueError('Clean window identities changed')
    for item in prepared:
        row = by_chunk[item['metadata']['chunk']]
        path = folder/row['cache']
        if file_sha256(path) != row['cache_sha256']:
            raise ValueError('Frozen clean cache changed')
        with np.load(path, allow_pickle=False) as cached:
            # Compare the persisted JSON representation: source parser tuples
            # become arrays when the original cache writer serializes them.
            expected_metadata = json.loads(json.dumps(item['metadata'], ensure_ascii=False))
            if json.loads(str(cached['metadata_json'])) != expected_metadata:
                raise ValueError('Default clean metadata changed')
            if set(cached.files) != set(item['payload']) | {'metadata_json'}:
                raise ValueError('Default clean fields changed')
            for key, value in item['payload'].items():
                if cached[key].dtype != value.dtype or not np.array_equal(cached[key], value):
                    raise ValueError('Default clean payload changed: '+key)


def supervised_identity(items: list[dict]) -> dict:
    """Identity-keyed targets let windows gain context without changing labels."""
    voices, staffs = {}, {}
    for item in items:
        p = item['payload']
        for pair, label, mask in zip(p['potential_edges'].T, p['voice'], p['voice_mask']):
            if mask:
                key = tuple(str(p['event_ids'][i]) for i in pair)
                if key in voices:
                    raise ValueError('Duplicated successor supervision across windows')
                voices[key] = bool(label)
        for identifier, label, mask in zip(p['event_ids'], p['staff'], p['staff_mask']):
            if mask:
                if str(identifier) in staffs:
                    raise ValueError('Duplicated staff supervision across windows')
                staffs[str(identifier)] = int(label)
    return dict(voice=voices, staff=staffs)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--clean-dataset', type=Path, required=True)
    cli.add_argument('--source-report', type=Path,
                     default=ROOT/'test-data/piano-svsep/musicxml-events-v2/report.json')
    cli.add_argument('--dataset-manifest', type=Path,
                     default=ROOT/'test-data/piano-svsep/dataset-v2/manifest.json')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous overlap caches')
    clean_path = args.clean_dataset/'manifest.json'
    if file_sha256(clean_path) != CLEAN_MANIFEST_SHA256:
        raise ValueError('Unexpected frozen clean dataset')
    if file_sha256(args.source_report) != SOURCE_REPORT_SHA256:
        raise ValueError('Unexpected frozen source-event report')
    clean = json.loads(clean_path.read_text(encoding='utf-8'))
    report = json.loads(args.source_report.read_text(encoding='utf-8'))
    if file_sha256(args.dataset_manifest) != report['identity']['dataset_manifest_sha256']:
        raise ValueError('Original split manifest changed')
    original = {row['name']: row for row in json.loads(args.dataset_manifest.read_text(encoding='utf-8'))['cases']}
    frozen_cases = {row['name']: row for row in report['cases'] if row['status'] == 'complete'
                    and row['split'] in ('train', 'validation')}
    if (clean['schema'] != SCHEMA or any(row['status'] != 'ready' or row['split'] not in ('train','validation')
            for row in clean['cases']) or {row['name'] for row in clean['cases']} != set(frozen_cases)):
        raise ValueError('Unexpected clean/source case coverage')
    grouped = defaultdict(list)
    for row in clean['items']:
        if row['name'] not in frozen_cases or row['split'] != frozen_cases[row['name']]['split']:
            raise ValueError('Unexpected clean window split')
        grouped[row['name']].append(row)
    code = {'runner': Path(__file__), 'builder': ROOT/'scripts/piano_event_training.py',
            'augmentation': ROOT/'scripts/piano_event_overlap.py'}
    identity = dict(clean_manifest_sha256=CLEAN_MANIFEST_SHA256,
        source_report_sha256=SOURCE_REPORT_SHA256,
        dataset_manifest_sha256=file_sha256(args.dataset_manifest),
        code_sha256={key:file_sha256(path) for key,path in code.items()},
        extra_quarters=list(EXTRA_QUARTERS), pitch_cap_scope='all events, same pitch, next strictly later attack',
        no_shortening=True, labels_from='unchanged source events', input_scope='train/validation only',
        test_read_by_runner=False, model_loaded=False, acoustic_release_supervision=False)
    args.output.mkdir(parents=True)
    for key,path in code.items():
        shutil.copyfile(path,args.output/(key+'-snapshot.py'))
    variants = {q:dict(schema=SCHEMA, identity=dict(identity, extra_quarters=q), cases=[], items=[],
        enabled_in_application=False, musical_acceptance=False) for q in EXTRA_QUARTERS}
    for q in EXTRA_QUARTERS:
        (args.output/f'plus{q}').mkdir()
    for clean_case in clean['cases']:
        name = clean_case['name']; record = frozen_cases[name]; original_row = original[name]
        if (record['split'] != clean_case['split'] or original_row['split'] != record['split']
                or record['input_sha256'] != clean_case['source_sha256']
                or original_row['sha256'] != record['input_sha256']):
            raise ValueError('Source/split identities differ')
        detail_path = args.source_report.parent/record['detail_file']
        if file_sha256(detail_path) != report['artifacts_sha256'][record['detail_file']]:
            raise ValueError('Frozen event detail changed')
        source_path = Path(original_row['path'])
        if file_sha256(source_path) != original_row['sha256']:
            raise ValueError('Original XML changed')
        source = json.loads(detail_path.read_text(encoding='utf-8'))['source']
        meter = read_meter_map(source_path.read_bytes(), source)
        options = dict(piece_id=clean_case['source_sha256'],
            max_sounding_measures=clean['identity']['max_sounding_measures'],
            core_bars=clean['identity']['core_bars'], halo_bars=clean['identity']['halo_bars'])
        baseline = build_piece_items(source,meter,**options)
        assert_clean_cache(baseline,grouped[name],args.clean_dataset)
        labels = supervised_identity(baseline)
        for q in EXTRA_QUARTERS:
            observed = augment_events(source,q)
            prepared = build_piece_items(source,meter,observed_events=observed,**options)
            if supervised_identity(prepared) != labels:
                raise ValueError('Input augmentation changed supervised identities/targets')
            counts = Counter()
            folder = args.output/f'plus{q}'
            for item in prepared:
                meta,payload=item['metadata'],item['payload']
                meta['augmentation'] = dict(extra_quarters=q, kind='fixed_tail_extension',
                    same_pitch_cap='whole_piece_next_attack', shortened_source_events=False)
                filename=f"{name}-chunk-{meta['chunk']:03d}.npz"
                np.savez_compressed(folder/filename,**payload,
                    metadata_json=np.asarray(json.dumps(meta,ensure_ascii=False)))
                variants[q]['items'].append(dict(name=name,split=record['split'],chunk=meta['chunk'],
                    cache=filename,cache_sha256=file_sha256(folder/filename),counts=meta['counts']))
                counts.update(meta['counts'])
            case=dict(name=name,split=record['split'],status='ready',source_sha256=record['input_sha256'],
                detail_sha256=file_sha256(detail_path),clean_cache_reproduced=True,
                supervised_identities_unchanged=True,chunks=len(prepared),counts=dict(counts),
                augmentation=summarize_augmentation(source,observed))
            variants[q]['cases'].append(case)
        print(json.dumps(dict(name=name,split=record['split'],clean_reproduced=True,
            variants={str(q):variants[q]['cases'][-1]['counts'] for q in EXTRA_QUARTERS})),flush=True)
    if any(file_sha256(path) != identity['code_sha256'][key] for key,path in code.items()):
        raise ValueError('Preparation code changed while running')
    manifests={}
    for q,value in variants.items():
        value.update(source_files=len(value['cases']),ready_files=len(value['cases']))
        path=args.output/f'plus{q}'/'manifest.json'
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        manifests[f'plus{q}']=dict(path=f'plus{q}/manifest.json',sha256=file_sha256(path),
                                 files=value['ready_files'],windows=len(value['items']))
    top=dict(schema=OVERLAP_SCHEMA,identity=identity,variants=manifests,
        clean_payloads_verified=len(clean['items']),enabled_in_application=False,musical_acceptance=False)
    (args.output/'manifest.json').write_text(json.dumps(top,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(variants=manifests,manifest_sha256=file_sha256(args.output/'manifest.json'))),flush=True)


if __name__ == '__main__':
    main()
