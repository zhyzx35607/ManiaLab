"""Publish verified aggregate stream experiments without score/audio events."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_provenance import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified_report(folder):
    report=read(folder/'report.json')
    if sha256(folder/'runner-snapshot.py')!=report['identity']['runner_sha256']:
        raise ValueError(f'Changed experiment snapshot: {folder.name}')
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=ROOT/'test-data/piano-svsep')
    cli.add_argument('--expression',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve prior public reports')
    root=args.root
    manifest=read(root/'dataset-v2/manifest.json')
    if sha256(root/'dataset-v2/runner-snapshot.py')!=manifest['runner_sha256']:
        raise ValueError('Dataset preparation snapshot changed')
    probe=verified_report(root/'probe-v3')
    trained=verified_report(root/'adaptation-v1')
    evaluated=verified_report(root/'adaptation-evaluation-v1')
    exported=verified_report(args.expression)
    preflight=read(root/'training-check-v1.json')
    if preflight['dataset_sha256']!=sha256(root/'dataset-v2/manifest.json'):
        raise ValueError('Preflight and final dataset differ')
    if preflight['training_runner_sha256']!=trained['identity']['runner_sha256']:
        raise ValueError('Preflight and actual training code differ')
    if trained['identity']['dataset_sha256']!=preflight['dataset_sha256']:
        raise ValueError('Training dataset identity differs')
    selected_sha=sha256(root/'adaptation-v1/selected.pt')
    if trained['selected_sha256']!=selected_sha or evaluated['identity']['selected_sha256']!=selected_sha:
        raise ValueError('Evaluation weights differ from training selection')
    if evaluated['identity']['training_report_sha256']!=sha256(root/'adaptation-v1/report.json'):
        raise ValueError('Evaluation training report differs')
    if exported['identity']['stream_report_sha256']!=sha256(root/'adaptation-evaluation-v1/report.json'):
        raise ValueError('Export evaluation identity differs')
    if exported['identity']['selected_model_sha256']!=selected_sha:
        raise ValueError('Export weights differ from selected model')
    for name,digest in exported['identity']['snapshots'].items():
        if sha256(args.expression/(Path(name).stem+'-snapshot.py'))!=digest:
            raise ValueError('Export runtime snapshot changed')
    cases=[]
    safe_fields=('name','model','grouping','keys','elapsed_seconds','selection_optimal','source_counts',
        'expressed_source_counts','expression_changes','drums','large_attack_shifts','omission_reasons',
        'newly_held_diagnostics','original_candidates_unchanged','non_piano_expression_unchanged',
        'weak_and_omitted_candidates_unchanged','selected_endpoints_preserved','roundtrip_endpoints_equal',
        'baseline_exactly_reproduced','artifacts_sha256')
    for case in exported['cases']:
        for filename,digest in case['artifacts_sha256'].items():
            if sha256(args.expression/filename)!=digest:raise ValueError(f'Changed export: {filename}')
        cases.append({key:case[key] for key in safe_fields}|dict(structurally_valid=case['structural']['valid'],
            off_grid=case['grid']['off_grid']))
    groups={split:len({r['work'] for r in manifest['cases'] if r['status']=='ready' and r['split']==split})
        for split in ('train','validation','test')}
    input_files=dict(dataset=root/'dataset-v2/manifest.json',probe=root/'probe-v3/report.json',
        preflight=root/'training-check-v1.json',training=root/'adaptation-v1/report.json',
        evaluation=root/'adaptation-evaluation-v1/report.json',expression=args.expression/'report.json',
        runtime=root/'requirements-resolved.txt')
    report=dict(date='2026-11-09',input_sha256={name:sha256(path) for name,path in input_files.items()},
        runner_sha256=sha256(__file__),
        data=dict(corpus_revision=manifest['corpus_revision'],score_files=len(manifest['cases']),
            split_scores=manifest['split_counts'],split_notes=manifest['split_notes'],split_works=groups,
            max_measures_per_score=16,excluded_related=manifest['excluded_related'],unusable=manifest['unusable'],
            unusable_reasons=dict(Counter(r['reason'] for r in manifest['cases'] if r['status']=='unusable')),
            work_disjoint=manifest['work_disjoint'],license=manifest['license']),
        preflight=preflight,frozen_probe=probe,training=trained,evaluation=evaluated,
        exports=dict(identity=exported['identity'],cases=cases),
        test_piece_deltas=read(root/'adaptation-evaluation-v1/test-piece-deltas.json'),
        test_component_cases={name:read(root/f'adaptation-evaluation-v1/{name}-test-components.json')['cases']
            for name in ('original','expanded','adapted')},
        enabled_in_application=False,musical_acceptance=False,
        limitations=['Score-engraving labels are not perceptual melody or finger-release truth.',
            'Work-disjoint adaptation does not establish an unseen pretraining test set.',
            'Synthetic tail augmentation is not a measured pedal signal.',
            'Frozen Sakura audio is not re-transcribed or automatically timed in this experiment.',
            'Component labels stop at bar boundaries; no full-song identity is claimed.',
            'No audio, score events or research weights are distributed in this report.'])
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),sha256=sha256(args.output),cases=len(cases))),flush=True)


if __name__=='__main__':main()
