"""Verify self-authored tie controls and publish only aggregate diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_provenance import sha256
from scripts.summarize_piano_stream_context import read,verify


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=ROOT/'test-data/piano-svsep')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous public reports')
    root=args.root;folder=root/'bar-identity-v1'
    report=read(folder/'report.json');identity=report['identity']
    verify(folder/'runner-snapshot.py',identity['runner_sha256'])
    verify(ROOT/'docs/piano-bar-identity-protocol.md',identity['protocol_sha256'])
    verify(root/'source/piano_svsep/utils/vocsep.py',identity['preprocessing_sha256'])
    installed=root/'venv/Lib/site-packages/partitura'
    verify(installed/'io/importmusicxml.py',identity['partitura_import_sha256'])
    verify(installed/'utils/music.py',identity['partitura_note_array_sha256'])
    revision=subprocess.check_output(['git','-C',str(root/'source'),'rev-parse','HEAD'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(root/'source'),'diff','HEAD','--name-only'],text=True).strip()
    if changes or revision!=identity['source_revision']:raise ValueError('Upstream identity changed')
    for filename,digest in report['artifacts_sha256'].items():
        if Path(filename).name!=filename:raise ValueError('Unexpected artifact path')
        verify(folder/filename,digest)
    if report['completed']!=7 or report['errors']!=0:raise ValueError('Incomplete audit')
    for flag in ('training_cache_read','training_cache_changed','weights_changed','enabled_in_application','musical_acceptance'):
        if report[flag] is not False:raise ValueError(f'Unexpected audit scope: {flag}')
    cases=[]
    safe=('name','scope','status','expected_raw_events','expected_preprocessed_events','raw_events',
          'processed_events','parsed_tie_count','processed_tie_count','added_event_count','removed_event_count',
          'returned_tie_count','checks','warnings','input_sha256')
    for case in report['cases']:
        detail=read(folder/(case['name']+'-detail.json'))
        verify(folder/(case['name']+'.musicxml'),case['input_sha256'])
        for field,values in (('raw_events',detail['raw_note_array']),('processed_events',detail['preprocessed_note_array']),
                ('parsed_tie_count',detail['parsed_ties']),('processed_tie_count',detail['processed_ties']),
                ('added_event_count',detail['new_event_keys']),('removed_event_count',detail['removed_event_keys'])):
            if case[field]!=len(values):raise ValueError(f'Inconsistent count: {case["name"]}.{field}')
        if case['returned_tie_count']!=len(detail['preprocessing_returned_ties'][0]):
            raise ValueError('Preprocessing return count differs')
        cases.append({key:case[key] for key in safe}|dict(expected_tie_count=len(case['expected_ties'])))
    result=dict(date='2026-11-09',runner_sha256=sha256(__file__),audit_report_sha256=sha256(folder/'report.json'),
        identity=identity,verification=dict(executed_snapshot=True,protocol=True,source_and_importer=True,
            artifacts=len(report['artifacts_sha256']),cases=len(cases),detail_counts_reproduced=True),
        cases=cases,observed_failed_expectations=report['observed_failed_expectations'],
        limitations=['Only seven self-authored controls were audited; corpus prevalence has not been measured.',
            'A tied written continuation and a new attack in the same musical voice are different relations.',
            'Voice/staff notation labels do not establish cross-bar perceptual identity.',
            'The parallel-unison import failure precedes upstream tie preprocessing.',
            'No existing training cache, model, audio transcription or application behavior was changed.',
            'No event arrays, note IDs, audio or research weights are distributed in this summary.'],
        training_cache_read=False,model_loaded=False,enabled_in_application=False,musical_acceptance=False)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),sha256=sha256(args.output),cases=len(cases),
        verified_artifacts=result['verification']['artifacts'])),flush=True)


if __name__=='__main__':main()
