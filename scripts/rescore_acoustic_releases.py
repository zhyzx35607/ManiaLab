"""Rescore frozen endpoint learning without retraining or recalibration."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_instrument_benchmark import verified_context,window_metrics
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/urmp/acoustic-releases-v1'))
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier evaluation')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    historical=read(args.root/'report.json')
    if sha(args.root/'model.npz')!=historical['model']['model_sha256']:
        raise ValueError('Frozen weights changed')
    cases={}
    for partition in ('benchmark','release-development'):
        manifest=read(Path('test-data/urmp')/partition/'manifest.json')
        for case in manifest['cases']:cases[case['name']]=case,partition
    rows=[]
    for old in historical['rows']:
        case,partition=cases[old['name']]
        base=(Path('test-data/urmp/benchmark/yourmt3')/case['name'] if partition=='benchmark'
              else Path('test-data/urmp/release-development-predictions')/case['name']/'baseline')
        report=read(base/'report.json')
        if (sha(base/'notes.json')!=old['identity']['baseline_notes_sha256'] or
            sha(case['labels_path'])!=case['labels_sha256'] or sha(case['audio'])!=case['audio_sha256']):
            raise ValueError('Frozen evidence changed')
        reference=read(Path(case['labels_path']))
        notes=read(base/'notes.json');learned=read(args.root/case['name']/'notes.json')
        decisions=read(args.root/case['name']/'decisions.json')
        reconstructed=[dict(n) for n in notes]
        for decision in decisions:
            if decision['selected']:reconstructed[decision['index']]['offset']=decision['new_offset']
        if (reconstructed!=learned or sum(d['selected'] for d in decisions)!=old['changed']):
            raise ValueError('Saved output differs from frozen decisions')
        context_start,context_end=verified_context(case,report)
        metrics={};scopes={}
        for backend,predicted in (('baseline',notes),('learned',learned)):
            metrics[backend],scopes[backend]=window_metrics(reference,predicted,begin=0,end=case['seconds'],
                context_start=context_start,context_end=context_end)
        rows.append(dict(name=case['name'],split=old['split'],changed=old['changed'],
            baseline=metrics['baseline'],learned=metrics['learned'],interval_scope=scopes,
            identity=old['identity'],learned_notes_sha256=sha(args.root/case['name']/'notes.json'),
            decisions_sha256=sha(args.root/case['name']/'decisions.json')))
    summary={}
    for split in sorted({r['split'] for r in rows}):
        local=[r for r in rows if r['split']==split]
        summary[split]=dict(cases=len(local),changed=sum(r['changed'] for r in local),
            macro_fixed_interval_f1={backend:sum(r[backend]['strict']['complete_intervals']['f1']
                for r in local)/len(local) for backend in ('baseline','learned')})
    result=dict(revision=2,runner_sha256=sha(__file__),
        scoring_sha256=sha('scripts/evaluate_instrument_benchmark.py'),
        historical_report_sha256=sha(args.root/'report.json'),frozen_model=historical['model'],
        rows=rows,summary=summary,retrained=False,recalibrated=False,reinferred=False,enabled_in_application=False,
        notice='All window attacks enter interval scores. The old 0.95 threshold stays frozen; historical calibration scores do not use this revised denominator.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
