"""Evaluate one frozen delayed-frame hypothesis without running an acoustic model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from manialab.piano import OUTPUTS
from scripts.build_provenance import sha256
from scripts.evaluate_piano_acoustics import metrics, counts
from scripts.piano_delayed_frames import diagnose


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier results')
    root=ROOT/'test-data/smd/selected-v1/benchmark'
    manifest=read(root/'manifest.json')
    frozen_path=ROOT/'test-data/piano-release-semantics-v1/report.json'
    frozen={r['name']:r for r in read(frozen_path)['cases']}
    args.output.mkdir(parents=True)
    save=lambda name,value:(args.output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    code=dict(runner=Path(__file__),diagnostic=ROOT/'scripts/piano_delayed_frames.py',
              runtime=ROOT/'manialab/piano.py',scoring=ROOT/'scripts/evaluate_piano_acoustics.py',
              protocol=ROOT/'docs/piano-delayed-frame-protocol.md')
    hashes={key:sha256(path) for key,path in code.items()}
    for key,path in code.items():
        shutil.copyfile(path,args.output/(key+'-snapshot'+path.suffix))
    cases=[]
    for case in manifest['cases']+[dict(name='sakura-nova-piano')]:
        name=case['name']
        folder=(ROOT/'test-data/piano-onnx/portable-v1' if name=='sakura-nova-piano'
                else root/'specialist-v2')
        path=folder/(name+'-activations.npz')
        if sha256(path)!=frozen[name]['activations_sha256']:
            raise ValueError('Frozen activations changed')
        with np.load(path,allow_pickle=False) as data:
            arrays={key:data[key] for key in OUTPUTS}
        original,proposed,diagnostics=diagnose(arrays)
        assert len(original)==frozen[name]['events']
        row=dict(name=name,events=len(original),activations_sha256=sha256(path),
                 status_counts=dict(Counter(r['status'] for r in diagnostics)),
                 changed=sum(a['offset']!=b['offset'] for a,b in zip(original,proposed)),
                 attack_pitch_velocity_preserved=True)
        save(name+'-events.json',dict(original=original,proposed=proposed,diagnostics=diagnostics))
        if name!='sakura-nova-piano':
            # Ground truth is loaded only after the acoustic proposal is fixed.
            scope=lambda notes:[n for n in notes if 0<=n['onset']<case['score_seconds']]
            refs={}
            for semantics,field in (('key','labels'),('sustain','sustain_labels')):
                if sha256(ROOT/case[field])!=case[field+'_sha256']:
                    raise ValueError('Reference changed')
                refs[semantics]=scope(read(ROOT/case[field]))
            row['models']={kind:{semantics:{tol:metrics(ref,scope(predicted),strict=strict)
                        for tol,strict in (('standard',False),('strict',True))}
                    for semantics,ref in refs.items()}
                for kind,predicted in (('original',original),('proposed',proposed))}
            row['scored_changed']=sum(a['offset']!=b['offset'] for a,b in zip(original,proposed)
                                     if 0<=a['onset']<case['score_seconds'])
        cases.append(row)
        print(json.dumps({key:value for key,value in row.items() if key!='models'}),flush=True)
    scored=[row for row in cases if 'models' in row]
    summary={}
    for kind in ('original','proposed'):
        summary[kind]={}
        for semantics in ('key','sustain'):
            summary[kind][semantics]={}
            for tol in ('standard','strict'):
                values=[row['models'][kind][semantics][tol] for row in scored]
                summary[kind][semantics][tol]=dict(
                    macro={m:float(np.mean([v[m]['f1'] for v in values])) for m in ('onsets','intervals')},
                    pooled={m:counts(*(sum(v[m][key] for v in values)
                        for key in ('matched','reference','predicted'))) for m in ('onsets','intervals')},
                    long_reference=sum(v['long_reference'] for v in values),
                    long_matched=sum(v['long_matched'] for v in values),
                    predicted_long=sum(v['predicted_long'] for v in values))
    if any(sha256(path)!=hashes[key] for key,path in code.items()):
        raise ValueError('Source changed during evaluation')
    report=dict(code_sha256=hashes,frozen_semantics_report_sha256=sha256(frozen_path),
        benchmark_manifest_sha256=sha256(root/'manifest.json'),cases=cases,summary=summary,
        model_rerun=False,application_enabled=False,musical_acceptance=False,
        limitations=['50ms hypothesis followed prior activation inspection; not a blind test.',
                     'Proposed acoustic tails are not chart LN release targets.'])
    report['artifacts_sha256']={p.name:sha256(p) for p in args.output.iterdir()}
    save('report.json',report)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
