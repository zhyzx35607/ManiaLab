"""One fixed joint-peak gain diagnostic for every prior Banquet case and control."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_banquet_separation as adapter
from scripts import check_banquet_separation as benchmark
from scripts.evaluate_guitar_vocal_benchmark import read_json,require,sha,verified_hash
from scripts.research_hf_instrument import write_new


def joint_gain(mixture,query):
    import numpy as np
    require(np.isfinite(mixture).all() and np.isfinite(query).all(),'Nonfinite level input')
    peak=max(float(np.max(abs(mixture))),float(np.max(abs(query))))
    return .95/peak if peak>0 else 1.


def levels(x):
    import numpy as np
    return dict(rms=float(np.sqrt(np.mean(x.astype(np.float64)**2))),peak=float(np.max(abs(x))))


def run(source,output):
    import numpy as np
    source,output=Path(source),Path(output)
    require(not output.exists(),'Preserve original evidence')
    baseline=read_json(source/'report.json')
    require([(r['case']['family'],r['case']['piece']) for r in baseline['rows']]==benchmark.EXPECTED,
            'Require all eight original cases')
    verified_hash(benchmark.__file__,baseline['plan']['runner_sha256'])
    verified_hash(adapter.__file__,baseline['plan']['adapter_sha256'])
    verified_hash(source/'model-identity.json',baseline['model_identity_sha256'])
    evidence=[]
    for row in baseline['rows']:
        folder=source/row['case']['name']
        verified_hash(folder/'inputs.npz',row['inputs_sha256'])
        require([v['variant'] for v in row['variants']]==['mixture','target','absent','silence'],
                'Incomplete baseline controls')
        for variant in row['variants']:
            verified_hash(folder/(variant['variant']+'.npy'),variant['estimate_sha256'])
        evidence.append(dict(name=row['case']['name'],inputs_sha256=row['inputs_sha256']))
    plan=dict(baseline_report_sha256=sha(source/'report.json'),baseline_path=str(source.resolve()),
              sources=evidence,policy='g=.95/max(peak(original mixture),peak(original query)); all four variants use same g',
              labels_used_for_gain=False,output_returned_to_original_scale=True,
              baseline_adapter_sha256=baseline['plan']['adapter_sha256'],runner_sha256=sha(__file__),
              settings_selected_after_observing_baseline=True,default_changed=False)
    write_new(output/'plan.json',plan)
    model,identity=adapter.load_model()
    require(identity==read_json(source/'model-identity.json'),'Model or runtime changed since baseline')
    rows=[]
    for original in baseline['rows']:
        name=original['case']['name'];folder=output/name;folder.mkdir()
        with np.load(source/name/'inputs.npz',allow_pickle=False) as saved:
            data={key:saved[key].copy() for key in saved.files}
        g=joint_gain(data['mixture'],data['query'])
        transformed={k:(v*g).astype(np.float32) for k,v in data.items()}
        np.savez_compressed(folder/'inputs-scaled.npz',**transformed)
        variants=[]
        for variant in ('mixture','target','absent','silence'):
            started=time.monotonic();raw=adapter.separate(model,transformed[variant],transformed['query'])
            seconds=time.monotonic()-started;estimate=(raw/g).astype(np.float32)
            np.save(folder/(variant+'-raw.npy'),raw,allow_pickle=False)
            np.save(folder/(variant+'.npy'),estimate,allow_pickle=False)
            reference=data['target'] if variant in ('mixture','target') else data['silence']
            scores={}
            for label,begin,end in (('full',0,6),('interior',2,4)):
                sl=slice(begin*adapter.RATE,end*adapter.RATE)
                scores[label]=dict(output=benchmark.wave_metrics(reference[:,sl],estimate[:,sl]),
                                   input=benchmark.wave_metrics(reference[:,sl],data[variant][:,sl]))
            item=dict(variant=variant,seconds=seconds,scores=scores,
                      raw_estimate_sha256=sha(folder/(variant+'-raw.npy')),
                      estimate_sha256=sha(folder/(variant+'.npy')))
            variants.append(item)
            print(name,variant,'g',round(g,4),round(seconds,3),scores['interior']['output'],flush=True)
        row=dict(case=original['case'],gain=g,original_levels={k:levels(v) for k,v in data.items()},
                 transformed_levels={k:levels(v) for k,v in transformed.items()},
                 transformed_inputs_sha256=sha(folder/'inputs-scaled.npz'),variants=variants)
        write_new(folder/'report.json',row);rows.append(row)
    require(sha(__file__)==plan['runner_sha256'],'Diagnostic runner changed during inference')
    verified_hash(adapter.__file__,plan['baseline_adapter_sha256'])
    write_new(output/'report.json',dict(plan=plan,rows=rows,seconds=sum(v['seconds'] for r in rows for v in r['variants']),
        enabled_in_application=False,musical_acceptance=False,
        limitations=['One post-baseline diagnostic setting; not selected on a blind evaluation.',
                     'Joint gain does not isolate query-level from mixture-level sensitivity.',
                     'Peak normalization does not guarantee training-distribution RMS.',
                     'All baseline limitations still apply; no per-case best-model selection.']))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();run(args.source,args.output)


if __name__=='__main__':main()
