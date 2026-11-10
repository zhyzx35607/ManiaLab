"""Compare the published ev-pre-fz-bal checkpoint on every frozen Banquet input."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_banquet_separation as adapter
from scripts import check_banquet_separation as benchmark
from scripts.evaluate_guitar_vocal_benchmark import read_json,require,sha,verified_hash
from scripts.research_hf_instrument import write_new

WEIGHT=ROOT/'test-data/banquet-model/ev-pre-fz-bal.ckpt'
WEIGHT_BYTES=551785899
WEIGHT_MD5='94691fb1c477ac811b5ab57ed5edcfe5'
WEIGHT_SHA256='527ba0b44f0e58abd805e246ebb3dac21d80d8741b75da4c4f4cf6c0853bc7f8'


def assets():
    # Retain the original verified architecture/config/source. Only the local
    # checkpoint identity changes; this is a new experiment, not baseline repair.
    identity=adapter.verify_assets()
    require(WEIGHT.stat().st_size==WEIGHT_BYTES,'Alternate checkpoint length differs')
    with WEIGHT.open('rb') as stream:digest=hashlib.file_digest(stream,'md5').hexdigest()
    require(digest==WEIGHT_MD5,'Alternate checkpoint differs from author manifest')
    verified_hash(WEIGHT,WEIGHT_SHA256)
    return dict(identity,checkpoint_sha256=sha(WEIGHT),checkpoint_bytes=WEIGHT_BYTES,
                checkpoint_variant='ev-pre-fz-bal.ckpt',checkpoint_md5=WEIGHT_MD5,
                variant_loader_sha256=sha(__file__))


def load_alternate(identity):
    # Process-local injection keeps the already frozen baseline adapter intact.
    # The complete model is still strictly loaded and every tensor compared by
    # that same loader; neither pretraining paths nor optimizer state are used.
    with patch.object(adapter,'CHECKPOINT',WEIGHT),patch.object(adapter,'verify_assets',lambda:dict(identity)):
        return adapter.load_model()


def prepare(source):
    source=Path(source);baseline=read_json(source/'report.json')
    require([(r['case']['family'],r['case']['piece']) for r in baseline['rows']]==benchmark.EXPECTED,
            'Require all eight frozen instrument families')
    verified_hash(benchmark.__file__,baseline['plan']['runner_sha256'])
    verified_hash(adapter.__file__,baseline['plan']['adapter_sha256'])
    verified_hash(source/'model-identity.json',baseline['model_identity_sha256'])
    identity=assets()
    for row in baseline['rows']:
        folder=source/row['case']['name']
        verified_hash(folder/'inputs.npz',row['inputs_sha256'])
        require([v['variant'] for v in row['variants']]==['mixture','target','absent','silence'],
                'Incomplete original controls')
        for v in row['variants']:verified_hash(folder/(v['variant']+'.npy'),v['estimate_sha256'])
    plan=dict(baseline_report_sha256=sha(source/'report.json'),baseline_directory=str(source.resolve()),
        baseline_model_identity_sha256=baseline['model_identity_sha256'],assets=identity,
        runner_sha256=sha(__file__),adapter_sha256=sha(adapter.__file__),benchmark_sha256=sha(benchmark.__file__),
        original_float_inputs_reused=True,gain=1.,query_changed=False,all_controls_required=True,
        prompted_by='https://github.com/kwatcharasupat/query-bandit/issues/6',
        reporter_claim_verified_locally=False,selected_after_baseline=True,enabled_in_application=False)
    return baseline,plan


def run(source,output):
    import numpy as np
    source,output=Path(source),Path(output)
    require(not output.exists(),'Preserve prior outputs')
    baseline,plan=prepare(source);write_new(output/'plan.json',plan)
    model,identity=load_alternate(plan['assets']);write_new(output/'model-identity.json',identity)
    original_identity=read_json(source/'model-identity.json')
    for key in ('source_files','source_commit','config','versions','state_keys','device','inference'):
        require(identity[key]==original_identity[key],'Architecture/runtime changed: '+key)
    rows=[]
    for original in baseline['rows']:
        name=original['case']['name'];folder=output/name;folder.mkdir()
        with np.load(source/name/'inputs.npz',allow_pickle=False) as stored:
            data={key:stored[key].copy() for key in stored.files}
        variants=[]
        for variant in ('mixture','target','absent','silence'):
            started=time.monotonic();estimate=adapter.separate(model,data[variant],data['query'])
            seconds=time.monotonic()-started;np.save(folder/(variant+'.npy'),estimate,allow_pickle=False)
            reference=data['target'] if variant in ('mixture','target') else data['silence']
            scores={}
            for label,begin,end in (('full',0,6),('interior',2,4)):
                sl=slice(begin*adapter.RATE,end*adapter.RATE)
                scores[label]=dict(output=benchmark.wave_metrics(reference[:,sl],estimate[:,sl]),
                                   input=benchmark.wave_metrics(reference[:,sl],data[variant][:,sl]))
            variants.append(dict(variant=variant,seconds=seconds,scores=scores,
                                 estimate_sha256=sha(folder/(variant+'.npy'))))
            print(name,variant,round(seconds,3),scores['interior']['output'],flush=True)
        row=dict(case=original['case'],inputs_sha256=original['inputs_sha256'],variants=variants)
        write_new(folder/'report.json',row);rows.append(row)
    verified_hash(__file__,plan['runner_sha256']);verified_hash(adapter.__file__,plan['adapter_sha256'])
    verified_hash(benchmark.__file__,plan['benchmark_sha256'])
    write_new(output/'report.json',dict(plan=plan,model_identity_sha256=sha(output/'model-identity.json'),rows=rows,
        seconds=sum(v['seconds'] for r in rows for v in r['variants']),musical_acceptance=False,default_changed=False,
        limitations=baseline['limitations']+['Checkpoint selected after seeing baseline and unconfirmed user issue.',
                    'One common alternative for all cases, no best-per-case picking or peak normalization.']))


def main():
    import json
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,required=True)
    cli.add_argument('--output',type=Path)
    cli.add_argument('--prepare-only',action='store_true')
    args=cli.parse_args();require(args.prepare_only or args.output is not None,'--output required')
    if args.prepare_only:
        _,plan=prepare(args.source);print(json.dumps(plan));require('torch' not in sys.modules,'Prepare loaded Torch')
    else:run(args.source,args.output)


if __name__=='__main__':main()
