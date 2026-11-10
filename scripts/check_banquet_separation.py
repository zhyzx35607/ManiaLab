"""Fixed eight-family URMP controlled mixtures for Banquet; not a Nova replacement."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_banquet_separation as adapter
from scripts.evaluate_guitar_vocal_benchmark import read_json,require,sha,verified_hash
from scripts.research_hf_instrument import write_new

MANIFEST=ROOT/'test-data/urmp/comparison-extension/manifest.json'
MANIFEST_SHA='e363a4e7b5f25628909eb13031251847492d9fa816e520b98fbc6d612bcda16a'
EXPECTED=[('vn',12),('vc',12),('va',13),('sax',25),('tpt',31),('hn',31),('tbn',31),('db',38)]


def prepare():
    from scipy.io import wavfile
    verified_hash(MANIFEST,MANIFEST_SHA)
    cases=read_json(MANIFEST)['cases']
    selected={};groups=defaultdict(list)
    for case in cases:
        if case['kind']=='isolated':
            groups[case['piece_id']].append(case)
            selected.setdefault(case['instrument'],case['piece_id'])
    require(list(selected.items())==EXPECTED,'Frozen first-occurrence family selection changed')
    rows=[]
    for family,piece in selected.items():
        sources=groups[piece]
        for source in sources:
            verified_hash(source['audio'],source['audio_sha256'])
            rate,wave=wavfile.read(source['audio'],mmap=True)
            require(rate==16000 and wave.ndim==1 and str(wave.dtype)=='int16' and len(wave)>26*rate,
                    'Expected original 16k mono PCM with complete query and test windows')
        rows.append(dict(name=f'{piece:02}-{family}',family=family,piece=piece,
                         sources=[{k:c[k] for k in ('name','instrument','audio','audio_sha256')} for c in sources]))
    return dict(cases=rows,manifest_sha256=MANIFEST_SHA,assets=adapter.verify_assets(),
                mixture_seconds=[20,26],query_seconds=[0,10],score_interiors_seconds=[0,6,2,4],
                gain='all source stems divided by the number of stems in that piece',
                same_family_parts_summed=True,mono_duplicated_to_stereo=True,
                query_is_oracle_isolated_audio=True,transcription_labels_used=False,
                models_parallel=False,enabled_in_application=False,
                runner_sha256=sha(__file__),adapter_sha256=sha(adapter.__file__))


def arrays(case):
    import numpy as np
    import torch
    import torchaudio.functional as af
    from scipy.io import wavfile
    waveforms=[];query=[];target=[]
    for source in case['sources']:
        verified_hash(source['audio'],source['audio_sha256'])
        rate,pcm=wavfile.read(source['audio'])
        x=pcm.astype(np.float32)/32768/len(case['sources'])
        waveforms.append(x[20*rate:26*rate])
        if source['instrument']==case['family']:
            target.append(waveforms[-1]);query.append(x[:10*rate])
    convert=lambda x:np.repeat(af.resample(torch.from_numpy(x.copy()),16000,adapter.RATE).numpy()[None],2,axis=0)
    reference=convert(np.sum(target,axis=0));mixture=convert(np.sum(waveforms,axis=0))
    q=convert(np.sum(query,axis=0))
    return dict(mixture=mixture,target=reference,absent=mixture-reference,query=q,
                silence=np.zeros_like(mixture))


def wave_metrics(reference,estimate):
    import numpy as np
    require(reference.shape==estimate.shape and np.isfinite(reference).all() and np.isfinite(estimate).all(),
            'Scoring waveforms differ')
    r=reference.astype(np.float64).ravel();x=estimate.astype(np.float64).ravel()
    energy=float(r@r)
    basic=dict(reference_rms=float(np.sqrt(np.mean(r*r))),output_rms=float(np.sqrt(np.mean(x*x))),
               mae=float(np.mean(abs(x-r))),peak=float(np.max(abs(x))))
    if energy<=1e-20:return dict(basic,snr_db=None,si_sdr_db=None,gain=None,reason='silent_reference')
    error=float((x-r)@(x-r))
    snr=float(10*np.log10(max(energy,1e-20)/max(error,1e-20)))
    r=r-r.mean();x=x-x.mean();centered=float(r@r)
    if centered<=1e-20:return dict(basic,snr_db=snr,si_sdr_db=None,gain=None,reason='constant_reference')
    if float(x@x)<=1e-20:return dict(basic,snr_db=snr,si_sdr_db=None,gain=0.,reason='constant_estimate')
    gain=float(x@r/centered);projected=gain*r;noise=x-projected
    score=float(10*np.log10(max(float(projected@projected),1e-20)/max(float(noise@noise),1e-20)))
    return dict(basic,snr_db=snr,si_sdr_db=score,gain=gain)


def run(output):
    import numpy as np
    output=Path(output);require(not output.exists(),'Preserve completed or incomplete evidence')
    plan=prepare();write_new(output/'plan.json',plan)
    model,identity=adapter.load_model();write_new(output/'model-identity.json',identity)
    rows=[]
    for case in plan['cases']:
        data=arrays(case);folder=output/case['name'];folder.mkdir()
        np.savez_compressed(folder/'inputs.npz',**data)
        variants=[]
        for variant in ('mixture','target','absent','silence'):
            started=time.monotonic();estimate=adapter.separate(model,data[variant],data['query'])
            elapsed=time.monotonic()-started
            np.save(folder/(variant+'.npy'),estimate,allow_pickle=False)
            reference=data['target'] if variant in ('mixture','target') else data['silence']
            scores={}
            for label,begin,end in (('full',0,6),('interior',2,4)):
                sl=slice(begin*adapter.RATE,end*adapter.RATE)
                scores[label]=dict(output=wave_metrics(reference[:,sl],estimate[:,sl]),
                                   input=wave_metrics(reference[:,sl],data[variant][:,sl]))
            item=dict(variant=variant,seconds=elapsed,scores=scores,
                      estimate_sha256=sha(folder/(variant+'.npy')))
            variants.append(item)
            print(case['name'],variant,round(elapsed,3),scores['interior']['output'],flush=True)
        row=dict(case=case,inputs_sha256=sha(folder/'inputs.npz'),variants=variants)
        write_new(folder/'report.json',row);rows.append(row)
    require(sha(__file__)==plan['runner_sha256'] and sha(adapter.__file__)==plan['adapter_sha256'],
            'Frozen code changed during inference')
    result=dict(plan=plan,model_identity_sha256=sha(output/'model-identity.json'),rows=rows,
                seconds=sum(v['seconds'] for r in rows for v in r['variants']),
                musical_acceptance=False,default_changed=False,
                limitations=['Oracle disjoint source query; no automatic query discovery.',
                    '16kHz mono public URMP sources upsampled and duplicated, not original stereo masters.',
                    'Controlled sums, not Nova residual stems or original mastered mixtures.',
                    'Single 6-second model window, not author CLI overlap-add.',
                    'No note transcription or LN evaluation; small non-blind cohort.'])
    write_new(output/'report.json',result)


def main():
    import json
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path)
    cli.add_argument('--prepare-only',action='store_true')
    args=cli.parse_args()
    require(args.prepare_only or args.output is not None,'--output required')
    if args.prepare_only:
        p=prepare();print(json.dumps(p));require('torch' not in sys.modules,'Preparation loaded Torch')
    else:run(args.output)


if __name__=='__main__':main()
