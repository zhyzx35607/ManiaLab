"""Train a small CPU network for feasible 6–10K chord preferences.

This learns lane choices under a fixed human timing/hold history. It cannot
transcribe music, choose LN lengths, or certify playability. Validation chooses
the epoch and enables each keycount independently; test only reports results.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from manialab.domain import parse_osu
from manialab.layout import chord_choices
from manialab.pattern_prior import NETWORK_FEATURES,network_features,mirror_features,score_network
from scripts.train_pattern_prior import split_for


def records_and_groups(root):
    records=[json.loads(line) for p in sorted(root.glob('manifest-*.jsonl')) for line in p.read_text(encoding='utf-8').splitlines() if line]
    selected,seen,excluded=[],set(),Counter()
    for r in records:
        if not r.get('eligible') or r['sha256'] in seen:
            continue
        seen.add(r['sha256'])
        raw=(root/'maps'/f"{r['beatmap_id']}.osu").read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:
            excluded['chart_hash_mismatch']+=1;continue
        style=' '.join(str(r.get(k,'')) for k in ('title','version','tags')).casefold()
        # A labelled 4+6K pack is not training evidence for a 5+5 hand layout.
        if r['keys']==10 and re.search(r'\b(?:4k?\s*\+\s*6k?|6k?\s*\+\s*4k?)\b',style):
            excluded['asymmetric_10k']+=1;continue
        if r['keys']==8 and re.search(r'\b(?:7k?\s*\+\s*1k?|1k?\s*\+\s*7k?|iidx)\b',style):
            excluded['scratch_8k']+=1;continue
        selected.append(r)
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for r in selected:
        song='song:'+r['song_key']
        if int(r.get('beatmapset_id',0))>0:union(song,'set:'+str(r['beatmapset_id']))
        if r.get('audio_hash'):union(song,'audio:'+r['audio_hash'])
    return [r|dict(group=find('song:'+r['song_key'])) for r in selected],dict(excluded)


def samples(parsed,seed,limit=250):
    keys=parsed['keys'];rows=defaultdict(list)
    for n in parsed['notes']:rows[n.start].append(n)
    beats=[]
    for line in parsed['timing']:
        p=line.split(',')
        if len(p)>=7 and p[6]=='1' and float(p[1])>0:beats.append((float(p[0]),float(p[1])))
    beats.sort();rng=np.random.default_rng(seed)
    take=set(rng.choice(len(rows),min(limit,len(rows)),replace=False))
    ends,last,history=np.full(keys,-100000.),np.full(keys,-100000.),[]
    index_beat=0;positive=[];negative=[]
    for index,(start,row) in enumerate(sorted(rows.items())):
        row.sort(key=lambda n:n.lane)
        while index_beat+1<len(beats) and beats[index_beat+1][0]<=start:index_beat+=1
        beat=beats[index_beat][1] if beats else 500
        if not 180<=beat<=1500:beat=500
        actual=np.array([n.lane for n in row])
        if index in take and len(row)<keys and len(set(actual))==len(row):
            choices=chord_choices(keys,len(row))
            choices=choices[~((ends[choices]>start)|(last[choices]==start)).any(axis=1)]
            choices=choices[np.any(choices!=actual[None],axis=1)]
            if len(choices):
                alternatives=choices[rng.choice(len(choices),min(4,len(choices)),replace=False)]
                duration=np.array([n.end-n.start for n in row])
                good=network_features(actual[None],keys,start,ends,last,history,duration,beat)[0]
                bad=network_features(alternatives,keys,start,ends,last,history,duration,beat)
                positive.extend([good]*len(bad));negative.extend(bad)
        for n in row:ends[n.lane],last[n.lane]=n.end,start
        history=(*history[-1:],(start,sum(1<<n.lane for n in row),len(row)))
    shape=(-1,len(NETWORK_FEATURES))
    return np.asarray(positive,np.float16).reshape(shape),np.asarray(negative,np.float16).reshape(shape)


def prepare(root,cache):
    selected,excluded=records_and_groups(root);cache.mkdir(parents=True,exist_ok=True)
    arrays=defaultdict(lambda:[[],[]]);count=Counter();manifest=[]
    for index,r in enumerate(selected):
        key,group=r['keys'],r['group'];split=split_for(group)
        remaining=5000-count[(key,group)]
        if remaining<=0:continue
        path=cache/(r['sha256']+'-network-v1.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as data:a,b=data['positive'],data['negative']
        else:
            parsed=parse_osu((root/'maps'/f"{r['beatmap_id']}.osu").read_text(encoding='utf-8-sig'))
            a,b=samples(parsed,int(r['beatmap_id']))
            np.savez_compressed(path,positive=a,negative=b)
        a,b=a[:remaining],b[:remaining]
        if not len(a):continue
        count[(key,group)]+=len(a)
        arrays[(key,split)][0].append(a);arrays[(key,split)][1].append(b)
        manifest.append(dict(id=r['beatmap_id'],sha256=r['sha256'],keys=key,group=group,split=split,pairs=len(a)))
        if index%100==0:print(json.dumps(dict(prepared=index,eligible=len(selected))),flush=True)
    data={k:tuple(np.concatenate(v) for v in pair) for k,pair in arrays.items()}
    return data,manifest,excluded


def evaluate(model,data,split):
    import torch
    result={}
    model.eval()
    with torch.inference_mode():
        for key in range(6,11):
            pair=data.get((key,split))
            if not pair:continue
            a,b=pair;margins=[]
            for start in range(0,len(a),4096):
                x,y=a[start:start+4096].astype(np.float32),b[start:start+4096].astype(np.float32)
                def score(z):
                    return (model(torch.from_numpy(z))+model(torch.from_numpy(mirror_features(z,key))))/2
                margins.extend((score(x)-score(y))[:,0].numpy())
            margin=np.asarray(margins)
            result[str(key)]=dict(pairs=len(margin),accuracy=round(float((margin>0).mean()+.5*(margin==0).mean()),4),
                                 loss=round(float(np.logaddexp(0,-margin).mean()),4))
    return result


def train(args):
    import torch
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(20261107);rng=np.random.default_rng(20261107)
    data,manifest,excluded=prepare(args.folder,args.cache)
    train_keys=[k for k in range(6,11) if (k,'train') in data]
    if not train_keys:raise ValueError('No training examples')
    model=torch.nn.Sequential(torch.nn.Linear(len(NETWORK_FEATURES),64),torch.nn.ReLU(),
                              torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.002)
    history=[];best_loss=float('inf');best=None;best_epoch=0;started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();losses=[]
        for step in range(args.steps):
            key=train_keys[step%len(train_keys)];a,b=data[(key,'train')]
            indices=rng.integers(0,len(a),size=args.batch)
            x,y=a[indices].astype(np.float32),b[indices].astype(np.float32)
            mirror=rng.random(len(x))<.5
            x[mirror]=mirror_features(x[mirror],key);y[mirror]=mirror_features(y[mirror],key)
            loss=torch.nn.functional.softplus(-(model(torch.from_numpy(x))-model(torch.from_numpy(y)))).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
        validation=evaluate(model,data,'validation')
        mean_loss=float(np.mean([m['loss'] for m in validation.values()])) if validation else float('inf')
        if mean_loss<best_loss:
            best_loss,best,best_epoch=mean_loss,copy.deepcopy(model.state_dict()),epoch+1
        item=dict(epoch=epoch+1,train_loss=round(float(np.mean(losses)),4),validation=validation,seconds=round(time.perf_counter()-started,1))
        history.append(item);print(json.dumps(item),flush=True)
    if best is None:raise ValueError('No validation groups; refusing to export an unvalidated model')
    model.load_state_dict(best)
    metrics={split:evaluate(model,data,split) for split in ('train','validation','test')}
    models={};statistics={}
    for key in range(6,11):
        splits={s:dict(maps=sum(r['keys']==key and r['split']==s for r in manifest),
            groups=len({r['group'] for r in manifest if r['keys']==key and r['split']==s}),
            pairs=sum(r['pairs'] for r in manifest if r['keys']==key and r['split']==s)) for s in ('train','validation','test')}
        validation=metrics['validation'].get(str(key),{})
        enabled=splits['train']['groups']>=8 and splits['validation']['groups']>=3 and validation.get('accuracy',0)>=.60
        models[str(key)]=dict(enabled=enabled)
        statistics[str(key)]=dict(enabled=enabled,splits=splits,metrics={s:metrics[s].get(str(key)) for s in metrics})
    layers=[dict(weight=np.round(m.weight.detach().numpy(),7).tolist(),bias=np.round(m.bias.detach().numpy(),7).tolist())
            for m in model if isinstance(m,torch.nn.Linear)]
    probe=np.random.default_rng(13).normal(size=(32,len(NETWORK_FEATURES))).astype(np.float32)
    with torch.inference_mode():expected=model(torch.from_numpy(probe))[:,0].numpy()
    error=float(np.max(np.abs(score_network(probe,layers)-expected)))
    if error>1e-4:raise AssertionError(f'Export mismatch: {error}')
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    payload=dict(revision=2,features=list(NETWORK_FEATURES),models=models,network=layers,
                 training_manifest_sha256=fingerprint,purpose='Feasible lane preference only; no audio or LN-duration prediction')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
    args.report.write_text(json.dumps(dict(models=statistics,manifest=manifest,excluded=excluded,epochs=history,best_epoch=best_epoch,
        training_manifest_sha256=fingerprint,export_max_error=error,seed=20261107,
        method='101-feature 64/32 MLP; pairwise softplus; key-balanced batches; mirror augmentation; song/audio/set grouped split'),indent=2),encoding='utf-8')
    print(json.dumps(dict(best_epoch=best_epoch,models=statistics,export_max_error=error)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,default=Path('test-data/layout-network.json'))
    parser.add_argument('--report',type=Path,default=Path('test-data/layout-network-report.json'))
    parser.add_argument('--cache',type=Path,default=Path('test-data/corpus/network-cache'))
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--batch',type=int,default=1024)
    parser.add_argument('--threads',type=int,default=2)
    train(parser.parse_args())
