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
from manialab.domain import parse_osu,validate
from manialab.layout import release_choices
from manialab.pattern_prior import NETWORK_FEATURES,network_features,mirror_features,score_network
from scripts.train_pattern_prior import split_for


def _attach_song_groups(selected):
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
        for alias in r.get('related_song_keys',[]):union(song,'song:'+alias)
        for alias in r.get('related_audio_hashes',[]):union(song,'audio:'+alias)
    return [r|dict(group=find('song:'+r['song_key'])) for r in selected]


def _style_exclusion(r):
    style=' '.join(str(r.get(k,'')) for k in ('title','version','tags')).casefold()
    # A labelled 4+6K pack is not training evidence for a 5+5 hand layout.
    if r['keys']==10 and re.search(r'\b(?:4k?\s*\+\s*6k?|6k?\s*\+\s*4k?)\b',style):
        return 'asymmetric_10k'
    if r['keys']==8 and re.search(r'\b(?:7k?\s*\+\s*1k?|1k?\s*\+\s*7k?|iidx)\b',style):
        return 'scratch_8k'
    return None


def records_and_groups(root, *, exclude_splits=(), split_seed=None):
    records=[json.loads(line) for p in sorted(root.glob('manifest-*.jsonl')) for line in p.read_text(encoding='utf-8').splitlines() if line]
    selected,seen,excluded=[],set(),Counter()
    if isinstance(exclude_splits,str):
        raise ValueError('exclude_splits must be an iterable of split names')
    exclude_splits=set(exclude_splits)
    if exclude_splits - {'train','validation','test'}:
        raise ValueError('exclude_splits contains an unknown split')
    if exclude_splits:
        candidates=[]
        for r in records:
            if r.get('approved') not in (1,2,4):
                excluded['status_not_ranked_loved']+=1;continue
            if not r.get('eligible'):
                excluded['ineligible']+=1;continue
            if r['sha256'] in seen:
                excluded['duplicate_bytes']+=1;continue
            seen.add(r['sha256'])
            style_exclusion=_style_exclusion(r)
            if style_exclusion:
                excluded[style_exclusion]+=1;continue
            candidates.append(r)

        # Group identities use manifest metadata first so split membership is
        # known before any chart file in an excluded split is opened.
        grouped=_attach_song_groups(candidates)
        for r in grouped:
            split=split_for(r['group'],seed=split_seed)
            if split in exclude_splits:
                excluded[f'split_{split}_excluded']+=1
                continue
            raw=(root/'maps'/f"{r['beatmap_id']}.osu").read_bytes()
            if hashlib.sha256(raw).hexdigest()!=r['sha256']:
                excluded['chart_hash_mismatch']+=1;continue
            selected.append(r)
        return selected,dict(excluded)

    for r in records:
        if r.get('approved') not in (1,2,4):
            excluded['status_not_ranked_loved']+=1;continue
        if not r.get('eligible'):
            excluded['ineligible']+=1;continue
        if r['sha256'] in seen:
            excluded['duplicate_bytes']+=1;continue
        seen.add(r['sha256'])
        raw=(root/'maps'/f"{r['beatmap_id']}.osu").read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:
            excluded['chart_hash_mismatch']+=1;continue
        style_exclusion=_style_exclusion(r)
        if style_exclusion:
            excluded[style_exclusion]+=1;continue
        selected.append(r)
    return _attach_song_groups(selected),dict(excluded)


def samples(parsed,seed,limit=250):
    keys=parsed['keys'];rows=defaultdict(list)
    for n in parsed['notes']:rows[n.start].append(n)
    beats=[]
    for line in parsed['timing']:
        p=line.split(',')
        if len(p)>=7 and p[6]=='1' and float(p[1])>0:beats.append((float(p[0]),float(p[1])))
    beats.sort();rng=np.random.default_rng(seed)
    # Give LN interactions and plain attacks equal sampling opportunities when
    # both exist. Scan the entire chart before sampling; no opening-only cap.
    context=[];scan_ends=np.full(keys,-100000.);scan_last=scan_ends.copy()
    for start,row in sorted(rows.items()):
        context.append(any(n.hold for n in row) or bool(np.any(scan_ends>start)) or
            bool(np.any((scan_ends>scan_last)&(start-scan_ends>=0)&(start-scan_ends<150))))
        for n in row:scan_ends[n.lane],scan_last[n.lane]=n.end,start
    context=np.asarray(context);take=set()
    for group,budget in zip((np.flatnonzero(context),np.flatnonzero(~context)),((limit+1)//2,limit//2)):
        take.update(rng.choice(group,min(budget,len(group)),replace=False).tolist())
    remaining=np.array(sorted(set(range(len(rows)))-take))
    if len(take)<min(limit,len(rows)):
        take.update(rng.choice(remaining,min(limit,len(rows))-len(take),replace=False).tolist())
    ends,last,history=np.full(keys,-100000.),np.full(keys,-100000.),[]
    index_beat=0;positive=[];negative=[]
    for index,(start,row) in enumerate(sorted(rows.items())):
        row.sort(key=lambda n:n.lane)
        while index_beat+1<len(beats) and beats[index_beat+1][0]<=start:index_beat+=1
        beat=beats[index_beat][1] if beats else 500
        if not 180<=beat<=1500:beat=500
        actual=np.array([n.lane for n in row])
        if index in take and len(row)<keys and len(set(actual))==len(row):
            duration=np.array([n.end-n.start for n in row])
            choices=release_choices(keys,duration)
            choices=choices[~((ends[choices]>start)|(last[choices]==start)).any(axis=1)]
            choices=choices[np.any(choices!=actual[None],axis=1)]
            if len(choices):
                alternatives=choices[rng.choice(len(choices),min(4,len(choices)),replace=False)]
                good=network_features(actual[None],keys,start,ends,last,history,duration,beat)[0]
                bad=network_features(alternatives,keys,start,ends,last,history,duration,beat)
                # Swapping equal-duration notes can yield the same observable
                # state. Such a tie is not a meaningful negative preference.
                bad=np.unique(bad[~np.all(bad==good,axis=1)],axis=0)
                positive.extend([good]*len(bad));negative.extend(bad)
        for n in row:ends[n.lane],last[n.lane]=n.end,start
        history=(*history[-1:],(start,sum(1<<n.lane for n in row),len(row)))
    shape=(-1,len(NETWORK_FEATURES))
    return np.asarray(positive,np.float16).reshape(shape),np.asarray(negative,np.float16).reshape(shape)


def ln_context(a):
    lanes=a[:,:90].reshape(-1,10,9)
    return np.any(((lanes[:,:,1]>0)|(lanes[:,:,5]>0)|(lanes[:,:,4]<.75))&(lanes[:,:,8]>0),axis=1)


def prepare(root,cache):
    selected,excluded=records_and_groups(root);cache.mkdir(parents=True,exist_ok=True)
    arrays=defaultdict(lambda:[[],[]]);manifest=[]
    chart_counts=Counter((r['keys'],r['group']) for r in selected)
    for index,r in enumerate(selected):
        key,group=r['keys'],r['group'];split=split_for(group)
        # Share each song's budget across difficulties; do not skip later maps
        # just because an earlier difficulty consumed all 5,000 pairs.
        budget=max(1,5000//chart_counts[(key,group)])
        path=cache/(r['sha256']+'-'+str(r['beatmap_id'])+'-network-ln-v2.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as data:a,b=data['positive'],data['negative']
        else:
            parsed=parse_osu((root/'maps'/f"{r['beatmap_id']}.osu").read_text(encoding='utf-8-sig'))
            if not validate(parsed['notes'],key)['valid']:
                raise ValueError('Invalid chart in training snapshot')
            a,b=samples(parsed,int(r['beatmap_id']))
            np.savez_compressed(path,positive=a,negative=b)
        pick=np.random.default_rng(int(r['beatmap_id'])).choice(len(a),min(budget,len(a)),replace=False)
        a,b=a[pick],b[pick]
        if not len(a):continue
        arrays[(key,split)][0].append(a);arrays[(key,split)][1].append(b)
        manifest.append(dict(id=r['beatmap_id'],sha256=r['sha256'],keys=key,group=group,split=split,pairs=len(a),ln_pairs=int(ln_context(a).sum()),ln_class=r.get('ln_class')))
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
            def metrics(mask):
                selected=margin[mask]
                return dict(pairs=len(selected),accuracy=round(float((selected>0).mean()+.5*(selected==0).mean()),4),
                    loss=round(float(np.logaddexp(0,-selected).mean()),4)) if len(selected) else dict(pairs=0)
            context=ln_context(a)
            result[str(key)]=dict(**metrics(np.ones(len(margin),dtype=bool)),ln=metrics(context),taps=metrics(~context))
    return result


def train(args):
    import torch
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(20261107);rng=np.random.default_rng(20261107)
    data,manifest,excluded=prepare(args.folder,args.cache)
    train_keys=[k for k in range(6,11) if (k,'train') in data]
    if not train_keys:raise ValueError('No training examples')
    pools={}
    for key in train_keys:
        context=ln_context(data[(key,'train')][0])
        pools[key]=[p for p in (np.flatnonzero(context),np.flatnonzero(~context)) if len(p)]
    model=torch.nn.Sequential(torch.nn.Linear(len(NETWORK_FEATURES),64),torch.nn.ReLU(),
                              torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.002)
    history=[];best_loss=float('inf');best=None;best_epoch=0;started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();losses=[]
        for step in range(args.steps):
            key=train_keys[step%len(train_keys)];a,b=data[(key,'train')]
            available=pools[key]
            indices=np.concatenate([rng.choice(pool,args.batch//len(available),replace=True) for pool in available])
            x,y=a[indices].astype(np.float32),b[indices].astype(np.float32)
            mirror=rng.random(len(x))<.5
            x[mirror]=mirror_features(x[mirror],key);y[mirror]=mirror_features(y[mirror],key)
            loss=torch.nn.functional.softplus(-(model(torch.from_numpy(x))-model(torch.from_numpy(y)))).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
        validation=evaluate(model,data,'validation')
        mean_loss=float(np.mean([np.mean([m[k]['loss'] for k in ('ln','taps') if m[k]['pairs']]) for m in validation.values()])) if validation else float('inf')
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
            ln_groups=len({r['group'] for r in manifest if r['keys']==key and r['split']==s and r['ln_pairs']>=20}),
            pairs=sum(r['pairs'] for r in manifest if r['keys']==key and r['split']==s)) for s in ('train','validation','test')}
        validation=metrics['validation'].get(str(key),{})
        enabled=splits['train']['groups']>=8 and splits['validation']['groups']>=3 and splits['validation']['ln_groups']>=3 and validation.get('accuracy',0)>=.60 and validation.get('ln',{}).get('accuracy',0)>=.60
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
        method='101-feature 64/32 MLP; pairwise softplus; key and LN-context balanced batches; release permutations; all difficulties share song budgets; mirror augmentation; song/audio/set grouped split; cache ln-v2'),indent=2),encoding='utf-8')
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
