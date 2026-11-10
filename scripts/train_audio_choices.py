"""Learn authored onset/sustain preferences from paired audio and maps.

Weak labels mean 'matches this author's chart', not 'is a real instrument note'.
No model from this experiment is enabled by default. All LN ratios are used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu

VOICES=('drums','bass','vocals','other','guitar','piano','mix')
FEATURES=['strength','log_duration','pitch','pitch_squared','source_pitch','local_density','voice_local_density',
          'time_in_song','sin_beat','cos_beat','sin_half_beat','cos_half_beat','keycount']+list(VOICES)


def normalize(text):
    return re.sub(r'[^\w]','',text.casefold())


def dataset(root, snapshot):
    records=[json.loads(l) for l in (snapshot/'manifest-snapshot.jsonl').read_text(encoding='utf-8').splitlines()]
    by_hash={r['sha256']:r for r in records if r.get('eligible') and r.get('reference_role')=='expression_and_layout'}
    by_id={str(r.get('embedded_beatmap_id') or r['beatmap_id']):r for r in by_hash.values()}
    samples=[];skipped=[]
    for path in sorted(root.glob('*/*/deep/analysis.json')):
        folder=path.parent.parent
        if not (folder/'reference.osu').exists():continue
        raw=(folder/'reference.osu').read_bytes();sha=hashlib.sha256(raw).hexdigest()
        record=by_hash.get(sha)
        chart=parse_osu(raw.decode('utf-8-sig'))
        newline_equivalent=False
        if record is None:
            candidate=by_id.get(chart['metadata'].get('BeatmapID'))
            if candidate:
                pinned=(snapshot/'maps'/f"{candidate['beatmap_id']}.osu").read_bytes()
                if hashlib.sha256(pinned).hexdigest()!=candidate['sha256']:
                    raise ValueError('Snapshot chart hash mismatch')
                if pinned.replace(b'\r\n',b'\n')==raw.replace(b'\r\n',b'\n'):
                    record=candidate;newline_equivalent=True
        if not record:
            skipped.append(dict(folder=str(folder),reason='not_in_verified_expression_snapshot'));continue
        a=json.loads(path.read_text(encoding='utf-8'))
        clip=a.get('research_clip',{})
        expected=a.get('separation',{}).get('audio_sha256')
        source_hash=clip.get('original_audio_sha256') if clip else expected
        # A research folder can contain both a re-encoded archive original and
        # the actual Opus analyzed by the detector. Match the recorded bytes,
        # never whichever filename the filesystem happens to enumerate first.
        audio=next((p for p in folder.iterdir() if p.suffix.lower() in ('.mp3','.opus','.ogg','.wav','.flac')
                    and hashlib.sha256(p.read_bytes()).hexdigest()==source_hash),None)
        if not audio:
            skipped.append(dict(folder=str(folder),reason='analysis_audio_hash_mismatch'));continue
        audio_sha=source_hash
        analyzed=path.parent/'clip.wav' if clip else audio
        if (not analyzed.is_file() or expected!=hashlib.sha256(analyzed.read_bytes()).hexdigest() or
                (clip and clip.get('original_audio_sha256')!=audio_sha)):
            skipped.append(dict(folder=str(folder),reason='analysis_audio_hash_mismatch'));continue
        red=[]
        for line in chart['timing']:
            p=line.split(',')
            if len(p)>=7 and p[6]=='1' and float(p[1])>0:red.append((float(p[0]),float(p[1])))
        if not red:continue
        clip_start=clip.get('start_ms',0);edge=clip.get('excluded_edge_ms',0)
        notes=[n for n in a['candidates'] if edge<=n['start']<a['duration']*1000-edge]
        all_times=np.sort([n['start'] for n in notes])
        voice_times={v:np.sort([n['start'] for n in notes if n.get('voice','mix')==v]) for v in VOICES}
        reference=chart['notes'];times=np.sort(list({n.start for n in reference}))
        holds=[n for n in reference if n.hold]
        hold_times=np.array([n.start for n in holds]);hold_ends=np.array([n.end for n in holds])
        x=[];labels=[]
        for n in notes:
            start=n['start']+clip_start-22.5;end=n['end']+clip_start-22.5
            _,beat=next(((o,b) for o,b in reversed(red) if o<=start),red[0])
            origin=next((o for o,b in reversed(red) if o<=start),red[0][0])
            phase=(start-origin)/beat*2*np.pi
            voice=n.get('voice','mix');local=voice_times[voice]
            density=np.searchsorted(all_times,n['start']+1000)-np.searchsorted(all_times,n['start']-1000)
            own=np.searchsorted(local,n['start']+1000)-np.searchsorted(local,n['start']-1000)
            pitch=(n['pitch']-60)/36
            x.append([n['strength'],np.log1p((end-start)/500),pitch,pitch*pitch,float(n['source']=='pitch'),
                      np.log1p(density/2),np.log1p(own/2),n['start']/max(1,a['duration']*1000),
                      np.sin(phase),np.cos(phase),np.sin(phase*2),np.cos(phase*2),chart['keys']/10,
                      *[float(voice==v) for v in VOICES]])
            hit=bool(len(times) and np.min(np.abs(times-start))<=40)
            sustain=bool(len(holds) and np.any((np.abs(hold_times-start)<=40)&(np.abs(hold_ends-end)<=80)))
            # Sustain is evaluated among plausible pitched intervals, not
            # rewarded for merely distinguishing drums/taps from pitch notes.
            labels.append([hit,float(sustain) if n['source']=='pitch' and end-start>=100 else float('nan')])
        song=normalize(chart['metadata'].get('Artist',''))+':'+normalize(chart['metadata'].get('Title',''))
        samples.append(dict(x=np.array(x),y=np.array(labels,dtype=float),song=song,audio_sha256=audio_sha,
            map_sha256=sha,snapshot_sha256=record['sha256'],newline_equivalent=newline_equivalent,
            clip=clip or None,
            beatmap_id=chart['metadata'].get('BeatmapID'),keys=chart['keys'],
            ln_ratio=sum(n.hold for n in reference)/len(reference),source=str(path),
            status=record.get('official_status') or record.get('approved')))
    return samples,skipped


def auc(y,score):
    positive=int(y.sum());negative=len(y)-positive
    if not positive or not negative:return None
    return round(float((rankdata(score)[y>0].sum()-positive*(positive+1)/2)/(positive*negative)),5)


def fit(x,y,weights):
    def loss(w):
        z=x@w
        value=np.sum(weights*(np.logaddexp(0,z)-y*z))+.005*np.sum(w[1:]**2)
        gradient=x.T@(weights*(expit(z)-y));gradient[1:]+=.01*w[1:]
        return value,gradient
    result=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-11})
    return result.x,dict(converged=bool(result.success),iterations=int(result.nit))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('test-data'))
    p.add_argument('--snapshot',type=Path,default=Path('test-data/corpus-expanded/snapshot-verified'))
    p.add_argument('--output',type=Path,default=Path('test-data/audio-choice-training'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    samples,skipped=dataset(args.root,args.snapshot)
    # Union both normalized song identities and exact audio hashes before
    # splitting: alternate encodings and multiple difficulties stay together.
    parent=list(range(len(samples)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,s in enumerate(samples):
        for key in ('song:'+s['song'],'audio:'+s['audio_sha256']):
            if key in seen:parent[find(i)]=find(seen[key])
            seen[key]=i
    identities={find(i):min(s['song'] for j,s in enumerate(samples) if find(j)==find(i)) for i in range(len(samples))}
    groups=sorted(identities,key=lambda g:hashlib.sha256(('audio-choice-20261108:'+identities[g]).encode()).hexdigest())
    cutoff=max(1,int(len(groups)*.75));training=set(groups[:cutoff])
    train=[i for i in range(len(samples)) if find(i) in training];test=[i for i in range(len(samples)) if find(i) not in training]
    if len(train)<6 or len(test)<3:raise ValueError('Not enough verified paired songs for a grouped experiment')
    mean=np.concatenate([samples[i]['x'] for i in train]).mean(axis=0)
    std=np.maximum(np.concatenate([samples[i]['x'] for i in train]).std(axis=0),.05)
    transformed=[np.column_stack([np.ones(len(s['x'])),(s['x']-mean)/std]) for s in samples]
    x=np.concatenate([transformed[i] for i in train]);y=np.concatenate([samples[i]['y'] for i in train])
    heads={};models={}
    for k,name in enumerate(('authored_onset','authored_sustain')):
        mask=np.isfinite(y[:,k])
        head_weights=[]
        for i in train:
            valid=np.isfinite(samples[i]['y'][:,k])
            head_weights.extend(np.full(int(valid.sum()),1/max(1,int(valid.sum()))/sum(find(j)==find(i) for j in train)))
        head_weights=np.asarray(head_weights);head_weights/=head_weights.sum()
        model,diagnostics=fit(x[mask],y[mask,k],head_weights);models[name]=model.tolist()
        # Beat phase alone can make chart matches easier without learning
        # instrumentation. Report a no-phase ablation on the same song split.
        columns=[j for j in range(x.shape[1]) if j not in (9,10,11,12)]
        no_phase,_=fit(x[mask][:,columns],y[mask,k],head_weights)
        rows=[]
        for i in test:
            valid=np.isfinite(samples[i]['y'][:,k])
            target=samples[i]['y'][valid,k];pred=expit(transformed[i][valid]@model)
            baseline=samples[i]['x'][valid,0] if k==0 else samples[i]['x'][valid,1]
            rows.append(dict(beatmap_id=samples[i]['beatmap_id'],candidates=len(target),positives=int(target.sum()),
                model_auc=auc(target,pred),baseline_auc=auc(target,baseline),
                no_phase_auc=auc(target,expit(transformed[i][valid][:,columns]@no_phase)),
                brier=float(np.mean((pred-target)**2)) if len(target) else None))
        valid=[r for r in rows if r['model_auc'] is not None]
        heads[name]=dict(**diagnostics,training_candidates=int(mask.sum()),heldout_songs=rows,
            macro_auc=round(float(np.mean([r['model_auc'] for r in valid])),5) if valid else None,
            macro_no_phase_auc=round(float(np.mean([r['no_phase_auc'] for r in valid])),5) if valid else None,
            macro_baseline_auc=round(float(np.mean([r['baseline_auc'] for r in valid])),5) if valid else None)
    manifest=[{k:v for k,v in s.items() if k not in ('x','y')}|dict(split='train' if i in train else 'test',candidates=len(s['x']),
             group=hashlib.sha256(identities[find(i)].encode()).hexdigest()) for i,s in enumerate(samples)]
    report=dict(revision=2,enabled_in_application=False,purpose=__doc__,
        protocol=dict(split='song identity and exact audio SHA256 union, fixed seed 75/25',
                      training_weight='equal song groups, equal charts within each song',
                      evaluation='macro average over held-out charts with both labels present',
                      sustain_candidates='source=pitch and duration >= 100ms',
                      onset_tolerance_ms=40,tail_tolerance_ms=80),
        counts=dict(songs=len(groups),charts=len(samples),train_charts=len(train),test_charts=len(test),
                    candidates=sum(len(s['x']) for s in samples),ln_bands=dict(Counter('ln' if s['ln_ratio']>=.5 else 'hybrid' if s['ln_ratio']>=.15 else 'tap' for s in samples))),
        heads=heads,manifest=manifest,skipped=skipped,
        limitations=['Human chart choices are weak labels; no instrument attribution truth.',
                    'Sustain evaluation includes only pitched intervals at least 100ms long.',
                    '40ms onset / 80ms tail matching can label multiple simultaneous voices positive.',
                    'Existing audio development material; grouped holdout is not an untouched product benchmark.',
                    'This is a small arrangement preference model, not new source-separation or note-transcription weights.'])
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'model.json').write_text(json.dumps(dict(features=FEATURES,mean=mean.tolist(),std=std.tolist(),heads=models)),encoding='utf-8')
    print(json.dumps(dict(counts=report['counts'],heads={k:{q:v[q] for q in ['converged','macro_auc','macro_baseline_auc']} for k,v in heads.items()})),flush=True)


if __name__=='__main__':main()
