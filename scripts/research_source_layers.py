"""Learn authored source selection/layer counts from frozen acoustic evidence.

This is same-song development. No beat grid or author timing enters features;
author onset/layer choices are supervision, not exhaustive acoustic truth.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha,event_similarity
from scripts.research_source_attack_classes import audio_features
from manialab.domain import parse_osu


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--role',choices=['piano','other'],required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();root=args.root;output=args.output
    if (output/'report.json').exists():raise ValueError('Preserve completed experiment')
    import numpy as np
    from scipy.io import wavfile
    from scipy.optimize import minimize,linear_sum_assignment
    from scipy.special import softmax,logsumexp
    folder=root/(args.role+'-yourmt3-full')
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    report=read(folder/'report.json');raw=read(folder/'notes.json');audit=read(folder/'release-audit.json')
    if sha(folder/'notes.json')!=audit['notes_sha256']:raise ValueError('Acoustic evidence changed')
    provenance=read(folder/'release-provenance.json')
    flags={p['note_index']:set(p['endpoint_evidence']) for p in provenance}
    groups=[]
    for index,note in sorted(enumerate(raw),key=lambda pair:pair[1]['onset']):
        if note['is_drum'] or note['program'] in (100,101):continue
        if not groups or note['onset']-raw[groups[-1][0]]['onset']>.03:groups.append([])
        groups[-1].append(index)
    times=np.asarray([np.median([raw[i]['onset'] for i in g])*1000-22.5 for g in groups])
    voices=('piano',) if args.role=='piano' else ('other','bass','guitar')
    signal=None;source_hashes={}
    for voice in voices:
        path=root/f'bundled/{voice}/preview.wav';rate,y=wavfile.read(path)
        if rate!=22050 or y.ndim!=1 or y.dtype!=np.int16:raise ValueError('Source PCM changed')
        source_hashes[str(path)]=sha(path)
        signal=y.astype(np.float32)/32768 if signal is None else signal+y.astype(np.float32)/32768
    if source_hashes!=report['identity']['sources']:raise ValueError('Transcription and acoustic sources differ')
    acoustic=audio_features(signal,rate,times+22.5)
    attributes=[]
    for group in groups:
        pitches=[raw[i]['pitch'] for i in group];durations=[raw[i]['offset']-raw[i]['onset'] for i in group]
        attributes.append([len(group),len(set(pitches)),min(pitches),max(pitches),np.median(pitches),np.ptp(pitches),
            *np.percentile(durations,[0,25,50,75,100]),
            sum('forced_segment_boundary' in flags[i] for i in group)/len(group),
            *[sum(lo<=raw[i]['program']<hi for i in group) for lo,hi in
                ((0,8),(8,24),(24,32),(32,40),(40,56),(56,80),(80,100))]])
    x=np.c_[acoustic,np.asarray(attributes)]
    reference_path=root/'reference.osu';manifest=read(root/'manifest.json')
    if sha(reference_path)!=manifest['charts']['reference']['chart_sha256']:raise ValueError('Author reference changed')
    lanes=range(4,6) if args.role=='piano' else range(6,10)
    reference=[n for n in parse_osu(reference_path.read_text(encoding='utf-8-sig'))['notes'] if n.lane in lanes]
    counts=Counter(n.start for n in reference);max_layers=2 if args.role=='piano' else 4
    if max(counts.values())>max_layers:raise ValueError('Reference exceeds source layer capacity')
    regions=[('fit',0,162340),('calibration',162340,188286),('later_sections',188286,283840)]
    labels=np.zeros(len(times),dtype=int)
    for _,left,right in regions:
        target=sorted(t for t in counts if left<=t<right);indices=np.flatnonzero((times>=left)&(times<right))
        distances=np.abs(np.asarray(target)[:,None]-times[indices][None,:])
        costs=np.c_[np.where(distances<=40,distances/(40*(len(target)+1)),1e6),np.ones((len(target),len(target)))]
        rows,cols=linear_sum_assignment(costs)
        for i,j in zip(rows,cols):
            if j<len(indices) and distances[i,j]<=40:labels[indices[j]]=counts[target[i]]
    fit=times<162340;mean=x[fit].mean(axis=0);scale=np.maximum(.05,x[fit].std(axis=0))
    x=np.c_[np.clip((x-mean)/scale,-8,8),np.ones(len(x))]
    present=np.unique(labels[fit]);train=x[fit];target=np.searchsorted(present,labels[fit])
    def objective(flat):
        weights=flat.reshape(x.shape[1],len(present));logits=train@weights;probability=softmax(logits,axis=1)
        loss=np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(target)),target])+.005*(weights[:-1]**2).sum()
        probability[np.arange(len(target)),target]-=1
        gradient=train.T@probability/len(target);gradient[:-1]+=.01*weights[:-1]
        return loss,gradient.ravel()
    optimized=minimize(objective,np.zeros(x.shape[1]*len(present)),jac=True,method='L-BFGS-B',options={'maxiter':500})
    if not optimized.success:raise ValueError('Layer model did not converge: '+optimized.message)
    weights=optimized.x.reshape(x.shape[1],len(present));logits=x@weights
    events=lambda chosen,left,right:[(float(t),) for t,c in zip(times,chosen) if left<=t<right for _ in range(int(c))]
    calibration=[(n.start,) for n in reference if 162340<=n.start<188286]
    options=[]
    for bias in np.linspace(-1,1,21):
        chosen=present[(logits+bias*present).argmax(axis=1)]
        score=event_similarity(calibration,events(chosen,162340,188286),[40])
        options.append((score['f1'],-abs(float(bias)),float(bias)))
    bias=max(options)[2]
    modes=dict(single=np.ones(len(groups),dtype=int),always_double=np.full(len(groups),2),
        pitch_count=np.asarray([min(max_layers,len(g)) for g in groups]),
        learned=present[(logits+bias*present).argmax(axis=1)])
    results={}
    for name,left,right in regions:
        human=[(n.start,) for n in reference if left<=n.start<right]
        results[name]={mode:event_similarity(human,events(chosen,left,right),[40]) for mode,chosen in modes.items()}
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'development-model.npz',weights=weights,mean=mean,scale=scale,classes=present,bias=bias)
    (output/'events.json').write_text(json.dumps([dict(start_ms=float(t),layers=int(count),source_note_indices=g)
        for t,count,g in zip(times,modes['learned'],groups)],indent=2),encoding='utf-8')
    result=dict(role=args.role,candidate_identity=report['identity'],notes_sha256=sha(folder/'notes.json'),
        reference_sha256=sha(reference_path),runner_sha256=sha(__file__),features_sha256=sha('scripts/research_source_attack_classes.py'),
        bias=bias,fit_class_counts=dict(Counter(str(c) for c in labels[fit])),results=results,
        enabled_in_application=False,
        protocol=['Fit 0-162340ms, calibrate only 162340-188286ms, evaluate later 188286-283840ms.',
            'Same development song with repeated musical material; no independent-song claim.',
            'Acoustic timbre, model pitch/program/duration evidence only; no author grid, time position or note density feature.',
            'Zero layers means author selection omission, not absence of physical sound.',
            'No new attacks, LN lengths or columns are generated by this experiment.'])
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(role=args.role,bias=bias,later=results['later_sections'])),flush=True)


if __name__=='__main__':main()
