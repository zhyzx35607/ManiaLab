"""Learn single/double snare expression from confirmed source-reference columns.

Candidates remain those selected by the existing frozen drum classifier.
Fit/calibration/later partitions follow the original same-song protocol.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_source_attack_classes import audio_features
from scripts.probe_mapper_adapter import sha,event_similarity
from manialab.domain import parse_osu


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output',type=Path,default=Path('test-data/sakura-benchmark/snare-layers-v1'))
    args=cli.parse_args();root=args.root
    if args.output.exists():raise ValueError('Preserve completed layer experiments')
    import numpy as np
    from scipy.io import wavfile
    from scipy.optimize import minimize,linear_sum_assignment
    from scipy.special import softmax,logsumexp
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    reference=root/'reference.osu';manifest=read(root/'manifest.json')
    source=root/'bundled/drums/preview.wav';frozen=root/'drum-classifier'
    identity=read(frozen/'report.json')
    if (sha(reference)!=manifest['charts']['reference']['chart_sha256'] or
        sha(source)!=identity['source_audio_sha256']):raise ValueError('Frozen source/reference changed')
    events=[(i,e) for i,e in enumerate(read(frozen/'events.json')) if e['kind']=='snare']
    times=np.asarray([e['start'] for _,e in events],float)
    rate,signal=wavfile.read(source)
    if rate!=22050 or signal.ndim!=1 or signal.dtype!=np.int16:raise ValueError('Source PCM differs')
    features=audio_features(signal.astype(np.float32)/32768,rate,times+22.5)
    reference_notes=[n for n in parse_osu(reference.read_text(encoding='utf-8-sig'))['notes'] if n.lane in (1,2)]
    counts=Counter(n.start for n in reference_notes)
    if max(counts.values())>2:raise ValueError('Snare source capacity changed')
    regions=[('fit',0,162340),('calibration',162340,188286),('later_sections',188286,283840)]
    labels=np.zeros(len(times),int)
    for _,left,right in regions:
        target=sorted(t for t in counts if left<=t<right);indices=np.flatnonzero((times>=left)&(times<right))
        distances=np.abs(np.asarray(target)[:,None]-times[indices][None,:])
        costs=np.c_[np.where(distances<=40,distances/(40*(len(target)+1)),1e6),np.ones((len(target),len(target)))]
        rows,cols=linear_sum_assignment(costs)
        for i,j in zip(rows,cols):
            if j<len(indices) and distances[i,j]<=40:labels[indices[j]]=counts[target[i]]
    fit=times<162340;mean=features[fit].mean(axis=0);scale=np.maximum(.05,features[fit].std(axis=0))
    x=np.c_[np.clip((features-mean)/scale,-8,8),np.ones(len(features))]
    classes=np.unique(labels[fit]);target=np.searchsorted(classes,labels[fit]);train=x[fit]
    def objective(flat):
        weights=flat.reshape(x.shape[1],len(classes));logits=train@weights;prob=softmax(logits,axis=1)
        loss=np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(target)),target])+.005*(weights[:-1]**2).sum()
        prob[np.arange(len(target)),target]-=1
        gradient=train.T@prob/len(target);gradient[:-1]+=.01*weights[:-1]
        return loss,gradient.ravel()
    optimized=minimize(objective,np.zeros(x.shape[1]*len(classes)),jac=True,method='L-BFGS-B',options={'maxiter':500})
    if not optimized.success:raise ValueError('Snare layer fit did not converge')
    weights=optimized.x.reshape(x.shape[1],len(classes));logits=x@weights
    predictions=lambda count,left,right:[(float(t),) for t,c in zip(times,count) if left<=t<right for _ in range(int(c))]
    calibration=[(n.start,) for n in reference_notes if 162340<=n.start<188286];options=[]
    for bias in np.linspace(-1,1,21):
        count=classes[(logits+bias*classes).argmax(axis=1)]
        score=event_similarity(calibration,predictions(count,162340,188286),[40])
        options.append((score['f1'],-abs(float(bias)),float(bias)))
    bias=max(options)[2];chosen=classes[(logits+bias*classes).argmax(axis=1)]
    results={}
    for name,left,right in regions:
        target=[(n.start,) for n in reference_notes if left<=n.start<right]
        results[name]={mode:event_similarity(target,predictions(count,left,right),[40]) for mode,count in
            dict(single=np.ones(len(times),int),always_double=np.full(len(times),2),learned=chosen).items()}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'model.npz',weights=weights,mean=mean,scale=scale,classes=classes,bias=bias)
    (args.output/'events.json').write_text(json.dumps([dict(source_event_index=i,start_ms=float(t),layers=int(count))
        for (i,_),t,count in zip(events,times,chosen)],indent=2),encoding='utf-8')
    report=dict(runner_sha256=sha(__file__),feature_source_sha256=sha('scripts/research_source_attack_classes.py'),
        source_sha256=sha(source),reference_sha256=sha(reference),candidates_sha256=sha(frozen/'events.json'),
        model_sha256=sha(args.output/'model.npz'),fit_class_counts=dict(Counter(str(c) for c in labels[fit])),
        bias=bias,results=results,enabled_in_application=False,
        protocol=['Same-song front fitted, middle calibrated, later reported; repeated musical material remains correlated.',
            'Existing classifier snare candidates only. No new onset times, BPM, bookmarks or desired density feature.',
            'Predict 0/1/2 authored objects. Double snare is expression, not proof of two physical drum sources.',
            'No hi-hat reclassification, LN durations or final performance lanes are generated.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
