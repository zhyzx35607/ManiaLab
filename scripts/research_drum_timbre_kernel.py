"""Matched-input nonlinear drum selection with calibration-only choices."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from manialab.timbre_kernel import fit,predict
from scripts.probe_mapper_adapter import sha,event_similarity
from scripts.research_source_attack_classes import CLASSES,audio_features,drum_event_counts,evaluate


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();root=args.root
    if args.output.exists():raise ValueError('Preserve earlier timbre experiments')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    source=root/'bundled/drums/preview.wav';reference_path=root/'reference.osu'
    prior=read(root/'drum-classifier/report.json');manifest=read(root/'manifest.json')
    if (sha(source)!=prior['source_audio_sha256'] or sha(reference_path)!=prior['reference_sha256']
            or sha(reference_path)!=manifest['charts']['reference']['chart_sha256']):raise ValueError('Benchmark identity changed')
    rate,samples=wavfile.read(source)
    if rate!=22050 or samples.ndim!=1 or samples.dtype!=np.int16:raise ValueError('Unexpected source PCM')
    candidates_path=root/'detector-baseline/drums-notes.json'
    times=np.asarray([n['start'] for n in read(candidates_path)],float)
    features=audio_features(samples.astype(np.float32)/32768,rate,times)
    with np.load(root/'drum-classifier/development-model.npz',allow_pickle=False) as saved:
        linear={k:saved[k] for k in saved.files}
    train_end,calibration_end=162340,188286
    chart_times=times-22.5;train=chart_times<train_end
    mean=features[train].mean(axis=0);scale=np.maximum(.05,features[train].std(axis=0))
    np.testing.assert_array_equal(mean,linear['mean']);np.testing.assert_array_equal(scale,linear['scale'])
    features=np.clip((features-mean)/scale,-8,8)
    linear_probability=expit(np.c_[features,np.ones(len(features))]@linear['weights'].T)
    reference=list(drum_event_counts(parse_osu(reference_path.read_text(encoding='utf-8-sig'))['notes']))
    probabilities=np.zeros((len(times),3));thresholds=[];selection=[]
    args.output.mkdir(parents=True)
    for c,kind in enumerate(CLASSES):
        labels=np.zeros(len(times));ambiguous=np.zeros(len(times),bool)
        for t,label in reference:
            if label!=kind or t>=train_end:continue
            distances=np.abs(chart_times-t);nearest=int(distances.argmin())
            if distances[nearest]<=40:ambiguous[distances<=40]=True;labels[nearest]=1
        valid=train&(~ambiguous|(labels>0));options=[];trained={};scores={}
        calibration=(chart_times>=train_end)&(chart_times<calibration_end)
        truth=[(t,) for t,label in reference if label==kind and train_end<=t<calibration_end]
        for width in (.5,1.,2.):
            model=fit(features[valid],labels[valid],width)
            probability=predict(features,model);trained[width]=model;scores[width]=probability
            for threshold in np.linspace(.1,.9,17):
                generated=[(t,) for t,p in zip(chart_times[calibration],probability[calibration]) if p>=threshold]
                score=event_similarity(truth,generated,[40])
                options.append(dict(width=width,threshold=float(threshold),score=score))
            print(json.dumps(dict(stage='fit',kind=kind,width=width,iterations=model['iterations'])),flush=True)
        best=max(options,key=lambda r:(r['score']['f1'],r['threshold'],-abs(np.log2(r['width']))))
        model=trained[best['width']];probabilities[:,c]=scores[best['width']];thresholds.append(best['threshold'])
        np.savez_compressed(args.output/(kind+'-model.npz'),**model,mean=mean,scale=scale,threshold=best['threshold'])
        selection.append(dict(kind=kind,chosen=best,calibration=options,fit_positive=int(labels[valid].sum()),fit_candidates=int(valid.sum())))
    events=[dict(start=round(t),kind=kind,score=float(probabilities[i,c])) for i,t in enumerate(chart_times)
            for c,kind in enumerate(CLASSES) if probabilities[i,c]>=thresholds[c]]
    (args.output/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    (args.output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    segments={name:{'kernel':evaluate(reference,chart_times,probabilities,thresholds,a,b),
                    'linear':evaluate(reference,chart_times,linear_probability,linear['thresholds'],a,b)}
              for name,a,b in [('fit',0,train_end),('calibration',train_end,calibration_end),('later_sections',calibration_end,len(samples)/rate*1000)]}
    for part in segments:
        for kind in CLASSES:
            if segments[part]['linear'][kind]!=prior['segments'][part][kind]:raise ValueError('Original linear baseline did not reproduce exactly')
    result=dict(source_audio_sha256=sha(source),reference_sha256=sha(reference_path),candidates_sha256=sha(candidates_path),
        script_sha256=sha(__file__),kernel_sha256=sha('manialab/timbre_kernel.py'),protocol_sha256=sha('docs/drum-timbre-kernel-protocol.md'),
        selection=selection,segments=segments,events_sha256=sha(args.output/'events.json'),enabled_in_application=False,
        notice='Same development song, repeated sections; fourth column includes hi-hat/tom/other. This predicts author selection, not exhaustive physical drum truth.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(later=segments['later_sections'],chosen=[s['chosen'] for s in selection])),flush=True)


if __name__=='__main__':main()
