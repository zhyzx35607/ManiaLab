"""Learn source attack selection/types on an explicitly annotated reference.

Sakura Grass is a development case, not unseen-song validation. Fit, threshold
calibration and later-section evaluation are separated in time. Features use
only local drum audio; chart timing, beat position and note density are absent.
Snare double notes become one physical class event, preserving chart counts
separately. No resulting model is enabled in the application.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile
from scipy.optimize import minimize
from scipy.signal import stft
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from scripts.probe_mapper_adapter import event_similarity, sha

CLASSES = ('kick','snare','hi-hat')
DRUM_PITCH_GROUPS=({35,36},{38,40},{2,37},{22,42,44},{26,46},
    {51,53,59},{49,52,55,57},{41,43,45,47,48,50,58})


def instrument_features(events,times):
    """Frozen acoustic predictions around an existing attack, no chart inputs."""
    columns=[]
    for pitches in DRUM_PITCH_GROUPS:
        onsets=np.asarray([n['onset']*1000 for n in events if n['is_drum'] and n['pitch'] in pitches])
        delta=onsets[None,:]-np.asarray(times)[:,None]
        for lo,hi in ((-50,-10),(-10,30),(30,80)):
            columns.append(((delta>=lo)&(delta<hi)).any(axis=1).astype(float))
    return np.asarray(columns).T


def drum_event_counts(notes):
    """Merge authored snare chords as one sound, retain their note multiplicity."""
    return Counter((n.start, CLASSES[0 if n.lane==0 else 1 if n.lane<3 else 2])
                   for n in notes if 0<=n.lane<4)


def audio_features(samples, rate, times):
    frequencies, frame_times, spectrum = stft(samples, rate, nperseg=1024, noverlap=896)
    power = abs(spectrum)**2
    limits = np.geomspace(35, rate/2, 33)
    bands = np.asarray([power[(frequencies>=a)&(frequencies<b)].sum(axis=0) for a,b in zip(limits,limits[1:])])
    features=[]
    for time in times:
        at=time/1000
        vectors=[]
        for lo,hi in ((-.045,-.005),(-.005,.025),(.025,.070),(.070,.140)):
            selected=(frame_times>=at+lo)&(frame_times<at+hi)
            vectors.append(bands[:,selected].mean(axis=1) if selected.any() else np.zeros(32))
        vectors=np.asarray(vectors)
        energy=vectors.sum(axis=1)
        # Timbre shape plus attack/sustain changes. Normalizing by this local
        # event prevents absolute loudness being the only usable discriminator.
        scale=max(float(energy.max()),1e-10)
        features.append(np.r_[np.log1p(vectors/scale*100).ravel(),
            np.log1p(np.maximum(0,vectors[1]-vectors[0])/scale*100),np.log1p(energy*1e6)])
    return np.asarray(features)


def fit_logistic(x, y):
    positive=max(1,int(y.sum()));negative=max(1,len(y)-positive)
    weight=np.where(y>0,len(y)/(2*positive),len(y)/(2*negative))
    def loss(w):
        z=x@w
        penalty=.01*(w[:-1]@w[:-1])/2
        objective=np.mean(weight*(np.logaddexp(0,z)-y*z))+penalty
        gradient=x.T@(weight*(expit(z)-y))/len(y)
        gradient[:-1]+=.01*w[:-1]
        return objective,gradient
    result=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':400})
    if not result.success:raise ValueError('Attack classifier did not converge: '+str(result.message))
    return result.x


def evaluate(reference, times, probability, thresholds, start, end):
    result={}
    for index,kind in enumerate(CLASSES):
        human=[(t,) for t,c in reference if c==kind and start<=t<end]
        generated=[(t,) for t,p in zip(times,probability[:,index]) if start<=t<end and p>=thresholds[index]]
        result[kind]=event_similarity(human,generated,[40])
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--drum-activations',type=Path,help='Optional frozen five-class audio model evidence; no chart conditioning')
    cli.add_argument('--instrument-probe',type=Path,help='Optional full-song frozen YourMT3 drum probe')
    cli.add_argument('--output',type=Path)
    args=cli.parse_args();root=args.root
    if args.drum_activations and args.instrument_probe:raise ValueError('Compare one frozen model at a time')
    source=root/'bundled/drums/preview.wav'
    rate,samples=wavfile.read(source);samples=samples.astype(np.float32)/32768
    notes=parse_osu((root/'reference.osu').read_text(encoding='utf-8-sig'))['notes']
    class_counts=drum_event_counts(notes)
    reference=list(class_counts)
    candidates=json.loads((root/'detector-baseline/drums-notes.json').read_text(encoding='utf-8'))
    times=np.asarray([n['start'] for n in candidates],dtype=float)
    x=audio_features(samples,rate,times)
    pretrained=None
    if args.drum_activations:
        with np.load(args.drum_activations,allow_pickle=False) as data:
            activity=data['activations'][0]
        if activity.ndim!=2 or activity.shape[1]!=5 or not np.isfinite(activity).all():
            raise ValueError('Expected finite five-class drum activations at 100fps')
        if abs(len(activity)/100-len(samples)/rate)>.03:raise ValueError('Drum activation clock mismatch')
        local=[]
        for t in times:
            center=round(t/10);values=[]
            for lo,hi in ((-5,-1),(-1,3),(3,8)):
                part=activity[max(0,center+lo):min(len(activity),center+hi)]
                values.extend(part.max(axis=0) if len(part) else np.zeros(5))
            local.append(values)
        x=np.c_[x,np.asarray(local)]
        pretrained=dict(activations_sha256=sha(args.drum_activations),features='five-class local max in three relative windows')
    if args.instrument_probe:
        folder=args.instrument_probe
        inference=json.loads((folder/'report.json').read_text())
        audit=json.loads((folder/'release-audit.json').read_text())
        identity=inference['identity']
        if (list(identity['sources'].values())!=[sha(source)] or identity['start_ms']!=0 or
            abs(identity['seconds']-len(samples)/rate)>.03 or not audit['decoded_notes_equal'] or
            sha(folder/'notes.json')!=audit['notes_sha256'] or sha(folder/'tokens.npz')!=audit['tokens_sha256']):
            raise ValueError('Frozen drum transcription identity/clock changed')
        events=json.loads((folder/'notes.json').read_text())
        x=np.c_[x,instrument_features(events,times)]
        pretrained=dict(identity=identity,notes_sha256=audit['notes_sha256'],
            features='eight fixed GM drum families, binary onset presence in three relative windows')
    chart_times=times-22.5
    train_end,calibration_end=162340,188286
    fit=chart_times<train_end;calibration=(chart_times>=train_end)&(chart_times<calibration_end)
    mean=x[fit].mean(axis=0);scale=np.maximum(.05,x[fit].std(axis=0))
    x=np.c_[np.clip((x-mean)/scale,-8,8),np.ones(len(x))]
    weights=[];thresholds=[];eligible=[]
    for index,kind in enumerate(CLASSES):
        y=np.zeros(len(times));ambiguous=np.zeros(len(times),dtype=bool)
        for t,c in reference:
            if c!=kind or t>=train_end:continue
            distances=abs(chart_times-t)
            nearest=int(distances.argmin())
            if distances[nearest]<=40:
                ambiguous[distances<=40]=True;y[nearest]=1
        valid=fit&(~ambiguous|(y>0))
        w=fit_logistic(x[valid],y[valid]);weights.append(w)
        score=expit(x@w)
        # Choose thresholds only on the middle calibration section.
        candidates_t=np.linspace(.1,.9,17)
        values=[]
        human=[(t,) for t,c in reference if c==kind and train_end<=t<calibration_end]
        for threshold in candidates_t:
            predicted=[(t,) for t,p in zip(chart_times[calibration],score[calibration]) if p>=threshold]
            metric=event_similarity(human,predicted,[40])
            values.append((metric['f1'],float(threshold)))
        threshold=max(values)[1];thresholds.append(threshold)
        eligible.append(dict(kind=kind,fit_positive=int(y[valid].sum()),fit_candidates=int(valid.sum()),threshold=threshold))
    weights=np.asarray(weights);probability=expit(x@weights.T)
    output=args.output or root/'drum-classifier'
    if (output/'report.json').exists():raise ValueError('Choose a fresh output directory to preserve evidence')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'development-model.npz',weights=weights,mean=mean,scale=scale,thresholds=thresholds)
    rows=[dict(start=round(t),kind=kind,score=float(probability[i,c])) for i,t in enumerate(chart_times)
          for c,kind in enumerate(CLASSES) if probability[i,c]>=thresholds[c]]
    (output/'events.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    report=dict(source_audio_sha256=sha(source),reference_sha256=sha(root/'reference.osu'),
        frozen_audio_evidence=pretrained,
        train_end_ms=train_end,calibration_end_ms=calibration_end,fit=eligible,raw_drum_objects=sum(class_counts.values()),
        physical_reference_events=len(reference),snare_double_groups=sum(v>1 for v in class_counts.values()),
        segments={name:evaluate(reference,chart_times,probability,thresholds,a,b) for name,a,b in
            [('fit',0,train_end),('calibration',train_end,calibration_end),('later_sections',calibration_end,len(samples)/rate*1000)]},
        enabled_in_application=False,
        notice='Same-song development only. Repeated musical material may overlap. Learned authored source attack selection, not exhaustive drum transcription. Occasional tom/other labels remain uncertain.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
