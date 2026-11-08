"""Distinguish authored piano layers from raw polyphonic pitch count.

Local development calibration only, on user-labeled Nova piano and Reference.
Known timing is explicitly supplied. Fit/calibration/later-song ranges match
the drum experiment; no time position or beat phase enters the classifier.
"""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, sections
from scripts.probe_mapper_adapter import event_similarity
from scripts.research_source_attack_classes import audio_features, fit_logistic


def snap_function(text, division=4):
    points=[]
    for line in sections(text)['TimingPoints']:
        fields=line.split(',')
        if len(fields)>=7 and fields[6]=='1' and float(fields[1])>0:
            points.append((float(fields[0]),float(fields[1])))
    points.sort()
    def snap(time):
        offset,beat=next((p for p in reversed(points) if p[0]<=time),points[0])
        return max(0,round(offset+round((time-offset)/(beat/division))*beat/division))
    return snap


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--notes',type=Path,help='Alternative piano transcription for a separately labelled experiment')
    cli.add_argument('--output',type=Path)
    args=cli.parse_args();root=args.root
    text=(root/'reference.osu').read_text(encoding='utf-8-sig');snap=snap_function(text)
    reference=[n for n in parse_osu(text)['notes'] if n.lane in (4,5)]
    counts=Counter(snap(n.start) for n in reference)
    note_path=args.notes or root/'detector-baseline/piano-notes.json'
    pitches=json.loads(note_path.read_text(encoding='utf-8'))
    groups=defaultdict(list)
    for n in pitches:groups[snap(n['start']-22.5)].append(n)
    times=np.asarray(sorted(groups))
    rate,samples=wavfile.read(root/'bundled/piano/preview.wav');samples=samples.astype(np.float32)/32768
    measured_times=np.asarray([np.median([n['start'] for n in groups[t]]) for t in times])
    acoustic=audio_features(samples,rate,measured_times)
    attributes=[]
    for t in times:
        local=groups[t];pitch=np.asarray([n['pitch'] for n in local]);strength=np.asarray([n['strength'] for n in local])
        attributes.append([len(local),len(set(pitch)),min(pitch),max(pitch),np.median(pitch),np.ptp(pitch),
            np.mean(strength),max(strength),sum(strength),sum(pitch<48),sum((pitch>=48)&(pitch<60)),sum(pitch>=72),
            np.median([n['end']-n['start'] for n in local]),np.ptp([n['start'] for n in local])])
    x=np.c_[acoustic,attributes]
    fit=times<162340;calibration=(times>=162340)&(times<188286)
    mean=x[fit].mean(axis=0);scale=np.maximum(.05,x[fit].std(axis=0))
    x=np.c_[np.clip((x-mean)/scale,-8,8),np.ones(len(x))]
    y=np.asarray([counts[t]>=2 for t in times],dtype=float)
    weights=fit_logistic(x[fit],y[fit]);score=expit(x@weights)
    thresholds=np.linspace(.1,.9,17)
    def count_events(mask,double):
        return [(float(t),) for i,t in enumerate(times) if mask[i] for _ in range(1+int(double[i]))]
    human=[(n.start,) for n in reference if 162340<=n.start<188286]
    threshold=max((event_similarity(human,count_events(calibration,score>=v),[40])['f1'],float(v)) for v in thresholds)[1]
    modes=dict(single=np.zeros(len(times),dtype=bool),always_double=np.ones(len(times),dtype=bool),
               pitch_count=np.asarray([len(groups[t])>=2 for t in times]),learned=score>=threshold)
    report=dict(confirmed_timing=True,audio_shift_ms=22.5,division=4,threshold=threshold,results={},
        enabled_in_application=False,notice='Same-song authored layer-count development, not independent musical quality or MIDI note transcription accuracy.')
    for name,a,b in [('fit',0,162340),('calibration',162340,188286),('later_sections',188286,283840)]:
        mask=(times>=a)&(times<b);human=[(n.start,) for n in reference if a<=n.start<b]
        report['results'][name]={mode:event_similarity(human,count_events(mask,double),[40]) for mode,double in modes.items()}
        matches=[i for i,t in enumerate(times) if mask[i] and counts[t]]
        report['results'][name]['layer_count_accuracy']={mode:sum(1+int(double[i])==counts[times[i]] for i in matches)/len(matches) for mode,double in modes.items()}
    output=args.output or root/'piano-layers';output.mkdir(exist_ok=True)
    from scripts.probe_mapper_adapter import sha
    report['candidate_sha256']=sha(note_path)
    report['reference_sha256']=sha(root/'reference.osu')
    np.savez_compressed(output/'development-model.npz',weights=weights,mean=mean,scale=scale,threshold=threshold)
    events=[dict(start=int(t),layers=1+int(score[i]>=threshold),score=float(score[i]),source_pitch_count=len(groups[t])) for i,t in enumerate(times)]
    (output/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
