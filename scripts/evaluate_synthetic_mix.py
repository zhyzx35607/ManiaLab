"""Known polyphonic synthesized events: diagnostic, not a natural-music benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.audio import analyze,analyze_deep
from manialab.arrange import parameters
from manialab.domain import Note,note_from
from manialab.separation import Separator
from scripts.evaluate_audio import LOADED_SOURCE,match_attacks


def synthesize(folder):
    rate=22050;duration=40;offset=.115;beat=.5
    stems={v:np.zeros(rate*duration,dtype=np.float64) for v in ('bass','lead','pad','drums')}
    events=[];rng=np.random.default_rng(20261108)
    def add(start,length,pitch,voice,amplitude):
        t=np.arange(round(length*rate))/rate;frequency=440*2**((pitch-69)/12)
        if voice=='bass':
            tone=np.sin(2*np.pi*frequency*t)+.2*np.sin(4*np.pi*frequency*t)
            envelope=np.minimum(1,t/.005)*np.minimum(1,(length-t)/.03)
        elif voice=='pad':
            tone=np.sin(2*np.pi*frequency*t)+.12*np.sin(4*np.pi*frequency*t)
            envelope=np.minimum(1,t/.025)*np.minimum(1,(length-t)/.06)
        else:
            tone=sum(np.sin(2*np.pi*frequency*h*t)/h**1.5 for h in (1,2,3))
            envelope=np.minimum(1,t/.005)*np.minimum(1,(length-t)/.03)*np.exp(-t/3)
        lo=round(start*rate);hi=min(rate*duration,lo+len(t))
        stems[voice][lo:hi]+=amplitude*tone[:hi-lo]*envelope[:hi-lo]
        events.append(Note(round(start*1000),round((start+length)*1000),pitch=pitch,source='pitch',voice=voice,id=f'{voice}-{len(events)}'))
    for bar,root in enumerate([48,45,41,43]*2):
        start=offset+bar*4
        for pitch in (root+12,root+16,root+19):add(start,3.75,pitch,'pad',.035)
        for when,pitch,length in ((0,root+24,.4),(.5,root+28,.4),(1,root+31,1.4),(2.5,root+28,.4),(3,root+26,.75)):
            add(start+when,length,pitch,'lead',.12)
        for number in range(8):
            add(start+number*beat,.4,root-12,'bass',.09)
            t=np.arange(round(rate*.15))/rate
            drum=np.sin(2*np.pi*(55*t+35*.025*(1-np.exp(-t/.025))))*np.exp(-t*25)
            if number%2:drum+=rng.normal(size=len(t))*.22*np.exp(-t*40)
            lo=round((start+number*beat)*rate)
            stems['drums'][lo:lo+len(t)]+=.13*drum
    # End with isolated sustained events after the polyphonic passage.
    add(34.115,2.75,72,'lead',.2);add(38.115,.75,60,'lead',.2)
    folder.mkdir(parents=True,exist_ok=True)
    for name,data in stems.items():wavfile.write(folder/(name+'.wav'),rate,(np.clip(data,-.98,.98)*32767).astype(np.int16))
    mixed=sum(stems.values());path=folder/'controlled-mix.wav'
    wavfile.write(path,rate,(np.clip(mixed,-.98,.98)*32767).astype(np.int16))
    (folder/'ground-truth.json').write_text(json.dumps([n.data() for n in events],indent=2),encoding='utf-8')
    return path,events


def pitched_match(predicted,reference,tail=False):
    predicted=[n for n in predicted if n.source=='pitch']
    pairs=[(i,j) for i,a in enumerate(predicted) for j,b in enumerate(reference)
           if a.pitch==b.pitch and abs(a.start-b.start)<=40 and (not tail or abs(a.end-b.end)<=80)]
    if pairs:
        i,j=zip(*pairs)
        graph=csr_matrix((np.ones(len(pairs)),(i,j)),shape=(len(predicted),len(reference)))
        count=int(np.count_nonzero(maximum_bipartite_matching(graph,perm_type='column')>=0))
    else:count=0
    return dict(predicted=len(predicted),reference=len(reference),matched=count,
                precision=round(count/max(1,len(predicted)),4),recall=round(count/max(1,len(reference)),4),
                f1=round(2*count/max(1,len(predicted)+len(reference)),4))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path('test-data/synthetic-mix'))
    parser.add_argument('--output',type=Path,default=Path('docs/synthetic-mix.json'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    path,reference=synthesize(args.work);results={};prefs=parameters()
    for mode in ('basic','deep'):
        folder=args.work/mode;folder.mkdir(exist_ok=True)
        print('Analyzing '+mode,flush=True)
        if mode=='deep':analysis=analyze_deep(path,folder,root/'assets/basic-pitch.onnx',prefs,Separator(root))
        else:analysis=analyze(path,folder,root/'assets/basic-pitch.onnx',prefs)
        (folder/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False),encoding='utf-8')
        notes=[note_from(n) for n in analysis['candidates']]
        results[mode]=dict(candidates=len(notes),cached=analysis['cached'],
            pitched_onsets=pitched_match(notes,reference),pitched_intervals=pitched_match(notes,reference,True),
            sustained_over_2s=dict(reference=sum(n.end-n.start>=2000 for n in reference),
                matched=pitched_match(notes,[n for n in reference if n.end-n.start>=2000],True)['matched']),
            timing={k:analysis['tempo'].get(k) for k in ('bpm','offset','stable','coverage','median_error_ms')})
        print(json.dumps(results[mode]),flush=True)
    report=dict(notice='Original deterministic synthesized fixture, not natural music or a player-quality benchmark. Includes overlapping chords, bass, harmonic melody, drums and two isolated ending events. No thresholds tuned after observation.',
        truth=dict(pitched_events=len(reference),duration_seconds=40,bpm=120,raw_offset_ms=115,
                   head_tolerance_ms=40,tail_tolerance_ms=80,exact_pitch_required=True),
        source_fingerprint=LOADED_SOURCE['sha256'],audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),modes=results)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
