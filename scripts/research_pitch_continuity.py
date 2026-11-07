"""Research sustained pitch activation across secondary onset detections.

Continuity is an alternate expression, not proof that repeated articulation
belongs to a single instrument. This script does not change the application.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import note_from,parse_osu
from manialab.arrange import parameters,select_notes
from manialab.timing import align_candidates,chart_clock_notes
from manialab.voices import apply_phrase_evidence
from scripts.evaluate_synthetic_mix import pitched_match
from scripts.evaluate_ln import compare,long_reference_recall
from scripts.evaluate_audio import LOADED_SOURCE
from scripts.audit_snapshot import assert_snapshot


def alternatives(frames,times,notes,threshold=.3):
    replacements={};step=float(np.median(np.diff(times)))*1000
    for pitch in range(frames.shape[1]):
        active=frames[:,pitch]>=threshold
        changes=np.diff(np.r_[False,active,False].astype(np.int8))
        runs=list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
        joined=[]
        for begin,finish in runs:
            if joined and (times[begin]-times[joined[-1][1]-1])*1000<=55:
                joined[-1][1]=finish
            else:joined.append([begin,finish])
        for begin,finish in joined:
            start=round(times[begin]*1000);end=round(times[finish-1]*1000+step)
            # Very long activation is ambiguous; preserve original detections.
            if not 800<=end-start<=6000 or active[begin:finish].mean()<.95:
                continue
            support=[n for n in notes if n.pitch==pitch+21 and n.start>=start-80 and n.start<end and n.end<=end+80]
            if len(support)<2:
                continue
            anchor=min(support,key=lambda n:abs(n.start-start))
            if abs(anchor.start-start)>80 or end-anchor.end<400:
                continue
            replacements[anchor.id]=replace(anchor,end=end,support=tuple(n.id for n in support),
                support_coverage=round(float(active[begin:finish].mean()),4))
    return replacements


def supported(folder,analysis):
    replacements={};raw=[note_from(n) for n in analysis['candidates']]
    for voice in ('bass','other'):
        with np.load(folder/'deep/stems'/f'{voice}-activations.npz') as data:
            replacements.update(alternatives(data['frames'],data['times'],[n for n in raw if n.voice==voice]))
    return raw,[replacements.get(n.id,n) for n in raw],len(replacements)


def evaluate(folder):
    audit=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    if not audit['reference']['constant_grid']:return None
    assert_snapshot(audit['source_fingerprint']['files'], LOADED_SOURCE['files'], Path(__file__).resolve().parents[1])
    analysis=json.loads((folder/'deep/analysis.json').read_text(encoding='utf-8'))
    baseline,extended,count=supported(folder,analysis)
    human=parse_osu((folder/'reference.osu').read_text(encoding='utf-8-sig'))
    prefs=parameters({'keys':human['keys']});offset,beat=audit['reference']['timing_points'][0]
    tempo=dict(bpm=60000/beat,offset=offset,division=4,stable=True);results={}
    for name,raw in [('baseline',baseline),('probe',extended)]:
        raw,_=apply_phrase_evidence(raw,[note_from(n) for n in analysis.get('phrase_candidates',[])],prefs['phrase_ln'])
        aligned,_=align_candidates(chart_clock_notes(raw,22.5),tempo)
        notes,selection=select_notes(aligned,prefs)
        results[name]=compare(notes,human['notes'])|{'long_reference':long_reference_recall(notes,human['notes']),
            'objects':len(notes),'ln':sum(n.hold for n in notes)}
    return dict(id=audit['beatmap_id'],extended=count,results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.synthetic:
        analysis=json.loads((args.folder/'deep/analysis.json').read_text(encoding='utf-8'))
        human=[note_from(n) for n in json.loads((args.folder/'ground-truth.json').read_text(encoding='utf-8'))]
        baseline,extended,count=supported(args.folder,analysis)
        result=dict(extended=count,modes={})
        for name,notes in [('baseline',baseline),('probe',extended)]:
            result['modes'][name]=dict(pitched_intervals=pitched_match(notes,human,True),
                over_2s=pitched_match(notes,[n for n in human if n.end-n.start>=2000],True)['matched'])
    else:
        rows=[];skipped=[]
        for p in sorted(args.folder.glob('*/audit.json')):
            row=evaluate(p.parent)
            if row is None:skipped.append(p.parent.name);continue
            rows.append(row);print(json.dumps(row),flush=True)
        result=dict(rows=rows,skipped=skipped)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.synthetic:print(json.dumps(result),flush=True)
