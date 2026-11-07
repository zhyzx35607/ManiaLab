"""Research-only release ranking inside an existing supported LN interval.

This never creates/extends an LN or changes an attack. Human endpoints label
training examples only; prediction sees separated candidates and the current
supported interval. Default generation and the application model are unchanged.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters, select_notes
from manialab.domain import note_from, parse_osu
from manialab.timing import align_candidates, chart_clock_notes
from manialab.voices import apply_phrase_evidence
from scripts.evaluate_audio import LOADED_SOURCE
from scripts.audit_snapshot import assert_snapshot
from scripts.evaluate_ln import compare, long_reference_recall

VOICES = ('drums', 'bass', 'vocals', 'other')
FEATURES = [
    'duration_log', 'remaining_log', 'support_fraction', 'original_endpoint',
    'strength', 'phrase', 'coverage', 'head_alignment', 'near_attack',
    'near_same_voice_attack', 'near_pitch_release', 'near_same_voice_release',
    'nearby_attack_density', 'duration_under_240', 'duration_over_800',
    'duration_over_2000', *VOICES,
]


def nearest(values, time):
    if not len(values):
        return 10000.
    at = np.searchsorted(values, time)
    return min(abs(time-values[min(at,len(values)-1)]), abs(time-values[max(0,at-1)]))


def dataset(folder, automatic=False):
    audit = json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    assert_snapshot(audit['source_fingerprint']['files'], LOADED_SOURCE['files'], Path(__file__).resolve().parents[1])
    if not automatic and not audit['reference']['constant_grid']:
        return None
    analysis = json.loads((folder/'deep/analysis.json').read_text(encoding='utf-8'))
    if automatic and not analysis['tempo']['stable']:
        return None
    human = parse_osu((folder/'reference.osu').read_text(encoding='utf-8-sig'))
    prefs = parameters({'keys':human['keys']})
    if automatic:
        tempo = analysis['tempo'] | {'offset':int(np.floor(analysis['tempo']['offset']-22.5+.5))}
    else:
        offset, beat = audit['reference']['timing_points'][0]
        tempo = dict(bpm=60000/beat, offset=offset, division=4, stable=True)
    raw, _ = apply_phrase_evidence([note_from(n) for n in analysis['candidates']],
        [note_from(n) for n in analysis.get('phrase_candidates',[])], prefs['phrase_ln'])
    aligned, _ = align_candidates(chart_clock_notes(raw,22.5),tempo)
    selected, selection = select_notes(aligned,prefs)
    starts = np.array(sorted({n.start for n in aligned}))
    ends = np.array(sorted({n.end for n in aligned if n.hold}))
    voice_starts = {v:np.array(sorted({n.start for n in aligned if n.voice==v})) for v in VOICES}
    voice_ends = {v:np.array(sorted({n.end for n in aligned if n.voice==v and n.hold})) for v in VOICES}
    refs = sorted({(n.start,n.end) for n in human['notes'] if n.hold})
    ref_starts = [r[0] for r in refs]
    tick = 60000/tempo['bpm']/tempo['division']; offset = tempo['offset']
    examples = []
    for index,n in enumerate(selected):
        if not n.hold:
            continue
        first = int(np.ceil((n.start+prefs['min_ln']-offset)/tick))
        last = int(np.floor((n.end-offset)/tick))
        indices = list(range(first,last+1))
        # Keep bounded work for exceptionally long sustained detections.
        if len(indices)>64:
            indices = [indices[i] for i in sorted(set(np.linspace(0,len(indices)-1,64).astype(int)))]
        options = sorted({n.end,*[int(np.floor(offset+i*tick+.5)) for i in indices
                                  if n.start+prefs['min_ln']<=int(np.floor(offset+i*tick+.5))<=n.end]})
        features = []
        for end in options:
            duration,remaining = end-n.start,n.end-end
            feature = [np.log1p(duration/500),np.log1p(remaining/500),duration/(n.end-n.start),
                end==n.end,n.strength,n.phrase,n.support_coverage,
                abs(n.start-n.evidence_start) / 100 if n.evidence_start is not None else 0,
                np.exp(-.5*(nearest(starts,end)/25)**2),
                np.exp(-.5*(nearest(voice_starts.get(n.voice,[]),end)/25)**2),
                np.exp(-.5*(nearest(ends,end)/25)**2),
                np.exp(-.5*(nearest(voice_ends.get(n.voice,[]),end)/25)**2),
                (np.searchsorted(starts,end+500)-np.searchsorted(starts,end-500))/12,
                duration<240,duration>=800,duration>=2000,*[n.voice==v for v in VOICES]]
            features.append(feature)
        local = refs[bisect_left(ref_starts,n.start-20):bisect_right(ref_starts,n.start+20)]
        labels = np.array([min((abs(end-r[1]) for r in local),default=10000)<=80 for end in options],float)
        examples.append(dict(index=index,options=options,x=np.array(features,float),y=labels))
    return dict(id=audit['beatmap_id'],keys=human['keys'],notes=selected,human=human['notes'],examples=examples,
                attack_optimal=selection['attack_optimal'],tempo=tempo)


def fit(items):
    matrices,targets,weights = [],[],[]
    for item in items:
        local = []
        for example in item['examples']:
            positive = example['x'][example['y']==1]; negative = example['x'][example['y']==0]
            if not len(positive) or not len(negative):
                continue
            pairs = (positive[:,None,:]-negative[None,:,:]).reshape(-1,len(FEATURES))
            # Each interval has equal weight regardless of how many grid tails it has.
            local.append(pairs)
        if not local:
            continue
        for pairs in local:
            matrices.append(pairs);targets.append(np.ones(len(pairs)))
            weights.append(np.full(len(pairs),1/len(pairs)/len(local)))
    if not matrices:
        raise ValueError('No supported positive/negative endpoint pairs')
    x=np.concatenate(matrices);y=np.concatenate(targets);w=np.concatenate(weights);w/=w.sum()
    scale=np.maximum(.15,np.sqrt(np.average(x*x,axis=0,weights=w)))
    z=np.clip(x/scale,-10,10);reg=.025
    def loss(beta):
        margin=z@beta
        return float(np.sum(w*np.logaddexp(0,-margin))+.5*reg*np.sum(beta**2)),z.T@(w*(expit(margin)-y))+reg*beta
    result=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':200})
    if not result.success:
        raise ValueError(result.message)
    return dict(scale=scale.tolist(),weights=result.x.tolist(),pairs=len(x))


def evaluate(item,model):
    predicted=list(item['notes']);changed=0;trim_ms=0
    for example in item['examples']:
        score=(example['x']/model['scale'])@model['weights']
        # An explicit fixed margin favors retaining the observed endpoint.
        original=example['options'].index(predicted[example['index']].end)
        score[original]+=.15
        best=int(np.argmax(score));end=example['options'][best]
        n=predicted[example['index']]
        assert n.start<end<=n.end
        changed+=end!=n.end;trim_ms+=n.end-end
        predicted[example['index']]=replace(n,end=end)
    result=dict(id=item['id'],keys=item['keys'],holds=len(item['examples']),changed=changed,trimmed_ms=trim_ms,
                attack_optimal=item['attack_optimal'])
    for name,notes in [('baseline',item['notes']),('probe',predicted)]:
        result[name]=compare(notes,item['human'])|{'long_reference':long_reference_recall(notes,item['human'])}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true');parser.add_argument('--evaluate',type=Path)
    parser.add_argument('--automatic',action='store_true');parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('test-data/ln-release-evaluation.json'))
    args=parser.parse_args()
    if args.train:
        items=[]
        for group in ('audio-audit','audio-holdout'):
            for p in sorted(Path('test-data',group).glob('*/audit.json')):
                item=dataset(p.parent)
                if item:
                    item['fold']=int(hashlib.sha256(str(item['id']).encode()).hexdigest()[:8],16)%5
                    items.append(item)
        rows=[]
        for fold in range(5):
            model=fit([item for item in items if item['fold']!=fold])
            for item in items:
                if item['fold']==fold:
                    row=evaluate(item,model)|{'fold':fold};rows.append(row);print(json.dumps(row),flush=True)
        model=fit(items)
        args.model.write_text(json.dumps(dict(features=FEATURES,model=model,training_ids=[i['id'] for i in items],
            source_fingerprint=LOADED_SOURCE,regularization=.025,original_endpoint_margin=.15,cross_validation=rows,
            notice='Research only. Tail shortening within supported durations; not enabled in the application.'),indent=2),encoding='utf-8')
    if args.evaluate:
        saved=json.loads(args.model.read_text(encoding='utf-8'));rows=[];skipped=[]
        for p in sorted(args.evaluate.glob('*/audit.json')):
            item=dataset(p.parent,args.automatic)
            if item is None:
                skipped.append(p.parent.name);continue
            if item['id'] in saved['training_ids']:
                raise ValueError('Training song included in evaluation')
            row=evaluate(item,saved['model']);rows.append(row);print(json.dumps(row),flush=True)
        args.output.write_text(json.dumps(dict(automatic=args.automatic,rows=rows,skipped=skipped),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
