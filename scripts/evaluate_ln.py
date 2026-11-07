"""Compare unique LN intervals with human mapping choices, ignoring lanes.

These are style similarities, not sustained-audio ground truth. Matching is
maximum-cardinality and one-to-one; a duplicated chord span counts only once.
Known-reference-grid and actual automatic-grid outputs remain separate.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import note_from, parse_osu
from manialab.arrange import parameters
from manialab.timing import align_candidates, chart_clock_notes
from manialab.voices import apply_phrase_evidence


def hold_spans(notes):
    return sorted({(n.start,n.end) for n in notes if n.hold})


def compare(predicted, reference, head_ms=20, tail_ms=80, overlap=.7):
    a, b = hold_spans(predicted), hold_spans(reference)
    starts = [t[0] for t in b]
    edge_rows, edge_cols, overlap_rows, overlap_cols = [], [], [], []
    for i,(start,end) in enumerate(a):
        for j in range(bisect_left(starts,start-head_ms),bisect_right(starts,start+head_ms)):
            left,right = b[j]
            if abs(end-right) <= tail_ms:
                edge_rows.append(i);edge_cols.append(j)
            intersection = max(0,min(end,right)-max(start,left))
            union = max(end,right)-min(start,left)
            if intersection/union >= overlap:
                overlap_rows.append(i);overlap_cols.append(j)

    def score(rows,columns):
        if a and b and rows:
            graph=csr_matrix((np.ones(len(rows),dtype=np.int8),(rows,columns)),shape=(len(a),len(b)))
            count=int(np.count_nonzero(maximum_bipartite_matching(graph,perm_type='column') >= 0))
        else:
            count=0
        precision=count/len(a) if a else None
        recall=count/len(b) if b else None
        f1=2*count/(len(a)+len(b)) if a or b else None
        return dict(predicted=len(a),reference=len(b),matched=count,
                    precision=round(precision,4) if precision is not None else None,
                    recall=round(recall,4) if recall is not None else None,
                    f1=round(f1,4) if f1 is not None else None)
    return dict(endpoints=score(edge_rows,edge_cols),overlap=score(overlap_rows,overlap_cols))


def long_reference_recall(predicted,reference):
    result={}
    for threshold in (800,2000):
        target=[n for n in reference if n.end-n.start>=threshold]
        scores=compare(predicted,target)
        result[str(threshold)]=dict(reference=scores['endpoints']['reference'],
            endpoints_matched=scores['endpoints']['matched'],overlap_matched=scores['overlap']['matched'],
            endpoints_recall=scores['endpoints']['recall'],overlap_recall=scores['overlap']['recall'])
    return result


def available_support(candidates, reference, head_ms=20, tail_ms=80):
    """An optimistic coverage bound, not a valid output or a matching score.

    A candidate may support more than one reference expression here. This
    deliberately asks whether a shorter supported release could be considered
    at all, before selection, density, simultaneous-LN or lane constraints.
    """
    candidates=sorted(candidates,key=lambda n:n.start)
    starts=[n.start for n in candidates]
    result=dict(reference_intervals=0,no_attack=0,insufficient_duration=0,
                sufficient_duration=0,existing_endpoint_nearby=0)
    for start,end in hold_spans(reference):
        result['reference_intervals']+=1
        local=candidates[bisect_left(starts,start-head_ms):bisect_right(starts,start+head_ms)]
        if not local:
            result['no_attack']+=1
        elif max(n.end for n in local) < end-tail_ms:
            result['insufficient_duration']+=1
        else:
            result['sufficient_duration']+=1
        if any(n.hold and abs(n.end-end)<=tail_ms for n in local):
            result['existing_endpoint_nearby']+=1
    return result


def run(folders):
    rows, fingerprints = [],set()
    for folder in folders:
        for path in sorted(folder.glob('*/audit.json')):
            audit=json.loads(path.read_text(encoding='utf-8'))
            fingerprints.add(audit['source_fingerprint']['sha256'])
            human=parse_osu((path.parent/'reference.osu').read_text(encoding='utf-8-sig'))['notes']
            automatic=json.loads((path.parent/'auto-audit.json').read_text(encoding='utf-8'))
            if automatic['source_fingerprint'] != audit['source_fingerprint']['sha256']:
                raise ValueError('Automatic and reference-grid snapshots differ')
            result=dict(group=folder.name,beatmap_id=audit['beatmap_id'],keys=audit['keys'],modes={})
            lengths=[end-start for start,end in hold_spans(human)]
            result['reference_lengths']=dict(unique_intervals=len(lengths),under_240ms=sum(n<240 for n in lengths),
                at_least_800ms=sum(n>=800 for n in lengths),at_least_2000ms=sum(n>=2000 for n in lengths))
            for mode in ('basic','deep'):
                comparisons={}
                for grid,generated,filename in (
                    ('reference', 'validation' in audit['modes'][mode]['generation'],'generated-reference-grid.osu'),
                    ('automatic', automatic['modes'][mode]['generated'],'generated-auto-grid.osu')):
                    if not generated:
                        comparisons[grid]=dict(skipped=True)
                        continue
                    notes=parse_osu((path.parent/mode/filename).read_text(encoding='utf-8'))['notes']
                    comparisons[grid]=compare(notes,human)|{'long_reference':long_reference_recall(notes,human)}
                result['modes'][mode]=comparisons
                if audit['reference']['constant_grid']:
                    analysis=json.loads((path.parent/mode/'analysis.json').read_text(encoding='utf-8'))
                    prefs=parameters({'keys':audit['keys']})
                    raw,_=apply_phrase_evidence([note_from(n) for n in analysis['candidates']],
                        [note_from(n) for n in analysis.get('phrase_candidates',[])],prefs['phrase_ln'])
                    offset,beat=audit['reference']['timing_points'][0]
                    aligned,_=align_candidates(chart_clock_notes(raw,audit['audio_shift_ms']),
                        dict(bpm=60000/beat,offset=offset,division=4,stable=True))
                    comparisons['available_support']=available_support(aligned,human)
            rows.append(result)
    if len(fingerprints) != 1:
        raise ValueError('Expected one consistent analysis and model fingerprint')
    groups={}
    for group in dict.fromkeys(r['group'] for r in rows):
        groups[group]={}
        for mode in ('basic','deep'):
            groups[group][mode]={}
            support=[r['modes'][mode]['available_support'] for r in rows
                     if r['group']==group and 'available_support' in r['modes'][mode]]
            groups[group][mode]['available_support']={k:sum(s[k] for s in support)
                                                      for k in support[0]} if support else {}
            for grid in ('reference','automatic'):
                subset=[r['modes'][mode][grid] for r in rows if r['group']==group and not r['modes'][mode][grid].get('skipped')]
                summary=dict(generated=len(subset))
                for method in ('endpoints','overlap'):
                    values=[r[method]['f1'] for r in subset if r[method]['f1'] is not None]
                    summary[method+'_median_f1']=round(median(values),4) if values else None
                groups[group][mode][grid]=summary
    return dict(purpose='Similarity to human LN mapping choices, not complete audio or player quality evaluation',
                protocol=dict(head_tolerance_ms=20,tail_tolerance_ms=80,minimum_iou=.7,
                              units='Unique (start,end) intervals; lanes and duplicate chord spans ignored',
                              matching='Maximum-cardinality one-to-one bipartite matching',
                              generation='Reference grid and automatic grid evaluated separately; stopped generations remain skipped'),
                source_fingerprint=next(iter(fingerprints)),
                evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),groups=groups,songs=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('docs/ln-audit.json'))
    args=parser.parse_args();report=run(args.folders)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['groups'],ensure_ascii=False,indent=2))
