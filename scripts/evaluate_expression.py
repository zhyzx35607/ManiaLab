"""Reproducible source-selection audit on cached audio, separate from listening QA."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from collections import Counter

import numpy as np
from scipy.io import wavfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters, select_notes, assign_lanes
from manialab.domain import note_from, parse_osu, export_osu, validate
from manialab.structure import detect_structure, reference_marks
from manialab.expression import propose_expression, expression_evidence, select_expression
from manialab.timing import chart_clock_notes, align_candidates, grid_report
from manialab.voices import apply_phrase_evidence


def run(path, keys, output, reference=None):
    a=json.loads(path.read_text(encoding='utf-8'))
    rate,y=wavfile.read(path.parent/'preview.wav')
    if np.issubdtype(y.dtype,np.integer):y=y.astype(np.float32)/np.iinfo(y.dtype).max
    a['structure']=detect_structure(a,y,rate);a['segments']=a['structure']['segments']
    prefs=parameters({'keys':keys})
    if reference:
        parsed=parse_osu(reference.read_text(encoding='utf-8-sig'))
        red=next(l.split(',') for l in parsed['timing'] if len(l.split(','))>=7 and l.split(',')[6]=='1')
        tempo=dict(bpm=60000/float(red[1]),offset=float(red[0]),division=4,stable=True)
    else:tempo=dict(bpm=210,offset=93,division=4,stable=True)
    raw=[note_from(n) for n in a['candidates']]
    old_evidence,_=apply_phrase_evidence(raw,[note_from(n) for n in a.get('phrase_candidates',[])],prefs['phrase_ln'])
    old,_=select_notes(align_candidates(chart_clock_notes(old_evidence,22.5),tempo)[0],prefs)
    plan=propose_expression(a,prefs)
    evidence,meta,trace=expression_evidence(a,plan,prefs)
    aligned,alignment=align_candidates(chart_clock_notes(evidence,22.5),tempo)
    selected,selection=select_expression(aligned,meta,trace,prefs,60000/tempo['bpm'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(a,ensure_ascii=False),encoding='utf-8')
    (output/'selection.json').write_text(json.dumps(selection,ensure_ascii=False),encoding='utf-8')
    arranged=assign_lanes(selected,prefs,beat_ms=60000/tempo['bpm'])
    (output/'planned.osu').write_text(export_osu(arranged,keys,Path('vendor/mapperatorinator/template.osu'),
        bpm=tempo['bpm'],offset=tempo['offset'],grid_division=4,bookmarks=[round(t-22.5) for t in a['structure']['boundaries'][1:-1]]),encoding='utf-8')
    def summarize(notes):
        result={}
        for v in sorted({n.voice for n in raw}):
            local=[n for n in notes if n.voice==v]
            ids={n.id for n in local}
            eligible=[n for n in raw if n.voice==v and n.end-n.start>=prefs['min_ln']]
            result[v]=dict(objects=len(local),holds=sum(n.hold for n in local),
                sustained_candidates_turned_into_taps=sum(n.id in ids and not next(x for x in local if x.id==n.id).hold for n in eligible))
        return result
    report=dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),keys=keys,
        boundaries=a['structure']['boundaries'],baseline=summarize(old),planned=summarize(selected),
        selection={k:v for k,v in selection.items() if k!='decisions'},alignment=alignment,
        validation=validate(arranged,keys),grid=grid_report(arranged,tempo),
        boundary_method=a['structure']['source'],notice='Source consistency diagnostic. This is not transcription truth or a player quality score.')
    if reference:report['reference_marks']=reference_marks(reference.read_text(encoding='utf-8-sig'))
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('analysis',type=Path);p.add_argument('--keys',type=int,default=10)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    args=p.parse_args();run(args.analysis,args.keys,args.output,args.reference)
