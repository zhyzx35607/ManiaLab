"""Exercise local audio plans against saved real detections, without re-transcribing.

Reports coverage, unchanged evidence, structural validity and exported grid.
This is an integration test, not a listening judgement or transcription score.
Audio and generated charts stay in the selected private output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manialab.domain import parse_osu
from manialab.recommendations import evidence_signature
from manialab.server import create_app
from manialab.storage import Projects
from manialab import pattern_prior


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    original_loader=pattern_prior.load_prior
    prior_path=args.prior or ROOT/'assets/pattern-prior.json'
    if args.prior:
        prior=json.loads(prior_path.read_text(encoding='utf-8'))
        for layer in prior['network']:
            layer['weight']=np.asarray(layer['weight'],np.float32);layer['bias']=np.asarray(layer['bias'],np.float32)
        pattern_prior.load_prior=lambda:prior
    analysis=json.loads(args.analysis.read_text(encoding='utf-8'))
    original=evidence_signature(analysis)
    app=create_app(ROOT,args.output/'app');client=app.test_client()
    headers={'X-ManiaLab-Token':app.config['TOKEN']}
    store=Projects(args.output/'app/projects')
    raw_audio=args.audio.read_bytes() if args.audio else b'analysis-only fixture'
    state=client.post('/api/import',data={'files':(io.BytesIO(raw_audio),args.audio.name if args.audio else 'song.wav')},headers=headers).json
    state['analysis']=analysis;store.save(state)
    base=f"/api/projects/{state['id']}"
    report=dict(protocol=__doc__,analysis_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
        prior_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256((ROOT/'assets/arrangement-profiles.json').read_bytes()).hexdigest(),
        keys=args.keys,bpm=args.bpm,offset=args.offset,audio_shift_ms=args.shift,rows=[])
    known_ids={n['id'] for n in analysis['candidates']+analysis.get('phrase_candidates',[])}
    try:
        for keys in args.keys:
            response=client.post(base+'/recommendations',json=dict(settings=dict(keys=keys),bpm=args.bpm),headers=headers)
            assert response.status_code==200,response.json
            proposal=response.json
            for option in proposal['options']:
                selected=client.post(base+'/apply-recommendation',json=dict(settings=dict(keys=keys),bpm=args.bpm,
                    choice=option['id'],evidence_signature=proposal['evidence_signature'],context_signature=proposal['context_signature']),headers=headers)
                assert selected.status_code==200,selected.json
                settings=selected.json['settings']
                job=client.post(base+'/generate',json=dict(mode='audio',settings=settings,bpm=args.bpm,offset=args.offset,
                    division=args.division,audio_shift_ms=args.shift,snap=True,timing_confirmed=True),headers=headers)
                assert job.status_code==200,job.json
                deadline=time.monotonic()+900
                while time.monotonic()<deadline:
                    result=client.get('/api/jobs/'+job.json['id']).json
                    if result['status'] in ('done','error','cancelled'):break
                    time.sleep(.1)
                assert result['status']=='done',result
                saved=store.load(state['id']);variant=saved['variants'][saved['active']]
                assert evidence_signature(saved['analysis'])==original
                assert {n['id'] for n in variant['notes']}<=known_ids
                assert variant['report']['valid'] and variant['report']['grid']['off_grid']==0
                exported=client.get(base+'/export/osu');assert exported.status_code==200,exported.json
                parsed=parse_osu(exported.data.decode('utf-8-sig'))
                assert sorted((n.start,n.end,n.lane) for n in parsed['notes'])==sorted((n['start'],n['end'],n['lane']) for n in variant['notes'])
                name=f'{keys}K-{option["id"]}'
                (args.output/(name+'.osu')).write_bytes(exported.data)
                if args.audio and keys==10:
                    package=client.get(base+'/export/osz');assert package.status_code==200
                    (args.output/(name+'.osz')).write_bytes(package.data)
                pattern=variant['report']['ln_patterns'].copy()
                pattern.pop('locations',None);pattern.pop('sections',None)
                row=dict(keys=keys,option=option['id'],preferences=settings,
                    notes=variant['report']['notes'],ln=variant['report']['ln'],ln_ratio=variant['report']['ln_ratio'],
                    seconds=variant['elapsed_seconds'],off_grid=0,evidence_preserved=True,
                    ln_patterns=pattern,alignment=variant['selection']['alignment'])
                report['rows'].append(row)
                (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps({k:row[k] for k in ('keys','option','notes','ln','ln_ratio','seconds','off_grid')}),flush=True)
    finally:
        app.extensions['executor'].shutdown()
        pattern_prior.load_prior=original_loader


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis',type=Path);p.add_argument('--audio',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior',type=Path,help='Evaluate a candidate prior before promoting it to assets')
    p.add_argument('--keys',type=int,nargs='+',default=[6,7,8,9,10])
    p.add_argument('--bpm',type=float,required=True);p.add_argument('--offset',type=float,required=True)
    p.add_argument('--division',type=int,default=4);p.add_argument('--shift',type=float,default=22.5)
    run(p.parse_args())
