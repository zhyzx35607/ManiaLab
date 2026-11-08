"""Compare two learned priors on fixed full-chart development rhythms.

Acceptance: unchanged non-lane fields, zero conflicts, and no per-key aggregate
increase in tight represses or fast same-column repeats beyond max(5, 5%).
All regressions are reported. These known charts are not a player blind test.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from manialab import pattern_prior
from manialab.arrange import assign_lanes,parameters,layout_metrics
from manialab.domain import parse_osu,validate,invariant
from manialab.ln_patterns import analyze_ln_patterns


def run(args):
    priors={}
    for label,path in [('baseline',args.baseline),('candidate',args.candidate)]:
        value=json.loads(path.read_text(encoding='utf-8'))
        for layer in value['network']:
            layer['weight']=np.asarray(layer['weight'],np.float32);layer['bias']=np.asarray(layer['bias'],np.float32)
        priors[label]=value
    selection=json.loads(args.selection.read_text(encoding='utf-8')) if args.selection else json.loads((ROOT/'docs/full-remap-validation.json').read_text(encoding='utf-8'))['rows']
    report=dict(protocol=__doc__,models={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
        [('baseline',args.baseline),('candidate',args.candidate)]},
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('manialab/arrange.py','manialab/layout.py','manialab/pattern_prior.py','manialab/ln_patterns.py','manialab/domain.py','manialab/timing.py')},rows=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    original_loader=pattern_prior.load_prior
    try:
        for row in selection:
            raw=(args.corpus/'maps'/f"{row['beatmap_id']}.osu").read_bytes()
            assert hashlib.sha256(raw).hexdigest()==row['sha256']
            chart=parse_osu(raw.decode('utf-8-sig'));notes=chart['notes'];keys=chart['keys']
            item=dict(id=row['beatmap_id'],keys=keys,sha256=row['sha256'],results={})
            for label,value in priors.items():
                pattern_prior.load_prior=lambda v=value:v
                started=time.perf_counter()
                output=assign_lanes(notes,parameters({'keys':keys}),timing_points=chart['timing'])
                assert invariant(output)==invariant(notes)
                assert validate(output,keys)['valid']
                patterns=analyze_ln_patterns(output,keys,timing_points=chart['timing'])
                item['results'][label]=dict(seconds=round(time.perf_counter()-started,2),preserved=True,
                    layout=layout_metrics(output,keys,timing_points=chart['timing']),counts=patterns['counts'])
            report['rows'].append(item)
            args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(item),flush=True)
    finally:pattern_prior.load_prior=original_loader
    totals={}
    for keys in range(6,11):
        group=[r for r in report['rows'] if r['keys']==keys]
        if not group:continue
        result={}
        for label in priors:
            result[label]=dict(short_jacks=sum(r['results'][label]['layout']['short_jacks'] for r in group),
                tight_repress=sum(r['results'][label]['counts'].get('tight_repress_heads',0) for r in group))
        result['passed']=all(result['candidate'][metric]-result['baseline'][metric]<=max(5,result['baseline'][metric]*.05)
            for metric in ('short_jacks','tight_repress'))
        totals[str(keys)]=result
    report.update(totals=totals,passed=all(v['passed'] for v in totals.values()))
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(totals),flush=True)
    if not report['passed']:raise SystemExit('Layout regression threshold failed; do not publish this prior as default')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,default=Path('test-data/corpus'))
    p.add_argument('--selection',type=Path,help='Optional list of fixed chart IDs and SHA256 hashes')
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
