"""Compare the grid scoring fix on frozen audio attacks, without rerunning models.

Both implementations select their own tempo/phase/subdivision and source.
Reference notes are read only afterwards for residual diagnostics. This is a
development regression set, not independent-song acceptance or a new audio run.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.domain import note_from, parse_osu
from manialab import timing


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_residuals(notes, tempo, shift):
    tick = 60000 / tempo['bpm'] / tempo['division']
    offset = math.floor(tempo['offset'] - shift + .5)
    times = np.array([t for n in notes for t in ([n.start, n.end] if n.hold else [n.start])])
    distances = abs((times-offset+tick/2) % tick-tick/2)
    return dict(endpoints=len(times), within_15ms=float(np.mean(distances<=15)),
        within_30ms=float(np.mean(distances<=30)), median_ms=float(np.median(distances)),
        p95_ms=float(np.percentile(distances,95)), max_ms=float(np.max(distances)))


def estimate(module, deep, basic):
    sources = dict(drums=[note_from(n) for n in deep['candidates'] if n['voice']=='drums'],
                   mix=[note_from(n) for n in basic['timing_attacks']])
    beats = deep.get('separation',{}).get('beat_evidence',{}).get('beats_ms')
    tempos = {v:module.estimate_timing(notes,deep['duration'],beats) for v,notes in sources.items()}
    def quality(t):
        return next((c for c in t.get('precise_candidates',[]) if c['bpm']==t.get('exact_bpm')),t)
    d,m = quality(tempos['drums']),quality(tempos['mix'])
    selected = 'drums' if len(sources['drums'])>=24 and d.get('coverage',0)>=.7 and d.get('score',0)>=m.get('score',0)*.9 else 'mix'
    result = tempos[selected]
    if selected=='drums':
        result,_ = module.cover_missing_timing_windows(result,sources['drums'],sources['mix'])
    return dict(result,evidence_source=selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/timing-grid-support-validation.json')
    parser.add_argument('--baseline-revision',default='15ba1af',help='Fixed pre-fix source; never silently use a newer HEAD')
    args=parser.parse_args()
    commit=subprocess.check_output(['git','rev-parse','--verify',args.baseline_revision+'^{commit}'],cwd=ROOT,text=True).strip()
    source=subprocess.check_output(['git','show',commit+':manialab/timing.py'],cwd=ROOT)
    old=types.ModuleType('manialab_timing_baseline')
    sys.modules[old.__name__]=old
    exec(compile(source,commit+':manialab/timing.py','exec'),old.__dict__)
    groups={r['group'] for r in json.loads((ROOT/'docs/timing-acceptance.json').read_text(encoding='utf-8'))['rows']}
    rows=[]
    for group in sorted(groups):
        for path in sorted((ROOT/'test-data'/group).glob('*/audit.json')):
            audit=json.loads(path.read_text(encoding='utf-8'))
            files={v:path.parent/v/'analysis.json' for v in ('deep','basic')}
            deep,basic=(json.loads(files[v].read_text(encoding='utf-8')) for v in ('deep','basic'))
            results={name:estimate(module,deep,basic) for name,module in [('before',old),('after',timing)]}
            # Generation above has no access to reference events or timing.
            human=parse_osu((path.parent/'reference.osu').read_text(encoding='utf-8-sig'))['notes']
            row=dict(group=group,beatmap_id=audit['beatmap_id'],constant_reference=audit['reference']['constant_grid'],
                     input_sha256={v:digest(p) for v,p in files.items()})
            for name,result in results.items():
                row[name]={k:result[k] for k in ('bpm','offset','division','coverage','stable','evidence_source')}
                row[name]['reference_residuals']=reference_residuals(human,result,audit['audio_shift_ms'])
            rows.append(row)
            print(json.dumps({k:row[k] for k in ('group','beatmap_id','before','after')}),flush=True)
    sakura=ROOT/'test-data/sakura-benchmark/provided-pipeline/analysis.json'
    if sakura.exists():
        data=json.loads(sakura.read_text(encoding='utf-8'))
        attacks=[note_from(n) for n in data['timing_attacks']]
        results={name:module.estimate_timing(attacks,data['duration']) for name,module in [('before',old),('after',timing)]}
        human=parse_osu(sakura.parents[1].joinpath('reference.osu').read_text(encoding='utf-8-sig'))['notes']
        row=dict(group='sakura',beatmap_id=5969640,input_sha256=digest(sakura),constant_reference=False)
        for name,result in results.items():
            row[name]={k:result[k] for k in ('bpm','offset','division','coverage','stable')}
            row[name]['reference_residuals']=reference_residuals(human,result,22.5)
        rows.append(row)
    changed=[r for r in rows if any(r['before'][k]!=r['after'][k] for k in ('bpm','offset','division','stable'))]
    summary=dict(songs=len(rows),changed=len(changed),
        accepted={stage:sum(r[stage]['stable'] for r in rows) for stage in ('before','after')},
        changed_ids=[r['beatmap_id'] for r in changed],
        division_changes=dict(Counter(str(r['before']['division'])+'->'+str(r['after']['division']) for r in changed)))
    report=dict(protocol=__doc__,baseline_commit=commit,baseline_sha256=hashlib.sha256(source).hexdigest(),
        current_sha256=digest(ROOT/'manialab/timing.py'),script_sha256=digest(Path(__file__)),summary=summary,rows=rows)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
