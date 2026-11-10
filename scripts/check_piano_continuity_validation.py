"""Replay regression-only overlap repair on cached development and Nova audio."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manialab.piano import OUTPUTS,decode
from manialab.piano_continuity import continue_overlap
from scripts.check_expression_current_replay import read,save,sha
from scripts.evaluate_piano_acoustics import metrics,pairs


def run(output):
    output=output.resolve()
    assert not output.exists() and output.is_relative_to(ROOT/'test-data')
    cache=ROOT/'test-data/piano-onset-head-cache-v1'
    assert sha(cache/'report.json')=='29a84fe641efbda504cd815dfea777de81ae8452fb6fc3fad1ebb87e36ad0a8f'
    report=read(cache/'report.json')
    expected={'449','1046','518','1176','926','1222','1219','628','1134','752'}
    assert {r['id'] for r in report['records']}==expected=={r['id'] for r in report['windows']}
    sakura=ROOT/'test-data/piano-window-sakura-v1';nova=read(sakura/'report.json')
    assert sha(sakura/'plan.json')==nova['plan_sha256']
    for name,digest in read(sakura/'plan.json')['inputs'].items():assert sha(ROOT/name)==digest
    sources={p:sha(ROOT/p) for p in ('scripts/check_piano_continuity_validation.py',
        'manialab/piano_continuity.py','manialab/piano.py','scripts/evaluate_piano_acoustics.py')}
    save(output/'plan.json',dict(sources=sources,development_report_sha256=sha(cache/'report.json'),
        nova_report_sha256=sha(sakura/'report.json'),ids=sorted(expected),test_read=False,
        require_release_regression=True,read_labels_after_decoding=True,neural_inference=False))
    for name in sources:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    jobs=[dict(name=r['id'],split=r['split'],frames=18000,score_seconds=170,metadata=r) for r in report['records']]
    jobs.append(dict(name='sakura',split='nova',frames=28384,score_seconds=None))
    prepared=[]
    for job in jobs:
        name=job['name'];raw=[]
        if name=='sakura':
            rows=sorted(nova['windows'],key=lambda r:r['start_sample'])
            assert len(rows)==57 and all(r['name']=='sakura' for r in rows)
            for row in rows:
                assert sha(sakura/row['path'])==row['sha256']
                with np.load(sakura/row['path'],allow_pickle=False) as z:raw.append({k:z[k] for k in OUTPUTS})
        else:
            assert sha(cache/(name+'-16k.npy'))==job['metadata']['pcm_sha256']
            rows=sorted([r for r in report['windows'] if r['id']==name],key=lambda r:r['index'])
            assert [r['index'] for r in rows]==list(range(35))
            for row in rows:
                assert sha(cache/row['file'])==row['sha256']
                with np.load(cache/row['file'],allow_pickle=False) as z:raw.append({k:z[k][0] for k in OUTPUTS})
        fixed=[raw[0]];trace=[]
        for i in range(1,len(raw)):
            window,changes=continue_overlap(raw[i-1],raw[i],owned_left=0 if i==1 else 250,require_release_regression=True)
            fixed.append(window);trace.extend([dict(window_start_seconds=i*5,**c) for c in changes])
        predictions={}
        for variant,windows in (('baseline',raw),('candidate',fixed)):
            values={k:np.concatenate([w[k][0 if i==0 else 250:1000 if i==len(windows)-1 else 750]
                for i,w in enumerate(windows)])[:job['frames']] for k in OUTPUTS}
            if variant=='baseline' and name!='sakura':
                path=cache/(name+'-epoch0-outputs.npz');assert sha(path)==report['artifacts_sha256'][path.name]
                with np.load(path,allow_pickle=False) as z:
                    for k in OUTPUTS:np.testing.assert_array_equal(values[k],z[k])
            predictions[variant]=decode(values)
            save(output/name/(variant+'.json'),predictions[variant])
            np.savez_compressed(output/name/(variant+'-activations.npz'),**values)
        a,b=predictions['baseline'],predictions['candidate']
        assert len(a)==len(b) and all(all(x[k]==y[k] for k in ('onset','pitch','velocity','onset_score')) for x,y in zip(a,b))
        changes=[dict(index=i,before=x,after=y) for i,(x,y) in enumerate(zip(a,b)) if x!=y]
        save(output/name/'trace.json',trace);save(output/name/'changes.json',changes)
        prepared.append((job,predictions,trace,changes))
        print(name,'decoded',len(trace),'proposals',len(changes),'changed',flush=True)
    results=[]
    for job,predictions,trace,changes in prepared:
        name=job['name'];row=dict(name=name,split=job['split'],proposals=len(trace),changes=len(changes),scores={},paired_changes={})
        if name!='sakura':
            path=cache/(name+'-labels.json');assert sha(path)==job['metadata']['labels_sha256'];labels=read(path)
            scope=lambda notes:[n for n in notes if 0<=n['onset']<job['score_seconds']]
            for semantics,field in (('key','notes'),('pedal','sustain_notes')):
                reference=scope(labels[field])
                row['scores'][semantics]={variant:{mode:metrics(reference,scope(notes),strict=strict)
                    for mode,strict in (('standard',False),('strict',True))} for variant,notes in predictions.items()}
                indices=[i for i,n in enumerate(predictions['baseline']) if 0<=n['onset']<job['score_seconds']]
                matching={indices[j]:i for i,j in pairs(reference,[predictions['baseline'][i] for i in indices])}
                detailed=[]
                for change in changes:
                    if change['index'] not in indices:continue
                    ref=reference[matching[change['index']]] if change['index'] in matching else None
                    old=abs(change['before']['offset']-ref['offset'])*1000 if ref else None
                    new=abs(change['after']['offset']-ref['offset'])*1000 if ref else None
                    detailed.append(dict(**change,reference=ref,before_error_ms=old,after_error_ms=new,
                        outcome='unmatched' if ref is None else 'improved' if new<old else 'worsened' if new>old else 'unchanged'))
                save(output/name/(semantics+'-changes.json'),detailed)
                row['paired_changes'][semantics]=dict(Counter(r['outcome'] for r in detailed))
        results.append(row);print(name,row['paired_changes'],flush=True)
    assert sources=={p:sha(ROOT/p) for p in sources}
    save(output/'report.json',dict(plan_sha256=sha(output/'plan.json'),cases=results,
        test_read=False,neural_inference=False,default_enabled=False,musical_acceptance=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();run(args.output)
