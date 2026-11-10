"""Prepare fixed score windows with work-disjoint adaptation partitions."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import warnings

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_provenance import sha256


def work_group(path):
    family,stem=path.parent.name,path.stem.lower()
    if family in ('beethoven_piano_sonatas','mozart_sonatas'):
        work=stem.split('-')[0]
    elif family=='debussy_corpus':
        work=stem.split('-')[0].split('_')[0]
    elif family=='liszt_pelerinage':
        work=stem.split('.')[0]
    elif family=='schumann_kinderszenen':
        work='op15'
    else:
        match=re.search(r'op\d+[a-z]?',stem)
        work=match.group() if match else stem
    return family+'/'+work


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--corpus',type=Path,default=ROOT/'test-data/piano-svsep/corpus')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous preparations')
    revision=subprocess.check_output(['git','-C',str(args.corpus),'rev-parse','HEAD'],text=True).strip()
    if revision!='aa732b635bc75c6583d86c6ac488e77b08b1b37e':raise ValueError('Unexpected corpus revision')
    sys.path.insert(0,str(args.source))
    import partitura
    from piano_svsep.utils import remove_ties_acros_barlines,sanitize_staff_voices
    tree=ast.parse((args.source/'piano_svsep/data/dataset.py').read_text(encoding='utf-8'))
    tests=[]
    dcml=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DCMLPianoCorporaPolyVoiceSeparationDataset')
    for node in ast.walk(dcml):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.List):
            if any(isinstance(t,ast.Attribute) and t.attr=='test_files' for t in node.targets):
                tests.extend(ast.literal_eval(node.value))
    files=sorted((args.corpus/'scores').rglob('*.musicxml'))
    named={p.parent.name+'_'+p.stem:p for p in files}
    missing=sorted(set(tests)-set(named))
    if missing:raise ValueError(f'Missing declared test scores: {missing}')
    test_groups={work_group(named[name]) for name in tests}
    rows=[];args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    for index,path in enumerate(files):
        name=path.parent.name+'_'+path.stem;group=work_group(path)
        split=('test' if name in tests else 'excluded_related_work' if group in test_groups else
               'validation' if int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%5==0 else 'train')
        row=dict(name=name,work=group,split=split,path=str(path),sha256=sha256(path))
        if split!='excluded_related_work':
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    score=partitura.load_score(str(path))
                    remove_ties_acros_barlines(score)
                    array=score.note_array(include_time_signature=True,include_grace_notes=True,include_staff=True)
                    array=sanitize_staff_voices(array)
                row['warnings']=dict(Counter(str(w.message) for w in caught))
                clock_part=score[np.array([p._quarter_durations[0] for p in score]).argmax()]
                measures=np.asarray(clock_part.measure_number_map(array['onset_div'])).astype(int)
                bounds=np.asarray(clock_part.measure_map(array['onset_div']))
                if bounds.shape!=(len(array),2):raise ValueError('Invalid measure boundaries')
                first=np.sort(np.unique(measures))[:16]
                mask=np.isin(measures,first)
                array,measures,bounds=array[mask],measures[mask],bounds[mask]
                valid=(array['duration_div']>0)&(array['duration_beat']>0)&(array['pitch']>=21)&(array['pitch']<=108)
                row['dropped_zero_or_out_of_range']=int((~valid).sum())
                array,measures,bounds=array[valid],measures[valid],bounds[valid]
                if not len(array) or array.dtype.hasobject:raise ValueError('Empty or object note array')
                if set(array['staff'])!={1,2}:raise ValueError('Window must contain exactly two score staves')
                order=np.lexsort((array['pitch'],array['onset_div']))
                array,measures,bounds=array[order],measures[order],bounds[order]
                target=args.output/(name+'.npz')
                np.savez_compressed(target,notes=array,measures=measures,measure_bounds=bounds)
                row.update(status='ready',notes=len(array),measures=len(set(measures)),
                    cache=target.name,cache_sha256=sha256(target))
            except Exception as exc:
                row.update(status='unusable',reason=f'{type(exc).__name__}: {exc}')
        else:
            row['status']='excluded'
        rows.append(row)
        print(json.dumps(dict(index=index+1,total=len(files),name=name,split=split,status=row['status'],notes=row.get('notes'))),flush=True)
        (args.output/'progress.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    ready=[r for r in rows if r['status']=='ready']
    groups={split:{r['work'] for r in ready if r['split']==split} for split in ('train','validation','test')}
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test'] and not groups['validation']&groups['test']
    hashes={split:{r['sha256'] for r in ready if r['split']==split} for split in groups}
    assert not hashes['train']&hashes['validation'] and not hashes['train']&hashes['test']
    result=dict(corpus_revision=revision,runner_sha256=sha256(__file__),
        protocol_sha256=sha256(ROOT/'docs/piano-stream-adaptation-protocol.md'),
        published_test_names=tests,split_counts=dict(Counter(r['split'] for r in ready)),
        split_notes={s:sum(r['notes'] for r in ready if r['split']==s) for s in groups},
        excluded_related=sum(r['status']=='excluded' for r in rows),unusable=sum(r['status']=='unusable' for r in rows),
        work_disjoint=True,cases=rows,license='CC-BY-NC-SA-4.0; local research data and weights only')
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','published_test_names')}),flush=True)


if __name__=='__main__':main()
