"""Prepare identical rhythm/embedding samples; PCA fits training songs only."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from manialab.mapper_release_evidence import EmbeddingEvidence, MapperReleaseContext, EMBEDDING_FEATURES, EMBEDDING_REVISION, DIMENSIONS
from scripts.probe_mapper_adapter import sha
from scripts.train_anchor_gestures import add_examples


def load_hidden(record):
    path = Path(record['path'])/'hidden.npz'
    if sha(path) != record['identity']['sha256']: raise ValueError('Frozen encoder observations changed')
    with np.load(path,allow_pickle=False) as data:
        times, hidden = data['times_ms'],data['hidden'].astype(np.float32)
    if hidden.shape != (len(times),768) or not np.isfinite(hidden).all(): raise ValueError('Invalid encoder observations')
    return times,hidden


def load_evidence(record):
    path = Path(record['observations'])
    if sha(path) != record['observations_sha256']: raise ValueError('Projected observations changed')
    with np.load(path,allow_pickle=False) as data:
        return EmbeddingEvidence(data['times_ms'],data['values'])


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--embeddings',type=Path,required=True); cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists(): raise ValueError('Preserve earlier datasets')
    manifest=json.loads((args.embeddings/'manifest.json').read_text(encoding='utf-8'))
    train=[]; training_ids=[]
    for record in manifest['records']:
        if record['split']!='train':continue
        _,hidden=load_hidden(record)
        indices=np.random.default_rng(record['id']).choice(len(hidden),min(len(hidden),128),replace=False)
        train.append(hidden[indices]); training_ids.append(record['id'])
    data=np.concatenate(train).astype(np.float64);mean=data.mean(axis=0);centered=data-mean
    eigenvalues,vectors=np.linalg.eigh(centered.T@centered/len(centered))
    order=np.argsort(eigenvalues)[::-1][:DIMENSIONS];components=vectors[:,order]
    # Fix arbitrary eigenvector signs for repeatable serialization.
    for i in range(DIMENSIONS):
        if components[np.argmax(np.abs(components[:,i])),i]<0:components[:,i]*=-1
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'projection.npz',mean=mean.astype(np.float32),components=components.astype(np.float32))
    projection=dict(sha256=sha(args.output/'projection.npz'),training_ids=training_ids,training_frames=len(data),
                    retained_variance=float(eigenvalues[order].sum()/eigenvalues.sum()),dimensions=DIMENSIONS)
    group_sizes=Counter((r['keys'],r['group']) for r in manifest['records'])
    key_groups=Counter((k,next(r['split'] for r in manifest['records'] if r['group']==g)) for k,g in group_sizes)
    arrays=defaultdict(lambda:dict(x=[],y=[],lengths=[],weights=[]));records=[];exclusions=[]
    for record in manifest['records']:
        if sha(record['chart'])!=record['chart_sha256']:raise ValueError('Chart changed')
        times,hidden=load_hidden(record)
        values=(hidden-mean.astype(np.float32))@components.astype(np.float32)
        observation=args.output/'observations'/str(record['id']);observation.parent.mkdir(exist_ok=True)
        observation=observation.with_suffix('.npz')
        np.savez_compressed(observation,times_ms=times,values=values)
        evidence=EmbeddingEvidence(times,values)
        chart=parse_osu(Path(record['chart']).read_text(encoding='utf-8-sig'))
        notes=[replace(n,end=n.start,lane=0,voice='mix',raw='') for n in chart['notes']]
        ratio=sum(n.hold for n in chart['notes'])/len(notes)
        context=MapperReleaseContext(notes,chart['keys'],chart['timing'],ratio,evidence)
        eligible=[i for i in range(len(context.times)) if context.contained(i,times[0],times[-1]) and context.supported(i)]
        if not eligible:
            exclusions.append(dict(id=record['id'],reason='No fully contained candidate and rhythm context'));continue
        indices=sorted(np.random.default_rng(record['id']).choice(eligible,min(128,len(eligible)),replace=False).tolist())
        weight=1/(len(indices)*group_sizes[(record['keys'],record['group'])]*key_groups[(record['keys'],record['split'])])
        counts=add_examples(context,chart['notes'],indices,arrays[record['split']],weight)
        records.append(record|dict(observations=str(observation),observations_sha256=sha(observation),rows=len(indices),labels=counts,
            score_bounds=[int(context.times[eligible[0]]),int(context.times[eligible[-1]])+1]))
        print(json.dumps(dict(prepared=record['id'],split=record['split'],rows=len(indices))),flush=True)
    files={}
    for split,group in arrays.items():
        path=args.output/(split+'.npz')
        np.savez_compressed(path,x=np.concatenate(group['x']),y=np.concatenate(group['y']),
                            lengths=np.asarray(group['lengths']),weights=np.asarray(group['weights']))
        files[split]=dict(sha256=sha(path),rows=len(group['lengths']),options=sum(group['lengths']))
    result=dict(revision=MapperReleaseContext.revision,features=EMBEDDING_FEATURES,source='mapper_embeddings',
        scope='Authored attacks/Timing/counts plus frozen mix encoder; song-group-isolated development comparison.',
        evidence_revision=EMBEDDING_REVISION,files=files,records=records,exclusions=exclusions,projection=projection,
        embeddings_manifest_sha256=sha(args.embeddings/'manifest.json'),pretraining_dataset_sha256=manifest['pretraining_dataset_sha256'],
        feature_sha256=sha('manialab/anchor_gestures.py'),release_feature_sha256=sha('manialab/release_candidates.py'),
        embedding_feature_sha256=sha('manialab/mapper_release_evidence.py'),loader_sha256=sha(__file__),
        protocol_sha256=sha('docs/mapper-embedding-release-protocol.md'))
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(projection=projection,files=files,exclusions=exclusions)),flush=True)


if __name__=='__main__':main()
