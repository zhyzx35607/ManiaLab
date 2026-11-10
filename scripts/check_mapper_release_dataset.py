"""Verify paired controls, grouped splits, training-only PCA and cache identity."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha
from scripts.prepare_mapper_releases import load_evidence
from manialab.release_candidates import RELEASE_FEATURES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','control'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(args.dataset/'manifest.json');control=read(args.control/'manifest.json')
    if control['parent_dataset_sha256']!=sha(args.dataset/'manifest.json'):raise ValueError('Control parent mismatch')
    groups={s:{r['group'] for r in manifest['records'] if r['split']==s} for s in ('train','validation','test')}
    if any(groups[a]&groups[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):
        raise ValueError('Song group leakage')
    if set(manifest['projection']['training_ids'])!={r['id'] for r in manifest['records'] if r['split']=='train'}:
        raise ValueError('Projection does not exactly identify its training songs')
    if sha(args.dataset/'projection.npz')!=manifest['projection']['sha256']:raise ValueError('Projection changed')
    for record in manifest['records']:load_evidence(record)
    reports={}
    for split in groups:
        paths=[args.dataset/(split+'.npz'),args.control/(split+'.npz')]
        if sha(paths[0])!=manifest['files'][split]['sha256'] or sha(paths[1])!=control['files'][split]['sha256']:
            raise ValueError('Dataset bytes changed')
        with np.load(paths[0],allow_pickle=False) as a,np.load(paths[1],allow_pickle=False) as b:
            if not np.array_equal(a['x'][:,:len(RELEASE_FEATURES)],b['x']):raise ValueError('Rhythm features differ')
            for name in ('y','weights','lengths'):
                if not np.array_equal(a[name],b[name]):raise ValueError('Control samples/labels differ')
            if not all(np.isfinite(a[name]).all() for name in a.files):raise ValueError('Nonfinite data')
            offsets=np.r_[0,np.cumsum(a['lengths'])]
            sums=np.add.reduceat(a['y'],offsets[:-1])
            if not np.allclose(sums,1,atol=1e-6) or np.any(a['y']<0) or np.any(a['weights']<=0):
                raise ValueError('Invalid supervision probability/weight')
            reports[split]=dict(rows=len(a['lengths']),candidates=len(a['y']),groups=len(groups[split]),
                               identical_rhythm_prefix=True,identical_labels_and_weights=True)
    result=dict(dataset_sha256=sha(args.dataset/'manifest.json'),control_sha256=sha(args.control/'manifest.json'),
                projection_training_only=True,disjoint_song_groups=True,finite=True,splits=reports)
    (args.dataset/'integrity-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
