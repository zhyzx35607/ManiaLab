"""Identical paired examples with only their frozen rhythmic feature prefix."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.release_candidates import RELEASE_FEATURES
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset',type=Path,required=True); cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists(): raise ValueError('Preserve previous controls')
    manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    if manifest['features'][:len(RELEASE_FEATURES)] != RELEASE_FEATURES: raise ValueError('Wrong rhythmic prefix')
    args.output.mkdir(parents=True); files={}
    for split,metadata in manifest['files'].items():
        path=args.dataset/(split+'.npz')
        if sha(path)!=metadata['sha256']: raise ValueError('Paired data changed')
        with np.load(path,allow_pickle=False) as saved:
            data={k:saved[k] for k in saved.files}
        data['x']=data['x'][:,:len(RELEASE_FEATURES)].copy()
        output=args.output/(split+'.npz'); np.savez_compressed(output,**data)
        files[split]=metadata|dict(sha256=sha(output))
    manifest.update(features=RELEASE_FEATURES, source='paired_stems_rhythm_control', files=files,
        parent_dataset_sha256=sha(args.dataset/'manifest.json'), stem_feature_sha256=None,evidence_revision=None,
        script_sha256=sha(__file__),protocol_sha256=sha('docs/paired-stem-comparison-protocol.md'))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(files),flush=True)


if __name__=='__main__':main()
