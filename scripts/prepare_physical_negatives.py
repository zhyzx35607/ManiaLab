"""Add only absent-class labels from the original training-group solo recordings."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha
from scripts.physical_drum_features import load_audio,features,CLASSES,RATE


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--base',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier partial-label datasets')
    base=json.loads((args.base/'manifest.json').read_text(encoding='utf-8'))
    if sha('scripts/physical_drum_features.py')!=base['feature_code_sha256']:raise ValueError('Original audio features changed')
    args.output.mkdir(parents=True);records=[]
    for row in base['records']:
        path=args.base/(row['id']+'.npz')
        if sha(path)!=row['data_sha256']:raise ValueError('Original split data changed')
        shutil.copyfile(path,args.output/path.name);records.append(row)
    norm=args.base/'normalization.npz'
    if sha(norm)!=base['normalization_sha256']:raise ValueError('Original normalization changed')
    shutil.copyfile(norm,args.output/norm.name)
    seen={row['audio_sha256'] for row in records};duplicates=[]
    for row in base['records']:
        if row['split']!='train' or row['dataset']!='IDMT':continue
        original=Path(row['audio'])
        for code,kind in [('KD','kick'),('SD','snare'),('HH','hi-hat')]:
            path=original.with_name(original.name.replace('#MIX.wav','#'+code+'#train.wav'))
            digest=sha(path)
            if digest in seen:
                duplicates.append(dict(audio=str(path),sha256=digest));continue
            seen.add(digest)
            audio=load_audio(path);x,times=features(audio);targets=np.zeros((len(times),3),np.float32)
            targets[:,CLASSES.index(kind)]=-1
            name=row['id'].replace('#MIX','#'+code+'#negative')
            data=args.output/(name+'.npz');np.savez_compressed(data,features=x,times=times,targets=targets)
            records.append(dict(id=name,dataset='IDMT-solo-negative',group=row['group'],split='train',audio=str(path),
                audio_sha256=digest,annotation=None,annotation_sha256=None,data_sha256=sha(data),events=[],
                unknown_class=kind,known_absent_classes=[c for c in CLASSES if c!=kind],duration_ms=len(audio)/RATE*1000,
                frames=len(times),parent_record=row['id'],near_zero_annotations=[]))
    result={**base,'records':records,'base_manifest_sha256':sha(args.base/'manifest.json'),
        'counts':{s:sum(r['split']==s for r in records) for s in ('train','validation','test')},
        'train_frames':sum(r['frames'] for r in records if r['split']=='train'),
        'normalization_scope':'unchanged original mixed training recordings only',
        'duplicate_training_files_ignored':duplicates,'protocol_sha256':sha('docs/physical-drum-negative-protocol.md'),
        'runner_sha256':sha(__file__),'physical_target_code_sha256':sha('scripts/physical_drum_targets.py'),
        'test_status':'Fixed development regression set; already inspected after model-v1.'}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name in ('scripts/prepare_physical_negatives.py','docs/physical-drum-negative-protocol.md'):
        shutil.copyfile(name,args.output/Path(name).name)
    print(json.dumps(dict(counts=result['counts'],additional_solo_records=len(records)-len(base['records']),duplicate_solos=len(duplicates))),flush=True)


if __name__=='__main__':main()
