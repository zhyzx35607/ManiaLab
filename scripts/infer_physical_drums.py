"""Freeze validation-selected physical model, infer reserved songs and Sakura."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset',type=Path,required=True)
    cli.add_argument('--model',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier test inference')
    import torch
    from safetensors.torch import load_file
    from scripts.physical_drum_features import features,load_audio
    from scripts.physical_drum_network import DrumOnsetNet,infer
    torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(args.dataset/'manifest.json');trained=read(args.model/'report.json')
    if (sha(args.dataset/'manifest.json')!=trained['identity']['dataset_sha256']
        or sha(args.model/'model.safetensors')!=trained['model_sha256']
        or sha('scripts/physical_drum_network.py')!=trained['identity']['network_sha256']
        or sha('scripts/physical_drum_features.py')!=trained['identity']['feature_sha256']):
        raise ValueError('Frozen physical model/data changed')
    with np.load(args.dataset/'normalization.npz') as d:mean,scale=d['mean'],d['scale']
    model=DrumOnsetNet();model.load_state_dict(load_file(str(args.model/'model.safetensors')),strict=True);model.eval()
    args.output.mkdir(parents=True);records=[]
    for row in manifest['records']:
        if row['split']!='test':continue
        path=args.dataset/(row['id']+'.npz')
        if sha(path)!=row['data_sha256']:raise ValueError('Reserved feature record changed')
        with np.load(path) as data:logits=infer(model,data['features'],mean,scale);times=data['times']
        out=args.output/(row['id']+'.npz');np.savez_compressed(out,logits=logits,times=times)
        records.append(dict(id=row['id'],sha256=sha(out),frames=len(times)))
    sakura=Path('test-data/sakura-benchmark/raw-pcm-audit-v1/drums.wav')
    audit=read(sakura.parent/'report.json')
    if sha(sakura)!=next(r['raw_float_sha256'] for r in audit['rows'] if r['role']=='drums'):
        raise ValueError('Original Sakura float audio changed')
    x,times=features(load_audio(sakura));logits=infer(model,x,mean,scale)
    out=args.output/'sakura.npz';np.savez_compressed(out,logits=logits,times=times)
    report=dict(model_sha256=trained['model_sha256'],training_report_sha256=sha(args.model/'report.json'),
        dataset_sha256=sha(args.dataset/'manifest.json'),records=records,
        sakura=dict(audio_sha256=sha(sakura),sha256=sha(out),frames=len(times)),runner_sha256=sha(__file__),
        enabled_in_application=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(reserved_songs=len(records),sakura_frames=len(times))),flush=True)


if __name__=='__main__':main()
