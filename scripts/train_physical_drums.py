"""Fit physical onsets, selecting epoch only with fixed validation recordings."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--protocol',type=Path,default=Path('docs/physical-drum-training-protocol.md'))
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous physical drum fits')
    import torch
    from safetensors.torch import save_file,load_file
    from scripts.physical_drum_network import DrumOnsetNet,normalized,infer
    from scripts.physical_drum_targets import known_targets
    torch.set_num_threads(2);torch.manual_seed(42);rng=np.random.default_rng(42)
    root=args.dataset;manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if sha('scripts/physical_drum_features.py')!=manifest['feature_code_sha256']:
        raise ValueError('Feature code differs from prepared inputs')
    if sha(root/'normalization.npz')!=manifest['normalization_sha256']:raise ValueError('Training-only scale changed')
    with np.load(root/'normalization.npz') as norm:mean,scale=norm['mean'],norm['scale']
    rows={s:[] for s in ('train','validation')}
    for row in manifest['records']:
        if row['split'] not in rows:continue
        path=root/(row['id']+'.npz')
        if sha(path)!=row['data_sha256']:raise ValueError('Physical record changed')
        with np.load(path) as data:
            rows[row['split']].append(dict(id=row['id'],features=data['features'],targets=data['targets']))
    examples=[]
    for song in rows['train']:
        x=normalized(song['features'],mean,scale);y,known=known_targets(song['targets']);length=len(x)
        x=np.pad(x,((32,32+256),(0,0)));y=np.pad(y,((0,256),(0,0)));known=np.pad(known,((0,256),(0,0)))
        for start in range(0,length,256):
            count=min(256,length-start)
            mask=known[start:start+256].T.copy() & (np.arange(256)<count)[None,:]
            examples.append((x[start:start+320].T.copy(),y[start:start+256].T.copy(),mask))
    x=torch.from_numpy(np.stack([e[0] for e in examples]));y=torch.from_numpy(np.stack([e[1] for e in examples]))
    masks=torch.from_numpy(np.stack([e[2] for e in examples]))
    model=DrumOnsetNet();optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    def loss(logits,target,mask=None):
        values=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(10.),reduction='none')
        return (values*mask).sum()/mask.sum().clamp_min(1) if mask is not None else values.mean()
    # Unknown labels produce zero gradient; complete labels retain old loss.
    check=torch.zeros((2,3,4),requires_grad=True);target=torch.zeros_like(check);mask=torch.ones_like(check,dtype=torch.bool)
    torch.testing.assert_close(loss(check,target,mask),loss(check,target))
    mask[:,2,:]=False;loss(check,target,mask).backward()
    if not torch.equal(check.grad[:,2,:],torch.zeros_like(check.grad[:,2,:])):raise ValueError('Unknown targets leaked into gradients')
    def validation():
        model.eval();total=0.;count=0
        for row in rows['validation']:
            logits=infer(model,row['features'],mean,scale)
            score=loss(torch.from_numpy(logits),torch.from_numpy(row['targets']))
            total+=float(score)*len(logits);count+=len(logits)
        return total/count
    # Full-vs-chunk inference must agree away from the external zero context.
    test=rows['validation'][0]['features'][:1400]
    a=infer(model,test,mean,scale,center=256)
    b=infer(model,test,mean,scale,center=2048)
    np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)
    args.output.mkdir(parents=True);began=time.monotonic();history=[]
    identity=dict(dataset_sha256=sha(root/'manifest.json'),network_sha256=sha('scripts/physical_drum_network.py'),
        runner_sha256=sha(__file__),feature_sha256=sha('scripts/physical_drum_features.py'),
        protocol_sha256=sha(args.protocol),partial_labels_sha256=sha('scripts/physical_drum_targets.py'),seed=42,epochs=30,learning_rate=.0003,
        batch_size=16,center_frames=256,context_frames=32,positive_weight=10,torch=torch.__version__,
        train_examples=len(examples),parameters=sum(p.numel() for p in model.parameters()),inference_chunk_max_error=float(np.max(abs(a-b))))
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    for name in ('scripts/train_physical_drums.py','scripts/physical_drum_network.py',args.protocol):
        shutil.copyfile(name,args.output/Path(name).name)
    best=float('inf');best_epoch=0
    for epoch in range(31):
        training=None
        if epoch:
            model.train();total=0.;count=0
            order=rng.permutation(len(examples))
            for start in range(0,len(order),16):
                selected=order[start:start+16]
                optimizer.zero_grad(set_to_none=True)
                logits=model(x[selected])[:,:,32:288]
                value=loss(logits,y[selected],masks[selected])
                if not torch.isfinite(value):raise ValueError('Nonfinite physical training loss')
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();total+=float(value.detach())*len(selected);count+=len(selected)
            training=total/count
        score=validation()
        if score<best:
            best=score;best_epoch=epoch;save_file(model.state_dict(),str(args.output/'model.safetensors'))
        row=dict(epoch=epoch,train_loss=training,validation_loss=score,best_epoch=best_epoch,seconds=round(time.monotonic()-began,1))
        history.append(row)
        (args.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    model.load_state_dict(load_file(str(args.output/'model.safetensors')),strict=True);model.eval()
    predictions={row['id']:infer(model,row['features'],mean,scale) for row in rows['validation']}
    np.savez_compressed(args.output/'validation-logits.npz',**predictions)
    report=dict(identity=identity,best_epoch=best_epoch,validation_loss=best,model_sha256=sha(args.output/'model.safetensors'),
        validation_logits_sha256=sha(args.output/'validation-logits.npz'),elapsed_seconds=time.monotonic()-began,
        enabled_in_application=False,notice='From-scratch physical onset model; no Sakura or chart targets. Epoch choice is not music acceptance.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
