"""Validate adaptation inputs and forward equivalence before fitting weights."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.adapt_piano_streams import augment,forward,load_items,load_model,make_graph,pairs,verify_source
from scripts.build_provenance import sha256
from scripts.research_piano_streams import predict


def same_graph(a,b):
    assert set(a.node_types)==set(b.node_types)
    assert set(a.edge_types)==set(b.edge_types)
    for node in a.node_types:
        assert set(a[node].keys())==set(b[node].keys())
        for key in a[node].keys():
            assert torch.equal(a[node][key],b[node][key]),(node,key)
    for edge in a.edge_types:
        assert torch.equal(a[edge].edge_index,b[edge].edge_index),edge


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--dataset',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous checks')
    revision=verify_source(args.source)
    sys.path.insert(0,str(args.source.resolve()))
    torch.set_num_threads(2);torch.manual_seed(42)
    manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    ready=[row for row in manifest['cases'] if row['status']=='ready']
    for field in ('work','sha256','cache_sha256'):
        groups={split:{r[field] for r in ready if r['split']==split} for split in ('train','validation','test')}
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            assert not groups[a]&groups[b],(field,a,b)
    # Test metadata is inspected for identity only; no test graph/prediction is loaded.
    items=load_items(args.dataset,manifest,'train')+load_items(args.dataset,manifest,'validation')
    model=load_model(args.source/'pretrained_models/model.ckpt').eval()
    checked=0;label_checks=0;max_error=0.;extended=0;target_voice=0;target_chord=0
    last_graph=last_target=None
    for index,item in enumerate(items):
        original=item['notes'].copy()
        for kind in ('clean','half','whole'):
            augmented=augment(original,item['bounds'],kind)
            for field in original.dtype.names:
                if field not in ('duration_div','duration_beat'):
                    assert np.array_equal(original[field],augmented[field]),field
            assert np.all(augmented['duration_div']>=original['duration_div'])
            extended+=int((augmented['duration_div']>original['duration_div']).sum())
            graph,target=make_graph(item,kind,True)
            assert 'staff' not in graph['note'] and 'voice' not in graph['note']
            assert all('truth' not in edge[1] for edge in graph.edge_types)
            assert target['voice_edges']<=pairs(graph['note','potential','note'].edge_index.numpy())
            assert target['chord_edges']<=pairs(graph['note','chord_potential','note'].edge_index.numpy())
            target_voice+=len(target['voice_edges']);target_chord+=len(target['chord_edges'])
            # Re-label actual inputs, then reconstruct features/edges from scratch.
            # Check every chunk, including sparse single-staff windows.
            changed=dict(item,notes=original.copy())
            changed['notes']['staff']=3-original['staff']
            changed['notes']['voice']=np.arange(len(original))%7+1
            shuffled,_=make_graph(changed,kind,True)
            same_graph(graph,shuffled);label_checks+=1
            if index<3:
                official=predict(model,graph)
                with torch.inference_mode():raw=forward(model,graph)
                for a,key in ((raw[0],'voice_logits'),(raw[1],'staff_scores'),(raw[3],'chord_logits')):
                    value=a.cpu().numpy()
                    np.testing.assert_allclose(value,official[key],rtol=1e-6,atol=1e-6)
                    if value.size:max_error=max(max_error,float(np.max(np.abs(value-official[key]))))
                last_graph,last_target=graph,target
            checked+=1
        assert np.array_equal(item['notes'],original)
    # One disposable backward step catches shape, dtype and autograd failures.
    candidate=copy.deepcopy(model).train()
    voice,staff,_,chord=forward(candidate,last_graph)
    loss=torch.nn.functional.cross_entropy(staff,last_target['staff'])
    for logits,label,weight in ((voice,last_target['voice'],5.),(chord,last_target['chord'],2.)):
        if logits.numel():loss+=torch.nn.functional.binary_cross_entropy_with_logits(logits,label,pos_weight=torch.tensor(weight))
    assert torch.isfinite(loss)
    loss.backward()
    gradients=[p.grad for p in candidate.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer=torch.optim.AdamW(candidate.parameters(),lr=1e-4,weight_decay=5e-4)
    torch.nn.utils.clip_grad_norm_(candidate.parameters(),1.);optimizer.step()
    assert all(torch.isfinite(p).all() for p in candidate.parameters())
    result=dict(source_revision=revision,dataset_sha256=sha256(args.dataset/'manifest.json'),
        training_runner_sha256=sha256(ROOT/'scripts/adapt_piano_streams.py'),runner_sha256=sha256(__file__),
        chunks=len(items),graphs_checked=checked,label_invariance_checks=label_checks,
        forward_max_abs_difference=max_error,extended_durations=extended,
        covered_voice_targets=target_voice,covered_chord_targets=target_chord,
        finite_backward_update=True,original_inputs_unchanged=True,test_graphs_loaded=False,
        oversized_chunks=[dict(name=i['name'],chunk=i['chunk'],notes=len(i['notes'])) for i in items if len(i['notes'])>500])
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
