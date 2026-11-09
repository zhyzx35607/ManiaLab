"""Fit a fixed supervised successor-ranking auxiliary alongside the existing losses."""
import argparse
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.adapt_piano_event_overlap import paired_variants,KINDS
from scripts.adapt_piano_source_events import check_graphs,loss_terms
from scripts.adapt_piano_streams import forward,load_model,verify_source
from scripts.build_provenance import sha256
from scripts.check_piano_successor_decode import owned_arrays,join_owned,diagnose,aggregate
from scripts.piano_event_training import make_event_graph
from scripts.piano_successor_ranking import prepare_successor_ranking,successor_ranking_loss
from scripts.research_piano_streams import MODEL_SHA

ALPHA=.1


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ranking_inputs(payload):
    a,b=payload['potential_edges']
    return (payload['event_ids'][a],payload['notes']['onset_div'][b],
            payload['voice'],payload['voice_mask'])


def check_ranking_partition(items):
    owned={};counts=Counter()
    for item in items:
        p=item['payload'];sources,ticks,actual,mask=ranking_inputs(p)
        groups=prepare_successor_ranking(sources,ticks,actual,mask)
        counts['windows']+=1;counts['ranking_sources']+=len(groups)
        for source in set(map(str,sources[mask])):
            key=(item['name'],source)
            if key in owned:raise ValueError('Source supervision crosses owner windows')
            owned[key]=item['chunk']
        positive=set(map(str,sources[mask & (actual>.5)]))
        counts['positive_sources']+=len(positive)
        counts['supervised_sources']+=len(set(sources[mask]))
        counts['positive_edges']+=int((mask & (actual>.5)).sum())
    return dict(counts)


def combined_loss(outputs,targets,payload):
    base=loss_terms(outputs,targets)
    ranked=successor_ranking_loss(outputs[0],*ranking_inputs(payload))
    if base is None:
        if ranked is not None:raise ValueError('Ranking without underlying supervision')
        return None,None,None
    return base if ranked is None else base+ALPHA*ranked,base,ranked


def evaluate(model,groups,output,epoch):
    model.eval();variants={}
    for kind,items in groups.items():
        pieces=defaultdict(list);staffs=defaultdict(Counter)
        for index,item in enumerate(items):
            graph,target=make_event_graph(item)
            with torch.inference_mode():voice,staff,_,_=forward(model,graph)
            values=voice.numpy()
            if not np.isfinite(values).all() or not torch.isfinite(staff).all():
                raise ValueError('Nonfinite validation output')
            filename=f'epoch-{epoch}-{kind}-{index:03d}.npz'
            np.savez_compressed(output/filename,voice_logits=values,staff_scores=staff.numpy())
            pieces[item['name']].append(owned_arrays(item['payload'],values))
            sm=target['staff_mask']
            staffs[item['name']]['correct']+=int((staff[sm].argmax(dim=1)==target['staff'][sm]).sum())
            staffs[item['name']]['notes']+=int(sm.sum())
        rows=[]
        for name,parts in sorted(pieces.items()):
            metrics,_=diagnose(join_owned(parts))
            rows.append(dict(name=name,**metrics,staff_correct=staffs[name]['correct'],staff_notes=staffs[name]['notes']))
        totals=aggregate(rows)
        variants[kind]=dict(voice=totals['raw'],macro_voice_f1=totals['raw_macro_f1'],
            decoded=totals['decoded'],decoded_macro_voice_f1=totals['decoded_macro_f1'],
            decode_losses={key:value for key,value in totals.items() if key.startswith('removed_')},
            staff_correct=sum(r['staff_correct'] for r in rows),staff_notes=sum(r['staff_notes'] for r in rows),cases=rows)
    return dict(variants=variants,selection_score=float(np.mean([r['macro_voice_f1'] for r in variants.values()])))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--check-only',action='store_true')
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous runs')
    revision=verify_source(args.source);sys.path.insert(0,str(args.source.resolve()))
    torch.set_num_threads(2);torch.manual_seed(42)
    base=ROOT/'test-data/piano-svsep'
    reference=base/'source-event-overlap-adaptation-v1'
    reference_identity=read(reference/'identity.json')
    folders=dict(clean=base/'source-event-dataset-v1',plus1=base/'source-event-overlap-v1/plus1',
                 plus2=base/'source-event-overlap-v1/plus2')
    for kind,folder in folders.items():
        if sha256(folder/'manifest.json')!=reference_identity['dataset_sha256'][kind]:
            raise ValueError('Different dataset from BCE-only reference')
    train=paired_variants(folders,'train');validation=paired_variants(folders,'validation')
    coverage={split:{kind:check_ranking_partition(items) for kind,items in groups.items()}
              for split,groups in (('train',train),('validation',validation))}
    for split in coverage:
        assert coverage[split]['clean']==coverage[split]['plus1']==coverage[split]['plus2']
    if coverage['train']['clean']['ranking_sources']!=14212 or coverage['validation']['clean']['ranking_sources']!=981:
        raise ValueError('Ranking coverage differs from independent audit')
    model=load_model(args.source/'pretrained_models/model.ckpt')
    args.output.mkdir(parents=True)
    save=lambda name,value:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    code=dict(runner=Path(__file__),ranking=ROOT/'scripts/piano_successor_ranking.py',
        decoder=ROOT/'scripts/piano_successor_decode.py',scoring=ROOT/'scripts/check_piano_successor_decode.py',
        event_graph=ROOT/'scripts/piano_event_training.py',base_loss=ROOT/'scripts/adapt_piano_source_events.py',
        variant_loader=ROOT/'scripts/adapt_piano_event_overlap.py',forward=ROOT/'scripts/adapt_piano_streams.py',
        protocol=ROOT/'docs/piano-successor-ranking-protocol.md')
    identity=dict(source_revision=revision,pretrained_sha256=MODEL_SHA,
        reference_report_sha256=sha256(reference/'report.json'),reference_epoch0_sha256=sha256(reference/'validation-epoch-0.json'),
        dataset_sha256={k:sha256(v/'manifest.json') for k,v in folders.items()},
        code_sha256={k:sha256(v) for k,v in code.items()},seed=42,epochs=4,lr=1e-4,weight_decay=5e-4,
        voice_positive_weight=5.,rank_coefficient=ALPHA,rank_margin=0,threshold=.5,
        selection='Equal mean of clean/plus1/plus2 raw per-piece macro F1; epoch0 participates; earliest maximum.',
        input_schedule='kind[(original_window_index + epoch - 1) % 3]; seeded shuffled traversal',
        unknown_edge_rank_supervision=False,test_loaded=False,application_enabled=False,musical_acceptance=False)
    for name,path in code.items():shutil.copyfile(path,args.output/(name+'-snapshot'+path.suffix))
    save('identity.json',identity);save('coverage.json',coverage)
    for kind,items in validation.items():
        for index,item in enumerate(items):
            np.savez_compressed(args.output/f'{kind}-graph-{index:03d}.npz',**item['payload'],
                                metadata_json=np.asarray(json.dumps(item['metadata'])))
    checks=check_graphs([item for groups in (train,validation) for kind in KINDS for item in groups[kind]],model)
    disposable=copy.deepcopy(model).train()
    sample=next(item for item in train['clean'] if prepare_successor_ranking(*ranking_inputs(item['payload'])))
    graph,target=make_event_graph(sample)
    total,base_loss,rank_loss=combined_loss(forward(disposable,graph),target,sample['payload'])
    if rank_loss is None or not torch.isfinite(total):raise ValueError('Invalid ranking control')
    total.backward()
    gradients=[p.grad for p in disposable.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite ranking gradients')
    torch.nn.utils.clip_grad_norm_(disposable.parameters(),1.)
    optimizer=torch.optim.AdamW(disposable.parameters(),lr=1e-4,weight_decay=5e-4);optimizer.step()
    if not all(torch.isfinite(p).all() for p in disposable.parameters()):raise ValueError('Nonfinite ranking update')
    checks.update(rank_backward_finite=True,rank_control_loss=float(rank_loss.detach()),rank_control_base_loss=float(base_loss.detach()))
    del disposable,optimizer
    save('checks.json',checks);print(json.dumps(dict(stage='checks',**checks)),flush=True)
    if args.check_only:return
    started=time.monotonic()
    baseline=evaluate(model,validation,args.output,0)
    old=read(reference/'validation-epoch-0.json')
    for kind,row in baseline['variants'].items():
        for key in ('tp','fp','fn','tn'):assert row['voice'][key]==old['variants'][kind]['voice'][key]
        assert abs(row['macro_voice_f1']-old['variants'][kind]['macro_voice_f1'])<1e-12
        old_cases={r['name']:r for r in old['variants'][kind]['cases']}
        for case in row['cases']:
            assert all(case['raw'][k]==old_cases[case['name']]['voice'][k] for k in ('tp','fp','fn','tn'))
    save('validation-epoch-0.json',baseline)
    torch.save(dict(state_dict=model.state_dict(),epoch=0),args.output/'epoch-0.pt')
    best_score,best_epoch=baseline['selection_score'],0
    print(json.dumps(dict(epoch=0,selection_score=best_score)),flush=True)
    torch.manual_seed(42);rng=random.Random(42)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-4)
    history=[]
    for epoch in range(1,5):
        model.train();order=list(range(len(train['clean'])));rng.shuffle(order)
        losses=[];base_losses=[];rank_losses=[];used=Counter();skipped=Counter()
        for position,index in enumerate(order):
            kind=KINDS[(index+epoch-1)%len(KINDS)];item=train[kind][index]
            graph,target=make_event_graph(item);optimizer.zero_grad(set_to_none=True)
            total,base_loss,rank_loss=combined_loss(forward(model,graph),target,item['payload'])
            if total is None:skipped[kind]+=1;continue
            if not torch.isfinite(total):raise ValueError('Nonfinite training loss')
            total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(total.detach()));base_losses.append(float(base_loss.detach()))
            if rank_loss is not None:rank_losses.append(float(rank_loss.detach()))
            used[kind]+=1
            if (position+1)%50==0:
                print(json.dumps(dict(epoch=epoch,window=position+1,total=len(order),mean_loss=float(np.mean(losses)),
                    elapsed=time.monotonic()-started)),flush=True)
        if not losses:raise ValueError('No training supervision')
        result=evaluate(model,validation,args.output,epoch)
        save(f'validation-epoch-{epoch}.json',result)
        torch.save(dict(state_dict=model.state_dict(),epoch=epoch),args.output/f'epoch-{epoch}.pt')
        if result['selection_score']>best_score:best_score,best_epoch=result['selection_score'],epoch
        row=dict(epoch=epoch,selection_score=result['selection_score'],best_epoch=best_epoch,used=used,skipped=skipped,
            mean_total_loss=float(np.mean(losses)),mean_base_loss=float(np.mean(base_losses)),
            mean_rank_loss=float(np.mean(rank_losses)) if rank_losses else None,rank_updates=len(rank_losses),
            variants={kind:{key:value for key,value in report.items() if key!='cases'} for kind,report in result['variants'].items()})
        history.append(row);save('progress.json',history);print(json.dumps(row),flush=True)
    if any(sha256(path)!=identity['code_sha256'][name] for name,path in code.items()):raise ValueError('Training code changed')
    if any(sha256(folder/'manifest.json')!=identity['dataset_sha256'][kind] for kind,folder in folders.items()):
        raise ValueError('Dataset changed')
    shutil.copyfile(args.output/f'epoch-{best_epoch}.pt',args.output/'selected.pt')
    report=dict(identity=identity,coverage=coverage,checks=checks,baseline=baseline,history=history,selected_epoch=best_epoch,
        selected_sha256=sha256(args.output/'selected.pt'),elapsed_seconds=time.monotonic()-started,
        musical_acceptance=False,application_enabled=False)
    report['artifacts_sha256']={p.name:sha256(p) for p in args.output.iterdir()}
    save('report.json',report)
    print(json.dumps(dict(selected_epoch=best_epoch,selection_score=best_score,elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
