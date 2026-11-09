"""Adapt frozen symbolic voice prediction to overlapping note tails locally."""
import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_provenance import sha256
from scripts.research_piano_streams import MODEL_SHA,REVISION,predict


def verify_source(source):
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(source),'diff','HEAD','--name-only'],text=True).strip()
    if revision!=REVISION or changes:raise ValueError('Unverified upstream source')
    return revision


def augment(array,bounds,kind):
    if kind not in ('clean','half','whole'):raise ValueError('Unknown duration augmentation')
    result=array.copy()
    if kind=='clean':return result
    factor=2 if kind=='half' else 1
    for i,note in enumerate(array):
        left,right=bounds[i];step=(right-left)/factor
        if step<=0:raise ValueError('Invalid bar duration')
        end=note['onset_div']+note['duration_div']
        new_end=left+np.ceil((end-left)/step-1e-8)*step
        repeat=array['onset_div'][(array['pitch']==note['pitch'])&(array['onset_div']>note['onset_div'])]
        if len(repeat):new_end=min(new_end,repeat.min())
        duration=max(int(note['duration_div']),int(round(new_end-note['onset_div'])))
        result['duration_div'][i]=duration
        result['duration_beat'][i]=duration*note['duration_beat']/note['duration_div']
    return result


def pairs(edges):
    return {tuple(map(int,p)) for p in np.asarray(edges).reshape(2,-1).T}


def make_graph(item,kind,expanded):
    from piano_svsep.utils import (hetero_graph_from_note_array,get_vocsep_features,HeteroScoreGraph,
        score_graph_to_pyg,get_measurewise_truth_edges,get_truth_chords_edges,get_pot_chord_edges,
        get_measurewise_pot_edges)
    original,measures,bounds=item['notes'],item['measures'],item['bounds']
    truth_voice=np.asarray(get_measurewise_truth_edges(original.copy(),measures)).reshape(2,-1)
    _,original_edges=hetero_graph_from_note_array(original,pot_edge_dist=0)
    original_chords=get_pot_chord_edges(original,original_edges[:2,original_edges[2]==0])
    truth_chord=get_truth_chords_edges(original,original_chords)
    array=augment(original,bounds,kind)
    array['staff']=1;array['voice']=1
    _,edges=hetero_graph_from_note_array(array,pot_edge_dist=0)
    graph=HeteroScoreGraph(get_vocsep_features(array),edges,note_array=array,name=item['name'])
    onsets=graph.get_edges_of_type('onset').numpy()
    if expanded:
        allowed=(measures[:,None]==measures[None,:])&(array['onset_div'][:,None]<array['onset_div'][None,:])
        potential=np.vstack(np.where(allowed))
        chords=onsets
    else:
        potential=get_measurewise_pot_edges(array,measures)
        chords=get_pot_chord_edges(array,onsets)
    graph.pot_edges=torch.tensor(potential).long()
    graph.pot_chord_edges=torch.tensor(chords).long()
    data=score_graph_to_pyg(graph)
    del data['note'].staff;del data['note'].voice
    voice_set,chord_set=pairs(truth_voice),pairs(truth_chord)
    targets=dict(voice=torch.tensor([tuple(p) in voice_set for p in potential.T],dtype=torch.float32),
        chord=torch.tensor([tuple(p) in chord_set for p in chords.T],dtype=torch.float32),
        staff=torch.tensor(original['staff'].astype('int64')-1),
        voice_edges=voice_set,chord_edges=chord_set)
    return data,targets


def forward(model,graph):
    from piano_svsep.utils import add_reverse_edges
    graph=graph.clone();add_reverse_edges(graph,mode=model.rev_edges)
    edges=graph.edge_index_dict
    potential=edges.pop(('note','potential','note'));chords=edges.pop(('note','chord_potential','note'))
    n=graph['note']
    # The frozen configuration disables edge-feature feedback. Encoder inputs
    # are pitches/durations and observed relation edges, never target labels.
    return model.module(graph.x_dict,edges,potential,chords,
        torch.zeros(len(n.x),dtype=torch.long),n.onset_div,n.duration_div,n.pitch,
        n.onset_beat,n.duration_beat,n.ts_beats,edge_attr_dict=None)


def load_items(folder,manifest,split):
    result=[]
    for row in manifest['cases']:
        if row['status']!='ready' or row['split']!=split:continue
        path=folder/row['cache']
        if sha256(path)!=row['cache_sha256']:raise ValueError('Score cache changed')
        with np.load(path,allow_pickle=False) as data:
            notes,measures,bounds=data['notes'],data['measures'],data['measure_bounds']
        unique=np.unique(measures);chunks=[];pending=[];size=0
        for measure in unique:
            count=int((measures==measure).sum())
            if pending and (len(pending)==4 or size+count>500):
                chunks.append(pending);pending=[];size=0
            pending.append(measure);size+=count
        if pending:chunks.append(pending)
        for index,chunk in enumerate(chunks):
            mask=np.isin(measures,chunk)
            result.append(dict(name=row['name'],chunk=index,notes=notes[mask],measures=measures[mask],bounds=bounds[mask]))
    return result


def score_counts(predicted,truth):
    return dict(tp=len(predicted&truth),fp=len(predicted-truth),fn=len(truth-predicted))


def f1(c):
    denominator=2*c['tp']+c['fp']+c['fn']
    return 2*c['tp']/denominator if denominator else 1.


def evaluate(model,items,expanded):
    model.eval();by_piece={}
    for item in items:
        for kind in ('clean','whole'):
            graph,target=make_graph(item,kind,expanded)
            result=predict(model,graph)
            key=(item['name'],kind)
            if key not in by_piece:by_piece[key]=dict(voice=Counter(),chord=Counter(),staff_correct=0,notes=0)
            row=by_piece[key]
            row['voice'].update(score_counts(pairs(result['voice_edges']),target['voice_edges']))
            row['chord'].update(score_counts(pairs(result['chord_edges']),target['chord_edges']))
            row['staff_correct']+=int((result['staff']==target['staff'].numpy()).sum())
            row['notes']+=len(result['staff'])
    totals={kind:dict(voice=Counter(),chord=Counter(),staff_correct=0,notes=0) for kind in ('clean','whole')}
    cases=[]
    for (name,kind),row in by_piece.items():
        row=dict(row,voice_f1=f1(row['voice']),chord_f1=f1(row['chord']),staff_accuracy=row['staff_correct']/row['notes'])
        cases.append(dict(name=name,input=kind,**row))
        for field in ('voice','chord'):totals[kind][field].update(row[field])
        for field in ('staff_correct','notes'):totals[kind][field]+=row[field]
    for row in totals.values():
        row.update(voice_f1=f1(row['voice']),chord_f1=f1(row['chord']),staff_accuracy=row['staff_correct']/row['notes'])
    criterion=float(np.mean([r[k] for r in totals.values() for k in ('voice_f1','chord_f1','staff_accuracy')]))
    return dict(totals=totals,criterion=criterion,cases=cases)


def load_model(path):
    from piano_svsep.models.pl_models import PLPianoSVSep
    if sha256(path)!=MODEL_SHA:raise ValueError('Unexpected pretrained weights')
    with torch.serialization.safe_globals([torch.nn.functional.relu]):checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    accepted=inspect.signature(PLPianoSVSep).parameters
    extra={k:v for k,v in checkpoint['hyper_parameters'].items() if k not in accepted}
    if extra!={'edge_feature_feedback':False,'after_encoder_frontend':False}:
        raise ValueError('Unexpected legacy hyperparameters')
    model=PLPianoSVSep(**{k:v for k,v in checkpoint['hyper_parameters'].items() if k in accepted})
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--dataset',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous training outputs')
    revision=verify_source(args.source)
    sys.path.insert(0,str(args.source));torch.set_num_threads(2);torch.manual_seed(42)
    manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    train=load_items(args.dataset,manifest,'train');validation=load_items(args.dataset,manifest,'validation')
    if not train or not validation:raise ValueError('Empty fit/validation partition')
    model_path=args.source/'pretrained_models/model.ckpt';model=load_model(model_path)
    args.output.mkdir(parents=True);shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    save=lambda name,obj:(args.output/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    identity=dict(source_revision=revision,dataset_sha256=sha256(args.dataset/'manifest.json'),model_sha256=MODEL_SHA,
        runner_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'docs/piano-stream-adaptation-protocol.md'),
        train_chunks=len(train),validation_chunks=len(validation),epochs=8,seed=42,
        oversized_chunks=[dict(split=split,name=i['name'],chunk=i['chunk'],notes=len(i['notes']))
            for split,items in (('train',train),('validation',validation)) for i in items if len(i['notes'])>500])
    save('identity.json',identity)
    baseline=evaluate(model,validation,True);save('validation-epoch-0.json',baseline)
    best_score,best_epoch=baseline['criterion'],0
    torch.save(dict(state_dict=model.state_dict(),epoch=0),args.output/'epoch-0.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-4)
    history=[];rng=random.Random(42);started=time.monotonic()
    for epoch in range(1,9):
        model.train();order=list(range(len(train)));rng.shuffle(order);losses=[]
        for position,index in enumerate(order):
            kind='clean' if (position+epoch)%2==0 else 'half' if (position+epoch)%4==1 else 'whole'
            graph,target=make_graph(train[index],kind,True)
            optimizer.zero_grad(set_to_none=True)
            voice,staff,_,chord=forward(model,graph)
            voice_loss=F.binary_cross_entropy_with_logits(voice,target['voice'],pos_weight=torch.tensor(5.)) if voice.numel() else voice.sum()
            chord_loss=F.binary_cross_entropy_with_logits(chord,target['chord'],pos_weight=torch.tensor(2.)) if chord.numel() else chord.sum()
            loss=voice_loss+chord_loss+F.cross_entropy(staff,target['staff'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
            if (position+1)%50==0:
                print(json.dumps(dict(epoch=epoch,chunk=position+1,total=len(order),loss=float(np.mean(losses)),elapsed=time.monotonic()-started)),flush=True)
        result=evaluate(model,validation,True);save(f'validation-epoch-{epoch}.json',result)
        torch.save(dict(state_dict=model.state_dict(),epoch=epoch),args.output/f'epoch-{epoch}.pt')
        if result['criterion']>best_score:best_score,best_epoch=result['criterion'],epoch
        row=dict(epoch=epoch,mean_train_loss=float(np.mean(losses)),validation=result['totals'],criterion=result['criterion'],best_epoch=best_epoch)
        history.append(row);save('progress.json',history);print(json.dumps(row),flush=True)
    shutil.copyfile(args.output/f'epoch-{best_epoch}.pt',args.output/'selected.pt')
    # Lock validation-selected identity before reading any test graphs/results.
    save('selection.json',dict(epoch=best_epoch,validation_score=best_score,sha256=sha256(args.output/'selected.pt')))
    test=load_items(args.dataset,manifest,'test')
    model=load_model(model_path);original=evaluate(model,test,False);save('test-original.json',original)
    expanded=evaluate(model,test,True);save('test-expanded.json',expanded)
    model.load_state_dict(torch.load(args.output/'selected.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    adapted=evaluate(model,test,True);save('test-adapted.json',adapted)
    result=dict(identity=identity,selected_epoch=best_epoch,selected_sha256=sha256(args.output/'selected.pt'),history=history,
        test={name:data['totals'] for name,data in (('original',original),('expanded',expanded),('adapted',adapted))},
        test_chunks=len(test),elapsed_seconds=time.monotonic()-started,enabled_in_application=False,musical_acceptance=False)
    save('report.json',result);print(json.dumps(result['test']),flush=True)


if __name__=='__main__':main()
