"""Replay fixed stream controls and audio candidates after model selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.adapt_piano_streams import f1,load_items,load_model,make_graph,verify_source
from scripts.build_provenance import sha256
from scripts.research_piano_streams import controls,expression_choices,graph_from_rows,pair_metrics,predict
from manialab.domain import note_from,UNOBSERVED_RELEASES


def graph_for(rows,expanded):
    graph=graph_from_rows(rows)
    del graph['note'].staff;del graph['note'].voice
    if expanded:
        onset=np.asarray([r['tick'] for r in rows])
        allowed=(onset[:,None]//1920==onset[None,:]//1920)&(onset[:,None]<onset[None,:])
        graph['note','potential','note'].edge_index=torch.tensor(np.vstack(np.where(allowed))).long()
        graph['note','chord_potential','note'].edge_index=graph['note','onset','note'].edge_index.clone()
    return graph


def changed_pieces(folder):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    models={name:read(folder/f'test-{name}.json')['cases'] for name in ('original','expanded','adapted')}
    rows=[]
    comparisons=(('original','expanded'),('expanded','adapted'),('original','adapted'))
    for before,after in comparisons:
        old={(row['name'],row['input']):row for row in models[before]}
        for item in models[after]:
            prior=old[item['name'],item['input']]
            rows.append(dict(comparison=before+'-to-'+after,name=item['name'],input=item['input'],
                delta={key:item[key]-prior[key] for key in ('voice_f1','chord_f1','staff_accuracy')}))
    summary={}
    for before,after in comparisons:
        key=before+'-to-'+after;subset=[r for r in rows if r['comparison']==key]
        summary[key]={kind:{metric:dict(better=sum(r['delta'][metric]>1e-9 for r in subset if r['input']==kind),
            equal=sum(abs(r['delta'][metric])<=1e-9 for r in subset if r['input']==kind),
            worse=sum(r['delta'][metric]<-1e-9 for r in subset if r['input']==kind))
            for metric in ('voice_f1','chord_f1','staff_accuracy')} for kind in ('clean','whole')}
    return rows,summary


def component_pair_counts(item,groups):
    """Measure-local score-voice pairs, including transitive grouping errors."""
    result=Counter(tp=0,fp=0,fn=0)
    choose_two=lambda counts:sum(n*(n-1)//2 for n in counts.values())
    for measure in np.unique(item['measures']):
        mask=item['measures']==measure
        notes=item['notes'][mask];labels=np.asarray(groups)[mask]
        truth=Counter(zip(map(int,notes['staff']),map(int,notes['voice'])))
        predicted=Counter(map(int,labels))
        intersection=Counter(zip(map(int,notes['staff']),map(int,notes['voice']),map(int,labels)))
        tp=choose_two(intersection)
        result.update(tp=tp,fp=choose_two(predicted)-tp,fn=choose_two(truth)-tp)
    return result


def component_evaluation(model,items,expanded):
    pieces={}
    for item in items:
        for kind in ('clean','whole'):
            graph,_=make_graph(item,kind,expanded)
            predicted=predict(model,graph)
            key=(item['name'],kind)
            pieces.setdefault(key,Counter(tp=0,fp=0,fn=0)).update(component_pair_counts(item,predicted['components']))
    cases=[dict(name=name,input=kind,counts=dict(counts),f1=f1(counts)) for (name,kind),counts in pieces.items()]
    totals={}
    for kind in ('clean','whole'):
        counts=Counter(tp=0,fp=0,fn=0)
        for case in cases:
            if case['input']==kind:counts.update(case['counts'])
        totals[kind]=dict(counts=dict(counts),f1=f1(counts),piece_macro_f1=float(np.mean([r['f1'] for r in cases if r['input']==kind])))
    return dict(totals=totals,cases=cases)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--dataset',type=Path,default=ROOT/'test-data/piano-svsep/dataset-v2')
    cli.add_argument('--frozen',type=Path,default=ROOT/'test-data/piano-svsep/probe-v3')
    cli.add_argument('--analysis',type=Path,default=ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous evaluations')
    revision=verify_source(args.source);sys.path.insert(0,str(args.source.resolve()))
    torch.set_num_threads(2);torch.manual_seed(42)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    result=read(args.run/'report.json');selection=read(args.run/'selection.json')
    if sha256(args.run/'selected.pt')!=selection['sha256']:raise ValueError('Selected weights changed')
    if result['selected_sha256']!=selection['sha256']:raise ValueError('Training result identity mismatch')
    if sha256(args.run/'runner-snapshot.py')!=result['identity']['runner_sha256']:
        raise ValueError('Training snapshot changed')
    if sha256(args.dataset/'manifest.json')!=result['identity']['dataset_sha256']:
        raise ValueError('Evaluation dataset differs from selected training run')
    scores=[read(args.run/f'validation-epoch-{epoch}.json')['criterion'] for epoch in range(9)]
    if max(range(9),key=scores.__getitem__)!=selection['epoch']:
        raise ValueError('Selection does not match fixed validation criterion')
    frozen=read(args.frozen/'report.json')
    if sha256(args.analysis)!=frozen['identity']['analysis_sha256']:
        raise ValueError('Frozen audio candidates changed')
    model=load_model(args.source/'pretrained_models/model.ckpt').eval()
    original={k:v.clone() for k,v in model.state_dict().items()}
    adapted=torch.load(args.run/'selected.pt',map_location='cpu',weights_only=True)['state_dict']
    test=load_items(args.dataset,read(args.dataset/'manifest.json'),'test')
    args.output.mkdir(parents=True);shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    save=lambda name,obj:(args.output/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    identity=dict(source_revision=revision,training_report_sha256=sha256(args.run/'report.json'),
        selected_sha256=selection['sha256'],selected_epoch=selection['epoch'],
        frozen_report_sha256=sha256(args.frozen/'report.json'),analysis_sha256=sha256(args.analysis),
        runner_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'docs/piano-stream-adaptation-protocol.md'),
        grouping_protocol_sha256=sha256(ROOT/'docs/piano-stream-evaluation-protocol.md'),
        reference_chart_read=False,enabled_in_application=False)
    save('identity.json',identity)
    analysis=read(args.analysis)
    notes=[note_from(n) for n in analysis['candidates'] if n['voice']=='piano']
    before=[n.data() for n in notes]
    rows=read(args.frozen/'sakura-graph-input.json')
    assert {r['id'] for r in rows}=={n.id for n in notes}
    control_rows=controls();control_results=[];sakura_results=[];test_components={}
    for name,weights,expanded in (('original',original,False),('expanded',original,True),('adapted',adapted,True)):
        model.load_state_dict(weights,strict=True);model.eval()
        for control,events in control_rows.items():
            predicted=predict(model,graph_for(events,expanded))
            if name=='original':
                with np.load(args.frozen/(control+'.npz'),allow_pickle=False) as previous:
                    for key in previous.files:np.testing.assert_array_equal(predicted[key],previous[key])
            np.savez_compressed(args.output/(name+'-'+control+'.npz'),**predicted)
            row=dict(model=name,name=control,notes=len(events),components=len(set(predicted['components'])),
                voice_pairs=pair_metrics(events,predicted['components']),staff_pairs=pair_metrics(events,predicted['staff']))
            control_results.append(row);print(json.dumps(row),flush=True)
        start=time.monotonic();predicted=predict(model,graph_for(rows,expanded));elapsed=time.monotonic()-start
        if name=='original':
            with np.load(args.frozen/'sakura-predictions.npz',allow_pickle=False) as previous:
                for key in previous.files:np.testing.assert_array_equal(predicted[key],previous[key])
        np.savez_compressed(args.output/(name+'-sakura-predictions.npz'),**predicted)
        assignments={r['id']:dict(staff=int(predicted['staff'][i]),components=int(predicted['components'][i])) for i,r in enumerate(rows)}
        save(name+'-sakura-assignments.json',assignments)
        policies={kind:expression_choices(notes,analysis['structure']['segments'],assignments,kind) for kind in ('all','staff','components')}
        baseline=policies['all'];expression=[]
        for kind,choices in policies.items():
            assert set(choices)==set(baseline)
            for n in notes:
                if n.id not in choices:continue
                generated=choices[n.id]['note']
                assert generated['start']==n.start and generated['pitch']==n.pitch
                assert generated['end'] in (n.start,n.end)
                assert not choices[n.id]['hold'] or n.release_reason not in UNOBSERVED_RELEASES
            expression.append(dict(mode=kind,attacks=len(choices),holds=sum(c['hold'] for c in choices.values()),
                newly_held=sum(c['hold'] and not baseline[k]['hold'] for k,c in choices.items()),
                newly_tapped=sum(not c['hold'] and baseline[k]['hold'] for k,c in choices.items())))
            save(name+'-sakura-expression-'+kind+'.json',choices)
        sizes=Counter(map(int,predicted['components']))
        row=dict(model=name,seconds=elapsed,raw_notes=len(notes),voice_components=len(sizes),
            singleton_notes=sum(size==1 for size in sizes.values()),staff_counts=dict(Counter(map(int,predicted['staff']))),
            voice_edges=predicted['voice_edges'].shape[1],chord_edges=predicted['chord_edges'].shape[1],expression=expression)
        sakura_results.append(row);print(json.dumps(row),flush=True)
        groups=component_evaluation(model,test,expanded)
        save(name+'-test-components.json',groups);test_components[name]=groups['totals']
        print(json.dumps(dict(model=name,test_components=groups['totals'])),flush=True)
    assert [n.data() for n in notes]==before
    changes,change_summary=changed_pieces(args.run);save('test-piece-deltas.json',changes)
    save('report.json',dict(identity=identity,controls=control_results,sakura=sakura_results,
        test=result['test'],test_changes=change_summary,test_components=test_components,raw_candidates_unchanged=True,
        frozen_baseline_reproduced=True,musical_acceptance=False))


if __name__=='__main__':main()
