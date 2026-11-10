"""Measure fixed-weight context sensitivity without choosing an LN policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.adapt_piano_streams import augment,f1,load_model,make_graph,pairs,verify_source
from scripts.build_provenance import sha256
from scripts.evaluate_piano_stream_adaptation import component_pair_counts,graph_for
from scripts.research_piano_streams import expression_choices,predict
from manialab.domain import note_from


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def windows(measures):
    chunks=[];pending=[];size=0
    for measure in np.unique(measures):
        count=int((measures==measure).sum())
        if pending and (len(pending)==4 or size+count>500):
            chunks.append(np.flatnonzero(np.isin(measures,pending)));pending=[];size=0
        pending.append(measure);size+=count
    if pending:chunks.append(np.flatnonzero(np.isin(measures,pending)))
    return chunks


def full_items(folder,manifest):
    result=[]
    for row in manifest['cases']:
        if row['status']!='ready' or row['split']!='validation':continue
        path=folder/row['cache']
        if sha256(path)!=row['cache_sha256']:raise ValueError('Score cache changed')
        with np.load(path,allow_pickle=False) as data:
            result.append(dict(name=row['name'],notes=data['notes'],measures=data['measures'],bounds=data['measure_bounds']))
    return result


def contexts(model,count,windows_,build):
    full_graph=build(np.arange(count));full=predict(model,full_graph)
    observed=('onset','consecutive','during','rest')
    full_observed={kind:pairs(full_graph['note',kind,'note'].edge_index.numpy()) for kind in observed}
    local_observed={kind:set() for kind in observed};window_ids=np.full(count,-1,dtype=int)
    node_keys=('staff','staff_scores')
    edge_keys=('edges','voice_edges','chord_edges','potential_voice','potential_chord')
    chunked={key:np.empty_like(full[key]) for key in node_keys}
    accum={key:[] for key in (*edge_keys,'voice_logits','chord_logits')}
    visited=np.zeros(count,dtype=int)
    for window_id,indices in enumerate(windows_):
        graph=build(indices)
        for field in ('x','pitch','onset_div','duration_div','onset_beat','duration_beat','ts_beats'):
            assert torch.equal(graph['note'][field],full_graph['note'][field][indices]),field
        window_ids[indices]=window_id
        for kind in observed:
            local_observed[kind].update(pairs(indices[graph['note',kind,'note'].edge_index.numpy()]))
        local=predict(model,graph);visited[indices]+=1
        for key in node_keys:chunked[key][indices]=local[key]
        for key in edge_keys:accum[key].append(indices[local[key]])
        for key in ('voice_logits','chord_logits'):accum[key].append(local[key])
    assert np.all(visited==1),'Every note must be predicted exactly once'
    for key in edge_keys:chunked[key]=np.concatenate(accum[key],axis=1)
    for key in ('voice_logits','chord_logits'):chunked[key]=np.concatenate(accum[key])
    for key in ('potential_voice','potential_chord'):
        assert pairs(full[key])==pairs(chunked[key]),key
    edges=chunked['edges']
    _,chunked['components']=connected_components(coo_matrix((np.ones(edges.shape[1]),edges),shape=(count,count)),directed=False)
    audit={}
    for kind in observed:
        total=full_observed[kind];local=local_observed[kind]
        internal={(a,b) for a,b in total if window_ids[a]==window_ids[b]}
        audit[kind]=dict(full=len(total),window=len(local),full_cross_window=len(total-internal),
            full_internal_missing_in_windows=len(internal-local),window_only=len(local-total))
    def invalid_rest(edges):
        end=(full_graph['note'].onset_div+full_graph['note'].duration_div).numpy()
        onset=full_graph['note'].onset_div.numpy()
        return sum(onset[b]<=end[a] for a,b in edges)
    audit['rest']['full_nonforward']=int(invalid_rest(full_observed['rest']))
    audit['rest']['window_nonforward']=int(invalid_rest(local_observed['rest']))
    # A scalar unicode array keeps snapshots loadable with allow_pickle=False.
    chunked['observed_edge_audit_json']=np.asarray(json.dumps(audit))
    return full,chunked


def score_prediction(model,item,kind,expanded):
    original=item['notes'].copy()
    frozen=augment(original,item['bounds'],kind)
    def build(indices):
        local=dict(name=item['name'],notes=frozen[indices],measures=item['measures'][indices],bounds=item['bounds'][indices])
        # Only input graphs are used; the original labels are evaluated separately.
        graph,_=make_graph(local,'clean',expanded)
        return graph
    full,chunked=contexts(model,len(original),windows(item['measures']),build)
    assert np.array_equal(original,item['notes'])
    return full,chunked


def logits_delta(full,chunked,kind):
    edge_key='potential_'+kind;value_key=kind+'_logits'
    a={tuple(map(int,p)):float(v) for p,v in zip(full[edge_key].T,full[value_key])}
    b={tuple(map(int,p)):float(v) for p,v in zip(chunked[edge_key].T,chunked[value_key])}
    assert set(a)==set(b)
    values=np.asarray([abs(a[k]-b[k]) for k in a])
    return dict(edges=len(a),max_abs=float(values.max()) if values.size else 0.,
        mean_abs=float(values.mean()) if values.size else 0.,
        threshold_flips=sum((a[k]>0)!=(b[k]>0) for k in a))


def differences(full,chunked,measures):
    changed=total=0
    for measure in np.unique(measures):
        indices=np.flatnonzero(measures==measure)
        a,b=full['components'][indices],chunked['components'][indices]
        mask=np.triu(np.ones((len(indices),len(indices)),dtype=bool),1)
        changed+=int((((a[:,None]==a[None,:])!=(b[:,None]==b[None,:]))[mask]).sum())
        total+=int(mask.sum())
    staff_delta=np.abs(full['staff_scores']-chunked['staff_scores'])
    return dict(notes=len(measures),staff_label_flips=int((full['staff']!=chunked['staff']).sum()),
        staff_scores_max_abs=float(staff_delta.max()),staff_scores_mean_abs=float(staff_delta.mean()),
        voice_logits=logits_delta(full,chunked,'voice'),chord_logits=logits_delta(full,chunked,'chord'),
        predicted_voice_edges_changed=len(pairs(full['voice_edges'])^pairs(chunked['voice_edges'])),
        predicted_chord_edges_changed=len(pairs(full['chord_edges'])^pairs(chunked['chord_edges'])),
        same_measure_pairs=total,component_pair_disagreements=changed,
        observed_edges=json.loads(str(chunked['observed_edge_audit_json'])))


def validation_totals(cases):
    result={}
    for model in ('original','adapted'):
        result[model]={}
        for kind in ('clean','whole'):
            subset=[r for r in cases if r['model']==model and r['input']==kind]
            result[model][kind]={}
            for context in ('full','four_measures'):
                counts=Counter(tp=0,fp=0,fn=0)
                for row in subset:counts.update(row['scores'][context]['counts'])
                result[model][kind][context]=dict(counts=dict(counts),f1=f1(counts),
                    piece_macro_f1=float(np.mean([r['scores'][context]['f1'] for r in subset])))
            result[model][kind]['staff_label_flips']=sum(r['differences']['staff_label_flips'] for r in subset)
            result[model][kind]['notes']=sum(r['differences']['notes'] for r in subset)
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--dataset',type=Path,default=ROOT/'test-data/piano-svsep/dataset-v2')
    cli.add_argument('--run',type=Path,default=ROOT/'test-data/piano-svsep/adaptation-v1')
    cli.add_argument('--evaluation',type=Path,default=ROOT/'test-data/piano-svsep/adaptation-evaluation-v1')
    cli.add_argument('--frozen',type=Path,default=ROOT/'test-data/piano-svsep/probe-v3')
    cli.add_argument('--analysis',type=Path,default=ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous context runs')
    revision=verify_source(args.source);sys.path.insert(0,str(args.source.resolve()))
    torch.set_num_threads(2);torch.manual_seed(42)
    selection=read(args.run/'selection.json');trained=read(args.run/'report.json');evaluated=read(args.evaluation/'report.json')
    selected_sha=sha256(args.run/'selected.pt')
    assert selection['sha256']==trained['selected_sha256']==evaluated['identity']['selected_sha256']==selected_sha
    assert sha256(args.dataset/'manifest.json')==trained['identity']['dataset_sha256']
    assert sha256(args.analysis)==evaluated['identity']['analysis_sha256']
    items=full_items(args.dataset,read(args.dataset/'manifest.json'))
    model=load_model(args.source/'pretrained_models/model.ckpt').eval()
    original={k:v.clone() for k,v in model.state_dict().items()}
    adapted=torch.load(args.run/'selected.pt',map_location='cpu',weights_only=True)['state_dict']
    args.output.mkdir(parents=True);shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    save=lambda name,value:(args.output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    analysis=read(args.analysis);notes=[note_from(n) for n in analysis['candidates'] if n['voice']=='piano']
    rows=read(args.frozen/'sakura-graph-input.json');measures=np.asarray([r['tick']//1920 for r in rows])
    rebuilt=[];beat=60000/74
    for n in notes:
        start=max(0,round((n.start-22.5-178)/beat*4)*120)
        end=round((n.end-22.5-178)/beat*4)*120
        rebuilt.append(dict(id=n.id,pitch=n.pitch,tick=start,duration=max(120,end-start)))
    rebuilt.sort(key=lambda r:(r['tick'],r['pitch'],r['id']))
    assert rows==rebuilt and len({r['id'] for r in rows})==len(rows)
    before=[n.data() for n in notes];cases=[];sakura=[]
    for name,weights,expanded in (('original',original,False),('adapted',adapted,True)):
        model.load_state_dict(weights,strict=True);model.eval()
        for item in items:
            for kind in ('clean','whole'):
                full,chunked=score_prediction(model,item,kind,expanded)
                scores={}
                for context,pred in (('full',full),('four_measures',chunked)):
                    counts=component_pair_counts(item,pred['components'])
                    scores[context]=dict(counts=dict(counts),f1=f1(counts),
                        staff_accuracy=float(np.mean(pred['staff']==item['notes']['staff']-1)))
                    np.savez_compressed(args.output/f'{name}-{item["name"]}-{kind}-{context}.npz',**pred)
                cases.append(dict(model=name,name=item['name'],input=kind,scores=scores,
                    differences=differences(full,chunked,item['measures'])))
        full,chunked=contexts(model,len(rows),windows(measures),lambda indices:graph_for([rows[i] for i in indices],expanded))
        with np.load(args.evaluation/(name+'-sakura-predictions.npz'),allow_pickle=False) as baseline:
            for key in baseline.files:np.testing.assert_array_equal(full[key],baseline[key])
        policies={}
        for context,pred in (('full',full),('four_measures',chunked)):
            np.savez_compressed(args.output/f'{name}-sakura-{context}.npz',**pred)
            assignments={r['id']:dict(staff=int(pred['staff'][i]),components=int(pred['components'][i])) for i,r in enumerate(rows)}
            policies[context]={mode:expression_choices(notes,analysis['structure']['segments'],assignments,mode) for mode in ('staff','components')}
            for mode,choices in policies[context].items():save(f'{name}-sakura-{context}-{mode}.json',choices)
        changes={mode:dict(full_holds=sum(c['hold'] for c in policies['full'][mode].values()),
            window_holds=sum(c['hold'] for c in policies['four_measures'][mode].values()),
            hold_tap_flips=sum(c['hold']!=policies['four_measures'][mode][key]['hold'] for key,c in policies['full'][mode].items()))
            for mode in ('staff','components')}
        row=dict(model=name,differences=differences(full,chunked,measures),expression_changes=changes,
            full_prediction_reproduced=True)
        sakura.append(row);print(json.dumps(row),flush=True)
    assert [n.data() for n in notes]==before
    identity=dict(source_revision=revision,selected_sha256=selected_sha,dataset_sha256=sha256(args.dataset/'manifest.json'),
        runner_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'docs/piano-stream-context-protocol.md'),
        analysis_sha256=sha256(args.analysis),graph_rows_sha256=sha256(args.frozen/'sakura-graph-input.json'),
        evaluation_sha256=sha256(args.evaluation/'report.json'))
    report=dict(identity=identity,validation=validation_totals(cases),cases=cases,sakura=sakura,
        test_data_used=False,model_selection_changed=False,raw_candidates_unchanged=True,
        enabled_in_application=False,musical_acceptance=False)
    save('report.json',report);print(json.dumps(report['validation']),flush=True)


if __name__=='__main__':main()
