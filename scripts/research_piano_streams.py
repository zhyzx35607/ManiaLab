"""Frozen symbolic voice/staff model probe; never changes raw audio candidates."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256
from manialab.domain import Note, UNOBSERVED_RELEASES, note_from
from manialab.expression_articulation import articulation_policy

REVISION = '1462e7c28d7adaae033150883a7cb00238ee364a'
MODEL_SHA = '1f71ab8b6c7cd271e18adfba5ccbcd634a8530a44e46aabede27e15ce84dbe32'
DTYPE = [('pitch','i4'),('onset_div','i4'),('duration_div','i4'),('onset_beat','f4'),
         ('duration_beat','f4'),('ts_beats','i4'),('staff','i4'),('voice','i4')]


def note_array(rows):
    """Symbolic graph fields only; labels are placeholders, never targets."""
    data = np.zeros(len(rows), dtype=DTYPE)
    for i, row in enumerate(rows):
        onset, duration = row['tick'], row['duration']
        data[i] = (row['pitch'], onset, duration, onset/480, duration/480, 4, 1, 1)
    return data


def graph_from_rows(rows):
    from piano_svsep.utils import (hetero_graph_from_note_array, get_vocsep_features,
        HeteroScoreGraph, get_measurewise_pot_edges, get_pot_chord_edges, score_graph_to_pyg)
    array = note_array(rows)
    _, edges = hetero_graph_from_note_array(array, pot_edge_dist=0)
    graph = HeteroScoreGraph(get_vocsep_features(array), edges, note_array=array, name='unlabelled')
    graph.pot_edges = torch.tensor(get_measurewise_pot_edges(array, array['onset_div']//1920)).long()
    graph.pot_chord_edges = torch.tensor(get_pot_chord_edges(array, graph.get_edges_of_type('onset').numpy())).long()
    return score_graph_to_pyg(graph)


def predict(model, graph):
    captured = []
    handle = model.module.register_forward_hook(lambda m, args, output: captured.append(
        [v.detach().cpu().numpy() for v in (output[0], output[1], output[3])]))
    try:
        with torch.inference_mode():
            edges, staff, returned = model.predict_step(graph.clone(), return_graph=True)
    finally:
        handle.remove()
    edges, staff = edges.cpu().numpy(), staff.cpu().numpy()
    count = len(staff)
    _, components = connected_components(coo_matrix((np.ones(edges.shape[1]), edges), shape=(count,count)), directed=False)
    return dict(edges=edges, staff=staff, components=components,
        voice_edges=returned['note','predicted','note'].edge_index.cpu().numpy(),
        chord_edges=returned['note','chord_predicted','note'].edge_index.cpu().numpy(),
        potential_voice=returned['note','potential','note'].edge_index.cpu().numpy(),
        potential_chord=returned['note','chord_potential','note'].edge_index.cpu().numpy(),
        voice_logits=captured[0][0], staff_scores=captured[0][1], chord_logits=captured[0][2])


def pair_metrics(rows, groups):
    tp=fp=fn=0
    for i, a in enumerate(rows):
        for j in range(i):
            b=rows[j]
            if a['tick']//1920 != b['tick']//1920:
                continue
            same=a['part']==b['part']; predicted=groups[i]==groups[j]
            tp+=same and predicted; fp+=not same and predicted; fn+=same and not predicted
    tp,fp,fn=int(tp),int(fp),int(fn)
    return dict(tp=tp, fp=fp, fn=fn, f1=2*tp/max(1,2*tp+fp+fn))


def controls():
    cases={}
    for name in ('lower_hold','upper_hold','chord_layers','crossing','wide_arpeggio','pedal_overlap'):
        rows=[]
        def add(beat, duration, pitch, part):
            rows.append(dict(id=str(len(rows)),tick=round(beat*480),duration=round(duration*480),pitch=pitch,part=part))
        for bar in range(4):
            at=bar*4
            if name in ('lower_hold','pedal_overlap'):
                add(at,4,48,0)
                for i,pitch in enumerate((72,74,76,77,79,77,76,74)):
                    add(at+i*.5,1.5 if name=='pedal_overlap' else .5,pitch,1)
            elif name=='upper_hold':
                add(at,4,76,1)
                for i,pitch in enumerate((48,55,60,55,48,55,60,55)):add(at+i*.5,.5,pitch,0)
            elif name=='chord_layers':
                for beat in (0,2):
                    for pitch in (48,55):add(at+beat,2,pitch,0)
                for i in range(8):
                    for pitch in (72,76,79):add(at+i*.5,.5,pitch,1)
            elif name=='crossing':
                for i in range(4):
                    add(at+i,1,60+i*4,0); add(at+i,1,73-i*4,1)
            else:
                for i,pitch in enumerate((48,55,60,64,67,72,76,79)):add(at+i*.5,.5,pitch,0)
        cases[name]=sorted(rows,key=lambda n:(n['tick'],n['pitch'],n['id']))
    return cases


def expression_choices(notes, sections, assignments, kind):
    result={}
    for index, section in enumerate(sections):
        groups=defaultdict(list)
        for n in notes:
            if section['start_ms']<=n.start<section['end_ms'] and n.strength>=.35:
                key=0 if kind=='all' else assignments[n.id][kind]
                groups[key].append(n)
        for key, group in groups.items():
            choices=articulation_policy(group)
            for n in group:
                hold=n.end-n.start>=240 and n.release_reason not in UNOBSERVED_RELEASES and not choices[n.id]['tap']
                result[n.id]=dict(hold=hold,group=str(key),section=index,choice=choices[n.id],
                    note=replace(n,end=n.end if hold else n.start).data())
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--analysis',type=Path,default=ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve prior experiments')
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(args.source),'diff','HEAD','--name-only'],text=True).strip()
    model_path=args.source/'pretrained_models/model.ckpt'
    if revision!=REVISION or changes or sha256(model_path)!=MODEL_SHA:raise ValueError('Unverified model/source')
    sys.path.insert(0,str(args.source.resolve()))
    from piano_svsep.models.pl_models import PLPianoSVSep
    from launch_scripts.predict import prepare_score
    torch.set_num_threads(2);torch.manual_seed(42)
    with torch.serialization.safe_globals([torch.nn.functional.relu]):
        checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
    accepted=inspect.signature(PLPianoSVSep).parameters
    extra={k:v for k,v in checkpoint['hyper_parameters'].items() if k not in accepted}
    if extra!={'edge_feature_feedback':False,'after_encoder_frontend':False}:raise ValueError('Unexpected legacy hyperparameters')
    model=PLPianoSVSep(**{k:v for k,v in checkpoint['hyper_parameters'].items() if k in accepted}).eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    save=lambda name,obj:(args.output/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    identity=dict(source_commit=revision,model_sha256=MODEL_SHA,runner_sha256=sha256(__file__),
        analysis_sha256=sha256(args.analysis),protocol_sha256=sha256(ROOT/'docs/piano-stream-protocol.md'),
        runtime_sha256=sha256(ROOT/'test-data/piano-svsep/requirements-resolved.txt'),
        strict_weights=True,legacy_disabled_arguments=extra,enabled_in_application=False)
    save('identity.json',identity)
    graph,score,_=prepare_score(str(args.source/'artifacts/test_score.musicxml'),include_original=False)
    example=predict(model,graph)
    graph['note'].staff=torch.arange(len(graph['note'].x))%2
    graph['note'].voice=torch.arange(len(graph['note'].x))%7
    shuffled=predict(model,graph)
    assert all(np.array_equal(example[k],shuffled[k]) for k in example)
    np.savez_compressed(args.output/'official-example.npz',**example)
    example_summary=dict(notes=len(example['staff']),edges=example['edges'].shape[1],labels_do_not_affect_prediction=True)
    summaries=[]
    for name,rows in controls().items():
        predicted=predict(model,graph_from_rows(rows))
        np.savez_compressed(args.output/(name+'.npz'),**predicted)
        save(name+'-input.json',rows)
        summary=dict(name=name,notes=len(rows),components=len(set(predicted['components'])),
            voice_pairs=pair_metrics(rows,predicted['components']),staff_pairs=pair_metrics(rows,predicted['staff']),
            middle_c_pairs=pair_metrics(rows,[r['pitch']>=60 for r in rows]))
        summaries.append(summary);print(json.dumps(summary),flush=True)
    analysis=read(args.analysis)
    notes=[note_from(n) for n in analysis['candidates'] if n['voice']=='piano']
    original=[n.data() for n in notes];rows=[]
    beat=60000/74
    for n in notes:
        begin=round((n.start-22.5-178)/beat*4)*120
        end=round((n.end-22.5-178)/beat*4)*120
        rows.append(dict(id=n.id,pitch=n.pitch,tick=max(0,begin),duration=max(120,end-max(0,begin))))
    rows.sort(key=lambda r:(r['tick'],r['pitch'],r['id']))
    start=time.monotonic();predicted=predict(model,graph_from_rows(rows))
    elapsed=time.monotonic()-start
    assignments={r['id']:dict(staff=int(predicted['staff'][i]),components=int(predicted['components'][i]),
        middle_c=int(r['pitch']>=60)) for i,r in enumerate(rows)}
    save('sakura-graph-input.json',rows);save('sakura-assignments.json',assignments)
    np.savez_compressed(args.output/'sakura-predictions.npz',**predicted)
    policies={kind:expression_choices(notes,analysis['structure']['segments'],assignments,kind)
        for kind in ('all','staff','components','middle_c')}
    baseline=policies['all'];expression=[]
    for kind,choices in policies.items():
        assert set(choices)==set(baseline)
        expression.append(dict(mode=kind,attacks=len(choices),holds=sum(c['hold'] for c in choices.values()),
            newly_held=sum(c['hold'] and not baseline[k]['hold'] for k,c in choices.items()),
            newly_tapped=sum(not c['hold'] and baseline[k]['hold'] for k,c in choices.items())))
        save('sakura-expression-'+kind+'.json',choices)
    assert [n.data() for n in notes]==original
    sizes=Counter(predicted['components'])
    result=dict(identity=identity,example=example_summary,controls=summaries,
        sakura=dict(raw_notes=len(notes),seconds=elapsed,staff_counts=dict(Counter(map(int,predicted['staff']))),
            voice_components=len(sizes),singleton_notes=sum(v==1 for v in sizes.values()),
            voice_edges=predicted['voice_edges'].shape[1],chord_edges=predicted['chord_edges'].shape[1],
            unobserved_tails=sum(n.release_reason in UNOBSERVED_RELEASES for n in notes),
            expression=expression,raw_candidates_unchanged=True),musical_acceptance=False)
    save('report.json',result);print(json.dumps(result['sakura']),flush=True)


if __name__=='__main__':main()
