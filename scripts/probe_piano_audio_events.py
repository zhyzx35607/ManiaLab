"""Compare three fixed voice predictors on the same unlabelled piano audio events."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.io import wavfile
from scripts.adapt_piano_streams import forward,load_model,verify_source
from scripts.build_provenance import sha256
from scripts.piano_audio_event_graph import build_audio_event_items,load_frozen_sakura_rows
from scripts.piano_event_training import make_event_graph
from scripts.research_piano_streams import MODEL_SHA

WEIGHTS={'clean_adapted':'1d4e13e30ca34f36004b8b4765fa9c427322ebaf4a35167478f1fb0a8e630471',
         'overlap_adapted':'972a21c07066fbdd4a2d9f299c2d504d44704ee9278de36c6e4c392f96fcfbdd'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audio_ms(tick):
    return tick/480*(60000/74)+178+22.5


def select_clips(items,duration_ms):
    available=[item for item in items if item['payload']['node_output_owner'].any()]
    chosen=[]
    for fraction in (.125,.375,.625,.875):
        target=duration_ms*fraction
        if not available:
            break
        def distance(item):
            left,right=map(audio_ms,item['metadata']['core_ticks'])
            return (0 if left<=target<right else 1),max(left-target,0,target-right),left
        item=min(available,key=distance)
        if item['chunk'] in {r['chunk'] for r in chosen}:
            continue
        left,right=map(audio_ms,item['metadata']['core_ticks'])
        chosen.append(dict(chunk=item['chunk'],fraction=fraction,target_ms=target,
                           begin_ms=max(0.,left),end_ms=min(duration_ms,right)))
    return chosen


def describe_edges(rows,edges):
    lookup={row['id']:row for row in rows}
    following,preceding=defaultdict(set),defaultdict(set)
    for a,b in edges:
        following[a].add(lookup[b]['tick'])
        preceding[b].add(lookup[a]['tick'])
    return dict(edges=len(edges),sources_with_multiple_target_attack_groups=sum(len(v)>1 for v in following.values()),
                targets_with_multiple_source_attack_groups=sum(len(v)>1 for v in preceding.values()))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior comparisons')
    revision=verify_source(args.source)
    sys.path.insert(0,str(args.source.resolve()))
    torch.set_num_threads(2);torch.manual_seed(42)
    graph_path=ROOT/'test-data/piano-svsep/probe-v3/sakura-graph-input.json'
    frozen=read(graph_path.parent/'report.json')
    analysis_path=ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1/specialist-analysis.json'
    if sha256(analysis_path)!=frozen['identity']['analysis_sha256']:
        raise ValueError('Frozen source candidates changed')
    analysis=read(analysis_path)
    raw=[n for n in analysis['candidates'] if n['voice']=='piano']
    rows=load_frozen_sakura_rows(graph_path)
    raw_by_id={n['id']:n for n in raw}
    if len(raw)!=1557 or len(raw_by_id)!=1557 or set(raw_by_id)!={r['id'] for r in rows}:
        raise ValueError('Unmatched audio identities')
    for row in rows:
        note=raw_by_id[row['id']]
        begin=round((note['start']-22.5-178)/(60000/74)*4)*120
        end=round((note['end']-22.5-178)/(60000/74)*4)*120
        assert row==dict(id=note['id'],pitch=note['pitch'],tick=max(0,begin),duration=max(120,end-max(0,begin)))
    pcm_path=ROOT/'test-data/sakura-benchmark/raw-pcm-audit-v1/piano.wav'
    pcm_audit=read(pcm_path.parent/'report.json')
    expected=next(r['raw_float_sha256'] for r in pcm_audit['rows'] if r['role']=='piano')
    if sha256(pcm_path)!=expected:
        raise ValueError('Original piano PCM changed')
    rate,pcm=wavfile.read(pcm_path)
    if rate!=22050 or pcm.ndim!=1 or pcm.dtype!=np.float32:
        raise ValueError('Unexpected original PCM representation')
    items=build_audio_event_items(rows,piece_id='sakura-frozen-piano')
    selection=select_clips(items,len(pcm)/rate*1000)
    args.output.mkdir(parents=True)
    save=lambda name,value:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    code=dict(runner=Path(__file__),adapter=ROOT/'scripts/piano_audio_event_graph.py',
        event_graph=ROOT/'scripts/piano_event_training.py',forward=ROOT/'scripts/adapt_piano_streams.py',
        protocol=ROOT/'docs/piano-audio-event-protocol.md')
    identity=dict(source_revision=revision,pretrained_sha256=MODEL_SHA,
        graph_input_sha256=sha256(graph_path),analysis_sha256=sha256(analysis_path),pcm_sha256=sha256(pcm_path),
        code_sha256={k:sha256(v) for k,v in code.items()},weights_sha256=dict(original=MODEL_SHA,**WEIGHTS),
        diagnostic_clock=dict(bpm=74,offset_ms=178,audio_shift_ms=22.5,subdivision=4,ticks_per_quarter=480),
        waveform=dict(sample_rate=rate,samples=len(pcm)),model_threshold=.5,
        raw_acoustic_candidates_preserved=True,voice_ground_truth_available=False,
        automatic_timing_evaluated=False,application_enabled=False)
    for name,path in code.items():
        shutil.copyfile(path,args.output/(name+'-snapshot'+path.suffix))
    save('identity.json',identity)
    save('selection.json',selection)
    save('graph-input.json',rows)
    save('source-map.json',[dict(id=row['id'],graph=row,physical=raw_by_id[row['id']]) for row in rows])
    common=[];node_owners=[];edge_owners=set()
    for item in items:
        payload=item['payload'];meta=item['metadata'];chunk=item['chunk']
        assert not any(payload[k].any() for k in ('voice_mask','chord_mask','staff_mask'))
        ids=payload['event_ids']
        node_owners.extend(map(str,ids[payload['node_output_owner']]))
        for a,b in payload['potential_edges'][:,payload['voice_output_owner']].T:
            pair=(str(ids[a]),str(ids[b]))
            assert pair not in edge_owners
            edge_owners.add(pair)
        filename=f'graph-{chunk:03d}.npz'
        np.savez_compressed(args.output/filename,**payload,metadata_json=np.asarray(json.dumps(meta)))
        common.append(dict(chunk=chunk,file=filename,sha256=sha256(args.output/filename),metadata=meta))
    assert len(node_owners)==1557 and set(node_owners)==set(raw_by_id)
    save('graphs.json',common)
    paths=dict(clean_adapted=ROOT/'test-data/piano-svsep/source-event-adaptation-v1/selected.pt',
               overlap_adapted=ROOT/'test-data/piano-svsep/source-event-overlap-adaptation-v1/selected.pt')
    for name,path in paths.items():
        if sha256(path)!=WEIGHTS[name]:
            raise ValueError('Selected weights changed')
    model=load_model(args.source/'pretrained_models/model.ckpt').eval()
    original={key:value.clone() for key,value in model.state_dict().items()}
    outputs={};summaries=[]
    for name in ('original','clean_adapted','overlap_adapted'):
        model.load_state_dict(original if name=='original' else torch.load(paths[name],map_location='cpu',weights_only=True)['state_dict'],strict=True)
        model.eval();started=time.monotonic()
        local_edges=set();cross_edges=set();staffs={};chords=set();windows=[]
        local_candidates=cross_candidates=0
        for item in items:
            graph,target=make_event_graph(item)
            assert not any(target[k].any() for k in ('voice_mask','chord_mask','staff_mask'))
            with torch.inference_mode():
                voice,staff,_,chord=forward(model,graph)
            arrays=dict(voice_logits=voice.numpy(),staff_scores=staff.numpy(),chord_logits=chord.numpy())
            if not all(np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Nonfinite inference')
            payload=item['payload'];ids=payload['event_ids'];chunk=item['chunk']
            owner=payload['voice_output_owner'];same=payload['voice_same_bar']
            local_candidates+=int((owner & same).sum());cross_candidates+=int((owner & ~same).sum())
            for mask,destination in ((owner & same,local_edges),(owner & ~same,cross_edges)):
                for a,b in payload['potential_edges'][:,mask & (arrays['voice_logits']>=0)].T:
                    destination.add((str(ids[a]),str(ids[b])))
            for a,b in payload['chord_edges'][:,payload['chord_output_owner'] & (arrays['chord_logits']>=0)].T:
                chords.add((str(ids[a]),str(ids[b])))
            for i in np.flatnonzero(payload['node_output_owner']):
                assert str(ids[i]) not in staffs
                staffs[str(ids[i])]=int(arrays['staff_scores'][i].argmax())
            filename=f'{name}-{chunk:03d}.npz'
            np.savez_compressed(args.output/filename,**arrays)
            windows.append(dict(chunk=chunk,file=filename,sha256=sha256(args.output/filename)))
        outputs[name]=dict(local_edges=sorted(local_edges),unvalidated_cross_bar_edges=sorted(cross_edges),
                           chord_edges=sorted(chords),staff=staffs,windows=windows)
        save(name+'-predictions.json',outputs[name])
        row=dict(model=name,windows=len(items),owned_nodes=len(staffs),
            local_candidates=local_candidates,unvalidated_cross_bar_candidates=cross_candidates,
            local_positive=describe_edges(rows,local_edges),unvalidated_cross_bar_positive=len(cross_edges),
            chord_positive_directed=len(chords),chord_asymmetric_directed=sum((b,a) not in chords for a,b in chords),
            staff_counts=dict(Counter(staffs.values())),elapsed_seconds=time.monotonic()-started)
        summaries.append(row);print(json.dumps(row),flush=True)
    changes=[]
    for name in ('clean_adapted','overlap_adapted'):
        old=set(map(tuple,outputs['original']['local_edges']));new=set(map(tuple,outputs[name]['local_edges']))
        changes.append(dict(model=name,local_edges_added=len(new-old),local_edges_removed=len(old-new),
            staff_changed=sum(outputs[name]['staff'][k]!=v for k,v in outputs['original']['staff'].items())))
    if any(sha256(path)!=identity['code_sha256'][name] for name,path in code.items()):
        raise ValueError('Source changed during inference')
    report=dict(identity=identity,source_events=1557,graphs=len(items),selection=selection,models=summaries,
        changes_from_original=changes,raw_audio_fields_unchanged=True,
        musical_acceptance=False,trained_in_this_run=False,application_enabled=False,
        limitations=['Synthetic score clock is reused for diagnostic inputs; it is not automatic timing evidence.',
                     'No source voice truth is available; positive edges are hypotheses, not measured accuracy.',
                     'Cross-bar scores are unvalidated and no global melody or LN assignments were made.'])
    report['artifacts_sha256']={p.name:sha256(p) for p in args.output.iterdir()}
    save('report.json',report)


if __name__=='__main__':
    main()
