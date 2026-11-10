"""Evaluate a fixed one-target-attack decoder without labels in selection."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.adapt_piano_streams import forward,load_model,verify_source
from scripts.build_provenance import sha256
from scripts.check_piano_source_event_controls import control_item
from scripts.piano_event_training import load_event_items,make_event_graph
from scripts.piano_successor_decode import select_successor_groups
from scripts.probe_piano_audio_events import WEIGHTS
from scripts.research_piano_streams import MODEL_SHA,controls

MODELS=('original','clean_adapted','overlap_adapted')
KINDS=('clean','plus1','plus2')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_artifacts(folder):
    report=read(folder/'report.json')
    for name,digest in report['artifacts_sha256'].items():
        if sha256(folder/name)!=digest:
            raise ValueError('Saved input artifact changed: '+name)
    return report


def score(actual,predicted,mask):
    a,p=actual[mask],predicted[mask]
    tp=int((a & p).sum());fp=int((~a & p).sum());fn=int((a & ~p).sum());tn=int((~a & ~p).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn))


def owned_arrays(payload,logits):
    edges=payload['potential_edges']
    owner=(payload['voice_output_owner'] if 'voice_output_owner' in payload else
           payload['core_mask'][edges[1]])
    a,b=edges[:,owner]
    return dict(source_ids=payload['event_ids'][a],target_ids=payload['event_ids'][b],
                target_ticks=payload['notes']['onset_div'][b],logits=logits[owner],
                actual=payload['voice'][owner]>.5,score_mask=payload['voice_mask'][owner])


def join_owned(parts):
    data={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
    pairs=list(zip(data['source_ids'],data['target_ids']))
    if len(pairs)!=len(set(pairs)):
        raise ValueError('Duplicate owned candidate edge')
    return data


def diagnose(data,*,scored=True):
    # Only these three arrays enter the decision. Truth and scoring masks are
    # deliberately absent from the decoder's signature.
    selected=select_successor_groups(data['source_ids'],data['target_ticks'],data['logits'])
    raw=data['logits']>=0
    assert not (selected & ~raw).any()
    groups=defaultdict(set)
    for source,tick in zip(data['source_ids'][selected],data['target_ticks'][selected]):
        groups[str(source)].add(int(tick))
    assert all(len(ticks)==1 for ticks in groups.values())
    raw_groups=defaultdict(set)
    for source,tick in zip(data['source_ids'][raw],data['target_ticks'][raw]):
        raw_groups[str(source)].add(int(tick))
    result=dict(candidate_edges=len(raw),raw_positive=int(raw.sum()),selected_positive=int(selected.sum()),
        removed_positive=int((raw & ~selected).sum()),
        raw_sources_with_multiple_target_ticks=sum(len(v)>1 for v in raw_groups.values()),
        selected_sources_with_multiple_target_ticks=0)
    if scored:
        mask,actual=data['score_mask'],data['actual']
        removed=raw & ~selected & mask
        scored_winners=set(map(str,data['source_ids'][selected & mask]))
        unscored_winners=set(map(str,data['source_ids'][selected & ~mask]))-scored_winners
        robbed=removed & actual & np.asarray([str(s) in unscored_winners for s in data['source_ids']])
        maxima={}
        for i in np.flatnonzero(selected):
            source=str(data['source_ids'][i]);value=float(data['logits'][i])
            if source not in maxima or value>maxima[source][0]:
                maxima[source]=(value,{bool(mask[i])})
            elif value==maxima[source][0]:
                maxima[source][1].add(bool(mask[i]))
        maximum_only_unscored={s for s,(_,labels) in maxima.items() if labels=={False}}
        tied_maximum={s for s,(_,labels) in maxima.items() if labels=={False,True}}
        mixed_group=scored_winners & set(map(str,data['source_ids'][selected & ~mask]))
        lost_from=lambda sources:int((removed & actual & np.asarray(
            [str(s) in sources for s in data['source_ids']])).sum())
        result.update(raw=score(actual,raw,mask),decoded=score(actual,selected,mask),
            supervised_edges=int(mask.sum()),removed_true_edges=int((removed & actual).sum()),
            removed_false_edges=int((removed & ~actual).sum()),
            removed_true_with_only_unscored_winning_edges=int(robbed.sum()),
            removed_true_sources_with_only_unscored_winning_edges=len(set(map(str,data['source_ids'][robbed]))),
            removed_true_with_unscored_maximum_only=lost_from(maximum_only_unscored),
            removed_true_with_tied_scored_unscored_maximum=lost_from(tied_maximum),
            removed_true_with_mixed_winning_group=lost_from(mixed_group),
            unscored_selected_positive=int((selected & ~mask).sum()))
        assert result['raw']['tp']-result['decoded']['tp']==result['removed_true_edges']
        assert result['raw']['fp']-result['decoded']['fp']==result['removed_false_edges']
    return result,selected


def aggregate(cases):
    output={}
    for mode in ('raw','decoded'):
        counts={k:sum(row[mode][k] for row in cases) for k in ('tp','fp','fn','tn')}
        tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
        counts.update(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))
        output[mode]=counts
        eligible=[row[mode]['f1'] for row in cases if row[mode]['tp']+row[mode]['fn']>0]
        output[mode+'_macro_f1']=float(np.mean(eligible)) if eligible else None
    for key in ('removed_true_edges','removed_false_edges','removed_true_with_only_unscored_winning_edges',
                'removed_true_with_unscored_maximum_only','removed_true_with_tied_scored_unscored_maximum',
                'removed_true_with_mixed_winning_group'):
        output[key]=sum(row[key] for row in cases)
    output['pieces']=len(cases)
    return output


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve previous runs')
    revision=verify_source(args.source);sys.path.insert(0,str(args.source.resolve()))
    torch.set_num_threads(2);torch.manual_seed(42)
    base=ROOT/'test-data/piano-svsep'
    folders=dict(clean=base/'source-event-dataset-v1',plus1=base/'source-event-overlap-v1/plus1',
                 plus2=base/'source-event-overlap-v1/plus2')
    paths=dict(clean_adapted=base/'source-event-adaptation-v1/selected.pt',
               overlap_adapted=base/'source-event-overlap-adaptation-v1/selected.pt')
    expected_manifests=dict(clean='bfd8873a195bf0e4085d72c77cd6a409fbcc01cb3db2a2a9437d0933612b7999',
        plus1='cd81db7696445602fd98adee3304d2a8d0ec65af59a86a0f15f7f60336d02e20',
        plus2='8859f14593d884426117a695973ed6b485328fa442da0a02a0a8c10d8c900beb')
    for kind,folder in folders.items():
        if sha256(folder/'manifest.json')!=expected_manifests[kind]:raise ValueError('Changed dataset')
    for name,path in paths.items():
        if sha256(path)!=WEIGHTS[name]:raise ValueError('Changed weights')
    groups={kind:load_event_items(folder,'validation') for kind,folder in folders.items()}
    if any(len(items)!=36 or len({i['name'] for i in items})!=9 for items in groups.values()):
        raise ValueError('Unexpected validation partition')
    control_folders=dict(original=base/'source-event-overlap-controls-v1',
        clean_adapted=base/'source-event-controls-v1',overlap_adapted=base/'source-event-overlap-controls-v1')
    control_reports={str(folder):verify_artifacts(folder) for folder in set(control_folders.values())}
    for report in control_reports.values():
        for name in ('check_piano_source_event_controls.py','research_piano_streams.py'):
            if sha256(ROOT/'scripts'/name)!=report['code_sha256'][name]:
                raise ValueError('Control input constructor changed')
    audio_folder=base/'audio-events-v1';audio_report=verify_artifacts(audio_folder)
    args.output.mkdir(parents=True)
    save=lambda name,value:(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    code=dict(runner=Path(__file__),decoder=ROOT/'scripts/piano_successor_decode.py',
        event_graph=ROOT/'scripts/piano_event_training.py',forward=ROOT/'scripts/adapt_piano_streams.py',
        control=ROOT/'scripts/check_piano_source_event_controls.py',control_rows=ROOT/'scripts/research_piano_streams.py',
        protocol=ROOT/'docs/piano-successor-decode-protocol.md')
    identity=dict(source_revision=revision,weights_sha256=dict(original=MODEL_SHA,**WEIGHTS),
        dataset_sha256=expected_manifests,code_sha256={k:sha256(p) for k,p in code.items()},
        audio_report_sha256=sha256(audio_folder/'report.json'),
        control_report_sha256={folder.name:sha256(folder/'report.json') for folder in set(control_folders.values())},
        decoder_uses_labels=False,decoder_uses_supervision_mask=False,test_loaded=False,
        trained=False,application_enabled=False,musical_acceptance=False)
    for name,path in code.items():shutil.copyfile(path,args.output/(name+'-snapshot'+path.suffix))
    save('identity.json',identity)
    for kind,items in groups.items():
        for index,item in enumerate(items):
            np.savez_compressed(args.output/f'{kind}-graph-{index:03d}.npz',**item['payload'],
                metadata_json=np.asarray(json.dumps(item['metadata'])))
    model=load_model(args.source/'pretrained_models/model.ckpt').eval()
    original={key:value.clone() for key,value in model.state_dict().items()}
    validations=[];control_cases=[];audio_cases=[]
    previous={
        'original':read(base/'source-event-overlap-adaptation-v1/validation-epoch-0.json')['variants'],
        'clean_adapted':{'clean':read(base/'source-event-adaptation-v1/validation-epoch-1.json')},
        'overlap_adapted':read(base/'source-event-overlap-adaptation-v1/validation-epoch-4.json')['variants']}
    for name in MODELS:
        model.load_state_dict(original if name=='original' else torch.load(paths[name],map_location='cpu',weights_only=True)['state_dict'],strict=True)
        model.eval()
        for kind,items in groups.items():
            pieces=defaultdict(list)
            for index,item in enumerate(items):
                graph,_=make_event_graph(item)
                with torch.inference_mode():voice,_,_,_=forward(model,graph)
                logits=voice.numpy()
                if not np.isfinite(logits).all():raise ValueError('Nonfinite output')
                np.savez_compressed(args.output/f'{name}-{kind}-{index:03d}.npz',voice_logits=logits)
                pieces[item['name']].append(owned_arrays(item['payload'],logits))
            cases=[]
            for index,(piece,parts) in enumerate(sorted(pieces.items())):
                data=join_owned(parts);result,selected=diagnose(data)
                file=f'{name}-{kind}-piece-{index:02d}.npz'
                np.savez_compressed(args.output/file,**data,selected=selected)
                cases.append(dict(name=piece,file=file,**result))
            totals=aggregate(cases)
            if kind in previous[name]:
                old=previous[name][kind]
                for key in ('tp','fp','fn','tn'):
                    assert totals['raw'][key]==old['voice'][key],(name,kind,key)
                assert abs(totals['raw_macro_f1']-old['macro_voice_f1'])<1e-12
                old_pieces={r['name']:r for r in old['cases']}
                assert set(old_pieces)=={r['name'] for r in cases}
                for row in cases:
                    assert all(row['raw'][key]==old_pieces[row['name']]['voice'][key]
                               for key in ('tp','fp','fn','tn')),(name,kind,row['name'])
            row=dict(model=name,input=kind,cases=cases,aggregate=totals,
                     raw_reproduces_saved_validation=kind in previous[name])
            validations.append(row);print(json.dumps(dict(model=name,input=kind,**totals)),flush=True)
        folder=control_folders[name]
        stored='original' if name=='original' else 'source_event_adapted'
        for control,rows in controls().items():
            item=control_item(rows);payload=item['payload']
            payload['event_ids']=np.asarray([f'{control}:{i}' for i in range(len(rows))])
            payload['core_mask']=np.ones(len(rows),dtype=bool)
            with np.load(folder/f'{stored}-{control}.npz',allow_pickle=False) as data:
                assert np.array_equal(data['potential_edges'],payload['potential_edges'])
                owned=owned_arrays(payload,data['voice_logits'])
            result,selected=diagnose(owned)
            old=next(r for r in control_reports[str(folder)]['cases'] if r['model']==stored and r['control']==control)
            assert all(result['raw'][k]==old['voice'][k] for k in ('tp','fp','fn'))
            file=f'{name}-control-{control}.npz';np.savez_compressed(args.output/file,**owned,selected=selected)
            control_cases.append(dict(model=name,control=control,file=file,**result))
        parts=[];same_bar=[]
        for row in read(audio_folder/'graphs.json'):
            chunk=row['chunk']
            with np.load(audio_folder/row['file'],allow_pickle=False) as data:
                payload={k:data[k].copy() for k in data.files if k!='metadata_json'}
            with np.load(audio_folder/f'{name}-{chunk:03d}.npz',allow_pickle=False) as data:
                parts.append(owned_arrays(payload,data['voice_logits']))
            same_bar.extend(payload['voice_same_bar'][payload['voice_output_owner']])
        data=join_owned(parts);result,selected=diagnose(data,scored=False)
        same=np.asarray(same_bar,dtype=bool);raw=data['logits']>=0
        result.update(raw_same_bar=int((raw & same).sum()),decoded_same_bar=int((selected & same).sum()),
                      raw_cross_bar=int((raw & ~same).sum()),decoded_cross_bar=int((selected & ~same).sum()))
        old=next(r for r in audio_report['models'] if r['model']==name)
        assert result['raw_same_bar']==old['local_positive']['edges']
        assert result['raw_cross_bar']==old['unvalidated_cross_bar_positive']
        file=f'{name}-sakura.npz';np.savez_compressed(args.output/file,**data,selected=selected,same_bar=same)
        audio_cases.append(dict(model=name,file=file,**result))
    if any(sha256(path)!=identity['code_sha256'][key] for key,path in code.items()):
        raise ValueError('Code changed during experiment')
    report=dict(identity=identity,validation=validations,controls=control_cases,audio=audio_cases,
        scope='Local masked successors; unscored candidates compete in decoder. No global voice or LN truth.',
        artifacts_sha256={p.name:sha256(p) for p in args.output.iterdir()})
    save('report.json',report)


if __name__=='__main__':main()
