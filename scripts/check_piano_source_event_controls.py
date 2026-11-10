"""Stress selected source-event weights on fixed independent symbolic controls.

Known control layers define evaluation only; no test corpus or chart is read.
"""
import argparse
from collections import Counter
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
from scripts.piano_event_training import DTYPE,_geometry,make_event_graph
from scripts.research_piano_streams import controls


def control_item(rows):
    array=np.zeros(len(rows),dtype=DTYPE)
    events=[]
    for i,r in enumerate(rows):
        array[i]=(r['pitch'],r['tick'],r['duration'],r['tick']/480,r['duration']/480,4,1,1)
        events.append(dict(start_tick=r['tick'],end_tick=r['tick']+r['duration']))
    attack=array['onset_div']
    potential=np.vstack(np.where(attack[:,None]<attack[None,:]))
    chords=np.vstack(np.where((attack[:,None]==attack[None,:]) & ~np.eye(len(rows),dtype=bool)))
    # Evaluation asks whether an intended control layer continues at its next
    # onset, even under simulated pedal overlap. Duration is not its identity.
    truth=set()
    for i,r in enumerate(rows):
        later=[j for j,s in enumerate(rows) if s['part']==r['part'] and s['tick']>r['tick']
               and s['tick']//1920==r['tick']//1920]
        if later:
            first=min(rows[j]['tick'] for j in later)
            truth.update((i,j) for j in later if rows[j]['tick']==first)
    mask=np.array([rows[a]['tick']//1920==rows[b]['tick']//1920 for a,b in potential.T])
    voice=np.array([(a,b) in truth for a,b in potential.T],dtype=np.float32)
    chord=np.array([rows[a]['part']==rows[b]['part'] for a,b in chords.T],dtype=np.float32)
    return dict(name='synthetic-control',metadata=dict(piece_id='control',counts={}),payload=dict(
        notes=array,observed_edges=_geometry(events,list(range(len(rows)))),potential_edges=potential,
        chord_edges=chords,voice=voice,voice_mask=mask,chord=chord,
        chord_mask=np.ones(len(chord),dtype=bool),staff=np.zeros(len(rows),dtype=np.int64),
        staff_mask=np.zeros(len(rows),dtype=bool)))


def metrics(logits,target,mask):
    predicted=logits[mask]>=0;actual=target[mask].bool()
    tp=int((predicted & actual).sum());fp=int((predicted & ~actual).sum());fn=int((~predicted & actual).sum())
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',type=Path,required=True)
    cli.add_argument('--source',type=Path,default=ROOT/'test-data/piano-svsep/source')
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve prior controls')
    revision=verify_source(args.source);sys.path.insert(0,str(args.source.resolve()))
    torch.set_num_threads(2);torch.manual_seed(42)
    report=json.loads((args.run/'report.json').read_text(encoding='utf-8'))
    if sha256(args.run/'selected.pt')!=report['selected_sha256']:raise ValueError('Changed selected weights')
    model=load_model(args.source/'pretrained_models/model.ckpt').eval()
    original={key:v.clone() for key,v in model.state_dict().items()}
    adapted=torch.load(args.run/'selected.pt',map_location='cpu',weights_only=True)['state_dict']
    args.output.mkdir(parents=True)
    snapshots=[Path(__file__),ROOT/'scripts/piano_event_training.py',ROOT/'scripts/research_piano_streams.py']
    for path in snapshots:shutil.copyfile(path,args.output/(path.stem+'-snapshot.py'))
    result=[]
    for name,weights in (('original',original),('source_event_adapted',adapted)):
        model.load_state_dict(weights,strict=True);model.eval()
        for control,rows in controls().items():
            item=control_item(rows);graph,target=make_event_graph(item)
            with torch.inference_mode():voice,staff,_,chord=forward(model,graph)
            record=dict(model=name,control=control,events=len(rows),
                voice=metrics(voice,target['voice'],target['voice_mask']),
                chord=metrics(chord,target['chord'],target['chord_mask']))
            # Staff labels are only grouping diagnostics: upper/lower staff is
            # not the control's layer identity, especially in crossings.
            groups=staff.argmax(dim=1).numpy();tp=fp=fn=0
            for i,r in enumerate(rows):
                for j in range(i):
                    if r['tick']//1920!=rows[j]['tick']//1920:continue
                    actual=r['part']==rows[j]['part'];pred=groups[i]==groups[j]
                    tp+=actual and pred;fp+=not actual and pred;fn+=actual and not pred
            record['staff_layer_pairs']=dict(tp=int(tp),fp=int(fp),fn=int(fn),f1=2*tp/max(1,2*tp+fp+fn))
            result.append(record)
            np.savez_compressed(args.output/(name+'-'+control+'.npz'),
                voice_logits=voice.numpy(),staff=staff.numpy(),chord_logits=chord.numpy(),
                potential_edges=item['payload']['potential_edges'],chord_edges=item['payload']['chord_edges'])
            print(json.dumps(record),flush=True)
    summary=dict(source_revision=revision,training_report_sha256=sha256(args.run/'report.json'),
        selected_sha256=report['selected_sha256'],selected_epoch=report['selected_epoch'],
        code_sha256={p.name:sha256(p) for p in snapshots},cases=result,
        fixed_threshold=.5,test_corpus_loaded=False,weights_changed=False,musical_acceptance=False,
        note='Independent intended-layer controls; raw head classification, not phrase decoding. Pedal overlap is synthetic.')
    summary['artifacts_sha256']={p.name:sha256(p) for p in args.output.iterdir()}
    (args.output/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
