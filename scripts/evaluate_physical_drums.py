"""Select thresholds on calibration only; retain separate physical/author scores."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.audio import rhythmic_evidence
from manialab.domain import parse_osu
from scripts.physical_drum_features import load_audio,peaks,CLASSES
from scripts.probe_mapper_adapter import sha,event_similarity
from scripts.research_source_attack_classes import drum_event_counts


def aggregate(scores):
    reference=sum(r['reference'] for r in scores);predicted=sum(r['predicted'] for r in scores);matched=sum(r['matched'] for r in scores)
    return dict(reference=reference,predicted=predicted,matched=matched,
        precision=matched/predicted if predicted else 0,recall=matched/reference if reference else 0,
        f1=2*matched/(reference+predicted) if reference+predicted else 0)


def score(truth,predicted,tolerance):
    return {kind:event_similarity([(e['time_ms'],) for e in truth if e['kind']==kind],
        [(e['time_ms'],) for e in predicted if e['kind']==kind],[tolerance]) for kind in CLASSES}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset',type=Path,required=True)
    cli.add_argument('--model',type=Path,required=True)
    cli.add_argument('--inference',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier physical scores')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(args.dataset/'manifest.json');trained=read(args.model/'report.json');test=read(args.inference/'report.json')
    if (sha(args.dataset/'manifest.json')!=trained['identity']['dataset_sha256'] or test['dataset_sha256']!=sha(args.dataset/'manifest.json')
        or test['model_sha256']!=trained['model_sha256'] or sha(args.model/'validation-logits.npz')!=trained['validation_logits_sha256']):
        raise ValueError('Frozen calibration/model identity changed')
    validation=[]
    with np.load(args.model/'validation-logits.npz') as predictions:
        for row in manifest['records']:
            if row['split']!='validation':continue
            with np.load(args.dataset/(row['id']+'.npz')) as data:times=data['times']
            validation.append((row,expit(predictions[row['id']]),times))
    selections=[];thresholds=[]
    for kind in CLASSES:
        options=[]
        for threshold in np.linspace(.1,.9,17):
            scores=[]
            for row,probability,times in validation:
                proposed=peaks(probability,times,[threshold]*3)
                scores.append(score(row['events'],proposed,40)[kind])
            options.append(dict(threshold=float(threshold),score=aggregate(scores)))
        best=max(options,key=lambda e:(e['score']['f1'],e['threshold']))
        thresholds.append(best['threshold']);selections.append(dict(kind=kind,chosen=best,trials=options))
    args.output.mkdir(parents=True)
    (args.output/'selection.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    records=[];events={};test_hashes={row['id']:row['sha256'] for row in test['records']}
    for row in manifest['records']:
        if row['split'] not in ('validation','test'):continue
        if row['split']=='validation':
            _,probability,times=next(value for value in validation if value[0]['id']==row['id'])
        else:
            path=args.inference/(row['id']+'.npz')
            if sha(path)!=test_hashes[row['id']]:raise ValueError('Reserved inference changed')
            with np.load(path) as data:probability=expit(data['logits']);times=data['times']
        proposed=peaks(probability,times,thresholds);events[row['id']]=proposed
        if sha(row['audio'])!=row['audio_sha256']:raise ValueError('Physical control audio changed')
        candidates,_=rhythmic_evidence(load_audio(row['audio']),estimate=False)
        # This measures timing coverage only, assigning each candidate no class.
        coverage={kind:event_similarity([(e['time_ms'],) for e in row['events'] if e['kind']==kind],[(n.start,) for n in candidates],[40]) for kind in CLASSES}
        records.append(dict(id=row['id'],dataset=row['dataset'],split=row['split'],
            scores={str(t):score(row['events'],proposed,t) for t in (20,40)},unclassified_candidate_coverage=coverage))
    summaries={}
    for split in ('validation','test'):
        summaries[split]={}
        for dataset in ('IDMT','MDB','all'):
            rows=[r for r in records if r['split']==split and (dataset=='all' or r['dataset']==dataset)]
            summaries[split][dataset]={str(t):{kind:aggregate([r['scores'][str(t)][kind] for r in rows]) for kind in CLASSES} for t in (20,40)}
    path=args.inference/'sakura.npz'
    if sha(path)!=test['sakura']['sha256']:raise ValueError('Sakura inference changed')
    with np.load(path) as data:sakura=peaks(expit(data['logits']),data['times'],thresholds)
    events['sakura']=sakura
    reference_path=Path('test-data/sakura-benchmark/reference.osu')
    sakura_manifest=read(reference_path.parent/'manifest.json')
    if sha(reference_path)!=sakura_manifest['charts']['reference']['chart_sha256']:raise ValueError('Author reference changed')
    reference=list(drum_event_counts(parse_osu(reference_path.read_text(encoding='utf-8-sig'))['notes']))
    author={}
    for name,start,end in [('whole',0,283840),('later',188286,283840)]:
        human=[dict(time_ms=t,kind=c) for t,c in reference if start<=t<end]
        author[name]={}
        for label,shift in [('raw',0),('chart_lead_22_5ms',22.5)]:
            predicted=[{**e,'time_ms':e['time_ms']-shift} for e in sakura if start<=e['time_ms']-shift<end]
            author[name][label]=score(human,predicted,40)
    (args.output/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    report=dict(dataset_sha256=sha(args.dataset/'manifest.json'),training_sha256=sha(args.model/'report.json'),
        inference_sha256=sha(args.inference/'report.json'),selection_sha256=sha(args.output/'selection.json'),
        events_sha256=sha(args.output/'events.json'),reference_sha256=sha(reference_path),summaries=summaries,records=records,
        test_status=manifest.get('test_status','First held-out evaluation of this from-scratch model family.'),
        sakura_authored_selection=author,thresholds=thresholds,runner_sha256=sha(__file__),enabled_in_application=False,
        notice='IDMT/MDB scores use physical onset annotations. Sakura uses author-selected events with heterogeneous lane 4, not physical hi-hat truth. No model adaptation to Sakura.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(thresholds=thresholds,test=summaries['test'],sakura=author)),flush=True)


if __name__=='__main__':main()
