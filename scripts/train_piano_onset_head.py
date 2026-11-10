"""Fit only the final acoustic onset head, with frozen validation selection."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.piano_head_runtime import load_frozen_model, head_copy, replay_outputs, decode
from scripts.piano_adaptation_data import stitch
from scripts.evaluate_piano_acoustics import metrics, pairs, counts


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def pooled(rows, semantics='key', tolerance='50ms', field='onsets'):
    values=[r['metrics'][semantics][tolerance][field] for r in rows]
    return counts(*(sum(v[k] for v in values) for k in ('matched','reference','predicted')))


def eligible(baseline, current):
    by_id={r['id']:r for r in baseline}
    before, after=pooled(baseline),pooled(current)
    checks=dict(
        all_recording_f1_not_lower=all(r['metrics']['key']['50ms']['onsets']['f1']
            >=by_id[r['id']]['metrics']['key']['50ms']['onsets']['f1'] for r in current),
        pooled_precision_not_lower=after['precision']>=before['precision'],
        pooled_recall_not_lower=after['recall']>=before['recall'],
        extra_not_increased=after['predicted']-after['matched']<=before['predicted']-before['matched'],
        all_reattack_coverage_not_lower=all(r['metrics']['key']['50ms']['close_repeats']['both_onsets_matched']
            >=by_id[r['id']]['metrics']['key']['50ms']['close_repeats']['both_onsets_matched'] for r in current),
        sustain_interval_f1_not_lower=pooled(current,'sustain',field='intervals')['f1']
            >=pooled(baseline,'sustain',field='intervals')['f1'])
    return dict(checks=checks,eligible=all(checks.values()),pooled_onsets=after)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cache',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    cache,output=args.cache.resolve(),args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Preserve earlier training; use a new local output')
    report=read(cache/'report.json')
    if len(report['windows'])!=350 or not report['all_epoch0_outputs_identical']:
        raise ValueError('Complete epoch0 cache verification before training')
    for name,digest in report['artifacts_sha256'].items():
        if sha256(cache/name)!=digest:
            raise ValueError('Frozen cache changed: '+name)
    paths=dict(runner=Path(__file__),runtime=ROOT/'scripts/piano_head_runtime.py',
        data=ROOT/'scripts/piano_adaptation_data.py', scoring=ROOT/'scripts/evaluate_piano_acoustics.py',
        protocol=ROOT/'docs/piano-onset-head-adaptation-protocol.md')
    for name in ('runtime','data','protocol'):
        if sha256(paths[name])!=report['identity']['code'][name]:
            raise ValueError('Cache and training identities differ')
    model,model_identity=load_frozen_model()
    if model_identity!=report['identity']['model']:
        raise ValueError('Model differs from verified cache')
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    head=head_copy(model)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-5,weight_decay=1e-4)
    identity=dict(cache_report_sha256=sha256(cache/'report.json'),model=model_identity,
        code={key:sha256(path) for key,path in paths.items()},seed=42,epochs=2,
        batch_windows=4,parameters=45144,lr=1e-5,weight_decay=1e-4,clip_norm=1.,
        train_recordings=6,validation_recordings=4,train_seconds_per_recording=180,
        validation_input_seconds=180,validation_score_seconds=170,
        label='maximum triangular physical onset distance within 50ms',
        loss='unweighted BCEWithLogits; sum over owned frames and 88 pitches / their count',
        test_accessed=False,application_enabled=False)
    output.mkdir(parents=True)
    save(output/'identity.json',identity)
    for key,path in paths.items():
        shutil.copyfile(path,output/(key+'-snapshot'+path.suffix))
    train=[r for r in report['windows'] if r['split']=='train']
    val=[r for r in report['records'] if r['split']=='validation']
    if len(train)!=210 or len(val)!=4:
        raise ValueError('Unexpected train/validation scope')
    history=[]

    def validate(epoch):
        results=[]
        for recording in val:
            name=recording['id']
            labels=read(cache/(name+'-labels.json'))
            old=read(cache/(name+'-epoch0-events.json'))
            if epoch==0:
                prediction=old
                delta=dict(frame_max_abs=0.,onset_max_abs=0.,changed_offsets_on_paired_onsets=0)
            else:
                pieces={}
                for item in [r for r in report['windows'] if r['id']==name]:
                    with np.load(cache/item['file'],allow_pickle=False) as saved:
                        replay=replay_outputs(model,head,saved)
                    for key,value in replay.items():
                        pieces.setdefault(key,[]).append(value[0])
                combined=stitch(pieces)
                prediction=decode(combined)
                with np.load(cache/(name+'-epoch0-outputs.npz'),allow_pickle=False) as original:
                    delta=dict(frame_max_abs=float(np.max(abs(combined['frame_output']-original['frame_output']))),
                        onset_max_abs=float(np.max(abs(combined['reg_onset_output']-original['reg_onset_output']))))
                    for key in combined:
                        if key not in ('frame_output','reg_onset_output') and not np.array_equal(combined[key],original[key]):
                            raise ValueError('Frozen output changed: '+key)
                paired=pairs(old['notes'],prediction['notes'])
                delta['changed_offsets_on_paired_onsets']=sum(abs(old['notes'][i]['offset']-prediction['notes'][j]['offset'])>.005
                                                            for i,j in paired)
                np.savez(output/(name+f'-epoch{epoch}-outputs.npz'),**combined)
            save(output/(name+f'-epoch{epoch}-events.json'),prediction)
            scope=lambda ns:[n for n in ns if 0<=n['onset']<170]
            predicted=scope(prediction['notes'])
            references=dict(key=scope(labels['notes']),sustain=scope(labels['sustain_notes']))
            results.append(dict(id=name,split='validation',predictions_full=len(prediction['notes']),
                predicted_onsets_outside_input=sum(not 0<=n['onset']<180 for n in prediction['notes']),
                unobservable_reference_offsets={k:sum(n['offset']>180 for n in ns) for k,ns in references.items()},
                metrics={semantics:{label:metrics(ref,predicted,strict=strict)
                    for label,strict in [('50ms',False),('40ms',True)]} for semantics,ref in references.items()},
                acoustic_output_change=delta))
            print(json.dumps(dict(epoch=epoch,validation=name,onsets=results[-1]['metrics']['key']['50ms']['onsets'],
                                  changes=delta)),flush=True)
        return results

    generator=np.random.default_rng(42)
    for epoch in range(3):
        loss_sum,denominator=0.,0
        order=[]
        if epoch:
            order=generator.permutation(len(train)).tolist()
            for first in range(0,len(order),4):
                features,targets=[],[]
                for index in order[first:first+4]:
                    item=train[index]
                    with np.load(cache/item['file'],allow_pickle=False) as saved:
                        features.append(saved['features'][0,item['left']:item['right']].copy())
                        targets.append(saved['targets'][item['left']:item['right']].copy())
                h=torch.from_numpy(np.concatenate(features))
                y=torch.from_numpy(np.concatenate(targets))
                optimizer.zero_grad(set_to_none=True)
                logits=head(h)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='mean')
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradient')
                optimizer.step()
                loss_sum+=float(loss.detach())*y.numel()
                denominator+=y.numel()
                if first%40==0:
                    print(json.dumps(dict(epoch=epoch,trained_windows=min(first+4,len(order)),total=210,
                                          bce=loss_sum/denominator)),flush=True)
            if denominator!=6*18000*88:
                raise ValueError('Supervised frame ownership changed')
        weight_path=output/f'epoch{epoch}-head.pt'
        torch.save(head.state_dict(),weight_path)
        results=validate(epoch)
        decision=eligible(history[0]['validation'] if history else results,results)
        history.append(dict(epoch=epoch,loss=loss_sum/denominator if denominator else None,
            supervised_cells=denominator,order=order,head_sha256=sha256(weight_path),
            validation=results,selection=decision))
        save(output/'progress.json',history)
    acceptable=[row for row in history if row['selection']['eligible']]
    selected=max(acceptable,key=lambda row:(row['selection']['pooled_onsets']['f1'],-row['epoch']))
    source=ROOT/'test-data/piano-release-probe/model.pth'
    frozen=torch.load(source,map_location='cpu',weights_only=True)['model']
    for branch in ('note_model','pedal_model'):
        if any(not torch.equal(tensor,frozen[branch][key]) for key,tensor in getattr(model,branch).state_dict().items()):
            raise ValueError('Frozen model state changed')
    if any(param.grad is not None or param.requires_grad for param in model.parameters()):
        raise ValueError('Frozen model acquired gradients')
    if any(sha256(path)!=identity['code'][key] for key,path in paths.items()):
        raise ValueError('Frozen training code changed')
    save(output/'report.json',dict(identity=identity,epochs=history,selected_epoch=selected['epoch'],
        frozen_model_unchanged=True,training_performed=True,test_accessed=False,application_enabled=False,
        musical_acceptance=False,artifacts_sha256={p.name:sha256(p) for p in output.iterdir()}))
    print(json.dumps(dict(complete=True,selected_epoch=selected['epoch'],validation=selected['selection'])),flush=True)


if __name__=='__main__':
    main()
