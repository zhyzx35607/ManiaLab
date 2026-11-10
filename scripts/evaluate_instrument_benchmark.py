"""Score isolated and mixed URMP recordings on their physical sample clock."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def match(reference,predicted,*,offset=False,strict=False,instrument=False):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    pairs=[]
    if reference and predicted:
        ref=np.asarray([[n['onset'],n['offset'],n['pitch'],n['program']] for n in reference])
        pred=np.asarray([[n['onset'],n['offset'],n['pitch'],n['program']] for n in predicted])
        difference=np.abs(ref[:,None,:]-pred[None,:,:])
        allowed=(difference[:,:,0]<=(.04 if strict else .05))&(difference[:,:,2]==0)
        allowed&=np.asarray([not n['is_drum'] for n in predicted])[None,:]
        if offset:
            tolerance=np.full(len(ref),.08) if strict else np.maximum(.05,.2*(ref[:,1]-ref[:,0]))
            allowed&=difference[:,:,1]<=tolerance[:,None]
        if instrument:allowed&=difference[:,:,3]==0
        costs=np.concatenate((np.where(allowed,difference[:,:,0]/(len(ref)+1),1e6),np.ones((len(ref),len(ref)))),axis=1)
        rows,cols=linear_sum_assignment(costs)
        pairs=[(int(i),int(j)) for i,j in zip(rows,cols) if j<len(pred) and allowed[i,j]]
    errors=lambda key:[abs(reference[i][key]-predicted[j][key])*1000 for i,j in pairs]
    onset=errors('onset');release=errors('offset');count=len(pairs)
    return dict(reference=len(reference),predicted=len(predicted),matched=count,
        precision=count/len(predicted) if predicted else 0,recall=count/len(reference) if reference else 0,
        f1=2*count/(len(reference)+len(predicted)) if reference or predicted else 0,
        mean_onset_error_ms=sum(onset)/count if count else None,
        mean_release_error_ms=sum(release)/count if count else None)


def window_metrics(reference, predicted, *, begin, end, context_start, context_end):
    """Score every attack in the interval, including incorrectly extended tails.

    The scoring end selects attacks, not releases. All reference releases must
    be observable in the inference context; otherwise this protocol refuses to
    produce a full-interval score. Never select predictions using their tails.
    """
    if not 0 <= context_start <= begin < end <= context_end:
        raise ValueError('Inference context does not cover the scoring window')
    for notes in (reference, predicted):
        if any(not all(math.isfinite(n[k]) for k in ('onset', 'offset', 'pitch', 'program'))
               or n['offset'] <= n['onset'] for n in notes):
            raise ValueError('Invalid intervals must be audited before scoring')
    reference = [n for n in reference if begin <= n['onset'] < end]
    predicted = [n for n in predicted if begin <= n['onset'] < end]
    unobserved = sum(n['offset'] >= context_end for n in reference)
    if unobserved:
        raise ValueError(f'{unobserved} reference tails exceed available context; extend inference before scoring')
    metrics = {}
    for strict in (False, True):
        metrics['strict' if strict else 'standard'] = dict(
            pitched_onsets=match(reference,predicted,strict=strict),
            complete_intervals=match(reference,predicted,offset=True,strict=strict),
            instrument_onsets=match(reference,predicted,strict=strict,instrument=True),
            complete_instrument_intervals=match(reference,predicted,offset=True,strict=strict,instrument=True))
    scope = dict(begin_seconds=begin,end_seconds=end,context_start_seconds=context_start,
                 context_end_seconds=context_end,unobservable_reference_offsets=unobserved,
                 reference_crosses_end=sum(n['offset'] >= end for n in reference),
                 predicted_crosses_end=sum(n['offset'] >= end for n in predicted),
                 predicted_crosses_context_end=sum(n['offset'] >= context_end for n in predicted),
                 crossing_predictions_retained=True)
    return metrics,scope


def verified_context(case, report):
    """Check the saved inference clock against its source length and request."""
    from scipy.io import wavfile
    identity = report['identity']
    if (identity['sources'] != {case['audio']:case['audio_sha256']} or
        identity['start_ms'] != case['start_ms'] or identity['seconds'] != case['seconds']):
        raise ValueError('Frozen benchmark/input mismatch')
    rate,samples = wavfile.read(case['audio'],mmap=True)
    begin = case['start_ms']/1000
    context = identity['context_seconds']
    start = max(0,round((begin-context)*rate))/rate
    end = min(len(samples),round((begin+case['seconds']+context)*rate))/rate
    # Historical baseline runs read from zero and were only run at start_ms=0.
    if 'context_start_seconds' not in report and case['start_ms'] != 0:
        raise ValueError('Historical baseline has no verified nonzero context start')
    if (abs(report.get('context_start_seconds',start)-start) > 1e-8 or
        abs(report.get('context_end_seconds',end)-end) > 1e-8):
        raise ValueError('Reported context differs from the frozen source/request')
    return start,end


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/urmp/benchmark'))
    cli.add_argument('--output',type=Path,default=Path('test-data/urmp/benchmark/comparison-v2.json'))
    args=cli.parse_args();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    if args.output.exists():raise ValueError('Preserve earlier evaluation; use a new output')
    manifest=read(args.root/'manifest.json');rows=[]
    for case in manifest['cases']:
        folder=args.root/'yourmt3'/case['name'];report=read(folder/'report.json');identity=report['identity']
        if sha(case['audio'])!=case['audio_sha256'] or sha(case['labels_path'])!=case['labels_sha256']:
            raise ValueError('Frozen benchmark/input mismatch')
        context_start,context_end=verified_context(case,report)
        begin=case['start_ms']/1000;end=begin+case['seconds']
        reference=read(case['labels_path'])
        predicted=read(folder/'notes.json')
        metrics,scope=window_metrics(reference,predicted,begin=begin,end=end,
                                    context_start=context_start,context_end=context_end)
        row=dict(name=case['name'],piece_id=case['piece_id'],kind=case['kind'],instrument=case.get('instrument'),
            identity=identity,annotation_hashes=case['annotations'],metrics=metrics,
            decoding_errors=report['decoding_errors'],programs=report['by_program'],
            interval_scope=scope,evidence=dict(report_sha256=sha(folder/'report.json'),
                notes_sha256=sha(folder/'notes.json'),labels_sha256=case['labels_sha256']))
        if (folder/'release-audit.json').exists():
            row['release_audit']=read(folder/'release-audit.json')
            if row['release_audit']['notes_sha256']!=sha(folder/'notes.json'):
                raise ValueError('Audited predictions changed')
        baseline=args.root/'application'/case['name']
        if (baseline/'report.json').exists():
            base_report=read(baseline/'report.json')
            base_start,base_end=verified_context(case,base_report)
            if (base_start,base_end)!=(context_start,context_end):raise ValueError('Baseline context differs')
            base=read(baseline/'notes.json')
            base_metrics,base_scope=window_metrics(reference,base,begin=begin,end=end,
                                                  context_start=base_start,context_end=base_end)
            row['application_detector']=dict(identity=base_report['identity'],interval_scope=base_scope,
                notes_sha256=sha(baseline/'notes.json'),report_sha256=sha(baseline/'report.json'),
                metrics={name:{k:v for k,v in values.items() if 'instrument' not in k}
                         for name,values in base_metrics.items()})
        rows.append(row)
    result=dict(revision=2,runner_sha256=sha(__file__),manifest_sha256=sha(args.root/'manifest.json'),
        rows=rows,protocol=manifest['protocol'],enabled_in_application=False,
        tolerances=dict(standard='50ms onset; offset max(50ms,20% reference duration); exact semitone',
            strict='40ms onset; 80ms offset; exact semitone'),
        limits=['All predictions retained, including incorrect instrument families and spurious drum events.',
            'All attacks in the scoring window enter interval denominators, even if predicted tails exceed the window or context.',
            'Reference tails must be observable in inference context; no end-based prediction filtering.',
            'The legacy strict key means fixed 40/80ms, not universally stricter: 80ms is wider than 50ms for short tails.',
            'Instrument metrics require exact program; sax subtype labels in upstream preprocessing are uncertain.',
            'Physical audio endpoints are not authored LN-release choices.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('name','metrics')} for r in rows]),flush=True)


if __name__=='__main__':main()
