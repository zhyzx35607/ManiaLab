"""Score isolated and mixed URMP recordings on their physical sample clock."""
import argparse
import json
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


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/urmp/benchmark'))
    cli.add_argument('--output',type=Path,default=Path('test-data/urmp/benchmark/comparison.json'))
    args=cli.parse_args();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    manifest=read(args.root/'manifest.json');rows=[]
    for case in manifest['cases']:
        folder=args.root/'yourmt3'/case['name'];report=read(folder/'report.json');identity=report['identity']
        if (sha(case['audio'])!=case['audio_sha256'] or sha(case['labels_path'])!=case['labels_sha256'] or
            identity['sources']!={case['audio']:case['audio_sha256']} or identity['start_ms']!=case['start_ms'] or
            identity['seconds']!=case['seconds']):raise ValueError('Frozen benchmark/input mismatch')
        begin=case['start_ms']/1000;end=begin+case['seconds']
        reference=[n for n in read(case['labels_path']) if begin<=n['onset']<end]
        predicted=[n for n in read(folder/'notes.json') if begin<=n['onset']<end]
        complete_reference=[n for n in reference if n['offset']<end]
        complete_predicted=[n for n in predicted if n['offset']<end]
        metrics={}
        for strict in (False,True):
            name='strict' if strict else 'standard'
            metrics[name]=dict(pitched_onsets=match(reference,predicted,strict=strict),
                complete_intervals=match(complete_reference,complete_predicted,offset=True,strict=strict),
                instrument_onsets=match(reference,predicted,strict=strict,instrument=True),
                complete_instrument_intervals=match(complete_reference,complete_predicted,offset=True,strict=strict,instrument=True))
        row=dict(name=case['name'],piece_id=case['piece_id'],kind=case['kind'],instrument=case.get('instrument'),
            identity=identity,annotation_hashes=case['annotations'],metrics=metrics,
            decoding_errors=report['decoding_errors'],programs=report['by_program'],
            interval_scope=dict(reference_crosses_end=len(reference)-len(complete_reference),
                predicted_crosses_end=len(predicted)-len(complete_predicted)))
        if (folder/'release-audit.json').exists():row['release_audit']=read(folder/'release-audit.json')
        baseline=args.root/'application'/case['name']
        if (baseline/'report.json').exists():
            base_report=read(baseline/'report.json')
            if base_report['identity']['sources']!=identity['sources']:raise ValueError('Baseline audio differs')
            base=[n for n in read(baseline/'notes.json') if begin<=n['onset']<end]
            complete_base=[n for n in base if n['offset']<end]
            row['application_detector']=dict(identity=base_report['identity'],metrics={})
            for strict in (False,True):
                row['application_detector']['metrics']['strict' if strict else 'standard']=dict(
                    pitched_onsets=match(reference,base,strict=strict),
                    complete_intervals=match(complete_reference,complete_base,offset=True,strict=strict))
        rows.append(row)
    result=dict(rows=rows,protocol=manifest['protocol'],enabled_in_application=False,
        tolerances=dict(standard='50ms onset; offset max(50ms,20% reference duration); exact semitone',
            strict='40ms onset; 80ms offset; exact semitone'),
        limits=['All predictions retained, including incorrect instrument families and spurious drum events.',
            'Complete interval metrics exclude events crossing the evaluation end; onset metrics retain them.',
            'Instrument metrics require exact program; sax subtype labels in upstream preprocessing are uncertain.',
            'Physical audio endpoints are not authored LN-release choices.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('name','metrics')} for r in rows]),flush=True)


if __name__=='__main__':main()
