"""Learn when an alternative predicted endpoint has stronger acoustic support.

Fits URMP pieces 3/4/5, calibrates only 6, then reports existing comparisons 1/2.
No authored LN labels or timing grids. Both endpoints are original model
predictions; the model cannot synthesize an attack, pitch, program or tail.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_instrument_overlap import load_run, refine
from scripts.evaluate_instrument_overlap import metrics
from scripts.probe_mapper_adapter import sha


WINDOWS=((-0.16,-0.08),(-0.08,-0.02),(-0.02,0.02),(0.02,0.08),(0.08,0.16),(0.16,0.32))
FEATURES=[f'{kind}_{i}' for kind in ('harmonic_level','broadband_level','harmonic_share') for i in range(len(WINDOWS))]+[
    'harmonic_drop','broadband_drop','duration_log','forced_boundary','explicit_off','reattack_end',
    'boundary_distance','nearby_pitched_ends','active_same_program','active_other_program']
PROTOCOL=dict(fit_piece_ids=[3,4,5],calibration_piece_ids=[6],comparison_piece_ids=[1,2],
    seconds=30,context_seconds=2.1,onset_match_seconds=.05,label_min_error_difference_ms=10,
    thresholds=[.5,.6,.7,.8,.9,.95,1.],regularization=.05,
    features=FEATURES,windows_seconds=WINDOWS,
    decision='Calibrate macro strict-interval F1, then fewer edits, then higher threshold. No comparison-song fitting.',
    enabled_in_application=False)


class AcousticEvidence:
    def __init__(self,audio,end_seconds):
        import numpy as np
        from scipy.io import wavfile
        from scipy.signal import stft
        rate,signal=wavfile.read(audio,mmap=True)
        if rate!=16000 or signal.ndim!=1 or signal.dtype!=np.int16:raise ValueError('Expected frozen mono 16kHz PCM')
        self.duration=min(end_seconds,len(signal)/rate)
        signal=signal[:round(self.duration*rate)].astype(np.float32)/32768
        self.freq,self.times,z=stft(signal,rate,nperseg=2048,noverlap=1920)
        self.power=abs(z)**2;self.broad=self.power.sum(axis=0)
        self.cache={}

    def harmonic_trace(self,pitch):
        import numpy as np
        if pitch not in self.cache:
            f0=440*2**((pitch-69)/12);weights=np.zeros(len(self.freq))
            for harmonic in range(1,7):
                centre=f0*harmonic
                if centre>self.freq[-1]:break
                weights+=np.maximum(0,1-np.abs(self.freq-centre)/(1.5*(self.freq[1]-self.freq[0])))/np.sqrt(harmonic)
            self.cache[pitch]=weights@self.power
        return self.cache[pitch]

    def feature_pair(self,note,alternate,base_flags,alternate_flags,base_report,shift_report,notes):
        import numpy as np
        trace=self.harmonic_trace(note['pitch'])
        local=(self.times>=note['onset'])&(self.times<=min(self.duration,max(note['offset'],alternate['offset'])+.32))
        hscale=max(1e-10,float(np.percentile(trace[local],90))) if local.any() else 1e-10
        bscale=max(1e-10,float(np.percentile(self.broad[local],90))) if local.any() else 1e-10

        def features(end,flags,report):
            harmonic=[];broad=[];share=[]
            for lo,hi in WINDOWS:
                mask=(self.times>=end+lo)&(self.times<end+hi)
                h=float(trace[mask].mean()) if mask.any() else 0.
                b=float(self.broad[mask].mean()) if mask.any() else 0.
                harmonic.append(h);broad.append(b);share.append(h/max(1e-10,b))
            duration=report['segment_samples']/report['sample_rate']
            phase=(end-report['context_start_seconds'])%duration
            active=[n for n in notes if not n['is_drum'] and n['onset']<=end<n['offset']]
            return np.asarray([*np.log1p(np.asarray(harmonic)/hscale*10),
                *np.log1p(np.asarray(broad)/bscale*10),*share,
                np.clip(np.log((harmonic[1]+1e-10)/(harmonic[3]+1e-10)),-8,8),
                np.clip(np.log((broad[1]+1e-10)/(broad[3]+1e-10)),-8,8),
                np.log1p((end-note['onset'])/.5),'forced_segment_boundary' in flags,
                'explicit_model_note_off' in flags,'same_pitch_reattack' in flags,
                min(phase,duration-phase)/duration,
                sum(not n['is_drum'] and abs(n['offset']-end)<.05 for n in notes)/10,
                sum(n['program']==note['program'] for n in active)/10,
                sum(n['program']!=note['program'] for n in active)/10],dtype=float)
        return (features(note['offset'],base_flags,base_report),
                features(alternate['offset'],alternate_flags,shift_report))


def build_case(case,base_folder,shift_folder):
    import numpy as np
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    base_report,notes,flags=load_run(base_folder)
    shift_report,shifted,shift_flags=load_run(shift_folder)
    expected={case['audio']:case['audio_sha256']}
    for report in (base_report,shift_report):
        identity=report['identity']
        if (identity['sources']!=expected or identity['start_ms']!=0 or identity['seconds']!=30
            or identity['context_seconds']!=2.1):raise ValueError('Prediction does not match the frozen case')
    if (base_report['identity'].get('segment_phase',0.)!=0. or
        shift_report['identity'].get('segment_phase')!=.5):raise ValueError('Unexpected segmentation pair')
    if any(base_report['identity'][k]!=shift_report['identity'][k] for k in
        ('source_commit','checkpoint_sha256','task')):raise ValueError('Different transcription models')
    if sha(case['audio'])!=case['audio_sha256'] or sha(case['labels_path'])!=case['labels_sha256']:
        raise ValueError('Frozen audio or annotations changed')
    _,decisions=refine(notes,shifted,flags,shift_flags,
        shifted_start=shift_report['context_start_seconds'],
        segment_seconds=shift_report['segment_samples']/shift_report['sample_rate'],only_forced=False)
    acoustic=AcousticEvidence(case['audio'],base_report['context_end_seconds'])
    reference=[n for n in read(case['labels_path']) if 0<=n['onset']<30]
    examples=[]
    for decision in decisions:
        if decision['status']!='replaced_from_explicit_prediction':continue
        i,j=decision['note_index'],decision['alternate_note_index'];n=notes[i];alternate=shifted[j]
        # Keep the complete-interval denominator fixed: selecting a later tail
        # must not hide a false positive beyond the evaluation boundary.
        if (not 0<=n['onset']<30 or max(n['offset'],alternate['offset'])>=30 or
            max(n['offset'],alternate['offset'])>acoustic.duration-.32):continue
        original_features,new_features=acoustic.feature_pair(n,alternate,flags[i],shift_flags[j],base_report,shift_report,notes)
        possible=[r for r in reference if r['pitch']==n['pitch'] and abs(r['onset']-n['onset'])<=.05]
        label=None;improvement=None
        if len(possible)==1 and possible[0]['offset']<30:
            r=possible[0]
            improvement=(abs(n['offset']-r['offset'])-abs(alternate['offset']-r['offset']))*1000
            if abs(improvement)>=PROTOCOL['label_min_error_difference_ms']:label=int(improvement>0)
        examples.append(dict(index=i,alternate_index=j,x=new_features-original_features,
            new_offset=alternate['offset'],label=label,improvement_ms=improvement))
    return dict(case=case,notes=notes,reference=reference,examples=examples,
        identity=dict(baseline_notes_sha256=sha(base_folder/'notes.json'),shifted_notes_sha256=sha(shift_folder/'notes.json'),
            audio_sha256=case['audio_sha256'],labels_sha256=case['labels_sha256']))


def fit(items):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    examples=[(item,e) for item in items for e in item['examples'] if e['label'] is not None]
    if len(examples)<10 or len({e['label'] for _,e in examples})!=2:
        raise ValueError('Insufficient independently labelled endpoint alternatives')
    counts=Counter(item['case']['piece_id'] for item,_ in examples)
    weight=np.asarray([1/counts[item['case']['piece_id']] for item,_ in examples]);weight/=weight.sum()
    x=np.stack([e['x'] for _,e in examples]);y=np.asarray([e['label'] for _,e in examples])
    scale=np.maximum(.1,np.sqrt(np.average(x*x,axis=0,weights=weight)))
    z=np.c_[np.clip(x/scale,-8,8),np.ones(len(x))]
    reg=PROTOCOL['regularization']
    def loss(beta):
        logits=z@beta
        return (np.sum(weight*(np.logaddexp(0,logits)-y*logits))+reg/2*(beta[:-1]@beta[:-1]),
            z.T@(weight*(expit(logits)-y))+reg*np.r_[beta[:-1],0.])
    optimized=minimize(loss,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500})
    if not optimized.success:raise ValueError('Endpoint ranking did not converge: '+str(optimized.message))
    return dict(scale=scale,weights=optimized.x,training_pairs=len(examples),
        positive_pairs=int(y.sum()),negative_pairs=int(len(y)-y.sum()))


def apply(item,model,threshold):
    import numpy as np
    from scipy.special import expit
    result=[dict(n) for n in item['notes']];decisions=[]
    for example in item['examples']:
        x=np.r_[np.clip(example['x']/model['scale'],-8,8),1.]
        probability=float(expit(x@model['weights']));selected=probability>threshold
        if selected:result[example['index']]['offset']=example['new_offset']
        decisions.append(dict(index=example['index'],alternate_index=example['alternate_index'],
            probability=probability,selected=selected,new_offset=example['new_offset']))
    return result,decisions


def score(item,notes):
    return metrics(item['reference'],[n for n in notes if 0<=n['onset']<30],30)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,default=Path('test-data/urmp/acoustic-releases-v1'))
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Keep completed experiments; use a new output directory')
    import numpy as np
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    manifest=read('test-data/urmp/release-development/manifest.json')
    if manifest['selected_ids']!=[3,4,5,6]:raise ValueError('Frozen development pieces differ')
    root=Path('test-data/urmp/release-development-predictions')
    if not (root/'completed.json').exists():raise ValueError('Finish all declared transcriptions before fitting')
    items=[]
    for case in manifest['cases']:
        item=build_case(case,root/case['name']/'baseline',root/case['name']/'shifted');items.append(item)
        print(json.dumps(dict(stage='features',case=case['name'],examples=len(item['examples']),
            labelled=sum(e['label'] is not None for e in item['examples']))),flush=True)
    train=[item for item in items if item['case']['piece_id'] in PROTOCOL['fit_piece_ids']]
    calibration=[item for item in items if item['case']['piece_id'] in PROTOCOL['calibration_piece_ids']]
    model=fit(train);options=[]
    for threshold in PROTOCOL['thresholds']:
        scores=[];changed=0
        for item in calibration:
            notes,decisions=apply(item,model,threshold)
            scores.append(score(item,notes)['strict']['complete_intervals']['f1'])
            changed+=sum(d['selected'] for d in decisions)
        options.append(dict(threshold=threshold,macro_strict_interval_f1=float(np.mean(scores)),changed=changed))
    chosen=max(options,key=lambda r:(r['macro_strict_interval_f1'],-r['changed'],r['threshold']))
    model['threshold']=chosen['threshold']
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'model.npz',**model)
    (args.output/'frozen-model.json').write_text(json.dumps(dict(protocol=PROTOCOL,model_sha256=sha(args.output/'model.npz'),
        runner_sha256=sha(__file__),training_pairs=model['training_pairs'],positive_pairs=model['positive_pairs'],
        negative_pairs=model['negative_pairs'],calibration=options,chosen=chosen),indent=2),encoding='utf-8')
    # Comparison examples and labels are loaded only after weights and the
    # calibration threshold have been written. They cannot choose the model.
    comparison=read('test-data/urmp/benchmark/manifest.json')
    if comparison['selected_ids']!=PROTOCOL['comparison_piece_ids']:raise ValueError('Frozen comparison pieces differ')
    for case in comparison['cases']:
        case=dict(case,split='comparison')
        items.append(build_case(case,Path('test-data/urmp/benchmark/yourmt3')/case['name'],
            Path('test-data/yourmt3/overlap-v1')/('urmp-'+case['name'])/'shifted'))
    rows=[]
    for item in items:
        notes,decisions=apply(item,model,model['threshold']);folder=args.output/item['case']['name'];folder.mkdir()
        (folder/'notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
        (folder/'decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
        row=dict(name=item['case']['name'],piece_id=item['case']['piece_id'],split=item['case']['split'],
            identity=item['identity'],alternatives=len(item['examples']),changed=sum(d['selected'] for d in decisions),
            baseline=score(item,item['notes']),learned=score(item,notes))
        rows.append(row);print(json.dumps(dict(stage='scored',**row)),flush=True)
    report=dict(protocol=PROTOCOL,model=read(args.output/'frozen-model.json'),rows=rows,
        enabled_in_application=False,limits=['Existing comparison songs were previously inspected; not a new blind benchmark.',
            'Original transcription model was trained using this dataset; new endpoint ranking uses piece-disjoint supervision.',
            'Physical acoustic endpoints do not determine authored LN release or chord-layer intent.',
            'No attack additions, omissions or pitch/program edits; original first-pass misses remain.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
