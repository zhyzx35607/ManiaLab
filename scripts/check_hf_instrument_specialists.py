"""Fixed serial 17-bass/2-sax comparison; prepare-only never loads a network."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_hf_instrument as adapter
from scripts import evaluate_guitar_vocal_benchmark as evaluation
from scripts.evaluate_instrument_benchmark import window_metrics
from scripts.run_bass_model_comparison import validate_manifest, score_case, aggregate_extra

read, require, sha, verify = evaluation.read_json,evaluation.require,evaluation.sha,evaluation.verified_hash
BASS_MANIFEST = '5e575417b05674ba23a0d1614643197a97ace1a0d1d533fbdd689987fed3d46f'
BASS_BASELINE = '3bf7d0c9513b08396ef85c9b60209df662e8969222d66a9466361832360aa417'
SAX_MANIFEST = 'e363a4e7b5f25628909eb13031251847492d9fa816e520b98fbc6d612bcda16a'
SAX_BASELINE = 'ff98a2e3c53671c1bcf4ec16a6c3cdff3215e17e239101dfe6883f884fbe2e75'
CASE_NAMES = [f'bass-{i:03}' for i in range(1,18)]+['sax-25-4-sax','sax-39-4-sax']


def normalized_metrics(metrics, scope=None):
    """Adapt legacy URMP availability only with verified observable context."""
    result={}
    for tolerance,values in metrics.items():
        cells={}
        for key,value in values.items():
            if key not in evaluation.METRICS:
                continue
            if 'available' not in value:
                require(scope is not None and scope.get('unobservable_reference_offsets')==0
                        and 0<=scope['context_start_seconds']<=scope['begin_seconds']
                        <scope['end_seconds']<=scope['context_end_seconds'],
                        'Legacy metrics need verified observable reference context')
                value=dict(available=True,**value)
            cells[key]=value
        result['fixed_40_80' if tolerance=='strict' else tolerance]=cells
    return result


def verified_reuse(folder,identity,case):
    """Reuse complete matching evidence, never infer success from a directory."""
    report=read(folder/'report.json')
    require(report['identity']==identity,'Reused model/source/runtime identity differs')
    expected=dict(audio=str(Path(case['audio']).resolve()),audio_sha256=case['audio_sha256'],
                  original_sample_rate=case['original_rate'],original_samples=case['original_samples'])
    require(all(report['input'].get(key)==value for key,value in expected.items()),'Reused audio identity differs')
    require(report.get('raw_decoder_parity') is True and report.get('unknown_tails_retained') is True,
            'Reused output lacks event parity or tail provenance')
    for filename,key in (('notes.json','notes_sha256'),('raw-notes.json','raw_notes_sha256'),
                         ('activations.npz','activations_sha256'),('windows.npz','windows_sha256')):
        verify(folder/filename,report[key])
    notes=read(folder/'notes.json')
    evaluation.validate_notes(notes,reference=False)
    require(len(notes)==report['notes'],'Reused note count differs')
    return notes,report


def prepare():
    import numpy as np
    from scipy.io import wavfile
    bass_root=ROOT/'test-data/bass-benchmark-v1'
    bass_previous=ROOT/'test-data/bass-model-comparison-v1'
    sax_root=ROOT/'test-data/urmp/comparison-extension'
    sax_previous=ROOT/'test-data/tsumugi-urmp-comparison-v1'
    files={str(bass_root/'manifest.json'):BASS_MANIFEST,
           str(bass_previous/'comparison.json'):BASS_BASELINE,
           str(sax_root/'manifest.json'):SAX_MANIFEST,
           str(sax_previous/'comparison.json'):SAX_BASELINE}
    for path,digest in files.items(): verify(path,digest)
    bass_cases=validate_manifest(read(bass_root/'manifest.json'))
    bass_prior={row['name']:row for row in read(bass_previous/'comparison.json')['rows']}
    require(set(bass_prior)=={case['name'] for case in bass_cases},'Incomplete bass baseline')
    sax_cases=[c for c in read(sax_root/'manifest.json')['cases'] if c['kind']=='isolated' and c['instrument']=='sax']
    require([c['name'] for c in sax_cases]==['25-4-sax','39-4-sax'],'Sax cohort changed')
    sax_prior={row['name']:row for row in read(sax_previous/'comparison.json')['rows']}
    cases=[]
    for case in bass_cases:
        name=case['name']
        baseline=bass_prior[name]
        verify(case['labels'],case['labels_sha256'])
        reference=read(case['labels'])
        require(len(reference)==case['reference_count'] and
                sum(n['onset']<case['seconds'] for n in reference)==case['scoring_reference_count'],
                'Bass reference denominator changed')
        for engine,values in baseline['engines'].items():
            verify(bass_previous/engine/name/'notes.json',values['notes_sha256'])
        cases.append(dict(name='bass-'+name,source_name=name,model='bass',audio=case['audio'],
            audio_sha256=case['audio_sha256'],labels=case['labels'],labels_sha256=case['labels_sha256'],
            original_rate=case['sample_rate'],original_samples=case['samples'],
            context_end=case['duration_seconds'],scoring_end=case['seconds'],
            scoring_reference=case['scoring_reference_count'],baseline=baseline['engines']))
    for case in sax_cases:
        name=case['name'];baseline=sax_prior[name];clip=sax_previous/(name+'.wav')
        verify(clip,baseline['clip_sha256'])
        verify(case['labels_path'],case['labels_sha256'])
        verify(sax_previous/name/'notes.json',baseline['notes_sha256'])
        scope=baseline['interval_scope']
        require(scope['begin_seconds']==0 and scope['end_seconds']==30 and scope['context_start_seconds']==0,
                'Sax scoring window changed')
        cases.append(dict(name='sax-'+name,source_name=name,model='sax',audio=str(clip),
            audio_sha256=baseline['clip_sha256'],labels=case['labels_path'],labels_sha256=case['labels_sha256'],
            original_rate=16000,original_samples=baseline['clip_samples'],context_end=scope['context_end_seconds'],
            scoring_end=30,scoring_reference=sum(0<=n['onset']<30 for n in read(case['labels_path'])),
            baseline={engine:dict(metrics=normalized_metrics(baseline[key],scope),interval_scope=scope) for engine,key in
                      (('application','application'),('yourmt3','yourmt3'),('tsumugi','metrics'))}))
    require([c['name'] for c in cases]==CASE_NAMES,'Require all 19 cases in the frozen order')
    for case in cases:
        verify(case['audio'],case['audio_sha256'])
        rate,wave=wavfile.read(case['audio'])
        require(rate==case['original_rate'] and len(wave)==case['original_samples']
                and wave.ndim==1 and wave.dtype==np.int16 and len(wave)/rate==case['context_end'],
                'Original PCM clock differs: '+case['name'])
    source=adapter.verify_source()
    weights={model:dict(path=str(adapter.verify_checkpoint(model)),**adapter.MODELS[model]) for model in ('bass','sax')}
    return dict(revision=1,order=CASE_NAMES,cohort_files=files,source=source,weights=weights,
        code_sha256={str(path):sha(path) for path in (Path(__file__).resolve(),Path(adapter.__file__).resolve(),
            ROOT/'manialab/piano.py',ROOT/'scripts/run_bass_model_comparison.py',
            ROOT/'scripts/evaluate_guitar_vocal_benchmark.py',ROOT/'scripts/evaluate_instrument_benchmark.py')},
        thresholds=adapter.THRESHOLDS,device='cpu',intra_threads=2,inter_threads=1,
        normalization=False,timing_quantization=False,unknown_tails_retained=True,cases=cases)


def run(plan,output):
    import json
    output=Path(output)
    require(not output.exists(),'Preserve completed or incomplete comparison')
    output.mkdir(parents=True)
    adapter.write_new(output/'plan.json',plan)
    rows=[]
    network,identity,model=None,None,None
    for index,case in enumerate(plan['cases']):
        verify(case['audio'],case['audio_sha256']);verify(case['labels'],case['labels_sha256'])
        if model!=case['model']:
            # Release the previous network before allocating the next; no parallel models.
            del network
            model=case['model']
            network,identity=adapter.load_model(model,plan['weights'][model]['path'])
        reuse=Path(plan['reuse_root'])/case['name'] if plan.get('reuse_root') else None
        reused=bool(reuse is not None and (reuse/'report.json').is_file())
        folder=reuse if reused else output/case['name']
        if reused:
            notes,report=verified_reuse(folder,identity,case)
        else:
            notes,report=adapter.transcribe(network,identity,case['audio'],folder)
        require(report['input']['original_samples']==case['original_samples']
                and report['input']['original_sample_rate']==case['original_rate']
                and report['context_end_seconds']==case['context_end'],'Inference context changed')
        reference=read(case['labels'])
        if model=='bass':
            result=score_case(reference,notes,end=case['scoring_end'],context_end=case['context_end'])
        else:
            metrics,scope=window_metrics(reference,notes,begin=0,end=case['scoring_end'],
                                         context_start=0,context_end=case['context_end'])
            result=dict(metrics=normalized_metrics(metrics,scope),interval_scope=scope)
        require(result['metrics']['standard']['pitched_onsets']['reference']==case['scoring_reference'],
                'Scoring reference denominator differs')
        engines=dict(case['baseline'],hf_specialist=result)
        row=dict(name=case['name'],model=model,engines=engines,seconds=report['seconds'],
            reused=reused,inference_folder=str(folder.resolve()),
            report_sha256=sha(folder/'report.json'),notes_sha256=report['notes_sha256'],
            release_reasons=report['release_reasons'])
        adapter.write_new(output/case['name']/'metrics.json',row)
        rows.append(row)
        scores=result['metrics']['standard']
        print(json.dumps(dict(index=index+1,total=19,name=case['name'],seconds=report['seconds'],
            onset=scores['pitched_onsets']['f1'],interval=scores['complete_intervals']['f1'])),flush=True)
    require([row['name'] for row in rows]==CASE_NAMES,'Never aggregate a partial cohort')
    report=summarize(plan,rows)
    for path,digest in plan['code_sha256'].items():verify(path,digest)
    for path,digest in plan['cohort_files'].items():verify(path,digest)
    adapter.write_new(output/'report.json',report)
    print(json.dumps(dict(stage='complete',cases=len(rows),totals=report['totals'])),flush=True)


def summarize(plan,rows):
    require([row['name'] for row in rows]==CASE_NAMES,'Never aggregate a partial cohort')
    totals={}
    for instrument in ('bass','sax'):
        selected=[row for row in rows if row['model']==instrument]
        totals[instrument]={}
        for engine in selected[0]['engines']:
            results=[dict(name=row['name'],**row['engines'][engine]) for row in selected]
            totals[instrument][engine]=dict(nominal_pitched=evaluation.aggregate(results))
            if instrument=='bass':
                totals[instrument][engine].update(aggregate_extra(results))
    release_reasons=Counter()
    for row in rows:release_reasons.update(row['release_reasons'])
    report=dict(identity=plan,rows=rows,totals=totals,seconds=sum(row['seconds'] for row in rows),
        newly_computed_seconds=sum(row['seconds'] for row in rows if not row['reused']),
        enabled_in_application=False,musical_acceptance=False,training_performed=False,
        release_reasons=dict(release_reasons),
        limitations=['Public isolated sources with possible upstream training overlap; no blind generalization claim.',
                    'All decoded physical candidates are scored, including unobserved and decoder-limited tails.',
                    'FiloBass annotations include nominal pitches for dead notes, slides and harmonics.',
                    'Two sax recordings do not establish all-style or polyphonic sax performance.',
                    'Acoustic interval accuracy does not establish authored LN quality.'])
    return report


def main():
    import json
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path)
    cli.add_argument('--reuse-root',type=Path,help='Reuse complete per-case evidence after checking hashes and exact runtime identity')
    cli.add_argument('--prepare-only',action='store_true')
    args=cli.parse_args()
    require(args.prepare_only or args.output is not None,'--output is required for inference')
    sys.path[:0]=[str(adapter.SOURCE_ROOT/'dependencies'),str(ROOT/'test-data/rosvot-runtime')]
    plan=prepare()
    if args.reuse_root:plan['reuse_root']=str(args.reuse_root.resolve())
    if args.prepare_only:
        print(json.dumps(dict(prepared=True,network_loaded=False,cases=plan['order'],
            scoring_references={m:sum(c['scoring_reference'] for c in plan['cases'] if c['model']==m)
                                for m in ('bass','sax')},code_sha256=plan['code_sha256'])),flush=True)
    else:
        run(plan,args.output)


if __name__=='__main__':main()
