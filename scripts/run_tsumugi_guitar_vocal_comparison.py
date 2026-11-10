"""Compare fixed Tsumugi guitar/vocal weights on every prepared recording."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_guitar_vocal_benchmark import evaluate, window_metrics, aggregate, read_json, require, verified_hash
from scripts.research_tsumugi_transcription import SOURCE_COMMIT, MODEL_REVISION, MODELS, sha, verify_source


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/guitar-vocal-benchmark-v3'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--python', type=Path, default=Path('.venv-audio/Scripts/python.exe'))
    cli.add_argument('--source', type=Path, default=Path('test-data/other-instrument-model-survey-2026-11-10/tsumugi'))
    cli.add_argument('--dependencies', type=Path, default=Path('test-data/rosvot-runtime'))
    args = cli.parse_args()
    require(not (args.output/'completed.json').exists(), 'Preserve completed comparison')
    baseline = evaluate(args.root)
    source_files = verify_source(args.source)
    manifest = read_json(args.root/'manifest.json')
    cases = manifest['cases']
    models = {'GuitarSet': 'guitar_v1_5', 'vocadito': 'vocal_harmony_v1_5'}
    for model in models.values():
        verified_hash(args.source/'checkpoints'/('best_model_'+model+'.pth'), MODELS[model])
    runtime = json.loads(subprocess.check_output([str(args.python), '-c',
        "import sys,json; sys.path.insert(0,sys.argv[1]); "
        "from scripts.research_tsumugi_transcription import runtime_versions; print(json.dumps(runtime_versions()))",
        str(args.dependencies.resolve())], text=True))
    identity = dict(manifest_sha256=sha(args.root/'manifest.json'),
        baseline_completed_sha256=sha(args.root/'application/completed.json'),
        adapter_sha256=sha('scripts/research_tsumugi_transcription.py'), runner_sha256=sha(__file__),
        source_commit=SOURCE_COMMIT, source_files=source_files, model_revision=MODEL_REVISION,
        models={m:MODELS[m] for m in models.values()}, selection=models, runtime=runtime,
        evaluator_sha256=sha('scripts/evaluate_guitar_vocal_benchmark.py'),
        matcher_sha256=sha('scripts/evaluate_instrument_benchmark.py'),
        stem_filter='all', inference_scope='full_original_audio', normalization=False,
        intra_threads=2, inter_threads=1, amp=False, compile=False)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output/'plan.json'
    if plan.exists():
        require(read_json(plan) == identity, 'Frozen Tsumugi comparison changed')
    else:
        require(not any(args.output.iterdir()), 'Inspect existing incomplete output before starting')
        plan.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    baseline_rows = {r['name']:r for r in baseline['rows']}
    groups = {engine:{key:[] for key in ('GuitarSet/strings','vocadito/A1','vocadito/A2')}
              for engine in ('application','tsumugi')}
    rows = []
    for index, case in enumerate(cases):
        folder = args.output/case['name']
        report_path = folder/'report.json'
        model = models[case['dataset']]
        verified_hash(case['audio'], case['audio_sha256'])
        print(json.dumps(dict(stage='tsumugi_comparison', case=case['name'], index=index+1,total=100)),flush=True)
        if not report_path.exists():
            require(not folder.exists(), 'Inspect partial inference before retry: '+case['name'])
            with (args.output/(case['name']+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run([str(args.python),'scripts/research_tsumugi_transcription.py',
                    '--audio',case['audio'],'--model',model,'--stem','all',
                    '--source',str(args.source),'--dependencies',str(args.dependencies),'--output',str(folder)],
                    check=True,stdout=log,stderr=subprocess.STDOUT,
                    env=os.environ | {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUTF8':'1'})
        report = read_json(report_path)
        expected = dict(source_commit=SOURCE_COMMIT,source_files=source_files,runner_sha256=identity['adapter_sha256'],
            model_revision=MODEL_REVISION,model=model,model_sha256=MODELS[model],
            audio=str(Path(case['audio']).resolve()),audio_sha256=case['audio_sha256'],
            original_sample_rate=case['sample_rate'],original_samples=case['samples'],original_channels=1,
            resampled_sample_rate=22050,resampled_samples=(case['samples']*22050+case['sample_rate']-1)//case['sample_rate'],
            resampler='torchaudio.functional.resample/default',mono_policy='duplicate_to_stereo',
            normalization=False,inference_scope='full_original_audio',intra_threads=2,inter_threads=1,
            amp=False,compile=False,stem_filter='all',versions=runtime,seed=42,evidence_revision=2)
        require(all(report['identity'].get(k)==v for k,v in expected.items()),'Tsumugi prediction identity changed')
        require(report['context_start_seconds']==0 and
                abs(report['context_end_seconds']-case['duration_seconds'])<1e-10,'Physical input clock changed')
        for name in ('notes','raw_notes','window_notes','interval_constructions'):
            verified_hash(folder/(name.replace('_','-')+'.json'),report[name+'_sha256'])
        predicted = read_json(folder/'notes.json')
        require(len(predicted)==report['notes'],'Event count changed')
        references = {}
        for annotator, entry in case['references'].items():
            verified_hash(entry['path'],entry['sha256'])
            metrics,scope = window_metrics(read_json(entry['path']),predicted,begin=0,end=case['seconds'],
                context_start=0,context_end=case['duration_seconds'])
            old = baseline_rows[case['name']]['references'][annotator]
            require(scope['reference_indices']==old['interval_scope']['reference_indices'],'Reference scope changed')
            references[annotator] = dict(tsumugi=dict(metrics=metrics,interval_scope=scope),application=old)
            for engine,values in (('application',old['metrics']),('tsumugi',metrics)):
                groups[engine][case['dataset']+'/'+annotator].append(dict(name=case['name'],metrics=values))
        rows.append(dict(name=case['name'],model=model,references=references,report_sha256=sha(report_path),
            notes_sha256=report['notes_sha256'],instrument_classes=report['instrument_classes'],
            elapsed_seconds=report['elapsed_seconds']))
    result = dict(revision=1,identity=identity,cases=100,annotations=140,rows=rows,
        groups={engine:{key:aggregate(value) for key,value in group.items()} for engine,group in groups.items()},
        enabled_in_application=False,limitations=[
            'Specialized weights selected by known input corpus family; no per-song annotation or tuning.',
            'All classes decoded, including erroneous drums; no best-annotator selection.',
            'Public corpus may occur in upstream training; this is not an unseen blind test.',
            'Stitched model tails are scored but not accepted as validated LN releases.'])
    target = args.output/'comparison.json'
    with target.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False))
    (args.output/'completed.json').write_text(json.dumps(dict(identity=identity,plan_sha256=sha(plan),
        comparison_sha256=sha(target),cases=100),indent=2),encoding='utf-8')
    print(json.dumps(dict(stage='complete',groups=result['groups'])),flush=True)


if __name__=='__main__':
    main()
