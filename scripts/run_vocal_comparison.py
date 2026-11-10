"""Compare fixed ROSVOT against the completed generic baseline on all 40 vocals."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_guitar_vocal_benchmark import evaluate, window_metrics, aggregate, read_json, require, verified_hash
from scripts.research_vocal_transcription import SOURCE_COMMIT, FILES, sha, verify_source


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/guitar-vocal-benchmark-v3'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--python', type=Path, default=Path('.venv-audio/Scripts/python.exe'))
    cli.add_argument('--source', type=Path, default=Path('test-data/other-instrument-model-survey-2026-11-10/ROSVOT'))
    cli.add_argument('--dependencies', type=Path, default=Path('test-data/rosvot-runtime'))
    args = cli.parse_args()
    require(not (args.output/'completed.json').exists(), 'Comparison already completed; preserve it')
    # Verify the complete 100-recording baseline and both vocal annotators first.
    baseline = evaluate(args.root)
    source_files = verify_source(args.source)
    manifest = read_json(args.root/'manifest.json')
    cases = [c for c in manifest['cases'] if c['dataset'] == 'vocadito']
    require(len(cases) == 40, 'Require every frozen vocal recording')
    runtime = json.loads(subprocess.check_output([str(args.python), '-c',
        "import sys,json; sys.path.insert(0,sys.argv[1]); "
        "from scripts.research_vocal_transcription import runtime_versions; "
        "print(json.dumps(runtime_versions()))", str(args.dependencies.resolve())], text=True))
    identity = dict(manifest_sha256=sha(args.root/'manifest.json'),
        baseline_completed_sha256=sha(args.root/'application/completed.json'),
        adapter_sha256=sha('scripts/research_vocal_transcription.py'), runner_sha256=sha(__file__),
        source_commit=SOURCE_COMMIT, source_files=source_files, model_files=FILES,
        evaluator_sha256=sha('scripts/evaluate_guitar_vocal_benchmark.py'),
        matcher_sha256=sha('scripts/evaluate_instrument_benchmark.py'),
        note_threshold=.85, word_threshold=.5, intra_threads=2, inter_threads=1,
        inference_scope='full_original_audio', normalization=False, runtime=runtime)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output/'plan.json'
    if plan.exists():
        require(read_json(plan) == identity, 'Frozen vocal comparison changed')
    else:
        require(not any(args.output.iterdir()), 'Inspect existing incomplete output before starting')
        plan.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    baseline_rows = {row['name']: row for row in baseline['rows']}
    rows = []
    groups = {engine: {'A1': [], 'A2': []} for engine in ('application', 'rosvot')}
    for index, case in enumerate(cases):
        folder = args.output/case['name']
        report_path = folder/'report.json'
        verified_hash(case['audio'], case['audio_sha256'])
        print(json.dumps(dict(stage='vocal_comparison', case=case['name'], index=index+1, total=40)), flush=True)
        if not report_path.exists():
            require(not folder.exists(), 'Inspect partial ROSVOT inference before retrying: '+case['name'])
            command = [str(args.python), 'scripts/research_vocal_transcription.py', '--audio', case['audio'],
                '--source', str(args.source), '--dependencies', str(args.dependencies), '--output', str(folder)]
            with (args.output/(case['name']+'.log')).open('w', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                    env=os.environ | {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2', 'PYTHONUTF8':'1'})
        report = read_json(report_path)
        candidate_identity = report['identity']
        expected = dict(source_commit=SOURCE_COMMIT, source_files=source_files, model_files=FILES,
            runner_sha256=identity['adapter_sha256'], audio=str(Path(case['audio']).resolve()),
            audio_sha256=case['audio_sha256'], original_sample_rate=case['sample_rate'],
            original_samples=case['samples'], original_dtype=case['decoded_dtype'],
            note_threshold=.85, word_threshold=.5, intra_threads=2, inter_threads=1,
            normalization=False, inference_scope='full_original_audio', lyrics_supplied=False,
            strict_weights=True, rmvpe_mode='get_pitch_batch/min_gap=6',
            versions=runtime, resampler='librosa/soxr_hq/fix=True/scale=False',
            resampled_sample_rate=24000,
            resampled_samples=(case['samples']*24000+case['sample_rate']-1)//case['sample_rate'])
        require(all(candidate_identity.get(k) == v for k, v in expected.items()), 'ROSVOT prediction identity changed')
        require(report['context_start_seconds'] == 0 and
            abs(report['context_end_seconds']-case['duration_seconds']) < 1e-10, 'ROSVOT physical input clock differs')
        for name, suffix in (('notes','.json'), ('raw_intervals','.json'), ('activations','.npz')):
            verified_hash(folder/(name.replace('_','-')+suffix), report[name+'_sha256'])
        predicted = read_json(folder/'notes.json')
        require(len(predicted) == report['notes'], 'ROSVOT event count differs')
        references = {}
        for annotator, entry in case['references'].items():
            verified_hash(entry['path'], entry['sha256'])
            reference = read_json(entry['path'])
            metrics, scope = window_metrics(reference, predicted, begin=0., end=case['seconds'],
                context_start=0., context_end=case['duration_seconds'])
            old = baseline_rows[case['name']]['references'][annotator]
            require(scope['reference_indices'] == old['interval_scope']['reference_indices'], 'Reference scoring scope differs')
            references[annotator] = dict(rosvot=dict(metrics=metrics, interval_scope=scope), application=old)
            for engine, values in (('application', old['metrics']), ('rosvot', metrics)):
                groups[engine][annotator].append(dict(name=case['name'], metrics=values))
        rows.append(dict(name=case['name'], report_sha256=sha(report_path), notes_sha256=report['notes_sha256'],
            references=references, release_reasons=report['release_reasons'],
            output_reasons=report['output_reasons'], elapsed_seconds=report['elapsed_seconds']))
    result = dict(revision=1, identity=identity, cases=40, annotations=80, rows=rows,
        groups={engine: {annotator:aggregate(values) for annotator,values in annotations.items()}
                for engine,annotations in groups.items()}, enabled_in_application=False,
        limitations=['Fixed public solo-vocal corpus; not unseen songs or polyphonic choruses.',
                    'A1/A2 scored separately with all onset-window predictions; no best-annotator selection.',
                    'Model note/word boundaries are not authored LN gestures; unknown input endings stay unknown.'])
    result_path = args.output/'comparison.json'
    with result_path.open('x', encoding='utf-8') as destination:
        destination.write(json.dumps(result, indent=2, allow_nan=False))
    (args.output/'completed.json').write_text(json.dumps(dict(identity=identity,
        plan_sha256=sha(plan), comparison_sha256=sha(result_path), cases=40), indent=2), encoding='utf-8')
    print(json.dumps(dict(stage='complete', groups=result['groups'])), flush=True)


if __name__ == '__main__':
    main()
