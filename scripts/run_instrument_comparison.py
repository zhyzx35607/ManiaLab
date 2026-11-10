"""Run both frozen detectors serially on a prepared acoustic comparison set."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, required=True)
    cli.add_argument('--context-seconds', type=float, default=10.)
    cli.add_argument('--python', type=Path, default=Path('test-data/yourmt3/venv/Scripts/python.exe'))
    args = cli.parse_args()
    if not math.isfinite(args.context_seconds) or not 0 <= args.context_seconds <= 30:
        raise ValueError('Use a bounded inference context')
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    manifest = read(args.root / 'manifest.json')
    scripts = ['probe_instrument_baseline.py', 'research_instrument_transcription.py',
               'audit_instrument_releases.py', 'evaluate_instrument_benchmark.py',
               'run_instrument_comparison.py']
    model_source = read('test-data/yourmt3/model/source.json')
    model_path = Path('test-data/yourmt3/model') / model_source['path']
    identity = dict(manifest_sha256=sha(args.root / 'manifest.json'),
                    scripts={p: sha(Path('scripts') / p) for p in scripts},
                    audio_code_sha256=sha('manialab/audio.py'),
                    baseline_model_sha256=sha('assets/basic-pitch.onnx'),
                    yourmt3_model_sha256=sha(model_path),
                    context_seconds=args.context_seconds, segment_phase=0.,
                    intra_threads=2, inter_threads=1)
    if identity['yourmt3_model_sha256'] != model_source['sha256']:
        raise ValueError('Frozen model identity changed')
    for case in manifest['cases']:
        if sha(case['audio']) != case['audio_sha256']:
            raise ValueError('Frozen audio changed: ' + case['name'])
    plan = args.root / 'comparison-plan.json'
    if plan.exists():
        if read(plan)['identity'] != identity:
            raise ValueError('Earlier plan differs; preserve it and use a new run directory')
    else:
        plan.write_text(json.dumps(dict(identity=identity,
            created_utc=datetime.now(timezone.utc).isoformat()), indent=2), encoding='utf-8')

    def run(command, log):
        with log.open('w', encoding='utf-8') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)

    baselines = [args.root / 'application' / c['name'] / 'report.json' for c in manifest['cases']]
    if not any(p.exists() for p in baselines):
        print('Running Basic Pitch for every frozen case', flush=True)
        run([sys.executable, 'scripts/probe_instrument_baseline.py', '--root', str(args.root),
             '--context-seconds', str(args.context_seconds), '--threads', '2'], args.root / 'baseline.log')
    if not all(p.exists() for p in baselines):
        raise ValueError('Partial baseline run: preserve it and inspect before resuming')
    for case, path in zip(manifest['cases'], baselines):
        report = read(path)
        expected = dict(sources={case['audio']: case['audio_sha256']},
                        model_sha256=identity['baseline_model_sha256'],
                        audio_code_sha256=identity['audio_code_sha256'],
                        runner_sha256=identity['scripts']['probe_instrument_baseline.py'],
                        start_ms=case['start_ms'], seconds=case['seconds'],
                        context_seconds=args.context_seconds, intra_threads=2, inter_threads=1)
        if report['identity'] != expected or sha(path.parent / 'notes.json') != report['notes_sha256']:
            raise ValueError('Baseline evidence differs: ' + case['name'])
        if sha(path.parent / 'activations.npz') != report['activations_sha256']:
            raise ValueError('Baseline activations changed: ' + case['name'])

    output = args.root / 'yourmt3'
    output.mkdir(exist_ok=True)
    for i, case in enumerate(manifest['cases']):
        folder = output / case['name']
        print(json.dumps(dict(stage='yourmt3', case=case['name'], index=i+1,
                              total=len(manifest['cases']))), flush=True)
        if not (folder / 'report.json').exists():
            run([str(args.python), 'scripts/research_instrument_transcription.py',
                 '--audio', case['audio'], '--start-ms', str(case['start_ms']),
                 '--seconds', str(case['seconds']), '--context-seconds', str(args.context_seconds),
                 '--segment-phase', '0', '--threads', '2', '--output', str(folder)],
                 output / (case['name'] + '.log'))
        report = read(folder / 'report.json')
        expected = dict(sources={case['audio']: case['audio_sha256']},
                        checkpoint_sha256=identity['yourmt3_model_sha256'],
                        runner_sha256=identity['scripts']['research_instrument_transcription.py'],
                        start_ms=case['start_ms'], seconds=case['seconds'],
                        context_seconds=args.context_seconds, segment_phase=0.)
        if any(report['identity'].get(k) != v for k, v in expected.items()):
            raise ValueError('Cached YourMT3 prediction differs: ' + case['name'])
        for name in ('notes', 'tokens'):
            suffix = '.json' if name == 'notes' else '.npz'
            if sha(folder / (name + suffix)) != report[name + '_sha256']:
                raise ValueError('YourMT3 evidence changed: ' + case['name'])
        if not (folder / 'release-audit.json').exists():
            run([str(args.python), 'scripts/audit_instrument_releases.py', '--runs', str(folder)],
                output / (case['name'] + '-audit.log'))
        audit = read(folder / 'release-audit.json')
        if (audit['runner_sha256'] != identity['scripts']['audit_instrument_releases.py']
                or audit['notes_sha256'] != report['notes_sha256']
                or audit['tokens_sha256'] != report['tokens_sha256']):
            raise ValueError('YourMT3 endpoint audit differs: ' + case['name'])
    score = args.root / 'comparison.json'
    if not score.exists():
        run([sys.executable, 'scripts/evaluate_instrument_benchmark.py', '--root', str(args.root),
             '--output', str(score)], args.root / 'evaluation.log')
    if read(score)['manifest_sha256'] != identity['manifest_sha256']:
        raise ValueError('Evaluation manifest changed')
    (args.root / 'completed.json').write_text(json.dumps(dict(identity=identity,
        cases=len(manifest['cases']), report_sha256=sha(score)), indent=2), encoding='utf-8')
    print('Completed frozen comparison: ' + str(score), flush=True)


if __name__ == '__main__':
    main()
