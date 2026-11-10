"""Compare both published guitar CRNNs on all 60 frozen GuitarSet recordings."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_hf_instrument as adapter
from scripts import evaluate_guitar_vocal_benchmark as evaluation

read, require, sha, verify = (evaluation.read_json, evaluation.require,
                            evaluation.sha, evaluation.verified_hash)
MODELS = {
    'guitar_gaps': dict(filename='guitar-gaps.pth', bytes=99178877, iteration=5000,
        sha256='65483e7c0e340a90415b15b520687587698c8c728f5fa470a205f13ee45c6513'),
    'guitar_fl': dict(filename='guitar-fl.pth', bytes=98916957, iteration=100000,
        sha256='50d93dba89bdd3401849bc735614478e83d9f46d21fa3f71d8aca5acc0a52028'),
}
FROZEN = {
    'test-data/guitar-vocal-benchmark-v3/manifest.json':
        '68932cb2b6aae626c83b1f725ec14ebb58bf7984edbd07208bfb37b52d6e76e9',
    'test-data/redecode-current-guitar-vocal-v1/evaluation.json':
        'f36b6bb9ef0d04659a4ddb036bddb56807f78bf52cd40ee651fae7016592673f',
    'test-data/tsumugi-guitar-vocal-comparison-v1/comparison.json':
        'ff8e7a42d8dc92d18cd135e3b218fd9b3e02f75ef775acd0318e67ad7581047d',
}
ADAPTER_SHA = 'b734e0806442ce028d1e16acf98e0808d0c96aaa1d123a9dc885d300a4b45bc1'


def prepare():
    import numpy as np
    from scipy.io import wavfile
    for name, digest in FROZEN.items():
        verify(ROOT/name, digest)
    verify(adapter.__file__, ADAPTER_SHA)
    manifest = read(ROOT/'test-data/guitar-vocal-benchmark-v3/manifest.json')
    cases = [case for case in evaluation.verify_manifest(manifest) if case['dataset'] == 'GuitarSet']
    require(len(cases) == 60, 'Require every frozen guitar recording')
    require(Counter(c['metadata']['mode'] for c in cases) == {'comp': 30, 'solo': 30},
            'Accompaniment/solo cohort changed')
    require(Counter(re.match(r'[A-Za-z]+', c['metadata']['style']).group() for c in cases)
            == dict.fromkeys(('BN', 'Funk', 'Jazz', 'Rock', 'SS'), 12), 'Style cohort changed')
    app_root = ROOT/'test-data/redecode-current-guitar-vocal-v1'
    tsumugi_root = ROOT/'test-data/tsumugi-guitar-vocal-comparison-v1'
    app = {row['name']: row for row in read(app_root/'evaluation.json')['rows']}
    tsumugi = {row['name']: row for row in read(tsumugi_root/'comparison.json')['rows']}
    prepared = []
    for case in cases:
        name = case['name']
        verify(case['audio'], case['audio_sha256'])
        rate, wave = wavfile.read(case['audio'], mmap=True)
        require(wave.ndim == 1 and wave.dtype == np.int16 and rate == case['sample_rate']
                and len(wave) == case['samples'] and len(wave)/rate == case['duration_seconds'],
                'Original PCM clock differs: '+name)
        labels = case['references']['strings']
        verify(labels['path'], labels['sha256'])
        reference = read(labels['path'])
        require(len(reference) == labels['count'], 'Reference count changed')
        baseline = {}
        for engine, folder, old, report_hash, note_hash in (
            ('application', app_root/'application'/name, app[name]['references']['strings'],
             app[name]['evidence']['report_sha256'], app[name]['evidence']['notes_sha256']),
            ('tsumugi', tsumugi_root/name, tsumugi[name]['references']['strings']['tsumugi'],
             tsumugi[name]['report_sha256'], tsumugi[name]['notes_sha256']),
        ):
            verify(folder/'report.json', report_hash)
            verify(folder/'notes.json', note_hash)
            predicted = read(folder/'notes.json')
            metrics, scope = evaluation.window_metrics(reference, predicted, begin=0, end=case['seconds'],
                context_start=0, context_end=case['duration_seconds'])
            require(metrics == old['metrics'] and scope == old['interval_scope'],
                    'Frozen baseline does not reproduce: '+engine+'/'+name)
            require(scope['reference'] == labels['scored_count'], 'Scoring denominator changed')
            baseline[engine] = dict(metrics=metrics, interval_scope=scope,
                folder=str(folder.resolve()), notes_sha256=note_hash, report_sha256=report_hash)
        prepared.append(dict(case, baseline=baseline))
    source = adapter.verify_source()
    with patch.dict(adapter.MODELS, MODELS, clear=True):
        weights = {name: dict(path=str(adapter.verify_checkpoint(name)), **spec)
                   for name, spec in MODELS.items()}
    return dict(cases=prepared, source=source, weights=weights, cohort_files=FROZEN,
        code_sha256={str(path): sha(path) for path in (Path(__file__).resolve(),
            Path(adapter.__file__).resolve(), ROOT/'manialab/piano.py',
            Path(evaluation.__file__).resolve(), ROOT/'scripts/evaluate_instrument_benchmark.py')},
        model_order=list(MODELS), normalization=False, timing_quantization=False,
        thresholds=adapter.THRESHOLDS, device='cpu', intra_threads=2, inter_threads=1,
        inference_scope='full_original_audio', unknown_tails_retained=True)


def existing_prediction(folder, identity, case):
    report = read(folder/'report.json')
    require(report['identity'] == identity, 'Reused model or runtime identity differs')
    expected = dict(audio=str(Path(case['audio']).resolve()), audio_sha256=case['audio_sha256'],
                    original_samples=case['samples'], original_sample_rate=case['sample_rate'])
    require(all(report['input'].get(k) == v for k, v in expected.items()), 'Reused audio differs')
    require(report['context_start_seconds'] == 0 and report['context_end_seconds'] == case['duration_seconds']
            and report['raw_decoder_parity'] is True and report['unknown_tails_retained'] is True,
            'Incomplete physical-clock or provenance evidence')
    for filename, key in (('notes.json', 'notes_sha256'), ('raw-notes.json', 'raw_notes_sha256'),
                          ('windows.npz', 'windows_sha256'), ('activations.npz', 'activations_sha256')):
        verify(folder/filename, report[key])
    notes = read(folder/'notes.json')
    evaluation.validate_notes(notes, reference=False)
    require(len(notes) == report['notes'], 'Reused event count differs')
    return notes, report


def summarize(plan, rows):
    names = [case['name'] for case in plan['cases']]
    require(len(names) == 60 and len(set(names)) == 60, 'Require 60 distinct cases')
    require([(row['model'], row['name']) for row in rows]
            == [(model, name) for model in MODELS for name in names],
            'Require both models on the entire cohort in the fixed order')
    groups = {engine: evaluation.aggregate([dict(name=case['name'], **case['baseline'][engine])
              for case in plan['cases']]) for engine in ('application', 'tsumugi')}
    for model in MODELS:
        groups[model] = evaluation.aggregate([row for row in rows if row['model'] == model])
    strata = {}
    for kind in ('mode', 'style'):
        partitions = {}
        for case in plan['cases']:
            label = case['metadata'][kind]
            if kind == 'style':
                label = re.match(r'[A-Za-z]+', label).group()
            partitions.setdefault(label, []).append(case)
        strata[kind] = {}
        for label, cases in partitions.items():
            members = {case['name'] for case in cases}
            cells = {engine: evaluation.aggregate([dict(name=case['name'], **case['baseline'][engine])
                     for case in cases]) for engine in ('application', 'tsumugi')}
            cells.update({model: evaluation.aggregate([row for row in rows
                          if row['model'] == model and row['name'] in members]) for model in MODELS})
            strata[kind][label] = dict(names=[case['name'] for case in cases], engines=cells)
    reasons = {model: dict(sum((Counter(row['release_reasons']) for row in rows if row['model'] == model), Counter()))
               for model in MODELS}
    return dict(plan=plan, rows=rows, groups=groups, strata=strata, release_reasons=reasons,
        seconds=sum(row['seconds'] for row in rows),
        newly_computed_seconds=sum(row['seconds'] for row in rows if not row['reused']),
        enabled_in_application=False, musical_acceptance=False, training_performed=False,
        limitations=['Public GuitarSet may overlap upstream training; not an unseen blind test.',
            'Two complete model cohorts, no best-checkpoint selection per recording.',
            'All physical events retained, including unknown release boundaries.',
            'One full case has unobservable reference tails and no interval score; onset score retained.',
            'Isolated acoustic guitar results do not establish distorted guitar or mixed-song LN quality.',
            'Original wrapper librosa resampling is replaced by the existing scipy resample_poly adapter.'])


def run(plan, output):
    import json
    output = Path(output)
    require(not (output/'report.json').exists(), 'Preserve completed comparison')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        require(read(output/'plan.json') == plan, 'Resume plan differs')
    else:
        require(not any(output.iterdir()), 'Inspect incomplete output before starting')
        adapter.write_new(output/'plan.json', plan)
    rows = []
    # The immutable bass/sax adapter is reused within this process. Its source,
    # thresholds, inference windows and release-provenance decoder do not change.
    with patch.dict(adapter.MODELS, MODELS, clear=True):
        for model in MODELS:
            network, identity = adapter.load_model(model, plan['weights'][model]['path'])
            identity['guitar_runner_sha256'] = sha(__file__)
            for index, case in enumerate(plan['cases']):
                verify(case['audio'], case['audio_sha256'])
                labels = case['references']['strings']
                verify(labels['path'], labels['sha256'])
                folder = output/model/case['name']
                reused = (folder/'report.json').is_file()
                if not reused:
                    adapter.transcribe(network, identity, case['audio'], folder)
                notes, report = existing_prediction(folder, identity, case)
                metrics, scope = evaluation.window_metrics(read(labels['path']), notes,
                    begin=0, end=case['seconds'], context_start=0, context_end=case['duration_seconds'])
                require(scope['reference_indices'] == case['baseline']['application']['interval_scope']['reference_indices'],
                        'Reference denominator changed')
                row = dict(name=case['name'], model=model, metrics=metrics, interval_scope=scope,
                    report_sha256=sha(folder/'report.json'), notes_sha256=report['notes_sha256'],
                    release_reasons=report['release_reasons'], seconds=report['seconds'], reused=reused)
                rows.append(row)
                print(json.dumps(dict(model=model, index=index+1, total=60, name=case['name'],
                    seconds=report['seconds'], reused=reused,
                    onset_f1=metrics['standard']['pitched_onsets']['f1'],
                    interval_f1=metrics['standard']['complete_intervals']['f1'])), flush=True)
            del network
    for path, digest in plan['code_sha256'].items():
        verify(path, digest)
    for path, digest in FROZEN.items():
        verify(ROOT/path, digest)
    report = summarize(plan, rows)
    adapter.write_new(output/'report.json', report)
    print(json.dumps(dict(stage='complete', cases=len(rows), groups=report['groups'])), flush=True)


def main():
    import json
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--prepare-only', action='store_true')
    args = cli.parse_args()
    require(args.prepare_only or args.output is not None, '--output is required')
    plan = prepare()
    if args.prepare_only:
        require('torch' not in sys.modules, 'Prepare loaded Torch')
        print(json.dumps(dict(cases=len(plan['cases']), weights=plan['weights'],
            code_sha256=plan['code_sha256'], network_loaded=False)), flush=True)
    else:
        run(plan, args.output)


if __name__ == '__main__':
    main()
