"""Compare frozen Tsumugi transcription before/after Mega 53 separation.

Every predetermined family and control is retained. Target-directed mixture
precision is not the quality of all-instrument transcription; also compare
target-only preservation and complete source coverage for pieces 12 and 31.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_mega53_separation import CASES, RATE, save, sha
from scripts.evaluate_instrument_benchmark import match
from scripts.research_tsumugi_transcription import MODELS, SOURCE_COMMIT, verify_source

MODEL = 'other_v1_5'
MANIFEST = ROOT / 'test-data/urmp/comparison-extension/manifest.json'
MANIFEST_SHA = 'e363a4e7b5f25628909eb13031251847492d9fa816e520b98fbc6d612bcda16a'
BASE = ROOT / 'test-data/banquet-separation-v1'
BASE_REPORT_SHA = '1cb7aee0598cd5d535ea64aa12637ee144d4c92b34366ff408c8056c76236997'
FAMILY = dict(zip([n for n, _ in CASES], ['vn', 'vc', 'va', 'sax', 'tpt', 'hn', 'tbn', 'db']))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(reference, predicted):
    # Score the central two seconds so that cropping does not create attack
    # evidence. Reference tails must be observable before interval scoring.
    for notes in (reference, predicted):
        assert all(all(math.isfinite(n[k]) for k in ('onset', 'offset', 'pitch', 'program'))
                   and n['offset'] > n['onset'] for n in notes), 'Invalid interval evidence'
    ref = [n for n in reference if 2 <= n['onset'] < 4]
    pred = [n for n in predicted if 2 <= n['onset'] < 4]
    unobserved = sum(n['offset'] >= 6 for n in ref)
    return dict(pitched_onsets=match(ref, pred),
        complete_intervals=None if unobserved else match(ref, pred, offset=True),
        reference_tails_outside_context=unobserved,
        predicted_tails_at_context_end=sum(n['offset'] >= 6 for n in pred),
        all_context_predicted=len(predicted), central_predicted=len(pred))


def run(separation, output):
    import numpy as np
    from scipy.io import wavfile
    assert not output.exists(), 'Preserve earlier or partial experiments'
    assert sha(MANIFEST) == MANIFEST_SHA and sha(BASE / 'report.json') == BASE_REPORT_SHA
    separator = read(separation / 'report.json')
    assert [r['case']['name'] for r in separator['rows']] == [name for name, _ in CASES]
    assert separator['plan']['sample_rate'] == RATE and separator['plan']['window_seconds'] == 6
    source = ROOT / 'test-data/other-instrument-model-survey-2026-11-10/tsumugi'
    source_files = verify_source(source)
    assert sha(source / 'checkpoints' / ('best_model_' + MODEL + '.pth')) == MODELS[MODEL]
    adapter = ROOT / 'scripts/research_tsumugi_transcription.py'
    matcher = ROOT / 'scripts/evaluate_instrument_benchmark.py'
    plan = dict(separation_report_sha256=sha(separation / 'report.json'),
        model=MODEL, model_sha256=MODELS[MODEL], source_commit=SOURCE_COMMIT,
        source_files=source_files, adapter_sha256=sha(adapter), matcher_sha256=sha(matcher),
        runner_sha256=sha(__file__), manifest_sha256=MANIFEST_SHA,
        base_report_sha256=BASE_REPORT_SHA, scoring_seconds=[2, 4], context_seconds=[0, 6],
        physical_source_seconds=[20, 26], stem_filter='all', normalization=False,
        sample_rate=RATE, silence_gate_rms_dbfs=-72., cases=[name for name, _ in CASES],
        variants=['mixture', 'target', 'absent', 'silence'],
        models_parallel=False, onsets_tolerance_ms=50,
        interval_tolerance='max(50ms,20% reference duration)',
        pitch_tolerance_semitones=0, automatic_instrument_discovery=False)
    save(output / 'plan.json', plan)
    inputs = output / 'inputs'; inputs.mkdir()
    predictions = output / 'predictions'; predictions.mkdir()
    recordings, assignments = {}, {}
    for row in separator['rows']:
        name = row['case']['name']
        path = BASE / name / 'inputs.npz'
        assert sha(path) == row['case']['inputs_sha256']
        with np.load(path, allow_pickle=False) as arrays:
            for variant in plan['variants']:
                saved = next(v for v in row['variants'] if v['variant'] == variant)
                estimated_path = separation / name / (variant + '.npy')
                assert sha(estimated_path) == saved['estimate_sha256']
                for method, wave in [('original', arrays[variant]),
                                     ('separated', np.load(estimated_path, allow_pickle=False))]:
                    assert wave.shape == (2, RATE * 6) and wave.dtype == np.float32
                    assert np.isfinite(wave).all()
                    # Same physical samples and decoder settings may reuse an
                    # inference; no target class is passed to the transcriber.
                    pcm = np.ascontiguousarray(wave.T, dtype='<f4')
                    digest = hashlib.sha256(pcm.tobytes()).hexdigest()
                    assignments[(name, variant, method)] = digest
                    if digest not in recordings:
                        path = inputs / (digest + '.wav')
                        wavfile.write(path, RATE, pcm)
                        rate, restored = wavfile.read(path)
                        assert rate == RATE and restored.dtype == np.float32 and np.array_equal(pcm, restored)
                        recordings[digest] = dict(path=str(path), audio_sha256=sha(path), pcm_sha256=digest)
    save(output / 'inputs.json', dict(recordings=recordings,
        assignments=[dict(case=a, variant=b, method=c, pcm_sha256=d)
                     for (a, b, c), d in assignments.items()]))
    decoded = {}
    decoder_identity = None
    for index, (digest, recording) in enumerate(recordings.items(), 1):
        folder = predictions / digest
        print(json.dumps(dict(stage='transcribe', index=index, total=len(recordings), pcm_sha256=digest)), flush=True)
        with (predictions / (digest + '.log')).open('x', encoding='utf-8') as log:
            subprocess.run([str(ROOT / '.venv-audio/Scripts/python.exe'), '-B', str(adapter),
                '--audio', recording['path'], '--model', MODEL, '--stem', 'all',
                '--source', str(source), '--dependencies', str(ROOT / 'test-data/rosvot-runtime'),
                '--output', str(folder)], check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=os.environ | {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUTF8': '1'})
        report = read(folder / 'report.json'); identity = report['identity']
        expected = dict(audio_sha256=recording['audio_sha256'], pcm_sha256=digest,
            model=MODEL, model_sha256=MODELS[MODEL], source_commit=SOURCE_COMMIT,
            source_files=source_files, runner_sha256=plan['adapter_sha256'],
            original_sample_rate=RATE, original_samples=RATE * 6, original_channels=2,
            normalization=False, stem_filter='all', intra_threads=2, inter_threads=1)
        assert all(identity.get(k) == v for k, v in expected.items())
        assert report['context_start_seconds'] == 0 and report['context_end_seconds'] == 6
        decoder = {k: identity[k] for k in ('settings', 'model_config', 'versions')}
        if decoder_identity is None:
            decoder_identity = decoder
            save(output / 'decoder-identity.json', decoder_identity)
        assert decoder == decoder_identity, 'Decoder configuration changed between inputs'
        for name in ('notes', 'raw_notes', 'window_notes', 'interval_constructions'):
            assert sha(folder / (name.replace('_', '-') + '.json')) == report[name + '_sha256']
        decoded[digest] = read(folder / 'notes.json')
        assert len(decoded[digest]) == report['notes']

    # Read annotations only after all inference is complete; do not use them
    # for model routing, gains, output filtering or choosing a better model.
    manifest = read(MANIFEST)
    references, all_sources = defaultdict(list), defaultdict(list)
    for case in manifest['cases']:
        if case['kind'] != 'isolated':
            continue
        assert sha(case['labels_path']) == case['labels_sha256']
        piece = case['piece_id']
        notes = [n | dict(onset=n['onset'] - 20, offset=n['offset'] - 20)
                 for n in read(case['labels_path'])]
        references[(piece, case['instrument'])].extend(notes)
        all_sources[piece].extend(notes)
    rows = []
    for name, _ in CASES:
        piece = int(name.split('-')[0]); family = FAMILY[name]
        reference = references[(piece, family)]
        assert reference
        scores = {}
        for variant in plan['variants']:
            target = reference if variant in ('mixture', 'target') else []
            scores[variant] = {method: metrics(target, decoded[assignments[(name, variant, method)]])
                              for method in ('original', 'separated')}
        rows.append(dict(case=name, target_directed=scores))
    complete_pieces = []
    for piece in (12, 31):
        names = [name for name, _ in CASES if int(name.split('-')[0]) == piece]
        represented = {FAMILY[name] for name in names}
        actual = {c['instrument'] for c in manifest['cases'] if c['kind'] == 'isolated' and c['piece_id'] == piece}
        assert represented == actual
        mix_ids = {assignments[(name, 'mixture', 'original')] for name in names}
        assert len(mix_ids) == 1
        original = decoded[next(iter(mix_ids))]
        # Keep every separated prediction, including duplicates and leakage.
        combined = [n for name in names for n in decoded[assignments[(name, 'mixture', 'separated')]]]
        complete_pieces.append(dict(piece=piece, families=sorted(actual),
            original=metrics(all_sources[piece], original),
            separated_union=metrics(all_sources[piece], combined)))
    assert sha(adapter) == plan['adapter_sha256'] and sha(matcher) == plan['matcher_sha256']
    assert sha(__file__) == plan['runner_sha256']
    assert sha(separation / 'report.json') == plan['separation_report_sha256']
    save(output / 'report.json', dict(plan=plan, rows=rows, complete_pieces=complete_pieces,
        unique_inferences=len(recordings), assignments=len(assignments),
        decoder_identity_sha256=sha(output / 'decoder-identity.json'),
        musical_acceptance=False, default_changed=False,
        limitations=['Small public non-blind cohort, reused source performances.',
            'Six-second model context; central two-second onset scoring only.',
            'Target-directed mixture precision penalizes valid non-target notes; not overall transcription quality.',
            'Complete source coverage comparison is limited to pieces 12 and 31.',
            'Original and separated tails use the same raw decoder, not authored LN releases.',
            'No instrument-answer filtering or deduplication of separated predictions.']))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--separation', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    run(args.separation, args.output)
