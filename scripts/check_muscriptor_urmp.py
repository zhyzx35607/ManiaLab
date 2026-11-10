"""Compare recovered MuScriptor small against the frozen full URMP baseline."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_mega53_separation import save, sha
from scripts.evaluate_instrument_benchmark import match
from scripts.check_harmonica_transcription import BASELINE, BASELINE_SHA, MANIFEST, MANIFEST_SHA


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(reference, predicted, end):
    assert end >= 30
    for items in (reference, predicted):
        assert all(all(math.isfinite(n[k]) for k in ('onset', 'offset', 'pitch', 'program')) for n in items)
    ref = [n for n in reference if 0 <= n['onset'] < 30]
    pred = [n for n in predicted if 0 <= n['onset'] < 30]
    assert all(n['offset'] > n['onset'] and n['offset'] < end for n in ref)
    valid = [n for n in pred if n['offset'] > n['onset']]
    result = {}
    for strict in (False, True):
        onset = match(ref, pred, strict=strict)
        interval = match(ref, valid, strict=strict, offset=True)
        # Invalid durations remain failures in the original denominator. They
        # must not accidentally match a short reference under an offset tolerance.
        interval.update(predicted=len(pred), precision=interval['matched']/len(pred) if pred else 0,
                        f1=2*interval['matched']/max(1, len(ref)+len(pred)))
        result['strict' if strict else 'standard'] = dict(pitched_onsets=onset, complete_intervals=interval)
    return dict(metrics=result, invalid_intervals=len(pred)-len(valid),
                drum_predictions=sum(n['is_drum'] for n in pred),
                predicted_tails_at_or_after_context=sum(n['offset'] >= end for n in pred))


def run(weights, output):
    from scipy.io import wavfile
    import numpy as np
    assert not output.exists(), 'Preserve previous complete or partial evidence'
    assert sha(MANIFEST) == MANIFEST_SHA and sha(BASELINE/'comparison.json') == BASELINE_SHA
    manifest, baseline = read(MANIFEST), read(BASELINE/'comparison.json')
    assert len(manifest['cases']) == len(baseline['rows']) == 35
    adapter = ROOT/'scripts/research_muscriptor_transcription.py'
    plan = dict(runner_sha256=sha(__file__), adapter_sha256=sha(adapter),
        matcher_sha256=sha(ROOT/'scripts/evaluate_instrument_benchmark.py'),
        baseline_sha256=BASELINE_SHA, manifest_sha256=MANIFEST_SHA,
        weights_sha256=sha(weights), config_sha256=sha(weights.parent/'config.json'),
        model='small GGUF F16 conversion run with original PyTorch CPU code',
        scoring_seconds=[0, 30], context='unchanged frozen clips up to 40 seconds',
        normalization=False, sampling=False, prelude_forcing=True,
        models_parallel=False, instrument_filter=None,
        full_interval_policy='invalid predicted duration cannot match; retain original denominator',
        annotation_access='after all 35 real inputs and silence complete',
        cases=[c['name'] for c in manifest['cases']], enabled_in_application=False)
    save(output/'plan.json', plan)
    silence = output/'silence.wav'
    wavfile.write(silence, 16000, np.zeros(40*16000, dtype=np.float32))
    cases = [dict(name='silence-control', audio=str(silence), audio_sha256=sha(silence), samples=40*16000)]
    for case, row in zip(manifest['cases'], baseline['rows']):
        assert case['name'] == row['name']
        path = BASELINE/(case['name']+'.wav')
        assert sha(path) == row['clip_sha256']
        cases.append(dict(name=case['name'], audio=str(path), audio_sha256=sha(path), samples=row['clip_samples']))
    save(output/'inputs.json', cases)
    reports = {}
    for index, case in enumerate(cases):
        print(json.dumps(dict(stage='transcribe', index=index, total=35, case=case['name'])), flush=True)
        folder = output/case['name']
        with (output/(case['name']+'.log')).open('x', encoding='utf-8') as log:
            subprocess.run([str(ROOT/'.venv-audio/Scripts/python.exe'), '-B', str(adapter),
                '--audio', case['audio'], '--weights', str(weights), '--output', str(folder)],
                check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=os.environ | {'PYTHONUTF8':'1', 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2',
                                  'HF_HUB_OFFLINE':'1'})
        report = read(folder/'report.json'); identity = report['identity']
        expected = dict(audio_sha256=case['audio_sha256'], samples=case['samples'], sample_rate=16000,
                        runner_sha256=plan['adapter_sha256'], weights_sha256=plan['weights_sha256'],
                        recovered_config_sha256=plan['config_sha256'], device='cpu', intra_threads=2,
                        inter_threads=1, instruments=None, normalization=False, prelude_forcing=True,
                        source_code_unmodified=True, strict_load=True)
        assert all(identity[k] == v for k,v in expected.items())
        assert report['context_end_seconds'] == case['samples']/16000
        for name, digest in report['artifacts'].items():
            assert sha(folder/name) == digest
        assert sha(case['audio']) == case['audio_sha256']
        reports[case['name']] = report
    save(output/'inference-completed.json', dict(
        reports={c['name']:sha(output/c['name']/'report.json') for c in cases},
        seconds=sum(r['seconds'] for r in reports.values()), labels_read=False))
    rows = []
    groups = defaultdict(lambda: defaultdict(lambda: dict(reference=0, predicted=0, matched=0)))
    for case, prior in zip(manifest['cases'], baseline['rows']):
        assert sha(case['labels_path']) == case['labels_sha256']
        predicted = read(output/case['name']/'notes.json')
        result = metrics(read(case['labels_path']), predicted, reports[case['name']]['context_end_seconds'])
        rows.append(dict(name=case['name'], kind=case['kind'], instrument=case.get('instrument'),
                         **result, tsumugi=prior['metrics']))
        for engine, values in [('muscriptor', result['metrics']), ('tsumugi', prior['metrics'])]:
            for label, values in values['standard'].items():
                if 'instrument' in label:
                    continue
                for field in ('reference', 'predicted', 'matched'):
                    groups[case['kind']][engine+'/'+label][field] += values[field]
    for group in groups.values():
        for counts in group.values():
            counts['f1'] = 2*counts['matched']/max(1, counts['reference']+counts['predicted'])
    assert sha(__file__) == plan['runner_sha256'] and sha(adapter) == plan['adapter_sha256']
    assert sha(ROOT/'scripts/evaluate_instrument_benchmark.py') == plan['matcher_sha256']
    save(output/'report.json', dict(plan=plan, rows=rows, groups=groups,
        silence_notes=reports['silence-control']['notes'],
        missing_eos=sum(r['missing_eos'] for r in reports.values()),
        seconds=sum(r['seconds'] for r in reports.values()),
        inference_completed_sha256=sha(output/'inference-completed.json'),
        musical_acceptance=False, default_changed=False,
        limitations=['Public reused cohort; not an unseen blind test.',
            'Original Python implementation with third-party F16-converted weights, not original FP32 parity.',
            'Predicted programs preserved but not used for routing, filtering or grading instruments.',
            'Raw event endpoints retained; no beat alignment, MIDI cleanup or LN gestures.']))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--weights', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    run(args.weights, args.output)
