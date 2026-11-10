"""Replay all 60 complete guitar recordings in the application ONNX runtime."""
from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_hf_instrument as adapter
from scripts import evaluate_guitar_vocal_benchmark as evaluation
from scripts.check_hf_guitar_specialists import MODELS
from manialab.piano import decode


def compare_activations(actual, expected):
    """Unchanged tolerance, with complete evidence even for unused context."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    adapter.require(actual.shape == expected.shape and actual.size
                    and np.isfinite(actual).all() and np.isfinite(expected).all(), 'Invalid activation comparison')
    delta = abs(actual - expected)
    violations = delta > .0001 + .001 * abs(expected)
    return dict(preserved=not bool(violations.any()), maximum_absolute_error=float(delta.max()),
        violations=int(violations.sum()), total=delta.size,
        worst_index=list(map(int, np.unravel_index(delta.argmax(), delta.shape))))


def compare_events(actual, expected):
    adapter.require(len(actual) == len(expected), 'Runtime changed event count')
    maximum = 0.
    maximum_score = 0.
    for a, b in zip(actual, expected):
        adapter.require(all(a[k] == b[k] for k in ('pitch', 'velocity', 'release_reason', 'release_observed')),
                        'Runtime changed pitch, velocity or release provenance')
        delta = max(abs(a[k]-b[k]) for k in ('onset', 'offset'))
        adapter.require(delta <= .001, 'Runtime moved an endpoint by more than 1 ms')
        maximum = max(maximum, delta)
        maximum_score = max(maximum_score, abs(a['onset_score']-b['onset_score']))
    return dict(maximum_endpoint_error_ms=maximum*1000, maximum_onset_score_error=maximum_score)


def main():
    import hashlib
    import json
    import onnxruntime as ort
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--export', dest='export_root', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    require, sha, verify, read = adapter.require, adapter.sha, adapter.verified_hash, adapter.read_json
    require(not args.output.exists(), 'Preserve prior verification')
    export = read(args.export_root/'report.json')
    for filename, key in (('guitar.onnx', 'graph_sha256'), ('input.npy', 'input_sha256'),
                          ('torch-reference.npz', 'reference_sha256')):
        verify(args.export_root/filename, export[key])
    source_path = Path(export['source_report'])
    verify(source_path, export['source_report_sha256'])
    source = read(source_path)
    names = [case['name'] for case in source['plan']['cases']]
    require(len(names) == len(set(names)) == 60 and source['plan']['model_order'] == list(MODELS)
            and [(row['model'], row['name']) for row in source['rows']]
            == [(model, name) for model in MODELS for name in names], 'Incomplete original cohort')
    for path, digest in source['plan']['code_sha256'].items():
        verify(path, digest)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.export_root/'guitar.onnx'), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    plan = dict(export_report_sha256=sha(args.export_root/'report.json'),
        source_report_sha256=sha(source_path), graph_sha256=export['graph_sha256'],
        model=export['model'], runner_sha256=sha(__file__), decoder_sha256=sha(ROOT/'manialab/piano.py'),
        ort=ort.__version__, numpy=np.__version__, device='CPUExecutionProvider', intra_threads=2, inter_threads=1,
        activation_rtol=.001, activation_atol=.0001, endpoint_tolerance_seconds=.001,
        require_exact_event_inventory_and_provenance=True,
        failure_policy='retain_all_finite_results_and_complete_cohort_before_rejecting',
        activation_scopes=['all_windows_including_unused_context', 'consumed_physical_frames'],
        cases=[c['name'] for c in source['plan']['cases']])
    args.output.mkdir(parents=True)
    adapter.write_new(args.output/'plan.json', plan)
    rows = []
    for index, case in enumerate(source['plan']['cases']):
        started = time.monotonic()
        original = next(r for r in source['rows'] if r['model'] == export['model'] and r['name'] == case['name'])
        folder = source_path.parent/export['model']/case['name']
        verify(folder/'report.json', original['report_sha256'])
        report = read(folder/'report.json')
        for filename, key in (('notes.json', 'notes_sha256'), ('windows.npz', 'windows_sha256')):
            verify(folder/filename, report[key])
        verify(case['audio'], case['audio_sha256'])
        rate, integer = wavfile.read(case['audio'])
        require(rate == case['sample_rate'] and len(integer) == case['samples']
                and integer.dtype == np.int16 and integer.ndim == 1, 'Original PCM clock differs')
        common = gcd(rate, adapter.RATE)
        pcm = resample_poly(integer.astype(np.float32)/32768, adapter.RATE//common, rate//common).astype(np.float32)
        require(hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() == report['input']['model_input_sha256'],
                'Model input differs between runtimes')
        pieces = {key: [] for key in adapter.OUTPUTS}
        with np.load(folder/'windows.npz', allow_pickle=False) as expected:
            starts = expected['starts']
            expected_windows = {key: expected[key] for key in adapter.OUTPUTS}
            padded = ((len(pcm)+adapter.WINDOW-1)//adapter.WINDOW)*adapter.WINDOW
            require(starts.tolist() == list(range(0, padded-adapter.WINDOW+1, adapter.HOP)), 'Window clock mismatch')
            for j, start in enumerate(starts):
                chunk = np.zeros(adapter.WINDOW, dtype=np.float32)
                available = pcm[int(start):int(start)+adapter.WINDOW]
                chunk[:len(available)] = available
                values = session.run(list(adapter.OUTPUTS), {'audio': chunk[None]})
                for key, value in zip(adapter.OUTPUTS, values):
                    pieces[key].append(value[0])
        windows = {key: np.stack(value) for key, value in pieces.items()}
        window_comparisons = {key: compare_activations(windows[key], expected_windows[key]) for key in adapter.OUTPUTS}
        heads = adapter.stitch_windows(windows)
        physical, times, valid = adapter.physical_heads(heads, case['samples'], rate)
        expected_physical, _, _ = adapter.physical_heads(adapter.stitch_windows(expected_windows), case['samples'], rate)
        physical_comparisons = {key: compare_activations(physical[key], expected_physical[key]) for key in adapter.OUTPUTS}
        events = [dict(n, is_drum=False, program=-1, voice=export['model'])
                  for n in decode(physical, **adapter.THRESHOLDS)]
        try:
            comparison = dict(events_preserved=True, **compare_events(events, read(folder/'notes.json')))
        except ValueError as exc:
            comparison = dict(events_preserved=False, event_failure=str(exc))
        reference = case['references']['strings']
        verify(reference['path'], reference['sha256'])
        metrics, scope = evaluation.window_metrics(read(reference['path']), events, begin=0,
            end=case['seconds'], context_start=0, context_end=case['duration_seconds'])
        count_changes = []
        for tolerance in evaluation.TOLERANCES:
            for metric in metrics[tolerance]:
                for key in ('reference', 'predicted', 'matched', 'f1'):
                    a, b = metrics[tolerance][metric].get(key), original['metrics'][tolerance][metric].get(key)
                    if a != b:
                        count_changes.append(dict(tolerance=tolerance, metric=metric, key=key, before=b, after=a))
        out = args.output/case['name']
        out.mkdir()
        np.savez_compressed(out/'windows.npz', starts=starts, **windows)
        np.savez_compressed(out/'activations.npz', times=times, valid_frame=valid, **heads)
        adapter.write_new(out/'notes.json', events)
        row = dict(name=case['name'], events=len(events), windows=len(starts),
            maximum_absolute_errors={key: value['maximum_absolute_error'] for key, value in window_comparisons.items()},
            window_comparisons=window_comparisons, physical_comparisons=physical_comparisons, **comparison,
            windows_sha256=sha(out/'windows.npz'), notes_sha256=sha(out/'notes.json'),
            activations_sha256=sha(out/'activations.npz'), metrics=metrics, interval_scope=scope,
            original_metrics=original['metrics'], scoring_count_changes=count_changes,
            interval_scope_unchanged=scope == original['interval_scope'],
            physical_frames=int(valid.sum()), seconds=time.monotonic()-started)
        adapter.write_new(out/'report.json', row)
        rows.append(row)
        print(json.dumps(dict(index=index+1, total=60, name=case['name'], events=len(events),
            **comparison, seconds=row['seconds'], scoring_count_changes=count_changes,
            window_activations_preserved=all(v['preserved'] for v in window_comparisons.values()),
            physical_activations_preserved=all(v['preserved'] for v in physical_comparisons.values()))), flush=True)
    require('torch' not in sys.modules, 'Application verification loaded Torch')
    verify(__file__, plan['runner_sha256'])
    verify(ROOT/'manialab/piano.py', plan['decoder_sha256'])
    result = dict(plan=plan, rows=rows, completed_cases=len(rows),
        all_window_activations_preserved=all(v['preserved'] for row in rows for v in row['window_comparisons'].values()),
        all_physical_activations_preserved=all(v['preserved'] for row in rows for v in row['physical_comparisons'].values()),
        all_physical_events_preserved=all(row['events_preserved'] for row in rows),
        seconds=sum(row['seconds'] for row in rows),
        all_scoring_counts_and_scopes_preserved=all(not row['scoring_count_changes']
            and row['interval_scope_unchanged'] for row in rows),
        enabled_in_application=False, musical_acceptance=False,
        limitation='Original GuitarSet clock only; Nova float stems and application selection still require checks.')
    result['original_all_window_protocol_passed'] = all(result[key] for key in (
        'all_window_activations_preserved', 'all_physical_activations_preserved',
        'all_physical_events_preserved', 'all_scoring_counts_and_scopes_preserved'))
    adapter.write_new(args.output/'report.json', result)
    require(result['original_all_window_protocol_passed'], 'Full evidence saved; original portability protocol did not pass')


if __name__ == '__main__':
    main()
