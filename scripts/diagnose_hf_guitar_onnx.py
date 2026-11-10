"""Investigate a failed ONNX migration on the fixed first complete recording.

This diagnostic cannot grant portability acceptance or change tolerances.
Both optimization modes retain every window, including discarded context.
"""
from __future__ import annotations

import argparse
import hashlib
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
from scripts.verify_hf_guitar_onnx import compare_events
from manialab.piano import decode


def difference(actual, expected):
    adapter.require(actual.shape == expected.shape and np.isfinite(actual).all()
                    and np.isfinite(expected).all(), 'Invalid output comparison')
    delta = abs(actual - expected)
    violations = delta > .0001 + .001 * abs(expected)
    return dict(maximum_absolute_error=float(delta.max()), violations=int(violations.sum()),
        total=delta.size, worst_index=list(map(int, np.unravel_index(delta.argmax(), delta.shape))),
        first_violation=list(map(int, np.argwhere(violations)[0])) if violations.any() else None)


def main():
    import onnxruntime as ort
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--export', dest='export_root', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    require, read, sha, verify = adapter.require, adapter.read_json, adapter.sha, adapter.verified_hash
    require(not args.output.exists(), 'Preserve previous diagnosis')
    export = read(args.export_root / 'report.json')
    verify(args.export_root / 'guitar.onnx', export['graph_sha256'])
    source_path = Path(export['source_report'])
    verify(source_path, export['source_report_sha256'])
    source = read(source_path)
    case = source['plan']['cases'][0]
    row = next(r for r in source['rows'] if r['name'] == case['name'] and r['model'] == export['model'])
    folder = source_path.parent / export['model'] / case['name']
    verify(folder / 'report.json', row['report_sha256'])
    original = read(folder / 'report.json')
    for name, key in (('windows.npz', 'windows_sha256'), ('notes.json', 'notes_sha256')):
        verify(folder / name, original[key])
    verify(case['audio'], case['audio_sha256'])
    rate, integer = wavfile.read(case['audio'])
    require(rate == case['sample_rate'] and len(integer) == case['samples']
            and integer.dtype == np.int16 and integer.ndim == 1, 'Original PCM differs')
    common = gcd(rate, adapter.RATE)
    pcm = resample_poly(integer.astype(np.float32) / 32768, adapter.RATE // common, rate // common).astype(np.float32)
    require(hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() == original['input']['model_input_sha256'],
            'Model input differs')
    with np.load(folder / 'windows.npz', allow_pickle=False) as saved:
        starts = saved['starts'].copy()
        expected_windows = {key: saved[key].copy() for key in adapter.OUTPUTS}
    expected_heads, _, _ = adapter.physical_heads(adapter.stitch_windows(expected_windows), len(integer), rate)
    args.output.mkdir(parents=True)
    plan = dict(case=case['name'], source_report_sha256=sha(source_path),
        graph_sha256=export['graph_sha256'], runner_sha256=sha(__file__),
        window_starts=starts.tolist(), ort=ort.__version__, scope='diagnostic_first_complete_recording',
        unchanged_rtol=.001, unchanged_atol=.0001, no_acceptance=True)
    adapter.write_new(args.output / 'plan.json', plan)
    rows = []
    for mode, level in (('all', ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                        ('disabled', ort.GraphOptimizationLevel.ORT_DISABLE_ALL)):
        started = time.monotonic()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = level
        session = ort.InferenceSession(str(args.export_root / 'guitar.onnx'), sess_options=options,
            providers=['CPUExecutionProvider'])
        pieces = {key: [] for key in adapter.OUTPUTS}
        for start in starts:
            chunk = np.zeros(adapter.WINDOW, dtype=np.float32)
            available = pcm[int(start):int(start) + adapter.WINDOW]
            chunk[:len(available)] = available
            values = session.run(list(adapter.OUTPUTS), {'audio': chunk[None]})
            for key, value in zip(adapter.OUTPUTS, values):
                pieces[key].append(value[0])
        del session
        windows = {key: np.stack(value) for key, value in pieces.items()}
        physical, times, valid = adapter.physical_heads(adapter.stitch_windows(windows), len(integer), rate)
        events = decode(physical, **adapter.THRESHOLDS)
        try:
            event_result = dict(preserved=True, **compare_events(events, read(folder / 'notes.json')))
        except ValueError as exc:
            event_result = dict(preserved=False, error=str(exc))
        out = args.output / mode
        out.mkdir()
        np.savez_compressed(out / 'windows.npz', starts=starts, **windows)
        np.savez_compressed(out / 'physical.npz', times=times[valid], **physical)
        adapter.write_new(out / 'notes.json', events)
        result = dict(mode=mode, events=event_result, count=len(events),
            window_errors={key: difference(windows[key], expected_windows[key]) for key in adapter.OUTPUTS},
            physical_errors={key: difference(physical[key], expected_heads[key]) for key in adapter.OUTPUTS},
            windows_sha256=sha(out / 'windows.npz'), physical_sha256=sha(out / 'physical.npz'),
            notes_sha256=sha(out / 'notes.json'), seconds=time.monotonic()-started)
        adapter.write_new(out / 'report.json', result)
        rows.append(result)
        import json
        print(json.dumps(result), flush=True)
    adapter.write_new(args.output / 'report.json', dict(plan=plan, rows=rows,
        portable_acceptance=False, musical_acceptance=False))


if __name__ == '__main__':
    main()
