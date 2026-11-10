"""Save raw piano windows and verify their owned samples against frozen inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.piano import MODEL_SHA256, OUTPUTS, RATE, WINDOW, HOP
from scripts.check_expression_current_replay import read, save, sha


def run(output, names, starts, reuse):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'test-data'):
        raise ValueError('Use a new ignored test-data output directory')
    benchmark = ROOT / 'test-data/smd/selected-v1/benchmark'
    manifest = read(benchmark / 'manifest.json')
    model = ROOT / 'assets/deep/piano-regression.onnx'
    assert sha(model) == MODEL_SHA256
    previous = read(ROOT / 'test-data/piano-inactivity-context-v1/execution-plan.json')
    cases = [dict(c) for c in manifest['cases'] if c['name'] in names]
    if 'sakura' in names:
        audio = 'test-data/sakura-benchmark/raw-pcm-audit-v1/piano.wav'
        activation = [p for p in previous['inputs'] if p.endswith('piano-specialist-activations.npz')]
        assert len(activation) == 1
        cases.append(dict(name='sakura', audio=audio, audio_sha256=previous['inputs'][audio],
            activation=activation[0], duration_seconds=None))
    assert {c['name'] for c in cases} == set(names)
    inputs = {str(model.relative_to(ROOT)): MODEL_SHA256,
              str((benchmark / 'manifest.json').relative_to(ROOT)): sha(benchmark / 'manifest.json')}
    for c in cases:
        path = ROOT / c['audio']
        assert sha(path) == c['audio_sha256']
        inputs[str(path.relative_to(ROOT))] = c['audio_sha256']
        cache = ROOT / c['activation'] if 'activation' in c else benchmark / 'specialist-v2' / (c['name'] + '-activations.npz')
        digest = previous['inputs'][cache.relative_to(ROOT).as_posix()]
        assert sha(cache) == digest
        inputs[str(cache.relative_to(ROOT))] = digest
    sources = {p: sha(ROOT / p) for p in ('scripts/check_piano_window_overlap.py', 'manialab/piano.py')}
    reusable = {}
    if reuse:
        reuse = reuse.resolve()
        old_plan = read(reuse / 'plan.json')
        recovered = read(reuse / 'recovery.json')
        assert len(old_plan['cases']) == 1 and recovered['both_owned_windows_verified']
        assert old_plan['inputs'][str(model.relative_to(ROOT))] == MODEL_SHA256
        for name, digest in old_plan['inputs'].items():
            assert sha(ROOT / name) == digest
        for row in recovered['windows']:
            path = reuse / old_plan['cases'][0] / f"{row['start']:04d}.npz"
            assert sha(path) == row['sha256']
            reusable[(old_plan['cases'][0], row['start'])] = path
    save(output / 'plan.json', dict(inputs=inputs, sources=sources, cases=names, starts_seconds=starts,
        threads=2, interop_threads=1, read_note_labels=False, thresholds_changed=False,
        reuse=str(reuse) if reuse else None,
        reused_hashes={str(p):sha(p) for p in reusable.values()}))
    for p in sources:
        target = output / 'source' / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, target)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    rows = []
    for case in cases:
        rate, raw = wavfile.read(ROOT / case['audio'])
        assert rate == 22050 and raw.ndim == 1
        assert raw.dtype == (np.float32 if case['name'] == 'sakura' else np.int16)
        if case['duration_seconds'] is not None:
            raw = raw[:round(case['duration_seconds'] * rate)]
        samples = raw.astype(np.float32) / (32768 if raw.dtype == np.int16 else 1)
        pcm = resample_poly(samples, 320, 441).astype(np.float32)
        count = int(np.ceil(len(pcm) / WINDOW)) * WINDOW
        all_starts = list(range(0, count - WINDOW + 1, HOP))
        selected = all_starts if starts is None else [round(s * RATE) for s in starts]
        assert set(selected) <= set(all_starts)
        folder = output / case['name']
        folder.mkdir()
        np.save(folder / 'pcm.npy', pcm, allow_pickle=False)
        cache_path = ROOT / case['activation'] if 'activation' in case else benchmark / 'specialist-v2' / (case['name'] + '-activations.npz')
        with np.load(cache_path, allow_pickle=False) as cache:
            baseline = {key: cache[key] for key in OUTPUTS}
        for start in selected:
            chunk = np.zeros(WINDOW, np.float32)
            available = pcm[start:start + WINDOW]
            chunk[:len(available)] = available
            begun = time.monotonic()
            prior = reusable.get((case['name'], start // RATE))
            if prior:
                np.testing.assert_array_equal(np.load(prior.parent / 'pcm.npy', allow_pickle=False), pcm)
                with np.load(prior, allow_pickle=False) as z:
                    result = {key:z[key][None] for key in OUTPUTS}
            else:
                result = dict(zip(OUTPUTS, session.run(list(OUTPUTS), {'audio': chunk[None]})))
            seconds = time.monotonic() - begun
            index = all_starts.index(start)
            left = 0 if index == 0 else 250
            right = 1001 if len(all_starts) == 1 else 1000 if index == len(all_starts) - 1 else 750
            global_start = start // 160 + left
            right = min(right, left + len(baseline[OUTPUTS[0]]) - global_start)
            errors = {}
            for key, value in result.items():
                assert value.shape == (1, 1001, 88) and np.isfinite(value).all()
                reference = baseline[key][global_start:global_start + right - left]
                delta = np.abs(value[0, left:right] - reference)
                errors[key] = float(delta.max(initial=0))
                np.testing.assert_allclose(value[0, left:right], reference, rtol=1e-3, atol=1e-4)
            path = folder / f'{start // RATE:04d}.npz'
            np.savez_compressed(path, **{k:v[0] for k,v in result.items()})
            row = dict(name=case['name'], start_seconds=start / RATE,
                start_sample=start, pcm_sha256=sha(folder / 'pcm.npy'), path=str(path.relative_to(output)),
                sha256=sha(path), elapsed_seconds=seconds, owned_global_frames=[global_start, global_start + right - left],
                owned_max_absolute_error=errors, reused_from=str(prior) if prior else None)
            rows.append(row)
            save(folder / f'{start // RATE:04d}.json', row)
            print(f"{case['name']} {start / RATE:g}s: preserved raw window, verified owned frames", flush=True)
    assert all(sha(ROOT / p) == h for p, h in inputs.items())
    assert all(sha(ROOT / p) == h for p, h in sources.items())
    save(output / 'report.json', dict(plan_sha256=sha(output / 'plan.json'), windows=rows,
        read_note_labels=False, musical_acceptance=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--names', nargs='+', required=True)
    cli.add_argument('--starts', nargs='+', type=float)
    cli.add_argument('--reuse', type=Path)
    args = cli.parse_args()
    run(args.output, args.names, args.starts, args.reuse)
