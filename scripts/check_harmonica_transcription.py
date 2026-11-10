"""Research a pinned third-party Harmonica reproduction on all 35 URMP inputs.

This is not the paper authors' code or weights. No application integration,
automatic instrument labels, threshold tuning or authored LN claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_mega53_separation import save, sha
from scripts.evaluate_instrument_benchmark import window_metrics

ASSETS = ROOT / 'test-data/harmonica-source-review-v1'
SOURCE = ASSETS / 'source'
COMMIT = '6ad8feef7a540baf3a384712ae4b44abe4b7a734'
WEIGHT = SOURCE / 'src/harmonica/weights/harmonica-medium.pt'
WEIGHT_SHA = 'ecd76a690d6b6e4bbd46d8ddb97e1d03c6e7a413ef9eeafdaa51b0b169cd586c'
BASELINE = ROOT / 'test-data/tsumugi-urmp-comparison-v1'
BASELINE_SHA = 'ff98a2e3c53671c1bcf4ec16a6c3cdff3215e17e239101dfe6883f884fbe2e75'
MANIFEST = ROOT / 'test-data/urmp/comparison-extension/manifest.json'
MANIFEST_SHA = 'e363a4e7b5f25628909eb13031251847492d9fa816e520b98fbc6d612bcda16a'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(precision='upstream-bfloat16'):
    assert precision in ('upstream-bfloat16', 'float32')
    assert sha(WEIGHT) == WEIGHT_SHA
    assert sha(BASELINE / 'comparison.json') == BASELINE_SHA and sha(MANIFEST) == MANIFEST_SHA
    git = lambda *args: subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()
    assert git('rev-parse', 'HEAD') == COMMIT and not git('diff', 'HEAD', '--name-only')
    baseline, manifest = read(BASELINE / 'comparison.json'), read(MANIFEST)
    assert len(baseline['rows']) == len(manifest['cases']) == 35
    assert [r['name'] for r in baseline['rows']] == [c['name'] for c in manifest['cases']]
    cases = []
    for case, row in zip(manifest['cases'], baseline['rows']):
        clip = BASELINE / (case['name'] + '.wav')
        assert sha(clip) == row['clip_sha256']
        cases.append(dict(name=case['name'], kind=case['kind'], piece_id=case['piece_id'],
                          instrument=case.get('instrument'), audio=str(clip),
                          audio_sha256=sha(clip), samples=row['clip_samples']))
    return dict(source_commit=COMMIT, independent_reproduction=True, paper_authors_implementation=False,
        source_files={str(p.relative_to(SOURCE)).replace('\\', '/'): sha(p)
                      for p in (SOURCE / 'src/harmonica').rglob('*.py')},
        weight_sha256=WEIGHT_SHA, weight_bytes=WEIGHT.stat().st_size, model='medium',
        baseline_sha256=BASELINE_SHA, manifest_sha256=MANIFEST_SHA, cases=cases,
        runner_sha256=sha(__file__), matcher_sha256=sha(ROOT / 'scripts/evaluate_instrument_benchmark.py'),
        context='original frozen clips up to 40 seconds', scoring_seconds=[0, 30],
        model_chunk_seconds=5., model_batch_size=8, thresholds='checkpoint defaults; no sweep',
        normalization=False, sample_rate=16000, thread_counts={'intra': 2, 'inter': 1},
        precision=precision,
        amp=('upstream CPU bfloat16 autocast; CQT float32' if precision == 'upstream-bfloat16'
             else 'disable CPU autocast only inside forward_spectrogram; CQT float32'),
        compatibility_wrapper=precision == 'float32', models_parallel=False,
        source_license='No license file or pyproject license field found; research asset only.',
        notes_or_chart_labels_read=False, enabled_in_application=False)


def explain_notes(probabilities, notes, fps, on_threshold, frame_threshold, physical_end):
    """Reconstruct endpoint causes without changing upstream notes or timing."""
    import numpy as np
    on, sustained = probabilities['onset'], probabilities['sustain']
    assert on.shape == sustained.shape and on.shape[0] == 88
    frames = on.shape[1]
    evidence = []
    for row in range(88):
        starts = [t for t in range(frames) if on[row, t] >= on_threshold
                  and on[row, t] >= (on[row, t-1] if t else -1.)
                  and on[row, t] > (on[row, t+1] if t+1 < frames else -1.)]
        for index, start in enumerate(starts):
            future = np.flatnonzero(sustained[row, start+1:] < frame_threshold)
            decay = start+1+int(future[0]) if len(future) else frames
            repeat = starts[index+1] if index+1 < len(starts) else frames
            end = max(start+1, min(decay, repeat))
            causes = []
            if end == decay and decay < frames:
                causes.append('sustain_threshold')
            if end == repeat and repeat < frames:
                causes.append('same_pitch_reattack')
            if end == frames:
                causes.append('model_context_end')
            if end/fps >= physical_end:
                causes.append('at_or_after_audio_end')
            evidence.append(dict(onset=start/fps, offset=end/fps, pitch=21+row,
                onset_frame=start, offset_frame=end, endpoint_causes=causes,
                input_start_peak=start == 0, outside_audio_onset=start/fps >= physical_end,
                offset_head_used=False, authored_ln_verified=False))
    evidence.sort(key=lambda n: (n['onset'], n['offset'], n['pitch']))
    actual = [(float(a), float(b), int(c)) for a, b, c in zip(notes.onsets, notes.offsets, notes.pitches)]
    assert actual == [(n['onset'], n['offset'], n['pitch']) for n in evidence]
    return evidence


def decoder_controls(decode):
    import numpy as np
    rows = []
    for name in ('silence', 'decay', 'reattack', 'context_end', 'last_frame', 'flat_peak'):
        onset = np.zeros((88, 16), dtype=np.float32)
        sustain = np.zeros_like(onset)
        if name != 'silence':
            start = 15 if name == 'last_frame' else 2
            onset[40, start] = .9
            sustain[40, start:] = .9
        if name == 'decay':
            sustain[40, 9:] = 0
        elif name == 'reattack':
            onset[40, 10] = .9
        elif name == 'flat_peak':
            onset[40, 3:5] = .9
        probabilities = dict(onset=onset, sustain=sustain)
        notes = decode(onset, sustain, 31.25, .5, .5)
        details = explain_notes(probabilities, notes, 31.25, .5, .5, .5)
        rows.append(dict(control=name, details=details))
    assert not rows[0]['details']
    assert rows[1]['details'][0]['endpoint_causes'] == ['sustain_threshold']
    assert 'same_pitch_reattack' in rows[2]['details'][0]['endpoint_causes']
    assert 'model_context_end' in rows[3]['details'][0]['endpoint_causes']
    assert rows[4]['details'][0]['onset_frame'] == 15
    assert rows[5]['details'][0]['onset_frame'] == 4
    return rows


def configure_precision(model, precision, torch):
    """Keep upstream windowing/decoding; opt out of CPU AMP explicitly."""
    if precision == 'upstream-bfloat16':
        return
    assert precision == 'float32'
    original = model.forward_spectrogram
    def forward_float32(batch):
        assert batch.dtype == torch.float32 and batch.device.type == 'cpu'
        with torch.autocast(device_type='cpu', enabled=False):
            result = original(batch)
        assert all(value.dtype == torch.float32 for value in result.values())
        return result
    model.forward_spectrogram = forward_float32


def run(output, precision='upstream-bfloat16'):
    import numpy as np
    from scipy.io import wavfile
    assert not output.exists(), 'Preserve previous complete or partial evidence'
    plan = prepare(precision)
    save(output / 'plan.json', plan)
    for folder in (ROOT / 'test-data/rosvot-runtime', ASSETS / 'deps', SOURCE / 'src'):
        sys.path.insert(0, str(folder))
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(42)
    from harmonica.inference import model_from_checkpoint, transcribe_audio
    from harmonica.decoding import decode
    from harmonica.corpus import URMP_TEST_PIECES
    # Do not use upstream read_checkpoint(), which requests unrestricted pickle.
    stored = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    assert stored['pipeline'] == 'harmonica' and stored['checkpoint_format_version'] == 2
    model, config, step = model_from_checkpoint(stored, WEIGHT, torch.device('cpu'))
    assert config.model.cqt.sample_rate == 16000 and config.model.cqt.hop_length == 512
    for key, tensor in stored['model'].items():
        assert torch.equal(tensor, model.state_dict()[key.removeprefix('_orig_mod.')])
    configure_precision(model, precision, torch)
    thresholds = stored['thresholds']
    assert set(thresholds) == {'onset', 'frame'} and all(0 < v < 1 for v in thresholds.values())
    identity = dict(config=stored['config'], thresholds=thresholds, step=step,
        parameters=sum(p.numel() for p in model.parameters()), safe_tensor_load=True,
        precision=precision, compatibility_wrapper=precision == 'float32',
        model_tensors_exact=True, cqt_kernels='reconstructed from stored config using nnAudio0.3.4',
        declared_urmp_test_pieces=sorted(URMP_TEST_PIECES),
        all_source_pieces_declared_test=all(c['piece_id'] in URMP_TEST_PIECES for c in plan['cases']),
        training_manifest_verified=False,
        versions={p: importlib.metadata.version(p) for p in
                  ('torch', 'nnAudio', 'loguru', 'soundfile', 'numpy', 'scipy', 'omegaconf')})
    save(output / 'model-identity.json', identity)
    save(output / 'decoder-controls.json', decoder_controls(decode))
    captures = defaultdict(list)
    captured_dtypes = defaultdict(list)
    handles = []
    for name, head in model.heads.items():
        def capture(module, args, result, label=name):
            assert torch.isfinite(result).all()
            captured_dtypes[label].append(str(result.dtype))
            captures[label].append(result.detach().float().cpu().clone())
        handles.append(head.register_forward_hook(capture))
    fps = 16000/512
    chunk = 4*max(1, int(round(5.*fps/4)))
    keep, margin = chunk//2, chunk//4
    inference_rows = []
    recordings = [dict(name='silence-control', kind='control', audio=None, samples=16000*40)] + plan['cases']
    for index, case in enumerate(recordings):
        folder = output / case['name']; folder.mkdir()
        if case['audio'] is None:
            pcm = np.zeros(case['samples'], dtype=np.float32)
        else:
            assert sha(case['audio']) == case['audio_sha256']
            rate, integer = wavfile.read(case['audio'])
            assert rate == 16000 and integer.dtype == np.int16 and integer.ndim == 1
            assert len(integer) == case['samples']
            pcm = integer.astype(np.float32)/32768.
        captures.clear()
        captured_dtypes.clear()
        started = time.monotonic()
        probabilities = transcribe_audio(model, pcm, chunk_seconds=5., batch_size=8, device=torch.device('cpu'))
        elapsed = time.monotonic()-started
        frames = len(pcm)//512+1
        assert set(captures) == {'onset', 'sustain', 'offset', 'velocity'}
        if precision == 'float32':
            assert all(dtype == 'torch.float32' for dtypes in captured_dtypes.values() for dtype in dtypes)
        raw = {}
        for name, batches in captures.items():
            assert sum(len(b) for b in batches) == (frames+keep-1)//keep
            assert all(b.shape[1:] == (88, chunk) for b in batches)
            for i, values in enumerate(batches):
                path = folder / f'logits-{name}-{i}.npy'
                np.save(path, values.numpy(), allow_pickle=False)
                raw[path.name] = sha(path)
            middle = torch.cat([torch.sigmoid(b[..., margin:margin+keep]) for b in batches])
            joined = middle.permute(1, 0, 2).reshape(88, -1)[:, :frames].numpy()
            if name in probabilities:
                assert np.array_equal(probabilities[name], joined)
            else:
                probabilities[name] = joined
        assert all(v.shape == (88, frames) and np.isfinite(v).all() for v in probabilities.values())
        np.savez_compressed(folder / 'activations.npz', **probabilities)
        notes = decode(probabilities['onset'], probabilities['sustain'], fps,
                       thresholds['onset'], thresholds['frame'], probabilities['velocity'])
        evidence = explain_notes(probabilities, notes, fps, thresholds['onset'], thresholds['frame'], len(pcm)/16000)
        predicted = [dict(onset=float(a), offset=float(b), pitch=int(c), velocity=int(v),
                          program=0, instrument_label_known=False, is_drum=False)
                     for a, b, c, v in zip(notes.onsets, notes.offsets, notes.pitches, notes.velocities)]
        save(folder / 'notes.json', predicted)
        save(folder / 'endpoint-evidence.json', evidence)
        receipt = dict(case=case, seconds=elapsed, frames=frames, notes=len(predicted),
            precision=precision, logits_dtypes=dict(captured_dtypes),
            neural_batches=len(captures['onset']), input_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
            raw_logits=raw, activations_sha256=sha(folder / 'activations.npz'),
            notes_sha256=sha(folder / 'notes.json'), endpoint_evidence_sha256=sha(folder / 'endpoint-evidence.json'),
            endpoint_causes=dict(Counter(c for n in evidence for c in n['endpoint_causes'])),
            outside_audio_onsets=sum(n['outside_audio_onset'] for n in evidence),
            offset_head_used_for_decoding=False)
        save(folder / 'report.json', receipt)
        inference_rows.append(receipt)
        print(json.dumps(dict(index=index, total=35, case=case['name'], seconds=elapsed,
                              notes=len(predicted), batches=receipt['neural_batches'])), flush=True)
    for handle in handles:
        handle.remove()
    save(output / 'inference-completed.json', dict(
        reports={r['case']['name']: sha(output / r['case']['name'] / 'report.json') for r in inference_rows},
        seconds=sum(r['seconds'] for r in inference_rows), notes_or_chart_labels_read=False))
    # Only now read reference labels. Never route, tune or filter using them.
    manifest, old = read(MANIFEST), read(BASELINE / 'comparison.json')
    scores = []
    groups = defaultdict(lambda: defaultdict(lambda: dict(reference=0, predicted=0, matched=0)))
    for case, baseline in zip(manifest['cases'], old['rows']):
        assert sha(case['labels_path']) == case['labels_sha256']
        receipt = next(r for r in inference_rows if r['case']['name'] == case['name'])
        predicted = read(output / case['name'] / 'notes.json')
        metrics, scope = window_metrics(read(case['labels_path']), predicted,
            begin=0, end=30, context_start=0, context_end=receipt['case']['samples']/16000)
        metrics = {k: {label: value for label, value in values.items() if 'instrument' not in label}
                   for k, values in metrics.items()}
        scores.append(dict(name=case['name'], kind=case['kind'], instrument=case.get('instrument'),
                           metrics=metrics, tsumugi=baseline['metrics'], scope=scope))
        for engine, values in [('harmonica', metrics), ('tsumugi', baseline['metrics'])]:
            for metric, result in values['standard'].items():
                if 'instrument' in metric:
                    continue
                for field in ('reference', 'predicted', 'matched'):
                    groups[case['kind']][engine+'/'+metric][field] += result[field]
    for group in groups.values():
        for row in group.values():
            row['f1'] = 2*row['matched']/max(1, row['reference']+row['predicted'])
    assert sha(__file__) == plan['runner_sha256']
    assert sha(ROOT / 'scripts/evaluate_instrument_benchmark.py') == plan['matcher_sha256']
    assert all(sha(SOURCE / name) == digest for name, digest in plan['source_files'].items())
    save(output / 'report.json', dict(plan=plan, scores=scores, groups=groups,
        silence_notes=inference_rows[0]['notes'], seconds=sum(r['seconds'] for r in inference_rows),
        inference_completed_sha256=sha(output / 'inference-completed.json'),
        model_identity_sha256=sha(output / 'model-identity.json'),
        decoder_controls_sha256=sha(output / 'decoder-controls.json'),
        musical_acceptance=False, default_changed=False,
        limitations=['Third-party reproduction, not paper authors weights or validated paper metrics.',
            'Public reused cohort; declared source split is not a verified training manifest.',
            'Single-instrument design; mixed clips are stress tests, without instrument identities.',
            'Model offset head unused by original decoder; sustain decay is not authored LN release.',
            'No threshold tuning, pitch filtering, quantization, or note cleanup.']))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--prepare-only', action='store_true')
    cli.add_argument('--precision', choices=['upstream-bfloat16', 'float32'], default='upstream-bfloat16')
    args = cli.parse_args()
    if args.prepare_only:
        print(json.dumps(prepare(args.precision), indent=2, ensure_ascii=False))
        assert 'torch' not in sys.modules
    elif args.output is None:
        cli.error('--output required')
    else:
        run(args.output, args.precision)
