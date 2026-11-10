"""Pinned Timbre-Trap pitch-activity probe, without creating or deleting notes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from math import gcd
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'test-data/instrument-search-continuation-2026-11-10/timbre-trap-updates'
SOURCE_COMMIT = 'e645620938bff6ce1b69532f56addc31f16211ca'
HF_REVISION = '84fbd38582435c863a2d65197a75edd794888f19'
WEIGHT = SOURCE.parent/'timbre-trap-models--tt-orig.pt'
WEIGHT_SHA = '1c3bafd33a831d61e8ee9051d6c5b4c5d483e6a7669ca9df85ac6ab304cb9fe3'
BASELINE = ROOT/'test-data/tsumugi-urmp-comparison-v1'
BASELINE_SHA = 'ff98a2e3c53671c1bcf4ec16a6c3cdff3215e17e239101dfe6883f884fbe2e75'
MANIFEST = ROOT/'test-data/urmp/comparison-extension/manifest.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def peak_mask(activations):
    """Author's strict frequency peaks and inclusive 0.5 activity threshold."""
    a = np.asarray(activations)
    require(a.ndim == 2 and np.isfinite(a).all(), 'Invalid pitch activations')
    padded = np.pad(a, ((1, 1), (0, 0)))
    return (a >= .5) & (a > padded[:-2]) & (a > padded[2:])


def match_pitches(reference, estimated):
    """Maximum matching on ordered pitch intervals; retain duplicate estimates."""
    ref, est = sorted(set(reference)), sorted(estimated)
    require(all(np.isfinite(x) for x in ref+est), 'Nonfinite pitch')
    i = j = matched = 0
    while i < len(ref) and j < len(est):
        if est[j] < ref[i] - .5:
            j += 1
        elif est[j] > ref[i] + .5:
            i += 1
        else:
            matched += 1
            i += 1
            j += 1
    return len(ref), len(est), matched


def score_frames(reference, times, valid, estimated):
    """Frame evidence on [0,30), not note-onset or physical-release accuracy."""
    times, valid = np.asarray(times), np.asarray(valid)
    require(times.ndim == 1 and times.shape == valid.shape and valid.dtype == bool
            and len(estimated) == len(times) and np.isfinite(times).all()
            and (np.diff(times) > 0).all(), 'Invalid frame clock')
    counts = dict(frames=0, reference=0, predicted=0, matched=0,
                  rest_frames=0, rest_false_active=0, polyphonic_frames=0)
    for t, physical, est in zip(times, valid, estimated):
        if not physical or not 0 <= t < 30:
            continue
        ref = {n['pitch'] for n in reference if not n.get('is_drum', False)
               and n['onset'] <= t < n['offset']}
        r, p, m = match_pitches(ref, est)
        counts['frames'] += 1
        counts['reference'] += r
        counts['predicted'] += p
        counts['matched'] += m
        counts['rest_frames'] += int(r == 0)
        counts['rest_false_active'] += int(r == 0 and p > 0)
        counts['polyphonic_frames'] += int(r > 1)
    return counts


def metrics(counts):
    r, p, m = (counts[k] for k in ('reference', 'predicted', 'matched'))
    return dict(counts, precision=m/p if p else None, recall=m/r if r else None,
                f1=2*m/(r+p) if r+p else None)


def prepare():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()
    require(git('rev-parse', 'HEAD') == SOURCE_COMMIT and not git('diff', 'HEAD', '--name-only'),
            'Source differs from reviewed author demo branch')
    require(WEIGHT.stat().st_size == 11353410 and sha(WEIGHT) == WEIGHT_SHA, 'Weight mismatch')
    sources = {name: sha(SOURCE/name) for name in git('ls-files', '*.py', 'LICENSE*').splitlines()}
    require(sha(BASELINE/'comparison.json') == BASELINE_SHA, 'Baseline identity mismatch')
    baseline = read(BASELINE/'comparison.json')
    require(sha(MANIFEST) == baseline['identity']['manifest_sha256'], 'Manifest mismatch')
    cases, prior = read(MANIFEST)['cases'], {row['name']: row for row in baseline['rows']}
    require(len(cases) == 35 and len(prior) == 35, 'Require complete fixed cohort')
    for case in cases:
        row = prior[case['name']]
        require(sha(BASELINE/(case['name']+'.wav')) == row['clip_sha256']
                and sha(BASELINE/case['name']/'notes.json') == row['notes_sha256']
                and sha(case['labels_path']) == case['labels_sha256'], 'Input evidence changed')
    return dict(source_commit=SOURCE_COMMIT, source_files=sources, hf_revision=HF_REVISION,
        checkpoint_sha256=WEIGHT_SHA, runner_sha256=sha(__file__), baseline_sha256=BASELINE_SHA,
        manifest_sha256=sha(MANIFEST), cases=[c['name'] for c in cases],
        threshold=.5, peak_rule='strict_frequency_neighbors_with_zero_padding',
        pitch_tolerance_semitones=.5, scoring_seconds=[0, 30],
        sample_rate=22050, normalization=False, device='cpu', intra_threads=2, inter_threads=1,
        reference_semantics='unique_active_note_pitches_not_continuous_F0',
        prediction_semantics='pitch_activity_not_instrument_identity_or_LN_release',
        upstream_training_overlap='not_certified_unseen'), cases


def run(output):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    output = Path(output)
    require(not output.exists(), 'Preserve previous evidence')
    plan, cases = prepare()
    for path in (ROOT/'test-data/piano-release-probe/dependencies',
                 ROOT/'test-data/rosvot-runtime', ROOT/'test-data/timbre-trap-runtime', SOURCE):
        sys.path.insert(0, str(path))
    import torch
    from timbre_trap.framework import TimbreTrap
    import cqt_pytorch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    model = TimbreTrap(sample_rate=22050, n_octaves=9, bins_per_octave=60,
        secs_per_block=3, latent_size=128, model_complexity=2, skip_connections=False)
    state = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    require(all(torch.equal(v, state[k]) for k, v in model.state_dict().items()), 'State differs after load')
    model.eval().requires_grad_(False).to('cpu')
    plan.update(strict_weights=True, state_entries=len(state),
        versions={name: importlib.metadata.version(name) for name in
                  ('torch', 'torchaudio', 'numpy', 'scipy', 'librosa', 'cqt-pytorch', 'einops')},
        cqt_source={p.name: sha(p) for p in Path(cqt_pytorch.__file__).parent.glob('*.py')},
        block_samples=model.sliCQ.block_length, block_frames=model.sliCQ.max_window_length,
        hop_seconds=model.sliCQ.hop_length/22050)
    output.mkdir(parents=True)
    save(output/'plan.json', plan)

    def infer(pcm, rate, folder, identity):
        started = time.monotonic()
        require(pcm.ndim == 1 and pcm.dtype == np.float32 and np.isfinite(pcm).all(), 'Invalid PCM')
        original_samples = len(pcm)
        common = gcd(rate, 22050)
        model_pcm = resample_poly(pcm, 22050//common, rate//common).astype(np.float32) if rate != 22050 else pcm
        require(len(model_pcm) == (original_samples*22050+rate-1)//rate, 'Resampling clock mismatch')
        with torch.inference_mode():
            activations = model.transcribe(torch.from_numpy(model_pcm[None, None])).cpu().numpy()[0]
        expected_frames = ((len(model_pcm)+model.sliCQ.block_length-1)//model.sliCQ.block_length) \
            * model.sliCQ.max_window_length
        require(activations.shape == (540, expected_frames) and np.isfinite(activations).all(),
                'Unexpected whole-block output shape or invalid activations')
        times = model.sliCQ.get_times(activations.shape[1])
        pitches = model.sliCQ.get_midi_freqs()
        valid = np.arange(len(times), dtype=np.int64)*model.sliCQ.block_length*rate < \
            original_samples*model.sliCQ.max_window_length*22050
        numerator = original_samples*model.sliCQ.max_window_length*22050
        denominator = model.sliCQ.block_length*rate
        require(int(valid.sum()) == (numerator+denominator-1)//denominator,
                'Output does not cover exactly the physical frame clock')
        peaks = peak_mask(activations)
        estimated = [pitches[peaks[:, i]].tolist() for i in range(len(times))]
        folder.mkdir()
        np.savez_compressed(folder/'activity.npz', activations=activations, times=times,
                            valid=valid, pitches=pitches, peaks=peaks)
        report = dict(input=identity, original_samples=original_samples, original_rate=rate,
            original_pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
            model_samples=len(model_pcm), model_pcm_sha256=hashlib.sha256(model_pcm.astype('<f4').tobytes()).hexdigest(),
            resampler='scipy.signal.resample_poly/default' if rate != 22050 else 'identity',
            frames=len(times), physical_frames=int(valid.sum()), activity_sha256=sha(folder/'activity.npz'),
            seconds=time.monotonic()-started)
        save(folder/'report.json', report)
        return times, valid, estimated, report

    times, valid, est, control = infer(np.zeros(4*22050, dtype=np.float32), 22050,
        output/'silence', dict(control='zero_pcm'))
    control['counts'] = score_frames([], times, valid, est)
    rows = []
    for index, case in enumerate(cases):
        name = case['name']
        clip = BASELINE/(name+'.wav')
        rate, integer = wavfile.read(clip)
        require(rate == 16000 and integer.ndim == 1 and integer.dtype == np.int16, 'Unexpected frozen clip')
        times, valid, est, report = infer(integer.astype(np.float32)/32768, rate,
            output/name, dict(audio=str(clip), sha256=sha(clip)))
        reference = read(case['labels_path'])
        baseline = read(BASELINE/name/'notes.json')
        baseline_frames = [sorted({n['pitch'] for n in baseline if not n['is_drum']
            and n['onset'] <= t < n['offset']}) for t in times]
        row = dict(name=name, kind=case['kind'], instrument=case.get('instrument'),
            timbre_trap=metrics(score_frames(reference, times, valid, est)),
            tsumugi=metrics(score_frames(reference, times, valid, baseline_frames)),
            seconds=report['seconds'], report_sha256=sha(output/name/'report.json'))
        rows.append(row)
        save(output/name/'metrics.json', row)
        print(json.dumps(dict(index=index+1, total=len(cases), **row)), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['kind']].append(row)
        if row['instrument']:
            groups['instrument/'+row['instrument']].append(row)
    sums = {group: {variant: metrics({key: sum(row[variant][key] for row in members)
            for key in ('frames', 'reference', 'predicted', 'matched', 'rest_frames',
                        'rest_false_active', 'polyphonic_frames')})
            for variant in ('timbre_trap', 'tsumugi')} for group, members in groups.items()}
    require(sha(__file__) == plan['runner_sha256']
            and sha(WEIGHT) == WEIGHT_SHA
            and all(sha(SOURCE/name) == digest for name, digest in plan['source_files'].items())
            and all(sha(Path(cqt_pytorch.__file__).parent/name) == digest
                    for name, digest in plan['cqt_source'].items()),
            'Source changed during inference')
    save(output/'report.json', dict(plan=plan, rows=rows, groups=sums, silence=control,
        seconds=sum(row['seconds'] for row in rows), enabled_in_application=False,
        musical_acceptance=False, note_events_changed=False))
    print(json.dumps(dict(stage='complete', groups=sums)), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--prepare-only', action='store_true')
    args = cli.parse_args()
    require(args.prepare_only or args.output is not None, '--output is required')
    if args.prepare_only:
        plan, cases = prepare()
        require('torch' not in sys.modules, 'Preparation loaded Torch')
        print(json.dumps(dict(cases=len(cases), plan=plan, network_loaded=False)))
    else:
        run(args.output)


if __name__ == '__main__':
    main()
