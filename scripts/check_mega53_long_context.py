"""Author 20-second/50%-overlap Mega 53 inference on fixed control sources.

The comparison output remains physical seconds 20--26. Separation receives
seconds 10--30; transcription still receives the same six-second comparison
window. Research only, with no annotation-driven routing or application changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_mega53_separation as base
from scripts.check_banquet_separation import wave_metrics

BASE = ROOT / 'test-data/banquet-separation-v1'
BASE_SHA = '1cb7aee0598cd5d535ea64aa12637ee144d4c92b34366ff408c8056c76236997'
RATE = base.RATE
VARIANTS = ['mixture', 'target', 'absent', 'silence']
START, STOP = 10, 30
CROP = slice(10 * RATE, 16 * RATE)


def prepare():
    from scipy.io import wavfile
    plan = base.prepare()
    assert base.sha(BASE / 'report.json') == BASE_SHA
    old = json.loads((BASE / 'report.json').read_text(encoding='utf-8'))
    for case, row in zip(plan['cases'], old['rows']):
        case['sources'] = row['case']['sources']
        case['family'] = row['case']['family']
        for source in case['sources']:
            assert base.sha(source['audio']) == source['audio_sha256']
            rate, pcm = wavfile.read(source['audio'], mmap=True)
            assert rate == 16000 and pcm.dtype.name == 'int16' and pcm.ndim == 1
            assert len(pcm) >= STOP * rate
    plan.update(runner_sha256=base.sha(__file__),
        adapter_sha256=base.sha(base.__file__),
        upstream_demix_sha256=base.sha(base.SOURCE / 'utils/model_utils.py'),
        source_seconds=[20, 26], separator_source_seconds=[START, STOP],
        window_seconds=6, separator_chunk_seconds=20, overlap_factor=2,
        inference='upstream_demix_unmodified_with_selected_head_wrapper',
        comparison='same_original_six_second_baseline_and_score_window',
        resampling_boundary_difference='measure_full_crop_and_require_exact_central_two_seconds',
        identical_input_reuse='exact_float32_pcm_with_union_of_preselected_output_heads',
        nonfinite_policy='fail_before_upstream_nan_to_num')
    return plan


def author_demix():
    sys.path.insert(0, str(base.ASSETS / 'demix-deps'))
    path = base.SOURCE / 'utils/model_utils.py'
    spec = importlib.util.spec_from_file_location('mega53_author_model_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import numpy as np

    class CheckedNumpy:
        calls = 0

        def __getattr__(self, name):
            return getattr(np, name)

        def nan_to_num(self, array, *args, **kwargs):
            assert np.isfinite(array).all(), 'Nonfinite overlap result before upstream cleanup'
            self.calls += 1
            return np.nan_to_num(array, *args, **kwargs)

    # Only the upstream module's NumPy binding is wrapped. Its algorithm and
    # dependency modules are unchanged; this prevents NaN cleanup hiding errors.
    module.np = CheckedNumpy()
    return module


def configuration():
    import yaml
    from ml_collections import ConfigDict

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor('tag:yaml.org,2002:python/tuple',
                           lambda loader, node: tuple(loader.construct_sequence(node)))
    config = ConfigDict(yaml.load((base.ASSETS / 'full-config.txt').read_text(encoding='utf-8'), Loader=Loader))
    assert config.inference.chunk_size == RATE * 20
    assert config.inference.num_overlap == 2 and config.inference.batch_size == 1
    assert config.inference.normalize is False
    # CUDA autocast has no effect on the CPU experiment. Disable its warning;
    # the network and overlap accumulators remain float32 as in the prior run.
    config.training.use_amp = False
    return config


def check_stitching(module, config):
    import numpy as np
    import torch

    class Echo(torch.nn.Module):
        def forward(self, x):
            return torch.stack((x, -x), dim=1)

    config = config.copy_and_resolve_references()
    config.training.instruments = ['positive', 'negative']
    controls = []
    for seconds in (6, 20, 22):
        wave = np.random.default_rng(19).uniform(-.2, .2, (2, seconds * RATE)).astype(np.float32)
        result = module.demix(config, Echo(), wave, torch.device('cpu'), 'bs_roformer', False)
        errors = {name: float(np.max(abs(result[name] - sign * wave)))
                  for name, sign in [('positive', 1), ('negative', -1)]}
        assert max(errors.values()) <= 3e-8
        controls.append(dict(seconds=seconds, max_absolute_error=errors,
                             outer_reflection=seconds > 20))
    return controls


def make_inputs(case):
    import numpy as np
    import torch
    from scipy.io import wavfile
    import torchaudio.functional as af
    sources, targets = [], []
    for source in case['sources']:
        assert base.sha(source['audio']) == source['audio_sha256']
        rate, pcm = wavfile.read(source['audio'])
        wave = pcm.astype(np.float32) / 32768 / len(case['sources'])
        sources.append(wave[START * rate:STOP * rate])
        if source['instrument'] == case['family']:
            targets.append(sources[-1])

    def convert(wave):
        resampled = af.resample(torch.from_numpy(wave.copy()), 16000, RATE).numpy()
        return np.repeat(resampled[None], 2, axis=0)

    mixture = convert(np.sum(sources, axis=0))
    target = convert(np.sum(targets, axis=0))
    return dict(mixture=mixture, target=target, absent=mixture-target,
                silence=np.zeros_like(mixture))


def run(output):
    import numpy as np
    assert not output.exists(), 'Preserve complete or incomplete evidence'
    plan = prepare()
    base.save(output / 'plan.json', plan)
    model, identity = base.load_model()
    import torch
    module = author_demix()
    config = configuration()
    base.save(output / 'stitching-controls.json', check_stitching(module, config))
    identity.update(upstream_demix_sha256=plan['upstream_demix_sha256'],
                    additional_versions={name: importlib.metadata.version(name)
                                         for name in ('ml-collections', 'absl-py', 'torchaudio')})
    base.save(output / 'model-identity.json', identity)
    jobs = {}
    uses = []
    boundary_checks = []
    long_hashes = {}
    for head, case in enumerate(plan['cases']):
        folder = output / case['name']; folder.mkdir()
        arrays = make_inputs(case)
        np.savez_compressed(folder / 'long-inputs.npz', **arrays)
        long_hashes[case['name']] = base.sha(folder / 'long-inputs.npz')
        with np.load(case['inputs'], allow_pickle=False) as original:
            for variant in VARIANTS:
                wave = arrays[variant]
                assert wave.shape == (2, 20 * RATE) and wave.dtype == np.float32
                assert np.isfinite(wave).all()
                crop = wave[:, CROP]
                central_equal = np.array_equal(crop[:, 2*RATE:4*RATE], original[variant][:, 2*RATE:4*RATE])
                assert central_equal
                boundary_checks.append(dict(case=case['name'], variant=variant,
                    central_exact=central_equal,
                    crop_max_absolute_difference=float(np.max(abs(crop-original[variant]))),
                    crop_changed_samples=int(np.count_nonzero(crop != original[variant]))))
                digest = hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()
                use = dict(case=case['name'], variant=variant, head=head, pcm_sha256=digest)
                uses.append(use)
                if digest not in jobs:
                    jobs[digest] = dict(wave=wave, heads=set(), uses=[])
                else:
                    assert np.array_equal(jobs[digest]['wave'], wave)
                jobs[digest]['heads'].add(head)
                jobs[digest]['uses'].append(use)
    base.save(output / 'inputs.json', dict(uses=uses, distinct_pcm=len(jobs),
              long_input_hashes=long_hashes, boundary_checks=boundary_checks))
    records = []
    for index, (digest, job) in enumerate(jobs.items(), 1):
        heads = sorted(job['heads'])
        names = [base.CASES[h][1] for h in heads]
        config.training.instruments = names
        config.training.target_instrument = None
        calls = []

        class SelectedHeads(torch.nn.Module):
            def forward(self, value):
                assert value.dtype == torch.float32 and torch.isfinite(value).all()
                started = time.monotonic()
                prediction = model(value, active_stem_ids=heads)
                assert prediction.shape == (len(value), len(heads), 2, 20*RATE)
                assert torch.isfinite(prediction).all()
                receipt = dict(chunk=len(calls), seconds=time.monotonic()-started,
                    input_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                    output_sha256=hashlib.sha256(prediction.numpy().tobytes()).hexdigest())
                calls.append(receipt)
                print(json.dumps(dict(job=index, total_jobs=len(jobs), heads=names, **receipt)), flush=True)
                return prediction

        started = time.monotonic()
        with torch.inference_mode():
            separated = module.demix(config, SelectedHeads(), job['wave'], torch.device('cpu'), 'bs_roformer', False)
        elapsed = time.monotonic() - started
        assert len(calls) == 2 and list(separated) == names
        estimates = {}
        for use in job['uses']:
            name = base.CASES[use['head']][1]
            estimate = separated[name]
            assert estimate.shape == (2, 20*RATE) and np.isfinite(estimate).all()
            folder = output / use['case']
            long_path = folder / (use['variant'] + '.long.npy')
            crop_path = folder / (use['variant'] + '.npy')
            np.save(long_path, estimate, allow_pickle=False)
            np.save(crop_path, estimate[:, CROP], allow_pickle=False)
            estimates[use['case'] + '/' + use['variant']] = dict(
                long_sha256=base.sha(long_path), estimate_sha256=base.sha(crop_path))
        receipt = dict(pcm_sha256=digest, selected_heads=heads, stems=names,
                       uses=job['uses'], seconds=elapsed, calls=calls, estimates=estimates)
        base.save(output / 'jobs' / (digest + '.json'), receipt)
        records.append(receipt)
        # Long PCM is also on disk; release each job's retained input promptly.
        job['wave'] = None
    rows = []
    for case in plan['cases']:
        variants = []
        with np.load(case['inputs'], allow_pickle=False) as arrays:
            for variant in VARIANTS:
                path = output / case['name'] / (variant + '.npy')
                estimate = np.load(path, allow_pickle=False)
                reference = arrays['target'] if variant in ('mixture', 'target') else arrays['silence']
                scores = {}
                for label, begin, end in [('full', 0, 6), ('interior', 2, 4)]:
                    window = slice(begin*RATE, end*RATE)
                    scores[label] = dict(input=wave_metrics(reference[:, window], arrays[variant][:, window]),
                                         output=wave_metrics(reference[:, window], estimate[:, window]))
                variants.append(dict(variant=variant, scores=scores, estimate_sha256=base.sha(path)))
        row = dict(case=case, variants=variants)
        base.save(output / case['name'] / 'report.json', row)
        rows.append(row)
    assert base.sha(__file__) == plan['runner_sha256']
    assert base.sha(base.__file__) == plan['adapter_sha256']
    assert base.sha(base.SOURCE / 'utils/model_utils.py') == plan['upstream_demix_sha256']
    assert all(base.sha(base.SOURCE / name) == digest for name, digest in plan['source_files'].items())
    assert module.np.calls == 3 + len(records)
    base.save(output / 'report.json', dict(plan=plan, rows=rows,
        jobs=[dict(pcm_sha256=r['pcm_sha256'], seconds=r['seconds'],
                   receipt_sha256=base.sha(output / 'jobs' / (r['pcm_sha256'] + '.json'))) for r in records],
        neural_forwards=sum(len(r['calls']) for r in records),
        seconds=sum(r['seconds'] for r in records),
        model_identity_sha256=base.sha(output / 'model-identity.json'),
        inputs_sha256=base.sha(output / 'inputs.json'),
        stitching_controls_sha256=base.sha(output / 'stitching-controls.json'),
        musical_acceptance=False, default_changed=False,
        limitations=['Known target families; no automatic instrument discovery.',
            'Same small public cohort and scoring windows; training overlap unknown.',
            'Twenty-second source segment, not full-song inference.',
            'Author 20-second chunks and overlap; CPU float32 rather than CUDA AMP.',
            'Six-second comparison crop retains raw resampling edge differences, measured separately.',
            'No note labels used; downstream transcription required before adoption.']))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--prepare-only', action='store_true')
    args = cli.parse_args()
    if args.prepare_only:
        print(json.dumps(prepare(), indent=2, ensure_ascii=False))
        assert 'torch' not in sys.modules
    elif args.output is None:
        cli.error('--output required')
    else:
        run(args.output)
