"""Export a completed guitar candidate to the existing lightweight ONNX runtime.

Export does execute the network: run only in the single serial neural slot.
The graph is research output until the full-cohort runtime verification passes.
"""
from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_hf_guitar_specialists import MODELS
from scripts import research_hf_instrument as adapter


def main():
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--model', choices=tuple(MODELS), required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    require, sha, verify, read = adapter.require, adapter.sha, adapter.verified_hash, adapter.read_json
    require(not args.output.exists(), 'Preserve earlier export')
    source = read(args.source/'report.json')
    names = [case['name'] for case in source['plan']['cases']]
    require(len(names) == len(set(names)) == 60 and source['plan']['model_order'] == list(MODELS)
            and [(row['model'], row['name']) for row in source['rows']]
            == [(model, name) for model in MODELS for name in names],
            'Complete both unique original model cohorts in the fixed order before exporting')
    for path, digest in source['plan']['code_sha256'].items():
        verify(path, digest)
    case = source['plan']['cases'][0]
    original = next(r for r in source['rows'] if r['model'] == args.model and r['name'] == case['name'])
    folder = args.source/args.model/case['name']
    verify(folder/'report.json', original['report_sha256'])
    evidence = read(folder/'report.json')
    verify(folder/'windows.npz', evidence['windows_sha256'])
    verify(case['audio'], case['audio_sha256'])
    rate, integer = wavfile.read(case['audio'])
    require(rate == case['sample_rate'] and len(integer) == case['samples']
            and integer.dtype == np.int16 and integer.ndim == 1, 'Input clock differs')
    common = gcd(rate, adapter.RATE)
    pcm = resample_poly(integer.astype(np.float32)/32768, adapter.RATE//common, rate//common).astype(np.float32)
    import hashlib
    require(hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() == evidence['input']['model_input_sha256'],
            'Export input differs from completed research')
    audio = np.zeros(adapter.WINDOW, dtype=np.float32)
    audio[:min(len(pcm), len(audio))] = pcm[:len(audio)]
    with patch.dict(adapter.MODELS, MODELS, clear=True):
        network, identity = adapter.load_model(args.model)
    require(identity['checkpoint_sha256'] == evidence['identity']['checkpoint_sha256'], 'Weight mismatch')
    sys.path.append(str(ROOT/'test-data/piano-onnx/export-dependencies'))
    import torch
    import onnx

    class Outputs(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            result = self.model(audio)
            return tuple(result[key] for key in adapter.OUTPUTS)

    args.output.mkdir(parents=True)
    np.save(args.output/'input.npy', audio, allow_pickle=False)
    with np.load(folder/'windows.npz', allow_pickle=False) as saved:
        np.savez_compressed(args.output/'torch-reference.npz',
                            **{key: saved[key][0:1] for key in adapter.OUTPUTS})
    started = time.monotonic()
    torch.onnx.export(Outputs(network).eval(), torch.from_numpy(audio[None]),
        str(args.output/'guitar.onnx'), input_names=['audio'], output_names=list(adapter.OUTPUTS),
        dynamo=False, opset_version=17)
    onnx.checker.check_model(str(args.output/'guitar.onnx'))
    report = dict(model=args.model, identity=identity,
        source_report=str((args.source/'report.json').resolve()), source_report_sha256=sha(args.source/'report.json'),
        original_case=case['name'], original_report_sha256=original['report_sha256'],
        graph_sha256=sha(args.output/'guitar.onnx'), input_sha256=sha(args.output/'input.npy'),
        reference_sha256=sha(args.output/'torch-reference.npz'), output_names=list(adapter.OUTPUTS),
        torch=torch.__version__, onnx=onnx.__version__, runner_sha256=sha(__file__),
        seconds=time.monotonic()-started, enabled_in_application=False,
        verified_in_application_runtime=False, musical_acceptance=False)
    adapter.write_new(args.output/'report.json', report)
    import json
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
