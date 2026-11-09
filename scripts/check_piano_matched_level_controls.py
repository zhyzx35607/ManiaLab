"""Add same-amplitude solo comparisons for the fixed weak-high mixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.piano import MODEL_SHA256, infer, decode
from scripts.build_provenance import sha256
from scripts.piano_source_controls import compose_case, score_attacks

PAIRS = ((36, 48), (48, 60), (60, 72), (72, 84), (84, 96), (48, 67), (60, 79), (48, 66))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset, run, output = args.dataset.resolve(), args.run.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Preserve prior runs; use a new test-data folder')
    manifest, report = read(dataset/'manifest.json'), read(run/'report.json')
    if (report['real_network_inference_completed'] != 83
            or report['identity']['manifest_sha256'] != sha256(dataset/'manifest.json')):
        raise ValueError('Expected the completed original fixed run')
    for folder, hashes in ((dataset, manifest['artifacts_sha256']), (run, report['artifacts_sha256'])):
        for name, digest in hashes.items():
            if sha256(folder/name) != digest:
                raise ValueError('Frozen evidence changed')
    paths = dict(runner=Path(__file__), helper=ROOT/'scripts/piano_source_controls.py',
                 runtime=ROOT/'manialab/piano.py',
                 protocol=ROOT/'docs/piano-matched-level-controls-protocol.md')
    code = {key: sha256(path) for key, path in paths.items()}
    if any(code[k] != report['identity']['code'][k] for k in ('helper', 'runtime')):
        raise ValueError('Original code changed')
    if sha256(ROOT/'assets/deep/piano-regression.onnx') != MODEL_SHA256:
        raise ValueError('Model changed')
    sources = {}
    for row in manifest['sources']:
        rate, samples = wavfile.read(dataset/row['wav_file'])
        if rate != 16000 or samples.shape != (81600,) or samples.dtype != np.float32:
            raise ValueError('Source clock changed')
        sources[row['pitch']] = samples
    output.mkdir(parents=True)
    identity = dict(source_report_sha256=sha256(run/'report.json'),
                    manifest_sha256=sha256(dataset/'manifest.json'), code=code,
                    model_sha256=MODEL_SHA256, thresholds=dict(onset=.3, offset=.3, frame=.1),
                    provider='CPUExecutionProvider', intra_op_threads=2, inter_op_threads=1,
                    environment=dict(python=sys.version, numpy=np.__version__, onnxruntime=ort.__version__),
                    supplement_registered_after_original_inference_started=True)
    save(output/'identity.json', identity)
    for key, path in paths.items():
        shutil.copyfile(path, output/(key+'-snapshot'+path.suffix))
    plans = []
    for low, high in PAIRS:
        samples, metadata = compose_case({high: sources[high]},
            [dict(pitch=high, anchor_seconds=2., gain=.25)], source_anchor_samples={high:1600})
        name = f'g0.25-single-{high}'
        wavfile.write(output/(name+'.wav'), 16000, samples)
        plans.append(dict(name=name, pitch=high, pair=[low,high], gain=.25,
                          file=name+'.wav', wav_sha256=sha256(output/(name+'.wav')),
                          metadata=metadata, expected=[dict(pitch=high, anchor_seconds=2.)]))
    save(output/'inputs.json', plans)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(ROOT/'assets/deep/piano-regression.onnx'),
                                  sess_options=options, providers=['CPUExecutionProvider'])
    results = {}
    for plan in plans:
        rate, samples = wavfile.read(output/plan['file'])
        start = time.monotonic()
        activations = infer(samples, rate, session)
        events = decode(activations)
        row = dict(name=plan['name'], pitch=plan['pitch'], elapsed_seconds=time.monotonic()-start,
                   input_sha256=plan['wav_sha256'], expected=plan['expected'], events=events,
                   score=score_attacks(plan['expected'], events))
        save(output/(plan['name']+'-result.json'), row)
        np.savez_compressed(output/(plan['name']+'-activations.npz'), **activations)
        results[plan['pitch']] = row
        print(json.dumps(dict(name=plan['name'], matched=row['score']['matched'],
                              extra=row['score']['extra'], seconds=row['elapsed_seconds'])), flush=True)
    paired = []
    for gain in (1,4):
        for low, high in PAIRS:
            mixed = read(run/f'g{gain}-{low}-{high}-weak_high-result.json')
            solo = results[high] if gain == 1 else read(run/f'g1-single-{high}-result.json')
            same_component, _ = compose_case({high:sources[high]},
                [dict(pitch=high, anchor_seconds=2., gain=gain*.25)], source_anchor_samples={high:1600})
            solo_path = (output if gain == 1 else dataset)/(solo['name']+'.wav')
            rate, solo_samples = wavfile.read(solo_path)
            if not np.array_equal(same_component, solo_samples):
                raise ValueError('Solo and mixture high component are different')
            original_case = next(c for c in manifest['cases'] if c['name'] == mixed['name'])
            if original_case['components'][1] != dict(pitch=high, anchor_seconds=2., gain=gain*.25):
                raise ValueError('Expected exact matched high component')
            matched = {m['ref_index'] for m in mixed['score']['matches']}
            paired.append(dict(gain=gain, pair=[low,high], high_gain=gain*.25,
                solo_name=solo['name'], mixture_name=mixed['name'],
                high_component_pcm_sha256=hashlib.sha256(same_component.astype('<f4').tobytes()).hexdigest(),
                high_pcm_identical=True, solo_high_matched=solo['score']['matched']==1,
                mixture_high_matched=1 in matched, mixture_low_matched=0 in matched,
                solo_extra=solo['score']['extra'], mixture_extra=mixed['score']['extra']))
    grouped = {}
    for gain in (1,4):
        rows = [r for r in paired if r['gain']==gain]
        grouped[str(gain)] = dict(pairs=len(rows),
            high_solo_matched=sum(r['solo_high_matched'] for r in rows),
            high_mixture_matched=sum(r['mixture_high_matched'] for r in rows),
            detected_alone_lost_in_mixture=sum(r['solo_high_matched'] and not r['mixture_high_matched'] for r in rows),
            missed_alone_and_in_mixture=sum(not r['solo_high_matched'] and not r['mixture_high_matched'] for r in rows),
            missed_alone_detected_in_mixture=sum(not r['solo_high_matched'] and r['mixture_high_matched'] for r in rows))
    if any(sha256(path)!=code[key] for key,path in paths.items()):
        raise ValueError('Code changed during the run')
    save(output/'report.json', dict(identity=identity, real_network_inference_completed=8,
        groups=grouped, paired_cases=paired, application_changed=False, musical_acceptance=False,
        offsets_scored=False, direct_listening_performed=False,
        artifacts_sha256={p.name:sha256(p) for p in output.iterdir()}))
    print(json.dumps(grouped), flush=True)


if __name__ == '__main__':
    main()
