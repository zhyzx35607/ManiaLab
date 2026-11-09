"""Run the frozen piano model on source-known single-note and mixture controls."""
import argparse
from collections import Counter
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
from scripts.piano_source_controls import score_attacks


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def probe(activations, pitch, anchor):
    times = np.arange(len(activations['frame_output']))/100
    attack = (times >= anchor-.1) & (times <= anchor+.1)
    frame = (times >= anchor) & (times <= anchor+.15)
    return dict(pitch=pitch, anchor_seconds=anchor,
                onset_max_within_100ms=float(activations['reg_onset_output'][attack, pitch-21].max()),
                frame_max_first_150ms=float(activations['frame_output'][frame, pitch-21].max()))


def event_indices(events, pitch, onset):
    return [i for i, e in enumerate(events) if e['pitch'] == pitch and abs(e['onset']-onset) <= .1+1e-12]


def summarize(rows):
    grouped = {}
    for row in rows:
        key = f'g{row["gain"]}_{row["condition"]}'
        bucket = grouped.setdefault(key, Counter())
        bucket['cases'] += 1
        for field in ('expected', 'predicted', 'matched', 'missing', 'extra'):
            bucket[field] += row['score'][field]
        bucket['all_expected_matched_cases'] += row['score']['missing'] == 0
        bucket['no_extra_cases'] += row['score']['extra'] == 0
    by_name = {r['name']: r for r in rows}
    comparisons = []
    pairs = sorted({tuple(r['pair']) for r in rows if r['pair'] is not None})
    for gain in (1, 4):
        for low, high in pairs:
            lower, higher = by_name[f'g{gain}-single-{low}'], by_name[f'g{gain}-single-{high}']
            item = dict(gain=gain, pair=[low, high],
                low_alone_expected_matched=lower['score']['matched'] == 1,
                high_alone_expected_matched=higher['score']['matched'] == 1,
                low_alone_proposed_high=len(event_indices(lower['events'], high, 2.)), conditions={})
            for condition in ('simultaneous', 'weak_high', 'delayed_high', 'repeated_high'):
                row = by_name[f'g{gain}-{low}-{high}-{condition}']
                missing = set(row['score']['missing_ref_indices'])
                expected = row['expected']
                item['conditions'][condition] = dict(
                    low_matched=0 not in missing,
                    high_expected=sum(e['pitch'] == high for e in expected),
                    high_matched=sum(e['pitch'] == high and i not in missing for i, e in enumerate(expected)),
                    extra=row['score']['extra'])
            comparisons.append(item)
    totals = {}
    for gain in (1, 4):
        chosen = [p for p in comparisons if p['gain'] == gain]
        totals[str(gain)] = dict(pairs=len(chosen),
            low_alone_proposed_high_pairs=sum(p['low_alone_proposed_high'] > 0 for p in chosen),
            high_alone_missed_pairs=sum(not p['high_alone_expected_matched'] for p in chosen),
            conditions={c:dict(
                high_missed_pairs=sum(p['conditions'][c]['high_matched'] < p['conditions'][c]['high_expected'] for p in chosen),
                high_detected_alone_but_lost_in_mixture_pairs=sum(p['high_alone_expected_matched'] and
                    p['conditions'][c]['high_matched'] < p['conditions'][c]['high_expected'] for p in chosen),
                low_detected_alone_but_lost_in_mixture_pairs=sum(p['low_alone_expected_matched'] and
                    not p['conditions'][c]['low_matched'] for p in chosen))
                for c in ('simultaneous', 'weak_high', 'delayed_high', 'repeated_high')})
    return dict(groups={key: dict(value) for key, value in grouped.items()}, paired_totals=totals,
                paired_cases=comparisons, anchor_semantics='source envelope crossing, not MIDI key-on',
                offsets_scored=False, classification_threshold_search=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset', type=Path, required=True)
    cli.add_argument('--model', type=Path, default=ROOT/'assets/deep/piano-regression.onnx')
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--resume', action='store_true')
    args = cli.parse_args()
    output, dataset = args.output.resolve(), args.dataset.resolve()
    if not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use local test-data output')
    manifest = read(dataset/'manifest.json')
    if len(manifest['cases']) != 83 or manifest['anchor_tolerance_seconds'] != .1:
        raise ValueError('Fixed control scope changed')
    if sha256(args.model) != MODEL_SHA256:
        raise ValueError('Frozen ONNX model changed')
    for name, digest in manifest['artifacts_sha256'].items():
        if sha256(dataset/name) != digest:
            raise ValueError('Fixed input changed: '+name)
    paths = dict(runner=Path(__file__), helper=ROOT/'scripts/piano_source_controls.py',
                 runtime=ROOT/'manialab/piano.py', protocol=ROOT/'docs/piano-isolated-controls-protocol.md')
    identity = dict(code={key:sha256(path) for key,path in paths.items()},
        manifest_sha256=sha256(dataset/'manifest.json'), model_sha256=MODEL_SHA256,
        thresholds=dict(onset=.3, offset=.3, frame=.1),
        environment=dict(python=sys.version, numpy=np.__version__, onnxruntime=ort.__version__),
        provider='CPUExecutionProvider', intra_op_threads=2, inter_op_threads=1)
    if any(manifest['identity'][key] != identity['code'][key] for key in ('helper', 'protocol')):
        raise ValueError('Preparation used a different helper/protocol')
    rows = []
    if output.exists():
        if not args.resume or not (output/'identity.json').is_file() or read(output/'identity.json') != identity:
            raise ValueError('Preserve prior runs; resume requires the exact same identity')
        if (output/'progress.json').is_file():
            rows = read(output/'progress.json')
        for row in rows:
            for kind in ('events', 'activations', 'result'):
                suffix = '.npz' if kind == 'activations' else '.json'
                if sha256(output/(row['name']+'-'+kind+suffix)) != row[kind+'_sha256']:
                    raise ValueError('Completed result changed')
        if [r['name'] for r in rows] != [r['name'] for r in manifest['cases'][:len(rows)]]:
            raise ValueError('Resume case order changed')
    else:
        output.mkdir(parents=True)
        save(output/'identity.json', identity)
        for key, path in paths.items():
            shutil.copyfile(path, output/(key+'-snapshot'+path.suffix))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
    for case in manifest['cases'][len(rows):]:
        rate, samples = wavfile.read(dataset/case['file'])
        if rate != 16000 or samples.shape != (160000,) or samples.dtype != np.float32:
            raise ValueError('Expected exact 10-second 16k float32 mono input')
        if hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest() != case['pcm_sha256']:
            raise ValueError('Fixed PCM changed')
        began = time.monotonic()
        activations = infer(samples, rate, session)
        events = decode(activations)
        elapsed = time.monotonic()-began
        score = score_attacks(case['expected'], events, tolerance_seconds=.1)
        probes = [probe(activations, e['pitch'], e['anchor_seconds']) for e in case['expected']]
        row = dict(name=case['name'], gain=case['gain'], condition=case['condition'], pair=case['pair'],
                   expected=case['expected'], input_sha256=case['wav_sha256'],
                   elapsed_seconds=elapsed, events=events, score=score, expected_activation_probes=probes,
                   release_reasons=dict(Counter(e['release_reason'] for e in events)))
        save(output/(case['name']+'-events.json'), events)
        np.savez_compressed(output/(case['name']+'-activations.npz'), **activations)
        save(output/(case['name']+'-result.json'), row)
        record = dict(name=case['name'], elapsed_seconds=elapsed,
                      **{kind+'_sha256':sha256(output/(case['name']+'-'+kind+('.npz' if kind == 'activations' else '.json')))
                         for kind in ('events', 'activations', 'result')})
        rows.append(record)
        save(output/'progress.json', rows)
        print(json.dumps(dict(completed=len(rows), total=83, name=case['name'], seconds=elapsed,
                              expected=score['expected'], matched=score['matched'], extra=score['extra'])), flush=True)
    if any(sha256(path) != identity['code'][key] for key, path in paths.items()):
        raise ValueError('Code/protocol changed during inference')
    results = [read(output/(row['name']+'-result.json')) for row in rows]
    report = dict(identity=identity, cases=rows, summary=summarize(results),
                  real_network_inference_completed=len(rows), notes_changed=False,
                  application_enabled=False, musical_acceptance=False, direct_listening_performed=False,
                  artifacts_sha256={p.name:sha256(p) for p in output.iterdir() if p.name != 'report.json'})
    save(output/'report.json', report)
    print(json.dumps(report['summary']['paired_totals']), flush=True)


if __name__ == '__main__':
    main()
