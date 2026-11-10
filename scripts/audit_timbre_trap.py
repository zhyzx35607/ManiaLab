"""Independent, non-neural reconstruction of the complete Timbre-Trap scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def maximum_match(reference, predicted):
    """Augmenting paths, independent of the probe's ordered greedy matcher."""
    adjacency = [[j for j, pitch in enumerate(predicted) if abs(pitch - target) <= .5]
                 for target in reference]
    owners = {}
    def visit(index, seen):
        for target in adjacency[index]:
            if target not in seen:
                seen.add(target)
                if target not in owners or visit(owners[target], seen):
                    owners[target] = index
                    return True
        return False
    return sum(visit(i, set()) for i in range(len(adjacency)))


def active_notes(notes, times):
    active = [set() for _ in times]
    for note in notes:
        if note.get('is_drum', False):
            continue
        lo, hi = np.searchsorted(times, [note['onset'], note['offset']], side='left')
        for index in range(int(lo), int(hi)):
            active[index].add(note['pitch'])
    return active


def score(reference, estimated, times, valid):
    counts = dict(frames=0, reference=0, predicted=0, matched=0, rest_frames=0,
                  rest_false_active=0, polyphonic_frames=0)
    for index in np.flatnonzero(valid & (times >= 0) & (times < 30)):
        ref, pred = sorted(reference[index]), estimated[index]
        r, p = len(ref), len(pred)
        counts['frames'] += 1
        counts['reference'] += r
        counts['predicted'] += p
        counts['matched'] += maximum_match(ref, pred)
        counts['rest_frames'] += int(r == 0)
        counts['rest_false_active'] += int(r == 0 and p > 0)
        counts['polyphonic_frames'] += int(r > 1)
    r, p, m = (counts[k] for k in ('reference', 'predicted', 'matched'))
    return dict(counts, precision=m/p if p else None, recall=m/r if r else None,
                f1=2*m/(r+p) if r+p else None)


def main():
    runner_digest = sha(__file__)
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve previous audit')
    root = Path(__file__).resolve().parents[1]
    report = read(args.source / 'report.json')
    plan = report['plan']
    manifest = root / 'test-data/urmp/comparison-extension/manifest.json'
    require(sha(manifest) == plan['manifest_sha256'], 'Manifest identity differs')
    cases = read(manifest)['cases']
    names = [c['name'] for c in cases]
    require(len(names) == len(set(names)) == 35 and names == plan['cases']
            and names == [r['name'] for r in report['rows']], 'Incomplete or reordered cohort')
    baseline_root = root / 'test-data/tsumugi-urmp-comparison-v1'
    require(sha(baseline_root / 'comparison.json') == plan['baseline_sha256'], 'Baseline changed')
    baseline = {r['name']: r for r in read(baseline_root / 'comparison.json')['rows']}
    rows, shared_reference = [], defaultdict(list)
    for case, row in zip(cases, report['rows']):
        folder = args.source / case['name']
        evidence = read(folder / 'report.json')
        require(sha(folder / 'report.json') == row['report_sha256']
            and sha(folder / 'activity.npz') == evidence['activity_sha256'], 'Activity evidence changed')
        clip = baseline_root / (case['name'] + '.wav')
        require(sha(clip) == baseline[case['name']]['clip_sha256'] == evidence['input']['sha256'], 'Clip changed')
        rate, integer = wavfile.read(clip)
        require(rate == 16000 and integer.dtype == np.int16 and integer.ndim == 1
                and rate == evidence['original_rate'] and len(integer) == evidence['original_samples'], 'PCM clock changed')
        pcm = integer.astype(np.float32) / 32768
        common = gcd(rate, 22050)
        resampled = resample_poly(pcm, 22050//common, rate//common).astype(np.float32)
        require(hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() == evidence['original_pcm_sha256']
            and hashlib.sha256(resampled.astype('<f4').tobytes()).hexdigest() == evidence['model_pcm_sha256']
            and len(resampled) == evidence['model_samples'], 'Float or resampling differs')
        with np.load(folder / 'activity.npz', allow_pickle=False) as data:
            activity, pitches, times, valid = (data[k] for k in ('activations', 'pitches', 'times', 'valid'))
            require(activity.shape == (540, len(times)) and len(pitches) == 540
                and np.isfinite(activity).all(), 'Invalid activity tensor')
            expected_pitches = 69 + 12 * np.log2((22050 / 2 / 2**9) / 440) + np.arange(540) / 5
            np.testing.assert_allclose(pitches, expected_pitches, rtol=0, atol=1e-12)
            expected_frames = ((len(resampled) + plan['block_samples'] - 1) // plan['block_samples']) * plan['block_frames']
            require(len(times) == expected_frames, 'Block clock changed')
            expected_times = np.arange(len(times)) * (plan['block_samples'] / plan['block_frames']) / 22050
            np.testing.assert_allclose(times, expected_times, rtol=0, atol=1e-12)
            expected_valid = np.arange(len(times), dtype=np.int64) * plan['block_samples'] * rate \
                < len(integer) * plan['block_frames'] * 22050
            np.testing.assert_array_equal(valid, expected_valid)
            peaks = activity >= .5
            peaks[1:] &= activity[1:] > activity[:-1]
            peaks[:-1] &= activity[:-1] > activity[1:]
            np.testing.assert_array_equal(data['peaks'], peaks)
            estimated = [pitches[np.flatnonzero(peaks[:, i])].tolist() for i in range(len(times))]
        require(sha(case['labels_path']) == case['labels_sha256'], 'Reference changed')
        shared_reference[case['labels_sha256']].append(case['name'])
        prior = baseline_root / case['name'] / 'notes.json'
        require(sha(prior) == baseline[case['name']]['notes_sha256'], 'Prior notes changed')
        reference = active_notes(read(case['labels_path']), times)
        previous = active_notes(read(prior), times)
        calculated = dict(timbre_trap=score(reference, estimated, times, valid),
                          tsumugi=score(reference, previous, times, valid))
        for variant in calculated:
            require(calculated[variant] == row[variant], 'Independent scoring differs: ' + case['name'] + '/' + variant)
        rows.append(dict(name=case['name'], passed=True, **calculated))
    silence_path = args.source / 'silence/activity.npz'
    require(sha(silence_path) == report['silence']['activity_sha256'], 'Silence activations changed')
    require(report['silence']['original_samples'] == 4 * 22050
            and report['silence']['original_rate'] == 22050
            and report['silence']['model_pcm_sha256'] == hashlib.sha256(np.zeros(4 * 22050, dtype='<f4').tobytes()).hexdigest(),
            'Silence input identity differs')
    with np.load(silence_path, allow_pickle=False) as data:
        activity, pitches, times, valid = (data[k] for k in ('activations', 'pitches', 'times', 'valid'))
        require(activity.shape == (540, len(times)) and np.isfinite(activity).all(), 'Invalid silence activations')
        np.testing.assert_allclose(pitches, expected_pitches, rtol=0, atol=1e-12)
        np.testing.assert_allclose(times, np.arange(len(times)) * (plan['block_samples'] / plan['block_frames']) / 22050,
                                   rtol=0, atol=1e-12)
        np.testing.assert_array_equal(valid,
            np.arange(len(times), dtype=np.int64) * plan['block_samples'] < 4 * 22050 * plan['block_frames'])
        require(len(times) == ((4 * 22050 + plan['block_samples'] - 1) // plan['block_samples']) * plan['block_frames'],
                'Silence block count differs')
        peaks = activity >= .5
        peaks[1:] &= activity[1:] > activity[:-1]
        peaks[:-1] &= activity[:-1] > activity[1:]
        np.testing.assert_array_equal(peaks, data['peaks'])
        silence = score([set() for _ in times], [pitches[peaks[:, i]].tolist() for i in range(len(times))], times, valid)
        require(all(silence[key] == value for key, value in report['silence']['counts'].items()), 'Silence score differs')
    groups = defaultdict(list)
    for case, row in zip(cases, rows):
        groups[case['kind']].append(row)
        if case.get('instrument'):
            groups['instrument/' + case['instrument']].append(row)
    totals = {}
    for group, members in groups.items():
        totals[group] = {}
        for variant in ('timbre_trap', 'tsumugi'):
            counts = {key: sum(row[variant][key] for row in members) for key in
                ('frames', 'reference', 'predicted', 'matched', 'rest_frames', 'rest_false_active', 'polyphonic_frames')}
            r, p, m = (counts[key] for key in ('reference', 'predicted', 'matched'))
            totals[group][variant] = dict(counts, precision=m/p if p else None, recall=m/r if r else None,
                                         f1=2*m/(r+p) if r+p else None)
    require(totals == report['groups'], 'Group aggregates differ')
    require(sha(__file__) == runner_digest, 'Auditor source changed during the run')
    result = dict(source_report_sha256=sha(args.source / 'report.json'), runner_sha256=runner_digest,
        cases=35, passed=True, independent_matching='augmenting_paths', rows=rows, groups=totals, silence=silence,
        shared_reference_groups=[v for v in shared_reference.values() if len(v) > 1],
        dataset_independence='35 inputs are not independent songs; shared annotation groups are retained',
        neural_inference=False, musical_acceptance=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
