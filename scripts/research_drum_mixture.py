"""Frozen-candidate additive drum dictionary with calibration-only selection."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from manialab.timbre_mixture import attack_power, prototypes, decompose
from scripts.probe_mapper_adapter import sha, event_similarity
from scripts.research_source_attack_classes import CLASSES, audio_features, fit_logistic, drum_event_counts, evaluate

TRAIN_END, CALIBRATION_END = 162340, 188286


def selected_matches(human, predicted):
    if not human or not predicted:
        return set()
    delta = abs(np.asarray(human)[:, None] - np.asarray(predicted)[None, :])
    costs = np.c_[np.where(delta <= 40, delta / 40 / (len(human) + 1), 1e6), np.ones((len(human), len(human)))]
    left, right = linear_sum_assignment(costs)
    return {int(i) for i, j in zip(left, right) if j < len(predicted) and delta[i, j] <= 40}


def overlap_recalls(reference, times, scores, thresholds, start, end):
    at_time = defaultdict(set)
    for time, kind in reference:
        at_time[time].add(kind)
    result = {}
    for category, kind in enumerate(CLASSES):
        human = sorted(t for t, c in reference if c == kind and start <= t < end)
        predicted = [float(t) for t, score in zip(times, scores[:, category]) if start <= t < end and score >= thresholds[category]]
        matched = selected_matches(human, predicted)
        assert len(matched) == event_similarity([(t,) for t in human], [(t,) for t in predicted], [40])['matched']
        result[kind] = {}
        for name, overlap in [('author_single_class', False), ('author_simultaneous_classes', True)]:
            indices = {i for i, t in enumerate(human) if (len(at_time[t]) > 1) == overlap}
            result[kind][name] = dict(reference=len(indices), matched=len(indices & matched),
                recall=len(indices & matched) / len(indices) if indices else None)
    return result


def choose_threshold(times, scores, truth, thresholds):
    options = []
    for threshold in thresholds:
        events = [(float(t),) for t, score in zip(times, scores) if score >= threshold]
        options.append(dict(threshold=float(threshold), score=event_similarity(truth, events, [40])))
    return max(options, key=lambda row: (row['score']['f1'], row['threshold'])), options


def linear_control(samples, rate, times, reference, *, extra_features=None):
    features = audio_features(samples, rate, times)
    if extra_features is not None:
        extra_features=np.asarray(extra_features,float)
        if extra_features.ndim!=2 or len(extra_features)!=len(features) or not np.isfinite(extra_features).all():
            raise ValueError('Invalid frozen drum evidence')
        features=np.c_[features,extra_features]
    chart_times = times - 22.5
    fit = chart_times < TRAIN_END
    calibration = (chart_times >= TRAIN_END) & (chart_times < CALIBRATION_END)
    mean = features[fit].mean(axis=0)
    scale = np.maximum(.05, features[fit].std(axis=0))
    features = np.c_[np.clip((features - mean) / scale, -8, 8), np.ones(len(times))]
    weights, thresholds, selections = [], [], []
    for kind in CLASSES:
        labels = np.zeros(len(times))
        ambiguous = np.zeros(len(times), bool)
        for t, label in reference:
            if label != kind or t >= TRAIN_END:
                continue
            delta = abs(chart_times - t)
            nearest = int(delta.argmin())
            if delta[nearest] <= 40:
                labels[nearest] = 1
                ambiguous[delta <= 40] = True
        valid = fit & (~ambiguous | (labels > 0))
        coefficients = fit_logistic(features[valid], labels[valid])
        probability = expit(features @ coefficients)
        truth = [(t,) for t, label in reference if label == kind and TRAIN_END <= t < CALIBRATION_END]
        best, options = choose_threshold(chart_times[calibration], probability[calibration], truth, np.linspace(.1, .9, 17))
        weights.append(coefficients)
        thresholds.append(best['threshold'])
        selections.append(dict(kind=kind, chosen=best, options=options))
    return dict(weights=np.asarray(weights), mean=mean, scale=scale, thresholds=np.asarray(thresholds)), expit(features @ np.asarray(weights).T), selections


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier timbre experiments')
    root = args.root
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    source = root / 'raw-pcm-audit-v1/drums.wav'
    reference_path = root / 'reference.osu'
    candidates_path = root / 'detector-baseline/drums-notes.json'
    audit_path = root / 'raw-pcm-audit-v1/report.json'
    audit = read(audit_path)
    manifest = read(root / 'manifest.json')
    drum = next(row for row in audit['rows'] if row['role'] == 'drums')
    original = next(row for row in manifest['audio'] if row['id'] == 'drums')
    if (sha(source) != drum['raw_float_sha256'] or sha(root / original['path']) != drum['original_sha256']
            or sha(reference_path) != manifest['charts']['reference']['chart_sha256']):
        raise ValueError('Benchmark identity changed')
    rate, samples = wavfile.read(source)
    if rate != 22050 or samples.ndim != 1 or samples.dtype != np.float32:
        raise ValueError('Expected original floating-point PCM')
    reference = list(drum_event_counts(parse_osu(reference_path.read_text(encoding='utf-8-sig'))['notes']))
    times = np.asarray([n['start'] for n in read(candidates_path)], float)
    chart_times = times - 22.5
    fit = (times + 140 < TRAIN_END) & (times >= 45)
    calibration = (chart_times >= TRAIN_END) & (chart_times < CALIBRATION_END)
    features = attack_power(samples, rate, times)
    average = features[fit].mean(axis=0)
    scale = np.sqrt(np.maximum(average, max(float(average.max()) * 1e-4, 1e-12)))
    features /= scale
    annotations = [set() for _ in times]
    for t, label in reference:
        if t >= TRAIN_END:
            continue
        delta = abs(chart_times - t)
        for index in np.flatnonzero(delta <= 40):
            annotations[index].add(label)
    examples = {}
    for kind in CLASSES:
        indices = set()
        for t, label in reference:
            if label != kind or t >= TRAIN_END:
                continue
            delta = abs(chart_times - t)
            index = int(delta.argmin())
            if delta[index] <= 40 and fit[index] and annotations[index] == {kind}:
                indices.add(index)
        examples[kind] = sorted(indices)
    args.output.mkdir(parents=True)
    identity = dict(source_audio_sha256=sha(source), original_sha256=drum['original_sha256'],
        reference_sha256=sha(reference_path), candidates_sha256=sha(candidates_path), audit_sha256=sha(audit_path),
        runner_sha256=sha(__file__), model_code_sha256=sha('manialab/timbre_mixture.py'),
        protocol_sha256=sha('docs/drum-mixture-protocol.md'), enabled_in_application=False)
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    predictions, dictionaries, residuals = {}, {}, {}
    for count in (1, 4, 8):
        atoms = np.concatenate([prototypes(features[examples[kind]], count) for kind in CLASSES])
        class_ids = np.concatenate([np.full(min(count, len(examples[kind])), i, int) for i, kind in enumerate(CLASSES)])
        scores, residual, coefficients = decompose(features, atoms, class_ids, len(CLASSES))
        np.savez_compressed(args.output / f'dictionary-{count}.npz', atoms=atoms, class_ids=class_ids,
            scale=scale, evidence=scores, residual=residual, coefficients=coefficients, times=times)
        predictions[count], dictionaries[count], residuals[count] = scores, atoms, residual
        print(json.dumps(dict(stage='dictionary',templates_per_class=count, examples={k:len(v) for k,v in examples.items()})), flush=True)
    scores = np.zeros((len(times), 3))
    thresholds, selections = [], []
    for category, kind in enumerate(CLASSES):
        truth = [(t,) for t, label in reference if label == kind and TRAIN_END <= t < CALIBRATION_END]
        options = []
        for count in predictions:
            _, trials = choose_threshold(chart_times[calibration], predictions[count][calibration, category], truth, np.arange(1, 46) * .02)
            options.extend(dict(templates=count, **row) for row in trials)
        best = max(options, key=lambda row: (row['score']['f1'], row['threshold'], -row['templates']))
        scores[:, category] = predictions[best['templates']][:, category]
        thresholds.append(best['threshold'])
        selections.append(dict(kind=kind, chosen=best, calibration=options))
    linear, linear_scores, linear_selection = linear_control(samples, rate, times, reference)
    np.savez_compressed(args.output / 'linear-float-control.npz', **linear, evidence=linear_scores, times=times)
    np.savez_compressed(args.output / 'selected-evidence.npz', evidence=scores, thresholds=thresholds, times=times)
    segments = {}
    for name, start, end in [('fit', 0, TRAIN_END), ('calibration', TRAIN_END, CALIBRATION_END), ('later_sections', CALIBRATION_END, len(samples) / rate * 1000)]:
        segments[name] = {}
        for label, evidence, cuts in [('mixture', scores, thresholds), ('linear_float', linear_scores, linear['thresholds'])]:
            segments[name][label] = dict(scores=evaluate(reference, chart_times, evidence, cuts, start, end),
                author_overlap=overlap_recalls(reference, chart_times, evidence, cuts, start, end))
    events = [dict(start_ms=float(t), kind=kind, evidence=float(scores[i, c])) for i, t in enumerate(chart_times)
        for c, kind in enumerate(CLASSES) if scores[i, c] >= thresholds[c]]
    (args.output / 'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    (args.output / 'selection.json').write_text(json.dumps(dict(mixture=selections, linear_float=linear_selection), indent=2), encoding='utf-8')
    report = dict(identity=identity, segments=segments, selected=[s['chosen'] for s in selections],
        fit_example_indices=examples, events_sha256=sha(args.output / 'events.json'), selection_sha256=sha(args.output / 'selection.json'),
        notice='Same-song development only. The fourth authored column is heterogeneous; annotation silence is not physical silence. Component evidence is not a probability.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(later=segments['later_sections'], chosen=report['selected'])), flush=True)


if __name__ == '__main__':
    main()
