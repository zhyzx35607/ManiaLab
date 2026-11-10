"""Audit and score the complete frozen guitar/vocal baseline without inference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_instrument_benchmark as instrument_evaluation
from scripts.probe_mapper_adapter import sha


EXPECTED_COUNTS = {'GuitarSet': 60, 'vocadito': 40}
TOLERANCES = {
    'standard': '50ms onset; offset max(50ms,20% reference duration); exact semitone',
    'fixed_40_80': '40ms onset; 80ms offset; exact semitone',
}
METRICS = ('pitched_onsets', 'complete_intervals')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=invalid)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def verified_hash(path, expected):
    require(digest(expected), f'Missing/invalid SHA256 for {path}')
    actual = sha(path)
    require(actual == expected, f'Hash mismatch: {path}')
    return actual


def validate_notes(notes, *, reference):
    require(isinstance(notes, list), 'Notes must be a list')
    for index, note in enumerate(notes):
        require(isinstance(note, dict), f'Invalid note {index}')
        require(all(finite(note.get(k)) for k in ('onset', 'offset', 'pitch', 'program')),
                f'Nonfinite/missing note fields at {index}')
        require(note['onset'] >= 0 and note['offset'] > note['onset'], f'Invalid interval at {index}')
        require(integer(note['pitch']) and 0 <= note['pitch'] <= 127, f'Invalid MIDI pitch at {index}')
        require(integer(note['program']) and -1 <= note['program'] <= 128, f'Invalid program at {index}')
        if reference:
            require(note['program'] == -1, 'Reference does not certify a GM program; expected -1')
            if 'midi_continuous' in note:
                require(finite(note['midi_continuous']) and round(note['midi_continuous']) == note['pitch'],
                        f'Rounded reference pitch mismatch at {index}')
        else:
            require(type(note.get('is_drum')) is bool, f'Missing/invalid is_drum at {index}')


def window_metrics(reference, predicted, *, begin, end, context_start, context_end):
    """Select by onset only; unobserved reference tails invalidate the whole interval metric."""
    require(all(finite(x) for x in (begin, end, context_start, context_end)) and
            0 <= context_start <= begin < end <= context_end, 'Invalid scoring/context clock')
    validate_notes(reference, reference=True)
    validate_notes(predicted, reference=False)
    ref_indices = [i for i, n in enumerate(reference) if begin <= n['onset'] < end]
    pred_indices = [i for i, n in enumerate(predicted) if begin <= n['onset'] < end]
    refs = [reference[i] for i in ref_indices]
    preds = [predicted[i] for i in pred_indices]
    unobserved = [i for i in ref_indices if reference[i]['offset'] >= context_end]
    metrics = {}
    for name, strict in (('standard', False), ('fixed_40_80', True)):
        onset = dict(available=True, **instrument_evaluation.match(refs, preds, strict=strict))
        if unobserved:
            intervals = dict(available=False, reason='reference_release_reaches_input_boundary',
                             reference=len(refs), predicted=len(preds), matched=None,
                             precision=None, recall=None, f1=None,
                             unobservable_reference_indices=unobserved)
        else:
            intervals = dict(available=True, **instrument_evaluation.match(refs, preds, offset=True, strict=strict))
        metrics[name] = dict(pitched_onsets=onset, complete_intervals=intervals)
    scope = dict(begin_seconds=begin, end_seconds=end, context_start_seconds=context_start,
                 context_end_seconds=context_end, clock='original_audio_seconds',
                 reference_total=len(reference), predicted_total=len(predicted),
                 reference_indices=ref_indices, predicted_indices=pred_indices,
                 reference=len(refs), predicted=len(preds),
                 unobservable_reference_offsets=len(unobserved),
                 unobservable_reference_indices=unobserved,
                 reference_crosses_scoring_end=sum(n['offset'] >= end for n in refs),
                 predicted_crosses_scoring_end=sum(n['offset'] >= end for n in preds),
                 predicted_crosses_context_end=sum(n['offset'] >= context_end for n in preds),
                 predicted_drums=sum(n['is_drum'] for n in preds),
                 crossing_and_drum_predictions_retained=True)
    return metrics, scope


def aggregate(rows):
    """Report available-case scores with explicit full-cohort and excluded denominators."""
    result = {}
    for tolerance in TOLERANCES:
        result[tolerance] = {}
        for metric in METRICS:
            all_values = [(row['name'], row['metrics'][tolerance][metric]) for row in rows]
            available = [value for _, value in all_values if value['available']]
            missing = [dict(name=name, reason=value['reason'], reference=value['reference'],
                            predicted=value['predicted']) for name, value in all_values if not value['available']]
            n_ref = sum(value['reference'] for value in available)
            n_pred = sum(value['predicted'] for value in available)
            matched = sum(value['matched'] for value in available)
            micro = dict(reference=n_ref, predicted=n_pred, matched=matched,
                         precision=matched/n_pred if n_pred else 0.,
                         recall=matched/n_ref if n_ref else 0.,
                         f1=2*matched/(n_ref+n_pred) if n_ref+n_pred else 0.) if available else None
            macro = dict(case_count=len(available), **{
                key: sum(value[key] for value in available)/len(available)
                for key in ('precision', 'recall', 'f1')}) if available else None
            result[tolerance][metric] = dict(
                aggregation_scope='available_cases_only', full_cohort_available=not missing,
                total_cases=len(rows), available_cases=len(available), unavailable_cases=missing,
                cohort_denominators=dict(reference=sum(v['reference'] for _, v in all_values),
                                         predicted=sum(v['predicted'] for _, v in all_values)),
                micro=micro, macro=macro)
    return result


def verify_manifest(manifest):
    require(isinstance(manifest, dict) and manifest.get('revision') == 1, 'Unsupported manifest')
    cases = manifest.get('cases')
    require(isinstance(cases, list) and len(cases) == 100, 'Require all 100 frozen cases')
    require(all(isinstance(c, dict) for c in cases), 'Invalid manifest case')
    require(Counter(c.get('dataset') for c in cases) == EXPECTED_COUNTS, 'Require 60 GuitarSet and 40 vocadito cases')
    names = [c.get('name') for c in cases]
    require(all(isinstance(n, str) and n and '/' not in n and '\\' not in n and n not in ('.', '..')
                for n in names) and len(set(names)) == 100, 'Invalid/duplicate case names')
    for case in cases:
        expected = {'strings'} if case['dataset'] == 'GuitarSet' else {'A1', 'A2'}
        require(isinstance(case.get('references'), dict) and set(case['references']) == expected,
                f'Missing/extra annotation: {case["name"]}')
    return cases


def verify_clock(case, report):
    import numpy as np
    from scipy.io import wavfile
    rate, wave = wavfile.read(case['audio'], mmap=True)
    require(integer(case.get('sample_rate')) and integer(case.get('samples')) and
            rate == case['sample_rate'] and len(wave) == case['samples'] and
            wave.ndim == 1 and wave.dtype == np.int16 and case.get('decoded_dtype') == 'int16',
            f'Original source format/length changed: {case["name"]}')
    duration = len(wave)/rate
    del wave
    require(finite(case.get('duration_seconds')) and abs(case['duration_seconds']-duration) <= 1e-10,
            f'Frozen source clock differs: {case["name"]}')
    require(case.get('inference_scope') == 'full_original_audio' and case.get('start_ms') == 0 and
            finite(case.get('seconds')) and duration > 2 and
            abs(case['seconds']-min(20., duration-2.)) <= 1e-10, 'Frozen scoring-window rule differs')
    require(finite(report.get('context_start_seconds')) and report['context_start_seconds'] == 0 and
            finite(report.get('context_end_seconds')) and abs(report['context_end_seconds']-duration) <= 1e-10,
            f'Reported inference clock differs: {case["name"]}')
    # resample_poly uses ceil(original_samples*target_rate/original_rate).
    resampled = ((case['samples']*22050+rate-1)//rate)/22050
    require(finite(report.get('resampled_duration_seconds')) and
            abs(report['resampled_duration_seconds']-resampled) <= 1e-10,
            f'Resampled clock differs: {case["name"]}')
    return 0., case['seconds'], 0., duration


def evaluate(root, predictions=None):
    root = Path(root)
    predictions = Path(predictions) if predictions is not None else root/'application'
    manifest_path = root/'manifest.json'
    manifest = read_json(manifest_path)
    cases = verify_manifest(manifest)
    plan_path, completed_path = predictions/'plan.json', predictions/'completed.json'
    plan, completed = read_json(plan_path), read_json(completed_path)
    require(isinstance(plan, dict) and isinstance(completed, dict), 'Invalid prediction identity')
    manifest_hash = verified_hash(manifest_path, plan.get('manifest_sha256'))
    plan_hash = verified_hash(plan_path, completed.get('plan_sha256'))
    require(completed.get('identity') == plan and completed.get('cases') == 100,
            'Incomplete or inconsistent prediction completion')
    require(plan.get('normalization') is False and plan.get('inference_scope') == 'full_original_audio',
            'Prediction identity changes original-audio protocol')
    require(all(digest(plan.get(key)) for key in ('model_sha256', 'audio_code_sha256', 'runner_sha256')),
            'Missing model/code/runner identity')
    require(plan.get('intra_threads') == 2 and plan.get('inter_threads') == 1, 'Prediction thread plan differs')
    case_reports = completed.get('case_reports')
    require(isinstance(case_reports, dict) and set(case_reports) == {c['name'] for c in cases},
            'Completion must cover exactly all 100 case reports')
    rows = []
    groups = {'guitar_strings': [], 'vocals_A1': [], 'vocals_A2': []}
    for case in cases:
        name = case['name']
        folder = predictions/name
        linked = case_reports[name]
        require(isinstance(linked, dict), f'Invalid completion hashes: {name}')
        report_path, notes_path = folder/'report.json', folder/'notes.json'
        report_hash = verified_hash(report_path, linked.get('report_sha256'))
        report = read_json(report_path)
        require(isinstance(report, dict) and report.get('identity') == dict(plan, audio=case['audio'],
                audio_sha256=case['audio_sha256']), f'Prediction input/plan identity mismatch: {name}')
        audio_hash = verified_hash(case['audio'], case['audio_sha256'])
        notes_hash = verified_hash(notes_path, report.get('notes_sha256'))
        require(notes_hash == linked.get('notes_sha256'), f'Completion notes hash differs: {name}')
        activations_hash = verified_hash(folder/'activations.npz', report.get('activations_sha256'))
        require(activations_hash == linked.get('activations_sha256'), f'Completion activation hash differs: {name}')
        predicted = read_json(notes_path)
        validate_notes(predicted, reference=False)
        require(integer(report.get('notes')) and report['notes'] == len(predicted), f'Prediction count differs: {name}')
        begin, end, context_start, context_end = verify_clock(case, report)
        annotations = {}
        for annotator, entry in case['references'].items():
            require(isinstance(entry, dict), f'Invalid label identity: {name}/{annotator}')
            labels_hash = verified_hash(entry['path'], entry.get('sha256'))
            reference = read_json(entry['path'])
            metrics, scope = window_metrics(reference, predicted, begin=begin, end=end,
                                            context_start=context_start, context_end=context_end)
            expected_counts = dict(count=len(reference), scored_count=scope['reference'],
                offsets_beyond_audio=sum(n['offset'] >= context_end for n in reference),
                scored_offsets_beyond_audio=scope['unobservable_reference_offsets'])
            require(all(integer(entry.get(k)) and entry[k] == v for k, v in expected_counts.items()),
                    f'Frozen label denominators differ: {name}/{annotator}')
            annotations[annotator] = dict(metrics=metrics, interval_scope=scope, labels_sha256=labels_hash,
                                         labels_path=entry['path'], original_sha256=entry.get('original_sha256'))
            group = 'guitar_strings' if case['dataset'] == 'GuitarSet' else 'vocals_'+annotator
            groups[group].append(dict(name=name, metrics=metrics))
        rows.append(dict(name=name, dataset=case['dataset'], metadata=case.get('metadata', {}),
                         references=annotations, identity=report['identity'],
                         evidence=dict(audio_sha256=audio_hash, report_sha256=report_hash,
                                       notes_sha256=notes_hash, activations_sha256=activations_hash)))
    # No caller receives a partial normal summary if any case above failed.
    return dict(revision=1, complete=True, case_count=len(rows), annotation_case_count=140,
                runner_sha256=sha(__file__), matcher_sha256=sha(instrument_evaluation.__file__),
                manifest_sha256=manifest_hash, plan_sha256=plan_hash, completed_sha256=sha(completed_path),
                prediction_identity=plan, rows=rows, groups={name: aggregate(values) for name, values in groups.items()},
                tolerances=TOLERANCES, protocol=manifest.get('protocol', []), enabled_in_application=False,
                limits=[
                    'A1 and A2 are independent complete groups; neither is selected as a more favorable answer.',
                    'No GM accuracy is computed: references have program=-1.',
                    'All predictions with onsets in the scoring window enter denominators, including drums, duplicates and crossing tails.',
                    'An unobservable reference tail makes the entire case interval metric unavailable; onset scoring remains available.',
                    'Aggregate micro/macro scores use only explicitly listed available cases; full-cohort counts and excluded denominators remain visible.',
                    'Fixed 40/80ms is not universally stricter: 80ms exceeds the standard minimum 50ms tail tolerance.',
                    'Rounded semitone labels retain tuning and annotation uncertainty. Physical intervals are not authored LN releases.',
                    'These public recordings may have been used upstream for training; isolated-source scores do not establish mixed-song quality.'])


def write_evaluation(root, output, predictions=None):
    output = Path(output)
    require(not output.exists(), 'Preserve earlier evaluation; use a new output')
    result = evaluate(root, predictions)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as destination:
        destination.write(text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('test-data/guitar-vocal-benchmark-v3'))
    parser.add_argument('--predictions', type=Path, help='Baseline output folder; defaults to ROOT/application')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = write_evaluation(args.root, args.output, args.predictions)
    print(json.dumps(dict(case_count=result['case_count'], groups=result['groups']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
