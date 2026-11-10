import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from scripts.evaluate_guitar_vocal_benchmark import (
    aggregate, evaluate, read_json, verify_clock, window_metrics, write_evaluation,
)
from scripts.probe_mapper_adapter import sha


def note(onset=.2, offset=.6, pitch=60, **extra):
    return dict(onset=onset, offset=offset, pitch=pitch, program=-1, **extra)


def prediction(*args, **kwargs):
    return note(*args, **(dict(is_drum=False) | kwargs))


def score(reference, predicted, end=1., context_end=3.):
    return window_metrics(reference, predicted, begin=0., end=end,
                          context_start=0., context_end=context_end)


def test_matching_is_maximum_one_to_one_and_preserves_unison_denominators():
    refs = [note(.04, .5), note(0., .5)]
    preds = [prediction(.02, .5), prediction(.08, .5)]
    metrics, _ = score(refs, preds)
    assert metrics['standard']['pitched_onsets']['matched'] == 2
    unisons, _ = score([note(), note()], [prediction()])
    result = unisons['standard']['pitched_onsets']
    assert (result['reference'], result['predicted'], result['matched'], result['recall']) == (2, 1, 1, .5)


def test_onset_window_keeps_wrong_drums_duplicates_and_crossing_tails():
    reference = [note(.2, 1.8)]
    predicted = [prediction(.2, 1.8), prediction(.2, 6.), prediction(.2, 1.8, is_drum=True),
                 prediction(1., 1.8), prediction(1.1, 2.)]
    metrics, scope = score(reference, predicted, context_end=2.)
    assert scope['predicted_indices'] == [0, 1, 2]
    assert scope['predicted_crosses_scoring_end'] == 3
    assert scope['predicted_crosses_context_end'] == 1 and scope['predicted_drums'] == 1
    for value in metrics['standard'].values():
        assert value['available'] and (value['reference'], value['predicted'], value['matched']) == (1, 3, 1)
        assert value['precision'] == pytest.approx(1/3)
    only_drum, _ = score([note()], [prediction(is_drum=True)])
    assert only_drum['standard']['pitched_onsets']['matched'] == 0


@pytest.mark.parametrize('offset', [3., 5.])
def test_reference_reaching_input_end_keeps_onsets_but_disables_entire_interval_case(offset):
    refs = [note(.2, offset), note(.7, .9, pitch=62)]
    preds = [prediction(.2, offset), prediction(.7, .9, pitch=62)]
    metrics, scope = score(refs, preds)
    assert scope['unobservable_reference_indices'] == [0]
    for values in metrics.values():
        assert values['pitched_onsets']['matched'] == 2
        interval = values['complete_intervals']
        assert not interval['available'] and interval['f1'] is None and interval['matched'] is None
        assert interval['reference'] == interval['predicted'] == 2
        assert interval['reason'] == 'reference_release_reaches_input_boundary'


def test_both_tolerances_report_pitch_onset_and_interval_without_program_accuracy():
    p = prediction(0., .27)
    p['program'] = 88
    metrics, _ = score([note(0., .2)], [p])
    assert metrics['standard']['pitched_onsets']['matched'] == 1
    assert metrics['standard']['complete_intervals']['matched'] == 0
    assert metrics['fixed_40_80']['complete_intervals']['matched'] == 1
    assert set(metrics['standard']) == {'pitched_onsets', 'complete_intervals'}
    longer, _ = score([note(0., 1.)], [prediction(0., 1.15)])
    assert longer['standard']['complete_intervals']['matched'] == 1
    assert longer['fixed_40_80']['complete_intervals']['matched'] == 0
    wrong_pitch, _ = score([note()], [prediction(pitch=61)])
    assert wrong_pitch['standard']['pitched_onsets']['matched'] == 0


@pytest.mark.parametrize('field,value', [('onset', float('nan')), ('offset', float('inf')),
                                        ('offset', .1), ('is_drum', 'false'), ('pitch', 60.2)])
def test_invalid_predictions_are_rejected_instead_of_silently_removed(field, value):
    p = prediction()
    p[field] = value
    with pytest.raises(ValueError):
        score([note()], [p])


def test_unavailable_aggregate_exposes_full_and_available_denominators():
    good, _ = score([note()], [prediction(), prediction()])
    unavailable, _ = score([note(offset=3.)], [prediction()])
    result = aggregate([dict(name='good', metrics=good), dict(name='boundary', metrics=unavailable)])
    onset = result['standard']['pitched_onsets']
    interval = result['standard']['complete_intervals']
    assert onset['full_cohort_available'] and onset['micro']['reference'] == 2
    assert not interval['full_cohort_available'] and interval['available_cases'] == 1
    assert interval['cohort_denominators'] == {'reference': 2, 'predicted': 3}
    assert interval['micro']['reference'] == 1 and interval['micro']['predicted'] == 2
    assert interval['unavailable_cases'][0]['name'] == 'boundary'
    assert interval['macro']['case_count'] == 1
    all_unavailable = aggregate([dict(name='boundary', metrics=unavailable)])
    assert all_unavailable['standard']['complete_intervals']['micro'] is None


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


@pytest.fixture
def benchmark(tmp_path):
    """A complete synthetic 60/40 cohort; no external data or inference."""
    root = tmp_path/'benchmark'
    root.mkdir()
    audio = root/'source.wav'
    wavfile.write(audio, 8000, np.zeros(24000, dtype=np.int16))
    label_paths = {}
    for name, rows in {'normal': [note()], 'boundary': [note(offset=3.)], 'A2': [note(pitch=61)]}.items():
        path = root/(name+'.json')
        save(path, rows)
        label_paths[name] = path
    cases = []
    for index in range(100):
        guitar = index < 60
        refs = {}
        for annotator in (('strings',) if guitar else ('A1', 'A2')):
            key = 'boundary' if index == 0 else 'A2' if annotator == 'A2' else 'normal'
            path = label_paths[key]
            refs[annotator] = dict(path=str(path), sha256=sha(path), count=1, scored_count=1,
                offsets_beyond_audio=int(key == 'boundary'), scored_offsets_beyond_audio=int(key == 'boundary'),
                original_sha256='f'*64)
        cases.append(dict(name=f'case-{index:03}', dataset='GuitarSet' if guitar else 'vocadito',
            audio=str(audio), audio_sha256=sha(audio), sample_rate=8000, samples=24000, decoded_dtype='int16',
            duration_seconds=3., start_ms=0, seconds=1., inference_scope='full_original_audio', references=refs))
    manifest = dict(revision=1, cases=cases, protocol=['Synthetic fixture'])
    save(root/'manifest.json', manifest)
    application = root/'application'
    application.mkdir()
    plan = dict(manifest_sha256=sha(root/'manifest.json'), model_sha256='a'*64,
        audio_code_sha256='b'*64, runner_sha256='c'*64, intra_threads=2, inter_threads=1,
        normalization=False, inference_scope='full_original_audio')
    save(application/'plan.json', plan)
    case_reports = {}
    for case in cases:
        folder = application/case['name']
        folder.mkdir()
        save(folder/'notes.json', [prediction()])
        np.savez(folder/'activations.npz', times=np.array([0., .01]))
        report = dict(identity=dict(plan, audio=case['audio'], audio_sha256=case['audio_sha256']),
            context_start_seconds=0., context_end_seconds=3., resampled_duration_seconds=3., notes=1,
            notes_sha256=sha(folder/'notes.json'), activations_sha256=sha(folder/'activations.npz'))
        save(folder/'report.json', report)
        case_reports[case['name']] = dict(report_sha256=sha(folder/'report.json'),
            notes_sha256=report['notes_sha256'], activations_sha256=report['activations_sha256'])
    save(application/'completed.json', dict(identity=plan, cases=100,
        plan_sha256=sha(application/'plan.json'), case_reports=case_reports))
    return root


def test_complete_cohort_scores_both_annotators_separately_and_audits_partial_intervals(benchmark):
    output = benchmark/'result.json'
    result = write_evaluation(benchmark, output)
    assert result['complete'] and result['case_count'] == 100 and result['annotation_case_count'] == 140
    assert len(result['rows']) == 100
    assert set(result['groups']) == {'guitar_strings', 'vocals_A1', 'vocals_A2'}
    guitar = result['groups']['guitar_strings']['standard']
    assert guitar['pitched_onsets']['available_cases'] == 60
    assert guitar['complete_intervals']['available_cases'] == 59
    assert guitar['complete_intervals']['cohort_denominators'] == {'reference': 60, 'predicted': 60}
    assert guitar['complete_intervals']['micro']['reference'] == 59
    assert result['groups']['vocals_A1']['standard']['pitched_onsets']['micro']['f1'] == 1
    assert result['groups']['vocals_A2']['standard']['pitched_onsets']['micro']['f1'] == 0
    assert read_json(output) == result
    with pytest.raises(ValueError, match='Preserve'):
        write_evaluation(benchmark, output)


@pytest.mark.parametrize('target', ['manifest', 'plan', 'report', 'notes', 'labels', 'activations', 'audio'])
def test_hash_tampering_is_rejected_and_no_partial_summary_is_written(benchmark, target):
    folder = benchmark/'application/case-099'
    paths = dict(manifest=benchmark/'manifest.json', plan=benchmark/'application/plan.json',
                 report=folder/'report.json', notes=folder/'notes.json', labels=benchmark/'A2.json',
                 activations=folder/'activations.npz', audio=benchmark/'source.wav')
    with paths[target].open('ab') as destination:
        destination.write(b' ')
    output = benchmark/'not-created/result.json'
    with pytest.raises(ValueError, match='Hash mismatch'):
        write_evaluation(benchmark, output)
    assert not output.exists() and not output.parent.exists()


@pytest.mark.parametrize('corruption', ['missing_case', 'wrong_total', 'duplicate_name', 'missing_A2', 'manifest_missing_case'])
def test_incomplete_cohorts_and_missing_annotators_cannot_report_normal_summary(benchmark, corruption):
    if corruption in ('missing_case', 'wrong_total'):
        path = benchmark/'application/completed.json'
        data = read_json(path)
        if corruption == 'missing_case':
            del data['case_reports']['case-099']
        else:
            data['cases'] = 99
    else:
        path = benchmark/'manifest.json'
        data = read_json(path)
        if corruption == 'duplicate_name':
            data['cases'][-1]['name'] = data['cases'][0]['name']
        elif corruption == 'missing_A2':
            del data['cases'][-1]['references']['A2']
        else:
            data['cases'].pop()
    save(path, data)
    with pytest.raises(ValueError):
        evaluate(benchmark)


def test_hash_consistent_but_retimed_report_is_rejected(benchmark):
    folder = benchmark/'application/case-099'
    report = read_json(folder/'report.json')
    report['context_start_seconds'] = -.0225
    save(folder/'report.json', report)
    completed_path = benchmark/'application/completed.json'
    completed = read_json(completed_path)
    completed['case_reports']['case-099']['report_sha256'] = sha(folder/'report.json')
    save(completed_path, completed)
    with pytest.raises(ValueError, match='inference clock'):
        evaluate(benchmark)


def test_clock_checks_original_samples_scoring_window_and_resample_rounding(benchmark):
    case = read_json(benchmark/'manifest.json')['cases'][0]
    report = read_json(benchmark/'application/case-000/report.json')
    assert verify_clock(case, report) == (0., 1., 0., 3.)
    for key, value in [('samples', 24001), ('duration_seconds', 3.01), ('seconds', .9)]:
        bad = copy.deepcopy(case)
        bad[key] = value
        with pytest.raises(ValueError):
            verify_clock(bad, report)
    bad_report = dict(report, resampled_duration_seconds=3.001)
    with pytest.raises(ValueError, match='Resampled clock'):
        verify_clock(case, bad_report)


def test_nonfinite_or_duplicate_json_fields_are_rejected(tmp_path):
    path = tmp_path/'invalid.json'
    for value in ('{"pitch": NaN}', '{"pitch": 60, "pitch": 61}'):
        path.write_text(value, encoding='utf-8')
        with pytest.raises(ValueError):
            read_json(path)
