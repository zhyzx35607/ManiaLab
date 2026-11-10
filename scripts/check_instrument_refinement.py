"""Evaluate fixed timbre refinement on the existing 35 URMP recordings.

Original detections remain unchanged. All inference completes before reading
the note labels; broad instrument-family scores are separate from onset scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_mega53_separation import save, sha
from scripts.check_harmonica_transcription import BASELINE, BASELINE_SHA, MANIFEST, MANIFEST_SHA
from scripts.evaluate_instrument_benchmark import window_metrics
from scripts.research_instrument_refinement import CHECKPOINT_SHA256
from scripts.research_tsumugi_transcription import MODEL_REVISION, SOURCE_COMMIT, verify_source

SOURCE = ROOT/'test-data/other-instrument-model-survey-2026-11-10/tsumugi'
WEIGHT = ROOT/'test-data/tsumugi-instrument-refinement-v1/best_instrument_refinement.pth'
FIRST = ROOT/'test-data/instrument-refinement-first-v1'
FIRST_SHA = 'aa97000caa1d80be917e371bf376e998b191aee1a910043dc5d11db3434c71ce'
ADAPTER = ROOT/'scripts/research_instrument_refinement.py'
TAXONOMY = SOURCE/'instrument_agnostic_amt/taxonomy/instrument_merge.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def family_mapping(taxonomy):
    classes = sorted(taxonomy['labels']) + ['melody', 'vocal_harmony', 'wind_chimes']
    mapping = {}
    for name, info in taxonomy['labels'].items():
        for member in info['members']:
            program = member.get('program_number')
            if program is not None:
                assert program not in mapping, 'Ambiguous taxonomy program'
                mapping[program] = classes.index(name)
    return classes, mapping


def family_metrics(reference, predicted, predicted_classes, mapping, context_end):
    """Use taxonomy IDs only in temporary scoring records, never as MIDI programs."""
    if len(predicted) != len(predicted_classes):
        raise ValueError('Every original prediction requires one class decision')
    if any(n['program'] not in mapping for n in reference):
        raise ValueError('Unknown reference program; do not silently assign a family')
    refs = [dict(n, program=mapping[n['program']]) for n in reference]
    preds = [dict(n, program=family) for n, family in zip(predicted, predicted_classes)]
    return window_metrics(refs, preds, begin=0, end=30, context_start=0, context_end=context_end)


def verified_result(folder, source_folder, source_files, classes):
    report = read(folder/'report.json')
    identity = report['identity']
    original = read(source_folder/'report.json')
    assert identity['source_commit'] == SOURCE_COMMIT and identity['source_files'] == source_files
    assert identity['model_revision'] == MODEL_REVISION and identity['checkpoint_sha256'] == CHECKPOINT_SHA256
    assert identity['runner_sha256'] == sha(ADAPTER)
    assert identity['original_report_sha256'] == sha(source_folder/'report.json')
    assert identity['original_identity'] == original['identity']
    settings = identity['settings']
    assert all(settings[k] == v for k, v in dict(window_seconds=8., stride_seconds=4.,
        window_batch_size=1, cluster_distance=.25, min_cluster_notes=2, global_logit_weight=0.,
        stem_name=None, use_stem_mask=False, intra_threads=2, inter_threads=1, seed=42,
        amp=False, normalization=False, prior_classes_used=False).items())
    for name, digest in report['evidence_sha256'].items():
        assert sha(folder/name) == digest
    decisions = read(folder/'decisions.json')
    events = read(source_folder/'notes.json')
    assert len(decisions) == len(events) == report['original_notes']
    for index, (event, decision) in enumerate(zip(events, decisions)):
        assert decision['original_index'] == index and decision['original_event'] == event
        if event['is_drum']:
            assert decision['decision'] == 'unsupported_drum_prediction'
        else:
            assert decision['decision'] == 'classified' and decision['observations'] > 0
            assert decision['refined_class'] == classes[decision['refined_class_id']]
            assert decision['ungrouped_class'] in classes
    return report, decisions


def run(output):
    assert not output.exists(), 'Preserve completed or partial comparisons'
    assert sha(MANIFEST) == MANIFEST_SHA and sha(BASELINE/'comparison.json') == BASELINE_SHA
    assert sha(WEIGHT) == CHECKPOINT_SHA256 and sha(FIRST/'report.json') == FIRST_SHA
    manifest, baseline = read(MANIFEST), read(BASELINE/'comparison.json')
    assert len(manifest['cases']) == len(baseline['rows']) == 35
    source_files = verify_source(SOURCE)
    classes, mapping = family_mapping(read(TAXONOMY))
    inputs, cases = {}, []
    for case, row in zip(manifest['cases'], baseline['rows']):
        assert case['name'] == row['name']
        source_folder = BASELINE/case['name']
        original = read(source_folder/'report.json')
        assert sha(source_folder/'report.json') == row['report_sha256']
        for name, key in [('notes.json', 'notes_sha256'), ('raw-notes.json', 'raw_notes_sha256')]:
            assert sha(source_folder/name) == original[key]
            inputs[str(source_folder/name)] = original[key]
        audio = Path(original['identity']['audio'])
        assert sha(audio) == row['clip_sha256'] == original['identity']['audio_sha256']
        inputs[str(source_folder/'report.json')] = row['report_sha256']
        inputs[str(audio)] = row['clip_sha256']
        cases.append(dict(name=case['name'], kind=case['kind'], instrument=case.get('instrument'),
            context_end_seconds=original['context_end_seconds'], original_folder=str(source_folder),
            result_folder=str(FIRST if case['name'] == '12-1-vn' else output/case['name']),
            reused=case['name'] == '12-1-vn'))
    plan = dict(runner_sha256=sha(__file__), adapter_sha256=sha(ADAPTER), source_files=source_files,
        checkpoint_sha256=CHECKPOINT_SHA256, baseline_sha256=BASELINE_SHA, manifest_sha256=MANIFEST_SHA,
        taxonomy_sha256=sha(TAXONOMY), classes=classes, program_to_family=mapping,
        matcher_sha256=sha(ROOT/'scripts/evaluate_instrument_benchmark.py'),
        cached_first_report_sha256=FIRST_SHA, cases=cases, inputs=inputs,
        settings='Pinned adapter defaults: 8s windows, 4s stride, cluster distance .25, minimum 2 notes.',
        scoring_seconds=[0, 30], original_onsets_offsets_pitches_preserved=True,
        models_parallel=False, thresholds_tuned=False, labels_read_before_inference=False,
        actual_instrument_scope=['violin', 'viola', 'cello', 'double_bass', 'saxophone',
                                 'trumpet', 'horn', 'trombone'],
        family_metrics_scope='strings / sax / brass; cannot distinguish individual players or string subtypes',
        enabled_in_application=False, musical_acceptance=False)
    save(output/'plan.json', plan)
    shutil_files = [Path(__file__), ADAPTER, ROOT/'scripts/evaluate_instrument_benchmark.py']
    for source in shutil_files:
        target = output/'source'/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    reports = {}
    for index, case in enumerate(cases):
        print(json.dumps(dict(stage='refinement', index=index+1, total=35,
                              case=case['name'], reused=case['reused'])), flush=True)
        folder = Path(case['result_folder'])
        if not case['reused']:
            with (output/(case['name']+'.log')).open('x', encoding='utf-8') as log:
                subprocess.run([str(ROOT/'.venv-audio/Scripts/python.exe'), '-B', str(ADAPTER),
                    '--input', case['original_folder'], '--output', str(folder), '--source', str(SOURCE),
                    '--checkpoint', str(WEIGHT), '--dependencies', str(ROOT/'test-data/rosvot-runtime')],
                    cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT,
                    env=os.environ | {'PYTHONUTF8':'1', 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2',
                                      'HF_HUB_OFFLINE':'1'})
        report, _ = verified_result(folder, Path(case['original_folder']), source_files, classes)
        reports[case['name']] = report
    save(output/'inference-completed.json', dict(reports={c['name']:sha(Path(c['result_folder'])/'report.json')
        for c in cases}, reused=1, newly_inferred=34, labels_read=False))
    rows, groups = [], defaultdict(lambda: defaultdict(lambda: dict(reference=0, predicted=0, matched=0)))
    for case, original_row, planned in zip(manifest['cases'], baseline['rows'], cases):
        assert sha(case['labels_path']) == case['labels_sha256']
        reference = read(case['labels_path'])
        events = read(BASELINE/case['name']/'notes.json')
        frozen_events = copy.deepcopy(events)
        decisions = read(Path(planned['result_folder'])/'decisions.json')
        labels = dict(original=[n['instrument_class_id'] for n in events],
            ungrouped=[n['instrument_class_id'] if n['is_drum'] else classes.index(d['ungrouped_class'])
                       for n, d in zip(events, decisions)],
            grouped=[n['instrument_class_id'] if n['is_drum'] else d['refined_class_id']
                     for n, d in zip(events, decisions)])
        measures = {}
        for mode, class_ids in labels.items():
            measures[mode], scope = family_metrics(reference, events, class_ids, mapping, planned['context_end_seconds'])
            for protocol in ('standard', 'strict'):
                for name in ('pitched_onsets', 'complete_intervals'):
                    assert measures[mode][protocol][name] == original_row['metrics'][protocol][name]
                for name, counts in measures[mode][protocol].items():
                    key = '/'.join((mode, protocol, name))
                    for field in ('reference', 'predicted', 'matched'):
                        groups[case['kind']][key][field] += counts[field]
        assert events == frozen_events
        row = dict(name=case['name'], kind=case['kind'], instrument=case.get('instrument'),
            metrics=measures, interval_scope=scope, classes={mode:dict(Counter(classes[i] for i in ids))
                for mode, ids in labels.items()},
            changed_classes={mode:sum(a != b for a,b in zip(labels['original'], ids)) for mode,ids in labels.items()},
            clusters=reports[case['name']]['clusters'], reference_sha256=case['labels_sha256'],
            original_notes_sha256=sha(BASELINE/case['name']/'notes.json'),
            refinement_report_sha256=sha(Path(planned['result_folder'])/'report.json'))
        save(output/'scores'/(case['name']+'.json'), row)
        rows.append(row)
    for group in groups.values():
        for counts in group.values():
            counts['f1'] = 2*counts['matched']/max(1, counts['reference']+counts['predicted'])
    assert inputs == {name:sha(name) for name in inputs}
    assert sha(__file__) == plan['runner_sha256'] and sha(ADAPTER) == plan['adapter_sha256']
    assert sha(TAXONOMY) == plan['taxonomy_sha256'] and verify_source(SOURCE) == source_files
    assert sha(ROOT/'scripts/evaluate_instrument_benchmark.py') == plan['matcher_sha256']
    save(output/'report.json', dict(plan_sha256=sha(output/'plan.json'),
        inference_completed_sha256=sha(output/'inference-completed.json'), rows=rows, groups=groups,
        new_inference_seconds=sum(reports[c['name']]['elapsed_seconds'] for c in cases if not c['reused']),
        original_notes_unchanged=True, onset_interval_scores_unchanged=True,
        enabled_in_application=False, musical_acceptance=False,
        limitations=['Public reused cohort, no verified exclusion from model training.',
            'Solo and mixture recordings reuse performances; do not pool them as independent samples.',
            'Broad family matching is not identity of an individual instrument or player.',
            'No flute, clarinet, guitar, synthesizer or vocal coverage in these recordings.',
            'Clustering cannot recover missing attacks or repair note durations.']))
    print(json.dumps(dict(stage='complete', output=str(output), cases=35)), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    run(cli.parse_args().output.resolve())
