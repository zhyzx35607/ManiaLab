"""Evaluate recovered AMT logits only after all fixed recordings have finished."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_tsumugi_releases import read_json as read
from scripts.audit_instrument_refinement import count_matches
from scripts.check_harmonica_transcription import BASELINE, BASELINE_SHA, MANIFEST, MANIFEST_SHA
from scripts.check_instrument_refinement import family_mapping, family_metrics
from scripts.check_mega53_separation import save, sha
from scripts.tsumugi_instrument_scores import ambiguity_bins, note_decisions

SOURCE = ROOT/'test-data/other-instrument-model-survey-2026-11-10/tsumugi'


def add_bins(target, values):
    for mode, result in values.items():
        for name, count in result['categories'].items():
            target[mode]['categories'][name] += count
        for key, bins in result['bins'].items():
            for i, counts in enumerate(bins):
                for name, count in counts.items():
                    target[mode]['bins'][key][i][name] += count


def empty_bins():
    return {mode: dict(categories=Counter(), bins={name:[Counter() for _ in range(4)]
            for name in ('top_sigmoid', 'logit_margin')}) for mode in ('first', 'mean')}


def verify_evidence(folder, case, completed, plan, classes):
    target, original = folder/case['name'], Path(case['original'])
    assert sha(target/'report.json') == completed['report_sha256']
    report = read(target/'report.json')
    assert report['name'] == case['name'] == completed['name']
    assert report['plan_sha256'] == sha(folder/'plan.json')
    assert report['runtime_sha256'] == sha(folder/'runtime.json')
    assert report['original_report_sha256'] == sha(original/'report.json') == case['report_sha256']
    assert report['labels_read'] is False and report['complete_class_rankings_reproduced'] is True
    for name, expected in report['evidence_sha256'].items():
        assert sha(target/name) == expected, name
    for path, expected in plan['inputs'].items():
        if Path(path).parent == original:
            assert sha(path) == expected, path
    with np.load(target/'evidence.npz', allow_pickle=False) as stored:
        arrays = {k:stored[k] for k in stored.files}
    constructions = read(original/'interval-constructions.json')
    raw, events = read(original/'raw-notes.json'), read(original/'notes.json')
    lineage, decisions = read(target/'lineage.json'), read(target/'decisions.json')
    assert [n['note'] for n in lineage] == raw and len(events) == len(raw) == len(decisions) == report['notes']
    scores, rankings = arrays['instrument_logits'], arrays['rankings']
    assert scores.shape == rankings.shape == (len(constructions), len(classes))
    assert np.isfinite(scores).all() and scores.dtype == np.float32
    assert arrays['boundary_logits'].shape == arrays['boundary_values'].shape == (len(constructions), 4)
    assert np.isfinite(arrays['boundary_logits']).all() and np.isfinite(arrays['boundary_values']).all()
    assert arrays['entries'].shape == (len(constructions), 5) and np.all(arrays['coverage'] == 1)
    # Independent checks of source interval indices and numerical rank order.
    slots = read(original/'report.json')['identity']['model_config']['num_pitch_slots']
    track_counts = Counter()
    for i, row in enumerate(constructions):
        args = row['arguments']
        track = (args['pair_id'] % 88)*slots+row['v1_slot_index']
        key = (args['window_start_frame'], track)
        assert arrays['entries'][i].tolist() == [0, track, track_counts[key], args['begin_frame'], args['end_frame']]
        track_counts[key] += 1
        assert rankings[i].tolist() == args['instrument_candidates']
        assert sorted(rankings[i].tolist()) == list(range(len(classes)))
        # sigmoid is monotone; allow saturation ties in the original FP32 order.
        p = (1/(1+np.exp(-scores[i].astype(np.float64)))).astype(np.float32)
        assert np.all(np.diff(p[rankings[i]]) <= 1e-7)
        assert arrays['boundary_values'][i,:2].tolist() == args['boundary_flag'][:2]
        assert np.allclose(arrays['boundary_values'][i,2:], args['boundary_flag'][2:], atol=1e-6, rtol=0)
    assert decisions == note_decisions(lineage, scores, rankings)
    for event, original_note, decision in zip(events, raw, decisions):
        assert event['instrument_class_id'] == original_note['instrument_id'] == decision['first']['class_id']
    assert report['changed_classes'] == sum(d['changed'] for d in decisions)
    assert report['disagreeing_sources'] == sum(not d['sources_agree'] for d in decisions)
    return report, events, decisions


def run(folder, output):
    assert not output.exists(), 'Preserve earlier score reports'
    completed, plan = read(folder/'inference-completed.json'), read(folder/'plan.json')
    assert completed['labels_read'] is False and len(completed['cases']) == len(plan['cases']) == 35
    assert completed['plan_sha256'] == sha(folder/'plan.json')
    assert completed['runtime_sha256'] == sha(folder/'runtime.json')
    assert sha(BASELINE/'comparison.json') == BASELINE_SHA == plan['baseline_sha256']
    assert sha(MANIFEST) == MANIFEST_SHA == plan['manifest_sha256']
    for path, digest in plan['code_sha256'].items():
        assert sha(folder/'source'/path) == digest
    for path, digest in plan['inputs'].items():
        assert sha(path) == digest, path
    manifest, baseline = read(MANIFEST), read(BASELINE/'comparison.json')
    taxonomy_path = SOURCE/'instrument_agnostic_amt/taxonomy/instrument_merge.json'
    classes, mapping = family_mapping(read(taxonomy_path))
    assert classes == read(folder/'runtime.json')['classes']
    evidence = []
    # Verify the entire completed cohort before opening the first note label.
    for case, done, listed in zip(plan['cases'], completed['cases'], manifest['cases']):
        assert case['name'] == done['name'] == listed['name']
        evidence.append(verify_evidence(folder, case, done, plan, classes))
    for case in manifest['cases']:
        assert sha(case['labels_path']) == case['labels_sha256']
    evaluation_plan = dict(inference_completed_sha256=sha(folder/'inference-completed.json'),
        inference_plan_sha256=sha(folder/'plan.json'), runner_sha256=sha(__file__),
        helper_sha256=sha(ROOT/'scripts/tsumugi_instrument_scores.py'),
        matcher_sha256=sha(ROOT/'scripts/evaluate_instrument_benchmark.py'),
        independent_matcher_sha256=sha(ROOT/'scripts/audit_instrument_refinement.py'),
        taxonomy_sha256=sha(taxonomy_path), classes=classes, program_to_family=mapping,
        labels={c['labels_path']:c['labels_sha256'] for c in manifest['cases']})
    save(output/'plan.json', evaluation_plan)
    groups = defaultdict(lambda: defaultdict(lambda: Counter(reference=0, predicted=0, matched=0)))
    bins = defaultdict(empty_bins)
    rows, checked = [], 0
    for case, old, (result, events, decisions) in zip(manifest['cases'], baseline['rows'], evidence):
        assert case['name'] == old['name'] == result['name']
        reference = read(case['labels_path'])
        labels = {mode:[d[mode]['class_id'] for d in decisions] for mode in ('first', 'mean')}
        measures = {}
        ref_scored = [n for n in reference if 0 <= n['onset'] < 30]
        indices = [i for i,n in enumerate(events) if 0 <= n['onset'] < 30]
        preds = [events[i] for i in indices]
        diagnosed = ambiguity_bins(ref_scored, preds, [decisions[i] for i in indices], mapping)
        for mode, predicted_classes in labels.items():
            measures[mode], scope = family_metrics(reference, events, predicted_classes, mapping,
                                                   old['interval_scope']['context_end_seconds'])
            for protocol in ('standard', 'strict'):
                for name in ('pitched_onsets', 'complete_intervals'):
                    assert measures[mode][protocol][name] == old['metrics'][protocol][name]
                for name, counts in measures[mode][protocol].items():
                    independent = count_matches(ref_scored, preds, [mapping[n['program']] for n in ref_scored],
                        [predicted_classes[i] for i in indices], tolerance=.04 if protocol == 'strict' else .05,
                        offsets='intervals' in name, fixed_offset=protocol == 'strict', families='instrument' in name)
                    assert independent == counts['matched'], (case['name'], mode, protocol, name)
                    checked += 1
                    keys = [case['kind']] + (['instrument/'+case['instrument']] if case.get('instrument') else [])
                    for key in keys:
                        for field in ('reference', 'predicted', 'matched'):
                            groups[key]['/'.join((mode, protocol, name))][field] += counts[field]
        keys = [case['kind']] + (['instrument/'+case['instrument']] if case.get('instrument') else [])
        for key in keys:
            add_bins(bins[key], diagnosed)
        row = dict(name=case['name'], kind=case['kind'], instrument=case.get('instrument'),
            metrics=measures, bins=diagnosed, interval_scope=scope,
            reference_sha256=case['labels_sha256'], evidence_report_sha256=sha(folder/case['name']/'report.json'),
            changed_classes_full_context=result['changed_classes'],
            changed_classes_scored=sum(decisions[i]['changed'] for i in indices),
            multiple_sources=result['multiple_sources'], disagreeing_sources=result['disagreeing_sources'],
            notes=result['notes'], constructions=result['constructions'], windows=result['windows'])
        save(output/'cases'/(case['name']+'.json'), row)
        rows.append(row)
    for group in groups.values():
        for counts in group.values():
            counts['precision'] = counts['matched']/counts['predicted'] if counts['predicted'] else 0.
            counts['recall'] = counts['matched']/counts['reference'] if counts['reference'] else 0.
            counts['f1'] = 2*counts['matched']/max(1, counts['reference']+counts['predicted'])
    assert all(sha(path) == digest for path, digest in plan['inputs'].items())
    assert all(sha(path) == digest for path, digest in evaluation_plan['labels'].items())
    save(output/'report.json', dict(plan_sha256=sha(output/'plan.json'), rows=rows,
        groups=groups, ambiguity=bins, independent_counts_checked=checked,
        original_notes_unchanged=True, enabled_in_application=False, musical_acceptance=False,
        totals={key:sum(row[key] for row in rows) for key in
            ('notes', 'constructions', 'windows', 'multiple_sources', 'disagreeing_sources',
             'changed_classes_full_context', 'changed_classes_scored')},
        limitations=['Reused public performances; not a blind test or 35 independent pieces.',
            'Reference covers three broad families, not individual instruments or voices.',
            'Class scores are not calibrated note-existence or musical-importance probabilities.',
            'High class scores cannot repair missing attacks or uncertain releases.']))
    print(json.dumps(dict(stage='complete', cases=len(rows), independent_counts=checked,
                          output=str(output))), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--evidence', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    run(args.evidence.resolve(), args.output.resolve())
