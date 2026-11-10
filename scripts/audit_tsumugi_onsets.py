"""Audit all 156 frozen Tsumugi inputs and rescore onset-source filters offline.

No neural/model imports, checkpoint reads, inference, audio decoding or source
rewrites. A frozen copy of the pure release auditor is saved in the new output.
All reported event and matching indices refer to original saved arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
MODES = ('all', 'raw_has_onset', 'exact_boundary_onset', 'input_active_or_observed',
         'remove_input_first_frame', 'remove_internal_unobserved', 'remove_input_later_frame')
COMPLETIONS = {
    'tsumugi-urmp-comparison-v1': '3acc38b972759f68b2aa2aec2e8b5b95d9945ce8e925211586d8ac9085656621',
    'bass-model-comparison-v1': 'f16c9fce73e25e046fb67f83e5a614dbb3948fd5d54f8378c1ceb6db94aec095',
    'tsumugi-guitar-vocal-comparison-v1': 'f3132291bc494f5dfbe51ce1c37077bfa74cd6cafa8a3aaf2c6c983bebf3264d',
    'sakura-tsumugi-fullsong-v1': '7c55343972a59a5a1eeb99af2b088293e907a96038a1434042c460660d2ddfc8',
}
INPUT_HASHES = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    require(expected is None or digest == expected, 'Changed frozen input: '+str(path))
    INPUT_HASHES[str(path)] = digest
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def completed(folder, count, manifest_path):
    done = read(folder/'completed.json', COMPLETIONS[folder.name])
    plan = read(folder/'plan.json', done['plan_sha256'])
    comparison = read(folder/'comparison.json', done['comparison_sha256'])
    require(done['identity'] == plan == comparison['identity'], 'Completion identity mismatch')
    require(done['cases'] == comparison['cases'] == count, 'Missing frozen cases')
    manifest = read(manifest_path, plan['manifest_sha256'])
    require(len(manifest['cases']) == len(comparison['rows']) == count, 'Missing manifest rows')
    require([c['name'] for c in manifest['cases']] == [r['name'] for r in comparison['rows']], 'Changed case order')
    return manifest['cases'], comparison['rows'], plan


def get_cases():
    """Require all completion chains before loading any annotation arrays."""
    cases = []
    folder = ROOT/'test-data/tsumugi-urmp-comparison-v1'
    manifest, rows, plan = completed(folder, 35, ROOT/'test-data/urmp/comparison-extension/manifest.json')
    for case, row in zip(manifest, rows):
        cases.append(dict(name='urmp/'+case['name'], family='urmp', subgroup=case['kind'],
            input=folder/case['name'], report_sha256=row['report_sha256'],
            annotations={'URMP': dict(path=case['labels_path'], sha256=case['labels_sha256'],
                baseline=row['metrics'], begin=0, end=30)}, plan_source=plan))
    folder = ROOT/'test-data/bass-model-comparison-v1'
    manifest, rows, plan = completed(folder, 17, ROOT/'test-data/bass-benchmark-v1/manifest.json')
    for case, row in zip(manifest, rows):
        cases.append(dict(name='bass/'+case['name'], family='bass', subgroup='bass',
            input=folder/'tsumugi'/case['name'], report_sha256=row['engines']['tsumugi']['report_sha256'],
            annotations={'bass': dict(path=case['labels'], sha256=case['labels_sha256'],
                baseline=row['engines']['tsumugi']['metrics'],
                pitch_agnostic_baseline=row['engines']['tsumugi']['pitch_agnostic'], begin=0, end=case['seconds'])},
            plan_source=plan['tsumugi']))
    folder = ROOT/'test-data/tsumugi-guitar-vocal-comparison-v1'
    manifest, rows, plan = completed(folder, 100, ROOT/'test-data/guitar-vocal-benchmark-v3/manifest.json')
    require(Counter(c['dataset'] for c in manifest) == {'GuitarSet': 60, 'vocadito': 40}, 'Changed guitar/vocal cohort')
    for case, row in zip(manifest, rows):
        cases.append(dict(name=case['dataset']+'/'+case['name'], family=case['dataset'], subgroup=case['dataset'],
            input=folder/case['name'], report_sha256=row['report_sha256'], annotations={
                annotator: dict(path=entry['path'], sha256=entry['sha256'], begin=0, end=case['seconds'],
                    baseline=row['references'][annotator]['tsumugi']['metrics'])
                for annotator, entry in case['references'].items()}, plan_source=plan))
    folder = ROOT/'test-data/sakura-tsumugi-fullsong-v1'
    done = read(folder/'completed.json', COMPLETIONS[folder.name])
    plan = read(folder/'execution-plan.json', done['plan_sha256'])
    require(done['song_inputs'] == len(done['results']) == 4, 'Missing full-song stems')
    require([r['voice'] for r in done['results']] == ['bass', 'guitar', 'other', 'vocals'], 'Changed full-song order')
    sources = {s['voice']: s for s in plan['sources']}
    for row in done['results']:
        cases.append(dict(name='sakura/'+row['voice'], family='sakura', subgroup=row['voice'],
            input=Path(row['output']), report_sha256=row['report_sha256'], annotations={},
            expected_audio=sources[row['voice']]['audio_sha256']))
    require(len(cases) == 156, 'Require all 156 frozen inputs')
    return cases


def source_onset(row):
    """Independent qualification at the exact original constructed sample."""
    args, note = row['arguments'], row['constructed_note']
    flag = args['boundary_flag']
    raw_start = args['window_start_frame'] + round((args['begin_frame'] + (flag[2] if flag is not None else 0.))*row['hop_length'])
    adjusted = raw_start != note['start_sample']
    supported = flag is not None and flag[0] is True and not adjusted
    return dict(boundary_head_has_onset=None if flag is None else flag[0],
        constructed_has_onset=note['has_onset'], raw_predicted_start_sample=raw_start,
        constructed_start_sample=note['start_sample'], onset_constructor_adjusted=adjusted,
        window_start_sample=args['window_start_frame'], begin_frame=args['begin_frame'],
        start_equals_window_start=note['start_sample'] == args['window_start_frame'],
        start_equals_input_start=note['start_sample'] == 0,
        fallback=flag is None,
        fallback_first_frame_forced_true=flag is None and args['window_start_frame'] == 0 and args['begin_frame'] == 0,
        exact_boundary_support=supported)


def analyze_events(replayed, constructions, predicted):
    rows = []
    for index, item in enumerate(replayed['notes']):
        note = item['note']
        require(note['start_sample'] >= 0, 'Negative raw onset')
        onset_ids = item['onset_sources']
        require(onset_ids == [i for i in item['sources'] if constructions[i]['constructed_note']['start_sample'] == note['start_sample']],
                'Source list does not cover all exact-onset sources')
        evidence = {str(i): source_onset(constructions[i]) for i in onset_ids}
        supported = [int(i) for i, e in evidence.items() if e['exact_boundary_support']]
        initial_active = [int(i) for i, e in evidence.items() if e['window_start_sample'] == 0
                          and e['begin_frame'] == 0 and not e['fallback'] and not e['onset_constructor_adjusted']]
        if supported:
            origin = 'observed'
        elif all(e['window_start_sample'] == 0 and e['begin_frame'] == 0
                 and e['boundary_head_has_onset'] is False for e in evidence.values()):
            origin = 'input_first_frame'
        elif all(e['window_start_sample'] == 0 and e['boundary_head_has_onset'] is False for e in evidence.values()):
            origin = 'input_later_frame'
        elif all(e['window_start_sample'] > 0 and e['boundary_head_has_onset'] is False for e in evidence.values()):
            origin = 'internal_first_frame' if all(e['begin_frame'] == 0 for e in evidence.values()) else 'internal_later_frame'
        else:
            origin = 'mixed_or_fallback'
        same_start_true = [int(i) for i, e in evidence.items() if e['constructed_has_onset']]
        later_true = [i for i in item['sources'] if constructions[i]['constructed_note']['has_onset']
                      and constructions[i]['constructed_note']['start_sample'] > note['start_sample']]
        require(note['has_onset'] == any(constructions[i]['constructed_note']['has_onset'] for i in item['sources']),
                'Final onset flag is not the OR of its original construction flags')
        raised = [m for m in item['merges'] if not m['existing_has_onset'] and m['incoming_has_onset']]
        rows.append(dict(note_index=index, onset=predicted[index]['onset'], offset=predicted[index]['offset'],
            pitch=note['pitch'], instrument_class=predicted[index]['instrument_class'],
            raw_has_onset=note['has_onset'], exact_boundary_onset=bool(supported),
            input_active_or_observed=bool(supported or initial_active),
            input_start_active_interval=bool(initial_active) and not supported,
            initial_active_source_indices=initial_active,
            unobserved_origin=origin,
            remove_input_first_frame=origin != 'input_first_frame',
            remove_internal_unobserved=not origin.startswith('internal_'),
            remove_input_later_frame=origin != 'input_later_frame',
            same_start_mixed_support=bool(supported) and len(supported) < len(onset_ids),
            raw_true_without_exact_boundary=note['has_onset'] and not supported,
            raw_true_without_same_start_constructed_onset=note['has_onset'] and not same_start_true,
            later_source_raises_onset=note['has_onset'] and not same_start_true and bool(later_true),
            later_true_source_indices=later_true, has_onset_false_to_true_merges=raised,
            any_start_equals_window_start=any(e['start_equals_window_start'] for e in evidence.values()),
            any_start_adjusted=any(e['onset_constructor_adjusted'] for e in evidence.values()),
            exact_onset_source_indices=onset_ids, exact_supported_source_indices=supported, source_evidence=evidence))
    built = [source_onset(r) for r in constructions if r['constructed_note'] is not None]
    counts = dict(raw_notes=len(rows), raw_has_onset_false=sum(not r['raw_has_onset'] for r in rows),
        exact_boundary_qualified=sum(r['exact_boundary_onset'] for r in rows),
        exact_boundary_unqualified=sum(not r['exact_boundary_onset'] for r in rows),
        input_active_or_observed_qualified=sum(r['input_active_or_observed'] for r in rows),
        input_start_active_interval=sum(r['input_start_active_interval'] for r in rows),
        raw_true_without_exact_boundary=sum(r['raw_true_without_exact_boundary'] for r in rows),
        raw_true_without_same_start_constructed_onset=sum(r['raw_true_without_same_start_constructed_onset'] for r in rows),
        later_source_raises_onset=sum(r['later_source_raises_onset'] for r in rows),
        false_to_true_merge_operations=sum(len(r['has_onset_false_to_true_merges']) for r in rows),
        same_start_mixed_support=sum(r['same_start_mixed_support'] for r in rows),
        final_start_equals_source_window_start=sum(r['any_start_equals_window_start'] for r in rows),
        final_any_onset_source_adjusted=sum(r['any_start_adjusted'] for r in rows),
        unobserved_input_first_frame=sum(r['unobserved_origin'] == 'input_first_frame' for r in rows),
        unobserved_input_later_frame=sum(r['unobserved_origin'] == 'input_later_frame' for r in rows),
        unobserved_internal_first_frame=sum(r['unobserved_origin'] == 'internal_first_frame' for r in rows),
        unobserved_internal_later_frame=sum(r['unobserved_origin'] == 'internal_later_frame' for r in rows),
        unobserved_mixed_or_fallback=sum(r['unobserved_origin'] == 'mixed_or_fallback' for r in rows),
        constructed=len(built), constructed_has_onset_false=sum(not b['constructed_has_onset'] for b in built),
        constructed_fallback=sum(b['fallback'] for b in built),
        constructed_fallback_first_frame_forced_true=sum(b['fallback_first_frame_forced_true'] for b in built),
        constructed_onset_adjusted=sum(b['onset_constructor_adjusted'] for b in built),
        constructed_start_equals_window_start=sum(b['start_equals_window_start'] for b in built))
    return rows, counts


def matching(reference, predicted, ref_indices, pred_indices, *, strict, offset=False, pitched=True):
    """Same frozen maximum-cardinality/minimum-onset-error assignment protocol."""
    pairs = []
    if ref_indices and pred_indices:
        ref = np.asarray([[reference[i][k] for k in ('onset','offset','pitch')] for i in ref_indices])
        pred = np.asarray([[predicted[i][k] for k in ('onset','offset','pitch')] for i in pred_indices])
        delta = np.abs(ref[:, None, :] - pred[None, :, :])
        allowed = delta[:, :, 0] <= (.04 if strict else .05)
        if pitched:
            allowed &= (delta[:, :, 2] == 0) & np.asarray([not predicted[i]['is_drum'] for i in pred_indices])[None, :]
        if offset:
            tolerance = np.full(len(ref), .08) if strict else np.maximum(.05, .2*(ref[:, 1]-ref[:, 0]))
            allowed &= delta[:, :, 1] <= tolerance[:, None]
        cost = np.concatenate((np.where(allowed, delta[:, :, 0]/(len(ref)+1), 1e6), np.ones((len(ref), len(ref)))), axis=1)
        ri, pi = linear_sum_assignment(cost)
        pairs = [[ref_indices[int(i)], pred_indices[int(j)]] for i, j in zip(ri, pi) if j < len(pred) and allowed[i, j]]
    nr, npred, hits = len(ref_indices), len(pred_indices), len(pairs)
    return dict(available=True, reference=nr, predicted=npred, matched=hits, true_positives=hits,
        false_positives=npred-hits, false_negatives=nr-hits, precision=hits/npred if npred else 0.,
        recall=hits/nr if nr else 0., f1=2*hits/(nr+npred) if nr+npred else 0.,
        mean_onset_error_ms=sum(abs(reference[i]['onset']-predicted[j]['onset'])*1000 for i, j in pairs)/hits if hits else None,
        matched_indices=pairs)


def change(before, after):
    if not before['available']:
        return dict(available=False, reason=before['reason'])
    rb, ra = {i for i, _ in before['matched_indices']}, {i for i, _ in after['matched_indices']}
    pb, pa = {j for _, j in before['matched_indices']}, {j for _, j in after['matched_indices']}
    return dict(available=True, removed_predictions=before['predicted']-after['predicted'],
        matched_delta=after['matched']-before['matched'], false_positive_delta=after['false_positives']-before['false_positives'],
        false_negative_delta=after['false_negatives']-before['false_negatives'], f1_delta=after['f1']-before['f1'],
        lost_reference_indices=sorted(rb-ra), gained_reference_indices=sorted(ra-rb),
        prior_matched_prediction_indices_no_longer_matched=sorted(pb-pa))


def score_annotation(annotation, predicted, events, context_end, family):
    reference = read(annotation['path'], annotation['sha256'])
    begin, end = annotation['begin'], annotation['end']
    require(0 <= begin < end <= context_end, 'Invalid original scoring clock')
    ri = [i for i, n in enumerate(reference) if begin <= n['onset'] < end]
    unobserved = [i for i in ri if reference[i]['offset'] >= context_end]
    pi = [i for i, n in enumerate(predicted) if begin <= n['onset'] < end]
    modes = {}
    for mode in MODES:
        selected = [i for i in pi if mode == 'all' or events[i][mode]]
        values = {}
        for tolerance, strict in (('standard', False), ('fixed_40_80', True)):
            metrics = {}
            for metric, tail, pitched in (('pitched_onsets', False, True), ('complete_intervals', True, True),
                                           ('pitch_agnostic_onsets', False, False)):
                if metric == 'pitch_agnostic_onsets' and family != 'bass':
                    continue
                if tail and unobserved:
                    result = dict(available=False, reason='reference_release_reaches_input_boundary',
                        reference=len(ri), predicted=len(selected), unobservable_reference_indices=unobserved)
                else:
                    result = matching(reference, predicted, ri, selected, strict=strict, offset=tail, pitched=pitched)
                metrics[metric] = result
                if mode == 'all':
                    old = (annotation['pitch_agnostic_baseline'][tolerance] if not pitched else
                        annotation['baseline']['strict' if tolerance == 'fixed_40_80' and 'strict' in annotation['baseline'] else tolerance][metric])
                    require(old.get('available', True) == result['available'], 'Frozen availability differs')
                    for key in ('reference', 'predicted'):
                        require(old[key] == result[key], 'Frozen denominator differs: '+key)
                    if result['available']:
                        require(old['matched'] == result['matched'], 'Frozen matching protocol differs')
                        require(abs(old['f1']-result['f1']) < 1e-12, 'Frozen F1 differs')
            values[tolerance] = metrics
        modes[mode] = dict(predicted_indices=selected, metrics=values)
    deltas = {mode: {t: {m: change(modes['all']['metrics'][t][m], after) for m, after in metrics.items()}
                    for t, metrics in modes[mode]['metrics'].items()} for mode in MODES[1:]}
    removed = {}
    for mode in MODES[1:]:
        dropped = sorted(set(pi)-set(modes[mode]['predicted_indices']))
        baseline_matches = {j: i for i, j in modes['all']['metrics']['standard']['pitched_onsets']['matched_indices']}
        retained_ref = {i for i, _ in modes[mode]['metrics']['standard']['pitched_onsets']['matched_indices']}
        removed[mode] = [dict(note_index=i, onset=predicted[i]['onset'], pitch=predicted[i]['pitch'],
            raw_has_onset=events[i]['raw_has_onset'], exact_boundary_onset=events[i]['exact_boundary_onset'],
            unobserved_origin=events[i]['unobserved_origin'],
            prior_match_reference_index=baseline_matches.get(i),
            prior_reference_still_matched=baseline_matches[i] in retained_ref if i in baseline_matches else None,
            prior_reference=reference[baseline_matches[i]] if i in baseline_matches else None) for i in dropped]
    return dict(labels_path=str(Path(annotation['path']).resolve()), labels_sha256=annotation['sha256'],
        begin_seconds=begin, end_seconds=end, context_end_seconds=context_end, reference_indices=ri,
        baseline_parity=True, modes=modes, deltas_from_all=deltas, removed_predictions_standard_pitch=removed)


def aggregate(rows):
    result = {}
    for mode in MODES:
        result[mode] = {}
        for tolerance in ('standard', 'fixed_40_80'):
            metrics = {}
            for metric in rows[0]['modes'][mode]['metrics'][tolerance]:
                all_values = [r['modes'][mode]['metrics'][tolerance][metric] for r in rows]
                values = [v for v in all_values if v['available']]
                nr, npred, hit = (sum(v[k] for v in values) for k in ('reference', 'predicted', 'matched'))
                metrics[metric] = dict(total_cases=len(rows), available_cases=len(values),
                    unavailable_cases=len(rows)-len(values), reference=nr, predicted=npred, matched=hit,
                    false_positives=npred-hit, false_negatives=nr-hit,
                    precision=hit/npred if npred else 0., recall=hit/nr if nr else 0.,
                    f1=2*hit/(nr+npred) if nr+npred else 0.,
                    macro_f1=sum(v['f1'] for v in values)/len(values) if values else None)
            result[mode][tolerance] = metrics
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--auditor', type=Path, default=ROOT/'scripts/audit_tsumugi_releases.py')
    cli.add_argument('--source', type=Path, default=ROOT/'test-data/other-instrument-model-survey-2026-11-10/tsumugi')
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve prior evidence; choose a new output directory')
    cases = get_cases()
    args.output.mkdir(parents=True)
    auditor_source = args.auditor.read_bytes()
    snapshot = args.output/'release-auditor-source.py'
    snapshot.write_bytes(auditor_source)
    module = ModuleType('_frozen_tsumugi_release_audit')
    module.__file__ = str(snapshot.resolve())
    sys.modules[module.__name__] = module
    exec(compile(auditor_source, str(snapshot.resolve()), 'exec'), module.__dict__)
    official, source_hashes, official_identity = None, None, None
    summaries, group_rows = [], defaultdict(list)
    for position, case in enumerate(cases, 1):
        folder = case['input']
        report = read(folder/'report.json', case['report_sha256'])
        identity = read(folder/'identity.json')
        require(identity == report['identity'], 'Report/identity mismatch')
        if 'expected_audio' in case:
            require(identity['audio_sha256'] == case['expected_audio'], 'Changed full-song audio identity')
        elif 'plan_source' in case:
            require(identity['source_files'] == case['plan_source']['source_files'], 'Changed model source identity')
        if official is None:
            official, source_hashes = module.load_official(args.source, identity)
            official_identity = (identity['source_commit'], identity['source_files'])
        require((identity['source_commit'], identity['source_files']) == official_identity, 'Mixed upstream sources')
        values = {name: read(folder/(name.replace('_', '-')+'.json'), report[name+'_sha256'])
                  for name in ('notes', 'raw_notes', 'interval_constructions', 'window_notes')}
        predicted, raw = values['notes'], values['raw_notes']
        require(len(predicted) == len(raw) == report['notes'], 'Changed model event count')
        rate = identity['resampled_sample_rate']
        for note, event in zip(raw, predicted):
            require(note['start_sample']/rate == event['onset'] and note['end_sample']/rate == event['offset']
                and note['pitch'] == event['pitch'] and note['has_onset'] == event['model_has_onset'],
                'Saved event mapping changed')
        replay = module.audit_replay(identity, values['interval_constructions'], values['window_notes'], raw, report['stats'], official)
        events, counts = analyze_events(replay, values['interval_constructions'], predicted)
        annotations = {name: score_annotation(entry, predicted, events, report['context_end_seconds'], case['family'])
                       for name, entry in case['annotations'].items()}
        for annotator, value in annotations.items():
            group = case['family']+'/'+annotator if case['family'] in ('GuitarSet', 'vocadito') else case['family']
            group_rows[group].append(value)
            if case['family'] == 'urmp':
                group_rows['urmp/'+case['subgroup']].append(value)
        name = case['name'].replace('/', '__')+'.json'
        evidence = dict(case=case['name'], family=case['family'], input=str(folder.resolve()),
            report_sha256=case['report_sha256'], identity=identity, counts=counts, events=events,
            annotations=annotations, unscored_no_labels=not annotations,
            official_finalize_equal=replay['official_finalize_equal'], saved_raw_notes_equal=replay['saved_raw_notes_equal'],
            window_snapshots_equal=replay['window_snapshots_equal'])
        write(args.output/name, evidence)
        summaries.append(dict(case=case['name'], family=case['family'], counts=counts, annotations=list(annotations),
            output=name, output_sha256=sha(args.output/name)))
        print(json.dumps(dict(index=position, total=len(cases), case=case['name'], counts=counts)), flush=True)
    totals = {}
    for family in ('urmp', 'bass', 'GuitarSet', 'vocadito', 'sakura', 'all'):
        selected = [r for r in summaries if family == 'all' or r['family'] == family]
        totals[family] = dict(cases=len(selected), **{k: sum(r['counts'][k] for r in selected) for k in selected[0]['counts']})
    require(not any(name in sys.modules for name in ('torch', 'torchaudio', 'onnxruntime')), 'Unexpected neural runtime import')
    result = dict(revision=3, completed=True, case_count=156, scored_cases=152, annotation_case_count=192,
        runner_sha256=sha(__file__), auditor_snapshot_sha256=sha(snapshot), verified_source_sha256=source_hashes,
        input_sha256=INPUT_HASHES, rows=summaries, counts=totals,
        groups={name: aggregate(rows) for name, rows in group_rows.items()},
        modes=dict(all='Every frozen raw final event', raw_has_onset='Keep final raw.has_onset=True',
            exact_boundary_onset='Keep >=1 source at exact final start with boundary_flag[0]=True and unchanged constructed sample',
            input_active_or_observed='Keep exact boundary onset OR window_start=0/begin_frame=0 with non-null boundary head and unchanged onset; label latter input_start_active_interval, not observed attack',
            remove_input_first_frame='Diagnostic: remove only unobserved head-false sources from original input window 0, frame 0',
            remove_internal_unobserved='Diagnostic: remove only unobserved head-false sources from nonzero-start windows',
            remove_input_later_frame='Diagnostic: remove only unobserved head-false sources from window 0 after frame 0'),
        onset_flag_semantics='Upstream fallback can synthesize has_onset; stitching ORs original source flags. No flag alone proves an acoustic attack.',
        neural_inference=False, neural_runtime_imports=False, checkpoint_reads=False, audio_decoding=False, source_modified=False,
        limitations=['Reference labels are fallible and these public corpora may overlap upstream training.',
            'URMP solo/mix cases reuse performances; vocal A1/A2 are separate full groups, not 80 independent recordings.',
            'Filtering only removes final events. It does not move starts, recover attacks or alter raw intervals.',
            'A removed baseline TP indicates loss under the fixed annotation protocol, not proof of acoustic correctness.',
            'All model classes remain in denominators. Standard onset tolerance is 50ms; fixed_40_80 uses 40ms.',
            'Complete-interval metrics remain unavailable when any scored reference release reaches the input boundary.',
            'Four Sakura stems have no independent labels and receive no accuracy score; the guitar stem is silent.',
            'Physical note intervals and onset scores do not establish authored LN quality.'])
    write(args.output/'report.json', result)
    write(args.output/'completed.json', dict(report_sha256=sha(args.output/'report.json'), cases=156,
        script_sha256=result['runner_sha256'], auditor_snapshot_sha256=result['auditor_snapshot_sha256']))
    print(json.dumps(dict(stage='complete', counts=totals, groups=result['groups'])), flush=True)


if __name__ == '__main__':
    main()
