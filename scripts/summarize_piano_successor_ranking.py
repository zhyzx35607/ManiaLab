"""Audit saved ranking-fit predictions against the frozen BCE/decode baseline.

No models are loaded. Public-facing JSON contains counts, not event records;
identity-paired prediction arrays are written only under local test-data.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_provenance import sha256
from scripts.piano_successor_decode import select_successor_groups

KINDS = ('clean', 'plus1', 'plus2')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_artifacts(folder, report):
    for name, digest in report['artifacts_sha256'].items():
        require(Path(name).name == name and sha256(folder/name) == digest,
                'Changed or invalid artifact: '+name)


def verify_snapshots(folder, report):
    for name, digest in report['identity']['code_sha256'].items():
        suffix = '.md' if name == 'protocol' else '.py'
        require(sha256(folder/(name+'-snapshot'+suffix)) == digest, 'Changed snapshot: '+name)
    require(read(folder/'identity.json') == report['identity'], 'Identity/report disagreement')


def same_arrays(left, right):
    require(set(left) == set(right), 'Different saved graph fields')
    for key in left:
        if key == 'metadata_json':
            require(json.loads(str(left[key])) == json.loads(str(right[key])), 'Changed graph metadata')
        else:
            require(left[key].dtype == right[key].dtype and np.array_equal(left[key], right[key]),
                    'Changed array: '+key)


def metrics(actual, predicted, mask):
    a, p = actual[mask], predicted[mask]
    tp = int((a & p).sum()); fp = int((~a & p).sum())
    fn = int((a & ~p).sum()); tn = int((~a & ~p).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/max(1, tp+fp),
                recall=tp/max(1, tp+fn), f1=2*tp/max(1, 2*tp+fp+fn))


def paired_transitions(actual, mask, before, after):
    """Count both directions of every changed, identity-paired scored edge."""
    a, b, n = actual[mask], before[mask], after[mask]
    return dict(tp_to_fn=int((a & b & ~n).sum()), fn_to_tp=int((a & ~b & n).sum()),
                fp_to_tn=int((~a & b & ~n).sum()), tn_to_fp=int((~a & ~b & n).sum()))


def truth_group_retention(source_ids, actual, mask, before, after):
    """Preserve all known chord targets, not just the easiest edge in a group."""
    groups = defaultdict(list)
    for index in np.flatnonzero(actual & mask):
        groups[str(source_ids[index])].append(int(index))
    counts = Counter(sources_with_truth=len(groups), true_edges=int((actual & mask).sum()),
                     multiple_target_sources=sum(len(group)>1 for group in groups.values()))
    for key in ('before_at_least_one', 'after_at_least_one', 'before_all', 'after_all',
                'at_least_one_lost', 'at_least_one_gained', 'all_lost', 'all_gained'):
        counts[key] = 0
    for indices in groups.values():
        old_any, new_any = bool(before[indices].any()), bool(after[indices].any())
        old_all, new_all = bool(before[indices].all()), bool(after[indices].all())
        for name, value in [('before_at_least_one', old_any), ('after_at_least_one', new_any),
                            ('before_all', old_all), ('after_all', new_all),
                            ('at_least_one_lost', old_any and not new_any),
                            ('at_least_one_gained', new_any and not old_any),
                            ('all_lost', old_all and not new_all), ('all_gained', new_all and not old_all)]:
            counts[name] += int(value)
    return dict(counts)


def align_baseline(reference, current):
    """Match unique source/target IDs, independent of window/file ordering."""
    old_pairs = list(zip(reference['source_ids'], reference['target_ids']))
    new_pairs = list(zip(current['source_ids'], current['target_ids']))
    require(len(set(old_pairs)) == len(old_pairs) and len(set(new_pairs)) == len(new_pairs),
            'Duplicate owned pair')
    require(set(old_pairs) == set(new_pairs), 'Different owned edge identities')
    positions = {pair: index for index, pair in enumerate(old_pairs)}
    order = np.asarray([positions[pair] for pair in new_pairs], dtype=np.int64)
    result = {key: value[order] for key, value in reference.items()}
    for key in ('source_ids', 'target_ids', 'target_ticks', 'actual', 'score_mask'):
        require(np.array_equal(result[key], current[key]), 'Changed paired identity/label: '+key)
    return result


def owned(payload, logits):
    edges = payload['potential_edges']
    owner = payload['core_mask'][edges[1]]
    a, b = edges[:, owner]
    return dict(source_ids=payload['event_ids'][a], target_ids=payload['event_ids'][b],
                target_ticks=payload['notes']['onset_div'][b], logits=logits[owner],
                actual=payload['voice'][owner]>.5, score_mask=payload['voice_mask'][owner])


def join(parts):
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    pairs = list(zip(data['source_ids'], data['target_ids']))
    require(len(pairs) == len(set(pairs)), 'Duplicate owner edge across windows')
    return data


def decoding_diagnostics(data, selected):
    raw = data['logits'] >= 0
    actual, mask = data['actual'], data['score_mask']
    require(not (selected & ~raw).any(), 'Decoder added an edge')
    raw_groups = defaultdict(set); winners = defaultdict(list)
    for index in np.flatnonzero(raw):
        raw_groups[str(data['source_ids'][index])].add(int(data['target_ticks'][index]))
    for index in np.flatnonzero(selected):
        winners[str(data['source_ids'][index])].append(int(index))
    require(all(len({int(data['target_ticks'][i]) for i in ids}) == 1 for ids in winners.values()),
            'Decoder selected multiple target ticks')
    loss = Counter(only_group=0, maximum_only=0, tied_maximum=0, mixed_group=0)
    only_sources = set()
    for index in np.flatnonzero(raw & ~selected & actual & mask):
        source = str(data['source_ids'][index]); indices = winners[source]
        labels = {bool(mask[i]) for i in indices}
        top = max(data['logits'][i] for i in indices)
        maximum_labels = {bool(mask[i]) for i in indices if data['logits'][i] == top}
        loss['only_group'] += labels == {False}
        loss['maximum_only'] += maximum_labels == {False}
        loss['tied_maximum'] += maximum_labels == {False, True}
        loss['mixed_group'] += labels == {False, True}
        if labels == {False}: only_sources.add(source)
    return dict(raw=metrics(actual, raw, mask), decoded=metrics(actual, selected, mask),
        candidate_edges=len(raw), raw_positive=int(raw.sum()), selected_positive=int(selected.sum()),
        removed_positive=int((raw & ~selected).sum()), supervised_edges=int(mask.sum()),
        raw_sources_with_multiple_target_ticks=sum(len(v)>1 for v in raw_groups.values()),
        selected_sources_with_multiple_target_ticks=0,
        removed_true_edges=int((raw & ~selected & actual & mask).sum()),
        removed_false_edges=int((raw & ~selected & ~actual & mask).sum()),
        removed_true_with_only_unscored_winning_edges=int(loss['only_group']),
        removed_true_sources_with_only_unscored_winning_edges=len(only_sources),
        removed_true_with_unscored_maximum_only=int(loss['maximum_only']),
        removed_true_with_tied_scored_unscored_maximum=int(loss['tied_maximum']),
        removed_true_with_mixed_winning_group=int(loss['mixed_group']),
        unscored_selected_positive=int((selected & ~mask).sum()))


def compare_record(actual, saved):
    for key, value in actual.items():
        require(saved[key] == value, 'Saved metric differs: '+key)


def validate_epochs(folder, report, ranked):
    epochs = []
    for epoch in range(report['identity']['epochs']+1):
        value = read(folder/f'validation-epoch-{epoch}.json')
        require(set(value['variants']) == set(KINDS), 'Unexpected validation variants')
        for kind, variant in value['variants'].items():
            cases = variant['cases']; key = 'raw' if ranked else 'voice'
            require(len(cases) == len({row['name'] for row in cases}) == 9, 'Unexpected validation pieces')
            totals = {name: sum(row[key][name] for row in cases) for name in ('tp', 'fp', 'fn', 'tn')}
            require(sum(totals.values()) == 8004 and totals['tp']+totals['fn'] == 2218,
                    'Changed validation labels')
            require(all(variant['voice'][name] == count for name, count in totals.items()),
                    'Pooled validation counts disagree')
            scores = []
            for row in cases:
                c = row[key]; tp, fp, fn = c['tp'], c['fp'], c['fn']
                require(c['f1'] == 2*tp/max(1, 2*tp+fp+fn), 'Changed per-piece F1')
                if tp+fn: scores.append(c['f1'])
            require(abs(float(np.mean(scores))-variant['macro_voice_f1'])<1e-12, 'Changed macro F1')
            require(sum(row['staff_correct'] for row in cases) == variant['staff_correct']
                    and sum(row['staff_notes'] for row in cases) == variant['staff_notes'],
                    'Changed staff counts')
        require(abs(float(np.mean([value['variants'][kind]['macro_voice_f1'] for kind in KINDS]))
                    -value['selection_score'])<1e-12, 'Changed selection score')
        epochs.append(value)
    selected = max(range(len(epochs)), key=lambda index: epochs[index]['selection_score'])
    require(selected == report['selected_epoch'], 'Wrong selected epoch or tie break')
    require(sha256(folder/'selected.pt') == sha256(folder/f'epoch-{selected}.pt') == report['selected_sha256'],
            'Changed selected checkpoint')
    require(report['baseline'] == epochs[0], 'Saved epoch-zero summary changed')
    for entry in report['history']:
        require(entry['selection_score'] == epochs[entry['epoch']]['selection_score'], 'History score changed')
    return epochs


def pooled_metrics(cases, side, mode):
    counts = {key: sum(row[side][mode][key] for row in cases) for key in ('tp', 'fp', 'fn', 'tn')}
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    counts.update(precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn), f1=2*tp/max(1,2*tp+fp+fn))
    eligible = [row[side][mode]['f1'] for row in cases if row[side][mode]['tp']+row[side][mode]['fn']]
    counts['macro_f1'] = float(np.mean(eligible)) if eligible else None
    return counts


def sum_fields(rows):
    return {key: sum(row[key] for row in rows) for key in rows[0]}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'baseline', 'decode-baseline', 'output'):
        cli.add_argument('--'+name, type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve previous summaries')
    require(args.output.resolve().is_relative_to((ROOT/'test-data').resolve()),
            'Paired research arrays must stay in local test-data')
    run, baseline, decoded = (read(path/'report.json') for path in (args.run, args.baseline, args.decode_baseline))
    verify_artifacts(args.run, run); verify_artifacts(args.decode_baseline, decoded)
    verify_snapshots(args.run, run); verify_snapshots(args.baseline, baseline); verify_snapshots(args.decode_baseline, decoded)
    ri, bi, di = run['identity'], baseline['identity'], decoded['identity']
    require(ri['reference_report_sha256'] == sha256(args.baseline/'report.json'), 'Wrong BCE reference')
    require(ri['reference_epoch0_sha256'] == sha256(args.baseline/'validation-epoch-0.json'), 'Changed reference initialization')
    require(baseline['selected_epoch'] == 4 and di['weights_sha256']['overlap_adapted'] == baseline['selected_sha256'],
            'Decode reference is not the fixed BCE epoch4')
    for key in ('source_revision', 'pretrained_sha256', 'seed', 'epochs', 'lr', 'weight_decay',
                'voice_positive_weight', 'threshold', 'input_schedule'):
        require(ri[key] == bi[key], 'Changed comparison setting: '+key)
    require(ri['rank_coefficient'] == .1 and ri['rank_margin'] == 0, 'Changed fixed ranking settings')
    require(ri['code_sha256']['decoder'] == di['code_sha256']['decoder']
            == sha256(ROOT/'scripts/piano_successor_decode.py'), 'Changed decoder implementation')
    for current, prior in [('base_loss','clean_runner'), ('event_graph','event_graph'),
                           ('variant_loader','runner'), ('forward','base_runner')]:
        require(ri['code_sha256'][current] == bi['code_sha256'][prior], 'Changed shared training implementation')
    epochs = validate_epochs(args.run, run, True); old_epochs = validate_epochs(args.baseline, baseline, False)
    selected = run['selected_epoch']; old_selected = old_epochs[4]
    for kind in KINDS:
        require(epochs[0]['variants'][kind]['voice'] == old_epochs[0]['variants'][kind]['voice'], 'Epoch0 predictions changed')
    folders = dict(clean=ROOT/'test-data/piano-svsep/source-event-dataset-v1',
                   plus1=ROOT/'test-data/piano-svsep/source-event-overlap-v1/plus1',
                   plus2=ROOT/'test-data/piano-svsep/source-event-overlap-v1/plus2')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'summarizer-snapshot.py')
    variants = []; graph_count = 0
    for kind, folder in folders.items():
        digest = sha256(folder/'manifest.json')
        require(digest == ri['dataset_sha256'][kind] == bi['dataset_sha256'][kind] == di['dataset_sha256'][kind],
                'Changed training data identity')
        rows = [row for row in read(folder/'manifest.json')['items'] if row['split'] == 'validation']
        require(len(rows) == 36, 'Unexpected validation graph count')
        parts = defaultdict(list); staff = defaultdict(Counter)
        for index, row in enumerate(rows):
            require(sha256(folder/row['cache']) == row['cache_sha256'], 'Changed validation graph')
            graph = load(args.run/f'{kind}-graph-{index:03d}.npz')
            same_arrays(graph, load(folder/row['cache']))
            same_arrays(graph, load(args.decode_baseline/f'{kind}-graph-{index:03d}.npz'))
            prediction = load(args.run/f'epoch-{selected}-{kind}-{index:03d}.npz')
            voice, scores = prediction['voice_logits'], prediction['staff_scores']
            require(voice.shape == (graph['potential_edges'].shape[1],) and scores.shape == (len(graph['notes']),2)
                    and np.isfinite(voice).all() and np.isfinite(scores).all(), 'Invalid saved prediction')
            parts[row['name']].append(owned(graph, voice))
            mask = graph['staff_mask']; name = row['name']
            staff[name]['correct'] += int((scores.argmax(axis=1)[mask] == graph['staff'][mask]).sum())
            staff[name]['notes'] += int(mask.sum()); graph_count += 1
        old_variant = next(row for row in decoded['validation'] if row['model']=='overlap_adapted' and row['input']==kind)
        old_cases = {row['name']:row for row in old_variant['cases']}
        old_fit = {row['name']:row for row in old_selected['variants'][kind]['cases']}
        current_fit = {row['name']:row for row in epochs[selected]['variants'][kind]['cases']}
        require(set(parts) == set(old_cases) == set(old_fit) == set(current_fit), 'Different per-piece coverage')
        cases = []
        for index, (name, windows) in enumerate(sorted(parts.items())):
            data = join(windows); old = align_baseline(load(args.decode_baseline/old_cases[name]['file']), data)
            new_selection = select_successor_groups(data['source_ids'], data['target_ticks'], data['logits'])
            old_selection = select_successor_groups(old['source_ids'], old['target_ticks'], old['logits'])
            require(np.array_equal(old_selection, old['selected']), 'Changed saved baseline decision')
            before = decoding_diagnostics(old, old_selection); after = decoding_diagnostics(data, new_selection)
            compare_record(before, old_cases[name]); compare_record(after, current_fit[name])
            require(before['raw'] == old_fit[name]['voice'], 'Old decode raw differs from selected BCE fit')
            require(staff[name]['correct'] == current_fit[name]['staff_correct']
                    and staff[name]['notes'] == current_fit[name]['staff_notes'] == old_fit[name]['staff_notes'],
                    'Staff counts or labels changed')
            change = {}; group = {}
            for mode, earlier, later in [('raw', old['logits']>=0, data['logits']>=0),
                                         ('decoded', old_selection, new_selection)]:
                change[mode] = paired_transitions(data['actual'], data['score_mask'], earlier, later)
                group[mode] = truth_group_retention(data['source_ids'], data['actual'], data['score_mask'], earlier, later)
                require(after[mode]['tp']-before[mode]['tp'] == change[mode]['fn_to_tp']-change[mode]['tp_to_fn'],
                        'Paired true-edge changes do not reconcile')
                require(after[mode]['fp']-before[mode]['fp'] == change[mode]['tn_to_fp']-change[mode]['fp_to_tn'],
                        'Paired false-edge changes do not reconcile')
            filename = f'{kind}-paired-{index:02d}.npz'
            np.savez_compressed(args.output/filename, **data, old_logits=old['logits'],
                                old_selected=old_selection, new_selected=new_selection)
            cases.append(dict(name=name, paired_file=filename, before=before, after=after,
                transitions=change, truth_groups=group, staff=dict(before_correct=old_fit[name]['staff_correct'],
                    after_correct=staff[name]['correct'], notes=staff[name]['notes'],
                    correct_delta=staff[name]['correct']-old_fit[name]['staff_correct'])))
        totals = dict(before={mode:pooled_metrics(cases,'before',mode) for mode in ('raw','decoded')},
                      after={mode:pooled_metrics(cases,'after',mode) for mode in ('raw','decoded')},
                      transitions={mode:sum_fields([row['transitions'][mode] for row in cases]) for mode in ('raw','decoded')},
                      truth_groups={mode:sum_fields([row['truth_groups'][mode] for row in cases]) for mode in ('raw','decoded')},
                      staff=sum_fields([row['staff'] for row in cases]))
        losses = [key for key in cases[0]['after'] if key.startswith('removed_')]
        totals['decode_losses'] = {side:{key:sum(row[side][key] for row in cases) for key in losses}
                                   for side in ('before','after')}
        variants.append(dict(input=kind, cases=cases, aggregate=totals))
    result = dict(training_report_sha256=sha256(args.run/'report.json'),
        baseline_report_sha256=sha256(args.baseline/'report.json'),
        decode_baseline_report_sha256=sha256(args.decode_baseline/'report.json'),
        summarizer_sha256=sha256(__file__), selected_epoch=selected, selected_sha256=run['selected_sha256'],
        baseline_selected_epoch=4, baseline_selected_sha256=baseline['selected_sha256'],
        graphs_verified=graph_count, selected_validation_cases=27,
        validation_reports_sha256={side:{str(epoch):sha256(folder/f'validation-epoch-{epoch}.json')
            for epoch in range(report['identity']['epochs']+1)}
            for side,folder,report in (('ranking',args.run,run),('bce',args.baseline,baseline))},
        epochs=[dict(epoch=i,raw_selection_score=value['selection_score']) for i,value in enumerate(epochs)],
        variants=variants, model_loaded=False, test_sources_read=False, application_enabled=False,
        musical_acceptance=False,
        limitations=['Known local successor labels only; unknown cross-bar edges remain unscored.',
                     'Validation was previously used for development; these are not held-out test results.',
                     'Staff comparison uses old per-piece counts; old per-node staff predictions were not saved.'])
    result['artifacts_sha256'] = {path.name:sha256(path) for path in args.output.iterdir()}
    (args.output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(selected_epoch=selected,graphs_verified=graph_count,
        variants=[dict(input=row['input'],**row['aggregate']) for row in variants]), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
