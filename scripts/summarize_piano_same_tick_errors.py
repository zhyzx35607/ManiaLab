"""Audit saved same-tick errors using validation-only written-note provenance.

No model or training cache is loaded. Local written staff/voice labels describe
this fixed successor task; they are not global melody or acoustic truth.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_provenance import sha256
from scripts.summarize_piano_successor_ranking import (
    align_baseline, load, read, require, sum_fields, truth_group_retention,
    verify_artifacts,
)

SOURCE_REPORT_SHA256 = '280ccd7a9e47a5b18b7f54a301b69924e7b267c75024796399bdfd2091bc1ecb'


def source_context(source):
    """Reconstruct known local successor truth from immutable source notation."""
    written = {row['note']['id']: row['note'] for row in source['written']}
    events = {event['id']: event for event in source['representation']['events']}
    require(len(written)==len(source['written']) and len(events)==len(source['representation']['events']),
            'Duplicate source identity')
    first, last, eligible, bad = {}, {}, {}, set()
    for identifier, event in events.items():
        members = [written[key] for key in event['written_ids']]
        require(bool(members) and members[0]['start_tick']==event['start_tick']
                and members[-1]['end_tick']==event['end_tick'], 'Changed source interval')
        first[identifier], last[identifier] = members[0], members[-1]
        eligible[identifier] = bool(event['supervision_eligible']) and all(n['voice'] for n in members)
        if not eligible[identifier]:
            bad.update((n['part'], n['measure']) for n in members)
    structural = {}
    for note in written.values():
        match = re.fullmatch(r'part\[(\d+)\]/measure\[(\d+)\]/note\[(\d+)\]', note['id'])
        if match:
            structural[int(match[1])] = note['part']
    for identifier, _ in source['excluded']:
        match = re.fullmatch(r'part\[(\d+)\]/measure\[(\d+)\]/note\[(\d+)\]', identifier)
        require(match is not None and int(match[1]) in structural, 'Unknown excluded-note provenance')
        bad.add((structural[int(match[1])], int(match[2])))
    key = lambda note: (note['part'], note['measure'], note['staff'], note['voice'])
    attacks = defaultdict(list)
    for identifier, note in first.items():
        if eligible[identifier] and (note['part'], note['measure']) not in bad:
            attacks[key(note)].append(identifier)
    truth = {}
    for identifier, note in last.items():
        if not eligible[identifier] or (note['part'], note['measure']) in bad:
            continue
        later = [other for other in attacks[key(note)] if other!=identifier
                 and events[other]['start_tick']>=events[identifier]['end_tick']]
        if later:
            tick = min(events[other]['start_tick'] for other in later)
            truth[identifier] = {other for other in later if events[other]['start_tick']==tick}
    return dict(events=events, first=first, last=last, eligible=eligible, bad=bad, truth=truth)


def classify_same_tick_negatives(data, context, piece_id):
    """Validate scored labels and classify known negatives at the true tick.

    Codes: 0 outside that subset; 1 same staff/different voice; 2 different
    staff. Cross-part or other unknown edges never become false positives.
    Source identity uses the final written fragment, including verified ties.
    """
    size = len(data['source_ids'])
    for key in ('target_ids', 'target_ticks', 'actual', 'score_mask'):
        require(data[key].shape==(size,), 'Invalid paired array shape')
    require(data['actual'].dtype==bool and data['score_mask'].dtype==bool, 'Expected Boolean labels/masks')
    prefix = piece_id+':'
    ids = []
    for field in ('source_ids', 'target_ids'):
        values = list(map(str, data[field]))
        require(all(value.startswith(prefix) for value in values), 'Wrong piece identity prefix')
        values = [value[len(prefix):] for value in values]
        require(all(value in context['events'] for value in values), 'Unknown source event identity')
        ids.append(values)
    sources, targets = ids
    positive_targets = defaultdict(set)
    codes = np.zeros(size, dtype=np.uint8)
    for index, (source, target) in enumerate(zip(sources, targets)):
        a, b = context['last'][source], context['first'][target]
        require(int(data['target_ticks'][index])==context['events'][target]['start_tick'], 'Changed target tick')
        require(context['events'][source]['start_tick']<context['events'][target]['start_tick'],
                'Expected strictly future attack candidate')
        expected_mask = bool(context['eligible'][source] and context['eligible'][target]
            and (a['part'],a['measure'])==(b['part'],b['measure'])
            and (b['part'],b['measure']) not in context['bad'])
        require(bool(data['score_mask'][index])==expected_mask, 'Scored mask differs from source notation')
        true_targets = context['truth'].get(source, set())
        expected_label = expected_mask and target in true_targets
        require(bool(data['actual'][index])==expected_label, 'Scored label differs from source notation')
        if expected_label:
            positive_targets[source].add(target)
        if expected_mask and not expected_label and true_targets:
            true_tick = context['events'][next(iter(true_targets))]['start_tick']
            if int(data['target_ticks'][index])==true_tick:
                if a['staff']!=b['staff']:
                    codes[index] = 2
                else:
                    require(a['voice']!=b['voice'], 'Known false edge shares the complete local voice key')
                    codes[index] = 1
    for source, targets in positive_targets.items():
        require(targets==context['truth'][source], 'Saved graph omitted part of a true successor chord group')
    return codes


def false_positive_transitions(subset, before, after):
    """Count both directions, with only the supplied known-negative subset."""
    require(subset.shape==before.shape==after.shape and subset.dtype==before.dtype==after.dtype==bool,
            'Expected equal Boolean prediction arrays')
    return dict(before_fp=int((subset & before).sum()), after_fp=int((subset & after).sum()),
                repaired_fp=int((subset & before & ~after).sum()),
                introduced_fp=int((subset & ~before & after).sum()),
                persistent_fp=int((subset & before & after).sum()))


def error_record(data, previous, categories):
    result = {}
    for name, code in (('same_staff_different_voice', 1), ('different_staff', 2), ('all_same_tick', None)):
        subset = categories>0 if code is None else categories==code
        result[name] = dict(known_negative_edges=int(subset.sum()),
            sources=len(set(map(str, data['source_ids'][subset]))),
            raw=false_positive_transitions(subset, previous['logits']>=0, data['logits']>=0),
            decoded=false_positive_transitions(subset, previous['new_selected'], data['new_selected']))
    counts = Counter(map(str, data['source_ids'][data['actual'] & data['score_mask']]))
    multi = np.asarray([counts[str(source)]>1 for source in data['source_ids']], dtype=bool)
    groups = {}
    for mode, before, after in (('raw', previous['logits']>=0, data['logits']>=0),
                                ('decoded', previous['new_selected'], data['new_selected'])):
        groups[mode] = dict(all=truth_group_retention(data['source_ids'], data['actual'], data['score_mask'], before, after),
            multiple_true_targets=truth_group_retention(data['source_ids'], data['actual'], data['score_mask'] & multi, before, after))
    return dict(negative_categories=result, true_successor_groups=groups)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('before-summary', 'after-summary', 'output'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--source-report', type=Path, default=ROOT/'test-data/piano-svsep/musicxml-events-v2/report.json')
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve previous diagnostics')
    require(args.output.resolve().is_relative_to((ROOT/'test-data').resolve()),
            'Per-edge research artifacts must remain in local test-data')
    before, after = read(args.before_summary/'report.json'), read(args.after_summary/'report.json')
    verify_artifacts(args.before_summary, before); verify_artifacts(args.after_summary, after)
    require(sha256(args.source_report)==SOURCE_REPORT_SHA256, 'Changed frozen source report')
    source_report = read(args.source_report)
    source_rows = {row['name']: row for row in source_report['cases']
                   if row['split']=='validation' and row['status']=='complete'}
    for key in ('baseline_report_sha256', 'decode_baseline_report_sha256', 'summarizer_sha256'):
        require(before[key]==after[key], 'Different paired-summary baseline')
    contexts, detail_hashes = {}, {}
    for row in after['variants'][0]['cases']:
        require(row['name'] in source_rows, 'Only completed validation sources may be read')
        source_row = source_rows[row['name']]
        path = args.source_report.parent/source_row['detail_file']
        digest = sha256(path)
        require(digest==source_report['artifacts_sha256'][source_row['detail_file']], 'Changed source detail')
        contexts[row['name']] = source_context(read(path)['source'])
        detail_hashes[row['name']] = digest
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    variants = []
    for kind in ('clean', 'plus1', 'plus2'):
        older = next(v for v in before['variants'] if v['input']==kind)
        newer = next(v for v in after['variants'] if v['input']==kind)
        old_cases = {row['name']:row for row in older['cases']}
        require(set(old_cases)=={row['name'] for row in newer['cases']}==set(contexts), 'Changed validation case coverage')
        rows = []
        for index, row in enumerate(newer['cases']):
            name = row['name']; data = load(args.after_summary/row['paired_file'])
            previous = align_baseline(load(args.before_summary/old_cases[name]['paired_file']), data)
            codes = classify_same_tick_negatives(data, contexts[name], source_rows[name]['input_sha256'])
            result = error_record(data, previous, codes)
            for mode in ('raw', 'decoded'):
                require(result['true_successor_groups'][mode]['all']['after_all']==row['truth_groups'][mode]['after_all']
                    and result['true_successor_groups'][mode]['all']['before_all']==old_cases[name]['truth_groups'][mode]['after_all'],
                    'True-group retention differs from audited summaries')
            filename = f'{kind}-categories-{index:02d}.npz'
            np.savez_compressed(args.output/filename, source_ids=data['source_ids'], target_ids=data['target_ids'], category=codes)
            rows.append(dict(name=name, category_file=filename, **result))
        negatives = {}
        for category in rows[0]['negative_categories']:
            records = [row['negative_categories'][category] for row in rows]
            negatives[category] = dict(known_negative_edges=sum(r['known_negative_edges'] for r in records),
                sources=sum(r['sources'] for r in records),
                **{mode:sum_fields([r[mode] for r in records]) for mode in ('raw','decoded')})
        groups = {mode:{group:sum_fields([r['true_successor_groups'][mode][group] for r in rows])
                        for group in ('all','multiple_true_targets')} for mode in ('raw','decoded')}
        variants.append(dict(input=kind,cases=rows,aggregate=dict(negative_categories=negatives,true_successor_groups=groups)))
    report = dict(identity=dict(before_summary_sha256=sha256(args.before_summary/'report.json'),
        after_summary_sha256=sha256(args.after_summary/'report.json'),source_report_sha256=sha256(args.source_report),
        source_detail_sha256=detail_hashes,runner_sha256=sha256(__file__),
        before_weights_sha256=before['selected_sha256'],after_weights_sha256=after['selected_sha256']),
        variants=variants,model_loaded=False,training_cache_read=False,test_sources_read=False,
        application_enabled=False,musical_acceptance=False,
        scope='Local written successor labels; cross-part unknowns excluded, multiple true targets retained.',
        artifacts_sha256={path.name:sha256(path) for path in args.output.iterdir()})
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([dict(input=v['input'],**v['aggregate']) for v in variants]),flush=True)


if __name__=='__main__':
    main()
