"""Pair two already audited ranking runs, preserving edge and chord-group losses."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_provenance import sha256
from scripts.piano_successor_decode import select_successor_groups
from scripts.summarize_piano_successor_ranking import (
    align_baseline, compare_record, decoding_diagnostics, load, paired_transitions,
    pooled_metrics, read, require, sum_fields, truth_group_retention, verify_artifacts,
    verify_snapshots,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('before-run', 'after-run', 'before-summary', 'after-summary', 'output'):
        cli.add_argument('--'+name, type=Path, required=True)
    args = cli.parse_args()
    require(args.output.resolve().is_relative_to((ROOT/'test-data').resolve()),
            'Event-level comparison must stay under local test-data')
    if args.output.exists():
        raise ValueError('Preserve prior paired comparisons')
    runs, summaries = {}, {}
    for side in ('before', 'after'):
        folder = getattr(args, side+'_run'); run = read(folder/'report.json')
        summary_folder = getattr(args, side+'_summary')
        summary = read(summary_folder/'report.json')
        verify_artifacts(folder, run); verify_snapshots(folder, run)
        verify_artifacts(summary_folder, summary)
        require(summary['training_report_sha256'] == sha256(folder/'report.json') and
                summary['selected_sha256'] == run['selected_sha256'], 'Different summarized run')
        runs[side], summaries[side] = run, summary
    before, after = (runs[side]['identity'] for side in ('before', 'after'))
    require(after['previous_ranking_report_sha256'] == sha256(args.before_run/'report.json'),
            'New experiment did not declare this previous ranking reference')
    require(after['previous_ranking_weights_sha256'] == runs['before']['selected_sha256'],
            'Previous selected weights differ')
    require(after['code_sha256']['ranking_common'] == before['code_sha256']['ranking'],
            'Ranking common helper differs from the frozen earlier helper')
    require(after['same_tick_known_negatives_included'] is True, 'Expected the same-tick intervention')
    for key in ('source_revision', 'pretrained_sha256', 'dataset_sha256', 'seed', 'epochs',
                'lr', 'weight_decay', 'voice_positive_weight', 'rank_coefficient', 'rank_margin',
                'threshold', 'selection', 'input_schedule', 'unknown_edge_rank_supervision'):
        require(before[key] == after[key], 'Changed training condition: '+key)
    for key in before['code_sha256'].keys()-{'runner', 'ranking', 'protocol'}:
        require(before['code_sha256'][key] == after['code_sha256'][key], 'Changed shared code: '+key)
    for key in ('baseline_report_sha256', 'decode_baseline_report_sha256', 'summarizer_sha256'):
        require(summaries['before'][key] == summaries['after'][key], 'Different audited baseline')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    variants = []
    for kind in ('clean', 'plus1', 'plus2'):
        earlier = next(v for v in summaries['before']['variants'] if v['input'] == kind)
        later = next(v for v in summaries['after']['variants'] if v['input'] == kind)
        old_cases = {row['name']: row for row in earlier['cases']}
        require(set(old_cases) == {row['name'] for row in later['cases']}, 'Changed works')
        cases = []
        for index, row in enumerate(later['cases']):
            name = row['name']; old_row = old_cases[name]
            data = load(args.after_summary/row['paired_file'])
            previous = align_baseline(load(args.before_summary/old_row['paired_file']), data)
            old_selected = select_successor_groups(previous['source_ids'], previous['target_ticks'], previous['logits'])
            new_selected = select_successor_groups(data['source_ids'], data['target_ticks'], data['logits'])
            require(np.array_equal(old_selected, previous['new_selected']) and
                    np.array_equal(new_selected, data['new_selected']), 'Changed decoded selection')
            old_metrics = decoding_diagnostics(previous, old_selected)
            new_metrics = decoding_diagnostics(data, new_selected)
            compare_record(old_metrics, old_row['after']); compare_record(new_metrics, row['after'])
            transitions, groups = {}, {}
            for mode, old_choice, new_choice in (
                    ('raw', previous['logits']>=0, data['logits']>=0),
                    ('decoded', old_selected, new_selected)):
                transitions[mode] = paired_transitions(data['actual'], data['score_mask'], old_choice, new_choice)
                groups[mode] = truth_group_retention(data['source_ids'], data['actual'], data['score_mask'],
                                                   old_choice, new_choice)
                require(new_metrics[mode]['tp']-old_metrics[mode]['tp'] ==
                        transitions[mode]['fn_to_tp']-transitions[mode]['tp_to_fn'], 'TP transfer mismatch')
                require(new_metrics[mode]['fp']-old_metrics[mode]['fp'] ==
                        transitions[mode]['tn_to_fp']-transitions[mode]['fp_to_tn'], 'FP transfer mismatch')
            filename = f'{kind}-{index:02d}.npz'
            fields = ('source_ids', 'target_ids', 'target_ticks', 'actual', 'score_mask')
            np.savez_compressed(args.output/filename, **{key: data[key] for key in fields},
                                before_logits=previous['logits'], after_logits=data['logits'],
                                before_selected=old_selected, after_selected=new_selected)
            cases.append(dict(name=name, before=old_metrics, after=new_metrics, transitions=transitions,
                truth_groups=groups, paired_file=filename,
                staff=dict(before_correct=old_row['staff']['after_correct'],
                           after_correct=row['staff']['after_correct'], notes=row['staff']['notes'])))
        totals = dict(before={mode: pooled_metrics(cases, 'before', mode) for mode in ('raw', 'decoded')},
                      after={mode: pooled_metrics(cases, 'after', mode) for mode in ('raw', 'decoded')},
                      transitions={mode: sum_fields([r['transitions'][mode] for r in cases]) for mode in ('raw', 'decoded')},
                      truth_groups={mode: sum_fields([r['truth_groups'][mode] for r in cases]) for mode in ('raw', 'decoded')},
                      staff=sum_fields([r['staff'] for r in cases]))
        totals['decode_losses'] = {side: {key: sum(r[side][key] for r in cases)
            for key in cases[0][side] if key.startswith('removed_')} for side in ('before', 'after')}
        variants.append(dict(input=kind, cases=cases, aggregate=totals))
    report = dict(identity={key: sha256(getattr(args, key)/'report.json')
                            for key in ('before_run', 'after_run', 'before_summary', 'after_summary')},
        selected_epochs={side: runs[side]['selected_epoch'] for side in runs},
        selected_weights={side: runs[side]['selected_sha256'] for side in runs},
        variants=variants, model_loaded=False, application_enabled=False, musical_acceptance=False,
        artifacts_sha256={p.name: sha256(p) for p in args.output.iterdir()})
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps([dict(input=v['input'], **v['aggregate']) for v in variants]), flush=True)


if __name__ == '__main__':
    main()
