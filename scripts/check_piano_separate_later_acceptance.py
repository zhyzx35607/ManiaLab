"""Apply the predeclared gate for proceeding to an audio connection experiment.

This does not accept a model for the application or certify musical quality.
All inputs are saved validation predictions and the existing symbolic controls.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.build_provenance import sha256
from scripts.summarize_piano_successor_ranking import load, read, require, verify_artifacts

KINDS = ('clean', 'plus1', 'plus2')


def complete_multiple_groups(data):
    """Count every member of every multi-target truth group in paired arrays."""
    groups = defaultdict(list)
    for index in np.flatnonzero(data['actual'] & data['score_mask']):
        groups[str(data['source_ids'][index])].append(int(index))
    groups = [indices for indices in groups.values() if len(indices) > 1]
    return dict(groups=len(groups),
                before=sum(bool(data['before_selected'][ids].all()) for ids in groups),
                after=sum(bool(data['after_selected'][ids].all()) for ids in groups))


def judge(raw_before, raw_after, variants, controls):
    """Evaluate fixed requirements; every failure is retained in the output."""
    require(np.isfinite([raw_before, raw_after]).all(), 'Nonfinite selection score')
    require(set(variants) == set(KINDS), 'Expected all three fixed inputs')
    require(len(controls) == 6 and 'wide_arpeggio' in controls, 'Expected six fixed controls')
    checks = []

    def check(name, before, after, relation):
        passed = after > before if relation == '>' else after >= before if relation == '>=' else after <= before
        checks.append(dict(name=name, before=before, after=after, required=relation, passed=bool(passed)))

    check('raw_macro_mean', raw_before, raw_after, '>')
    improved_competition = False
    for kind in KINDS:
        row = variants[kind]
        for key in ('decoded_macro_f1', 'complete_truth_groups', 'complete_multiple_groups'):
            check(kind+'/'+key, row['before'][key], row['after'][key], '>=')
        if kind != 'clean':
            key = 'true_edges_lost_to_proved_maximum'
            check(kind+'/'+key, row['before'][key], row['after'][key], '<=')
            improved_competition |= row['after'][key] < row['before'][key]
    checks.append(dict(name='overlap_competition_at_least_one_strict_improvement',
                       before=False, after=bool(improved_competition), required='True', passed=bool(improved_competition)))
    for name, row in sorted(controls.items()):
        for kind in ('voice', 'chord'):
            for count, relation in (('tp', '>='), ('fp', '<=')):
                check(name+'/'+kind+'/'+count, row['before'][kind][count], row['after'][kind][count], relation)
    row = controls['wide_arpeggio']
    for count, relation in (('tp', '>='), ('fp', '<=')):
        check('wide_arpeggio/staff_layer_pairs/'+count,
              row['before']['staff_layer_pairs'][count], row['after']['staff_layer_pairs'][count], relation)
    return dict(audio_connection_experiment_allowed=all(row['passed'] for row in checks), checks=checks,
                failed=[row['name'] for row in checks if not row['passed']],
                application_enabled=False, release_allowed=False, musical_acceptance=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'comparison', 'competition', 'controls', 'old-controls', 'output'):
        cli.add_argument('--'+name, type=Path, required=True)
    args = cli.parse_args()
    require(not args.output.exists(), 'Preserve earlier decisions')
    require(args.output.resolve().is_relative_to((ROOT/'test-data').resolve()), 'Keep research output local')
    reports = {}
    for name in ('run', 'comparison', 'competition', 'controls', 'old_controls'):
        folder = getattr(args, name)
        reports[name] = read(folder/'report.json')
        verify_artifacts(folder, reports[name])
    run, comparison, competition, controls, old_controls = (
        reports[name] for name in ('run', 'comparison', 'competition', 'controls', 'old_controls'))
    require(run['identity']['rank_normalization'] == 'independent original and proved-later source/tick/positive means'
            and run['identity']['rank_coefficient'] == run['identity']['added_rank_coefficient'] == .1,
            'Different training intervention')
    require(comparison['identity']['after_run'] == sha256(args.run/'report.json')
            and comparison['selected_weights']['after'] == run['selected_sha256'], 'Different compared run')
    require(comparison['identity']['before_run'] == run['identity']['previous_ranking_report_sha256'],
            'Different primary baseline')
    require(competition['identity']['comparison_sha256'] == sha256(args.comparison/'report.json')
            and competition['identity']['proof_report_sha256'] == run['identity']['proof_report_sha256'],
            'Different proved-competition comparison')
    for report, digest, weights in (
            (controls, sha256(args.run/'report.json'), run['selected_sha256']),
            (old_controls, run['identity']['previous_ranking_report_sha256'], run['identity']['previous_ranking_weights_sha256'])):
        require(report['training_report_sha256'] == digest and report['selected_sha256'] == weights,
                'Control weights or training reference changed')
    require(controls['code_sha256'] == old_controls['code_sha256']
            and controls['fixed_threshold'] == old_controls['fixed_threshold'] == .5, 'Different control definitions')
    competition_rows = {row['input']: row['aggregate'] for row in competition['variants']}
    variants = {}; raw = dict(before=[], after=[])
    for row in comparison['variants']:
        kind = row['input']; total = row['aggregate']; multi = dict(groups=0, before=0, after=0)
        require(len(row['cases']) == 9, 'Different validation works')
        for case in row['cases']:
            counts = complete_multiple_groups(load(args.comparison/case['paired_file']))
            for key, value in counts.items():
                multi[key] += value
        require(multi['groups'] == 550 and total['truth_groups']['decoded']['sources_with_truth'] == 1328
                and total['truth_groups']['decoded']['true_edges'] == 2218, 'Different truth-group population')
        variants[kind] = {}
        for side in ('before', 'after'):
            raw[side].append(total[side]['raw']['macro_f1'])
            variants[kind][side] = dict(decoded_macro_f1=total[side]['decoded']['macro_f1'],
                complete_truth_groups=total['truth_groups']['decoded'][side+'_all'],
                complete_multiple_groups=multi[side],
                true_edges_lost_to_proved_maximum=competition_rows[kind][side]['removed_true_with_proved_maximum_only'])
    old = {row['control']: row for row in old_controls['cases'] if row['model'] == 'source_event_adapted'}
    new = {row['control']: row for row in controls['cases'] if row['model'] == 'source_event_adapted'}
    require(set(old) == set(new), 'Different controls')
    control_rows = {name: dict(before=old[name], after=new[name]) for name in old}
    result = judge(float(np.mean(raw['before'])), float(np.mean(raw['after'])), variants, control_rows)
    result.update(identity={name: sha256(getattr(args, name)/'report.json') for name in reports},
                  variants=variants, controls=control_rows,
                  scope='Minimum gate to another research comparison, not model adoption; development validation and controls.')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    result['artifacts_sha256'] = {'runner-snapshot.py': sha256(args.output/'runner-snapshot.py')}
    (args.output/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(allowed=result['audio_connection_experiment_allowed'], failed=result['failed'])), flush=True)


if __name__ == '__main__':
    main()
