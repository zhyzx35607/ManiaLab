"""Audit unambiguous same-source decay evidence without changing guitar attacks."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_guitar_vocal_benchmark as evaluation
from scripts.research_hf_instrument import write_new

read, require, sha, verify = (evaluation.read_json, evaluation.require,
                            evaluation.sha, evaluation.verified_hash)
ONSET_TOLERANCE = .05
UNKNOWN = frozenset(('reattack', 'duration_limit', 'input_boundary', 'frame_inactive'))


def transfer_releases(targets, donors, *, context_end):
    """Transfer only a unique Basic Pitch decay to an explicitly unknown tail.

    Graph degrees use every non-drum event, including ineligible tails. Filtering
    those first could manufacture certainty for a rapid repeat or unison.
    Neither onset scores nor reference annotations participate in this decision.
    """
    evaluation.validate_notes(targets, reference=False)
    evaluation.validate_notes(donors, reference=False)
    require(evaluation.finite(context_end) and context_end > 0, 'Invalid source duration')
    edges = [[] for _ in targets]
    incoming = [[] for _ in donors]
    for i, target in enumerate(targets):
        for j, donor in enumerate(donors):
            if (not target['is_drum'] and not donor['is_drum']
                    and target['pitch'] == donor['pitch']
                    and abs(target['onset']-donor['onset']) <= ONSET_TOLERANCE):
                edges[i].append(j)
                incoming[j].append(i)
    output, trace = [], []
    for i, target in enumerate(targets):
        note = dict(target)
        crossed = []
        decision = 'target_tail_not_explicitly_unknown'
        if target.get('release_observed') is False and target.get('release_reason') in UNKNOWN:
            decision = 'no_unique_bidirectional_attack_match'
            if len(edges[i]) == 1 and len(incoming[edges[i][0]]) == 1:
                j = edges[i][0]
                donor = donors[j]
                decision = 'donor_has_no_independent_decay'
                if donor.get('release_reason') == 'frame_decay' and donor.get('release_observed') is not False:
                    decision = 'donor_tail_outside_valid_context'
                    if target['onset'] < donor['offset'] < context_end:
                        crossed = [k for k, other in enumerate(targets) if not other['is_drum']
                                   and other['pitch'] == target['pitch']
                                   and target['onset'] < other['onset'] < donor['offset']]
                        decision = 'donor_tail_crosses_later_target_attack'
                        if not crossed:
                            decision = 'transferred_model_decay'
                            note.update(offset=donor['offset'], release_reason='frame_decay', release_observed=True,
                                release_evidence=dict(source='application_basic_pitch', donor_index=j,
                                    donor_onset=donor['onset'], donor_offset=donor['offset'],
                                    donor_reason=donor['release_reason'], original_offset=target['offset'],
                                    original_reason=target['release_reason'],
                                    semantics='another_model_decay_hypothesis_not_physical_ground_truth'))
        output.append(note)
        trace.append(dict(target_index=i, matching_donor_indices=edges[i],
                          crossed_later_target_indices=crossed, decision=decision))
    require([(n['onset'], n['pitch'], n['is_drum']) for n in output]
            == [(n['onset'], n['pitch'], n['is_drum']) for n in targets], 'Changed source attacks')
    evaluation.validate_notes(output, reference=False)
    return output, trace


def run(source, output):
    source, output = Path(source), Path(output)
    require(not output.exists(), 'Preserve previous evidence')
    report = read(source/'report.json')
    plan = report['plan']
    require(plan['model_order'] == ['guitar_gaps', 'guitar_fl'] and len(plan['cases']) == 60,
            'Require complete two-model guitar comparison')
    names = [case['name'] for case in plan['cases']]
    require([(row['model'], row['name']) for row in report['rows']]
            == [(model, name) for model in plan['model_order'] for name in names], 'Incomplete source cohort')
    for path, digest in plan['cohort_files'].items():
        verify(ROOT/path, digest)
    identity = dict(source_report_sha256=sha(source/'report.json'), runner_sha256=sha(__file__),
        evaluator_sha256=sha(evaluation.__file__), matcher_sha256=sha(ROOT/'scripts/evaluate_instrument_benchmark.py'),
        onset_tolerance_seconds=ONSET_TOLERANCE, donor_family='same_source_basic_pitch',
        target_unknown_reasons=sorted(UNKNOWN), donor_reason='frame_decay',
        require_unique_both_directions=True, reject_donor_crossing_later_target_attack=True,
        uses_reference_to_choose_tail=False,
        new_neural_inference=False, selected_after_partial_model_observation=True)
    write_new(output/'plan.json', identity)
    cases = {case['name']: case for case in plan['cases']}
    rows = []
    for original in report['rows']:
        case, model = cases[original['name']], original['model']
        folder = source/model/case['name']
        verify(folder/'report.json', original['report_sha256'])
        verify(folder/'notes.json', original['notes_sha256'])
        donor = case['baseline']['application']
        donor_path = Path(donor['folder'])/'notes.json'
        verify(donor_path, donor['notes_sha256'])
        reference = case['references']['strings']
        verify(reference['path'], reference['sha256'])
        targets, donors = read(folder/'notes.json'), read(donor_path)
        notes, trace = transfer_releases(targets, donors, context_end=case['duration_seconds'])
        metrics, scope = evaluation.window_metrics(read(reference['path']), notes, begin=0, end=case['seconds'],
            context_start=0, context_end=case['duration_seconds'])
        for tolerance in evaluation.TOLERANCES:
            before = original['metrics'][tolerance]['pitched_onsets']
            after = metrics[tolerance]['pitched_onsets']
            require(all(before[k] == after[k] for k in ('reference', 'predicted', 'matched', 'f1', 'mean_onset_error_ms')),
                    'Attack scoring changed')
        out = output/model/case['name']
        write_new(out/'notes.json', notes)
        write_new(out/'trace.json', trace)
        rows.append(dict(name=case['name'], model=model, metrics=metrics, interval_scope=scope,
            before=original['metrics'], notes_sha256=sha(out/'notes.json'), trace_sha256=sha(out/'trace.json'),
            decisions=dict(Counter(item['decision'] for item in trace)),
            scored_transfers=sum(item['decision'] == 'transferred_model_decay' and targets[i]['onset'] < case['seconds']
                                 for i, item in enumerate(trace))))
    groups = {model: dict(before=report['groups'][model],
        after=evaluation.aggregate([row for row in rows if row['model'] == model])) for model in plan['model_order']}
    verify(__file__, identity['runner_sha256'])
    verify(source/'report.json', identity['source_report_sha256'])
    write_new(output/'report.json', dict(identity=identity, rows=rows, groups=groups,
        enabled_in_application=False, musical_acceptance=False,
        limitations=['A post-observation development probe, not unseen validation.',
            'Agreement on a pitch/attack is not proof that a donor tail is correct.',
            'Model decay evidence is not an authored LN endpoint; all raw predictions retained.',
            'Donor same-pitch cuts and input boundaries never supply independent decay evidence.',
            'A donor crossing a later same-pitch target attack is rejected, not truncated.',
            'Uniqueness and non-crossing do not prove individual guitar-string identity.']))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    run(args.source, args.output)


if __name__ == '__main__':
    main()
