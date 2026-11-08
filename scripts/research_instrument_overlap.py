"""Test a second segmentation of the same audio for truncated instrument tails.

The first pass owns all attacks, pitches and programs. Only forced window-end
tails may use an explicit note-off from a one-to-one matched second-pass note.
No authored chart, timing grid or desired LN ratio enters this experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


PROTOCOL = dict(onset_tolerance_seconds=.05, interior_margin_seconds=.08,
    shift_fraction=.5, changed_field='offset', maximum_note_seconds=10.,
    purpose='Research instrument endpoints, not authored LN expression')


def refine(base, shifted, base_flags, shifted_flags, *, shifted_start, segment_seconds):
    """Return copies with traceable tail edits; never join across a reattack."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    tolerance = PROTOCOL['onset_tolerance_seconds']
    margin = PROTOCOL['interior_margin_seconds']
    pairs = {}
    keys = {(n['program'], n['pitch']) for n in base if not n['is_drum']}
    # Match every note, including ones ineligible for editing: a second-pass
    # observation cannot be stolen from its closer, non-truncated neighbour.
    for program, pitch in sorted(keys):
        left = [i for i, n in enumerate(base)
            if not n['is_drum'] and (n['program'], n['pitch']) == (program, pitch)]
        right = [j for j, n in enumerate(shifted)
            if not n['is_drum'] and (n['program'], n['pitch']) == (program, pitch)]
        if not right:
            continue
        distance = np.abs(np.asarray([base[i]['onset'] for i in left])[:, None]
            - np.asarray([shifted[j]['onset'] for j in right])[None, :])
        costs = np.c_[np.where(distance <= tolerance, distance / (len(left) + 1), 1e6),
            np.ones((len(left), len(left)))]
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols):
            if j < len(right) and distance[i, j] <= tolerance:
                pairs[left[i]] = right[j]

    result = [dict(n) for n in base]
    decisions = []
    for i, n in enumerate(base):
        if n['is_drum'] or 'forced_segment_boundary' not in base_flags[i]:
            continue
        decision = dict(note_index=i, status='no_matching_attack')
        j = pairs.get(i)
        if j is not None:
            alternate = shifted[j]
            flags = shifted_flags[j]
            end = alternate['offset']
            decision['alternate_note_index'] = j
            decision['status'] = 'alternate_end_unreliable'
            if ('explicit_model_note_off' in flags and
                not {'forced_segment_boundary', 'unresolved_or_postprocessed'} & set(flags)):
                phase = (end-shifted_start) % segment_seconds
                if min(phase, segment_seconds-phase) < margin:
                    decision['status'] = 'alternate_end_near_boundary'
                elif not n['onset'] < end <= n['onset'] + PROTOCOL['maximum_note_seconds']:
                    decision['status'] = 'invalid_duration'
                else:
                    # A separate repeated attack in either pass prevents a
                    # proposed extension. A matched alternate onset is exempt.
                    def reattacks(notes, excluded):
                        return [x['onset'] for k, x in enumerate(notes) if k != excluded
                            and not x['is_drum']
                            and (x['program'], x['pitch']) == (n['program'], n['pitch'])
                            and n['onset'] < x['onset'] < end-1e-7]
                    if reattacks(base, i) or reattacks(shifted, j):
                        decision['status'] = 'intervening_reattack'
                    elif abs(end-n['offset']) < 1e-7:
                        decision['status'] = 'unchanged_explicit_end'
                    else:
                        result[i]['offset'] = end
                        decision.update(status='replaced_from_explicit_prediction',
                            original_offset=n['offset'], predicted_offset=end)
        decisions.append(decision)
    return result, decisions


def load_run(folder):
    read = lambda name: json.loads((folder/name).read_text(encoding='utf-8'))
    report, audit, notes = read('report.json'), read('release-audit.json'), read('notes.json')
    if not audit['decoded_notes_equal'] or audit['notes_sha256'] != sha(folder/'notes.json'):
        raise ValueError('Endpoint audit does not identify these predictions')
    if audit['tokens_sha256'] != sha(folder/'tokens.npz'):
        raise ValueError('Audited tokens changed')
    flags = {r['note_index']: r['endpoint_evidence'] for r in read('release-provenance.json')}
    if set(flags) != set(range(len(notes))):
        raise ValueError('Endpoint provenance is incomplete')
    return report, notes, flags


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--base', type=Path, required=True)
    cli.add_argument('--shifted', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory to preserve completed experiments')
    report, base, flags = load_run(args.base)
    alternate_report, alternate, alternate_flags = load_run(args.shifted)
    identity = report['identity']
    other = alternate_report['identity']
    for key in ('source_commit', 'checkpoint_sha256', 'sources', 'start_ms', 'seconds',
                'context_seconds', 'task', 'strict_weights'):
        if identity[key] != other[key]:
            raise ValueError('Pass inputs differ: '+key)
    if identity.get('segment_phase', 0.) != 0. or other.get('segment_phase') != .5:
        raise ValueError('Expected original and fixed half-window shifted passes')
    if any(report[key] != alternate_report[key] for key in ('segment_samples', 'sample_rate', 'context_end_seconds')):
        raise ValueError('Model/sample clocks differ')
    duration = report['segment_samples']/report['sample_rate']
    expected_shift = int(report['segment_samples']*.5)/report['sample_rate']
    if abs(alternate_report['context_start_seconds']-report['context_start_seconds']-expected_shift) > 1e-8:
        raise ValueError('Physical shift differs from declared segmentation')
    refined, decisions = refine(base, alternate, flags, alternate_flags,
        shifted_start=alternate_report['context_start_seconds'], segment_seconds=duration)
    edits = [r for r in decisions if r['status'] == 'replaced_from_explicit_prediction']
    result = dict(protocol=PROTOCOL, enabled_in_application=False,
        identity=identity, baseline_notes_sha256=sha(args.base/'notes.json'),
        alternate_notes_sha256=sha(args.shifted/'notes.json'), runner_sha256=sha(__file__),
        unchanged_attack_count=len(base), forced_endpoints_considered=len(decisions),
        decisions=dict(Counter(r['status'] for r in decisions)), changed=len(edits),
        extended=sum(r['predicted_offset'] > r['original_offset'] for r in edits),
        shortened=sum(r['predicted_offset'] < r['original_offset'] for r in edits),
        notice='Explicit second-pass note-offs are predictions, not confirmed acoustic or LN endpoints.')
    args.output.mkdir(parents=True)
    for name, value in (('notes.json', refined), ('decisions.json', decisions), ('report.json', result)):
        (args.output/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
