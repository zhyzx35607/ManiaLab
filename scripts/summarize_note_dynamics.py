"""Summarize the fixed dynamics controls without selecting a winning program."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_tsumugi_velocity import read, sha, write


def distribution(values):
    values = np.asarray(values)
    return dict(min=float(values.min()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), max=float(values.max()))


def summarize(folder):
    folder = Path(folder)
    report = read(folder/'report.json')
    rows, synthetic = [], []
    for case in report['cases']:
        source = folder/case['case']
        for name in ('report', 'identity', 'queries'):
            if sha(source/(name+'.json')) != case[name+'_sha256']:
                raise ValueError('Changed saved model evidence')
        identity, queries = read(source/'identity.json'), read(source/'queries.json')
        outputs = {}
        for result in read(source/'report.json')['outputs']:
            gain = result['gain_db']
            tag = 'silence' if gain is None else f'gain-{gain:g}'
            if sha(source/(tag+'.npz')) != result['evidence_sha256']:
                raise ValueError('Changed velocity/logit evidence')
            with np.load(source/(tag+'.npz'), allow_pickle=False) as data:
                if not np.all(data['observations'] == 1):
                    raise ValueError('Query missing or observed twice')
                outputs[gain] = data['expected_velocity'].copy()
        if identity['kind'] == 'synthetic_control':
            by_program = defaultdict(dict)
            for index, q in enumerate(queries):
                by_program[q['program']][q['original_index']] = float(outputs[0.][index])
            for program, predictions in by_program.items():
                difference = predictions[0] - predictions[1]
                expected = identity['relation']
                stronger_margin = (difference if expected == 'lower_stronger' else
                                   -difference if expected == 'higher_stronger' else None)
                synthetic.append(dict(case=case['case'], pitches=identity['pitches'],
                    relation=expected, assumed_program=program, expected_velocity=predictions,
                    component_rms=identity['component_rms'], lower_minus_higher=difference,
                    known_stronger_margin=stronger_margin,
                    known_stronger_ranked_higher=stronger_margin > 0 if stronger_margin is not None else None))
            continue
        stack = np.stack([outputs[g] for g in (-12., -6., 0., 6.)])
        deltas = np.diff(stack, axis=0)
        by_original = defaultdict(list)
        for i, q in enumerate(queries):
            by_original[q['original_index']].append(i)
        ranges = np.array([np.ptp(outputs[0.][indices]) for indices in by_original.values()])
        inversions = [dict(query=i, original_index=queries[i]['original_index'],
            pitch=queries[i]['pitch'], assumed_program=queries[i]['program'],
            predicted_velocities=stack[:,i].tolist())
            for i in range(len(queries)) if np.any(deltas[:,i] < -1e-5)]
        rows.append(dict(case=case['case'], original_notes=len(by_original), queries=len(queries),
            gain_deltas=distribution(deltas.flatten()),
            monotonically_nondecreasing_queries=int(np.sum(np.all(deltas>=-1e-5,axis=0))),
            strictly_increasing_queries=int(np.sum(np.all(deltas>1e-5,axis=0))),
            nonmonotonic_queries=inversions,
            baseline_velocity=distribution(outputs[0.]),
            within_broad_class_program_range=distribution(ranges),
            program_ranges_by_original=[dict(original_index=i, range=float(np.ptp(outputs[0.][indices])),
                assumed_programs=[queries[k]['program'] for k in indices],
                predicted_velocity=[float(outputs[0.][k]) for k in indices]) for i,indices in by_original.items()],
            silence_velocity=distribution(outputs[None]),
            zero_above_original=int(np.sum(outputs[None]>outputs[0.])),
            output_not_presence_probability=True))
    unequal = [row for row in synthetic if row['known_stronger_margin'] is not None]
    return dict(source_report_sha256=sha(folder/'report.json'),
        source_identity_sha256=sha(folder/'identity.json'), summary_script_sha256=sha(__file__),
        real_clips=rows, synthetic=synthetic,
        synthetic_stronger_correct=sum(row['known_stronger_ranked_higher'] for row in unequal),
        synthetic_stronger_comparisons=len(unequal),
        enabled_in_application=False, musical_acceptance=False,
        caution='Program is a hypothesis; velocity is conditional on queries, not note confidence, loudness truth or melody identity.')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args=cli.parse_args()
    result=summarize(args.input)
    write(args.output,result)
    print(json.dumps(dict(real_clips=[{k:v for k,v in row.items() if k not in
        ('nonmonotonic_queries','program_ranges_by_original')} for row in result['real_clips']],
        synthetic_stronger_correct=result['synthetic_stronger_correct'],
        synthetic_stronger_comparisons=result['synthetic_stronger_comparisons']),indent=2))


if __name__=='__main__':
    main()
