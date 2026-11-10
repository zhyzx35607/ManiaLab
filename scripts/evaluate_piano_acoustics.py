"""Score key and sustain endpoints separately, retaining every predicted note."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def pairs(reference, predicted, *, strict=False, offsets=False):
    result = []
    for pitch in sorted({n['pitch'] for n in reference}):
        ri = [i for i,n in enumerate(reference) if n['pitch'] == pitch]
        pi = [i for i,n in enumerate(predicted) if n['pitch'] == pitch]
        if not pi:
            continue
        ref = np.asarray([[reference[i]['onset'], reference[i]['offset']] for i in ri])
        pred = np.asarray([[predicted[i]['onset'], predicted[i]['offset']] for i in pi])
        delta = abs(ref[:, None]-pred[None, :])
        allowed = delta[:, :, 0] <= (.04 if strict else .05)
        if offsets:
            tolerance = np.full(len(ref), .08) if strict else np.maximum(.05, .2*(ref[:, 1]-ref[:, 0]))
            allowed &= delta[:, :, 1] <= tolerance[:, None]
        cost = np.concatenate((np.where(allowed, delta[:, :, 0]/(len(ref)+1), 2.),
                               np.ones((len(ref), len(ref)))), axis=1)
        row, col = linear_sum_assignment(cost)
        result.extend((ri[i], pi[j]) for i,j in zip(row,col) if j < len(pi) and allowed[i,j])
    return result


def counts(matched, reference, predicted):
    return dict(matched=matched, reference=reference, predicted=predicted,
                precision=matched/predicted if predicted else 0.,
                recall=matched/reference if reference else 0.,
                f1=2*matched/(reference+predicted) if reference+predicted else 0.)


def metrics(reference, predicted, *, strict=False):
    matched_onset = pairs(reference, predicted, strict=strict)
    matched_full = pairs(reference, predicted, strict=strict, offsets=True)
    long_indices = {i for i,n in enumerate(reference) if n['offset']-n['onset'] >= 2.}
    long_matches = sum(i in long_indices for i,j in matched_full)
    matched_map = dict(matched_onset)
    previous = {}
    repeats = []
    for i,note in sorted(enumerate(reference), key=lambda item: item[1]['onset']):
        if note['pitch'] in previous:
            j = previous[note['pitch']]
            before = reference[j]
            gap = note['onset']-before['offset']
            if 0 <= gap <= .08:
                repeated = i in matched_map and j in matched_map
                bridge = (i not in matched_map and any(p['pitch'] == note['pitch']
                    and abs(p['onset']-before['onset']) <= (.04 if strict else .05)
                    and p['offset'] > note['onset']+.08 for p in predicted))
                repeats.append(dict(both_onsets_matched=repeated, bridged_missing_reattack=bridge))
        previous[note['pitch']] = i
    onset_errors = [abs(reference[i]['onset']-predicted[j]['onset'])*1000 for i,j in matched_onset]
    offset_errors = [abs(reference[i]['offset']-predicted[j]['offset'])*1000 for i,j in matched_onset]
    return dict(onsets=counts(len(matched_onset),len(reference),len(predicted)),
                intervals=counts(len(matched_full),len(reference),len(predicted)),
                long_reference=len(long_indices), long_matched=long_matches,
                predicted_long=sum(n['offset']-n['onset'] >= 2. for n in predicted),
                mean_onset_error_ms=float(np.mean(onset_errors)) if onset_errors else None,
                median_offset_error_on_matched_onsets_ms=float(np.median(offset_errors)) if offset_errors else None,
                close_repeats=dict(reference=len(repeats),
                    both_onsets_matched=sum(r['both_onsets_matched'] for r in repeats),
                    bridged_missing_reattack=sum(r['bridged_missing_reattack'] for r in repeats)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=Path('test-data/smd/selected-v1/benchmark'))
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--specialist', default='specialist-v1')
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier evaluation')
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    manifest = read(args.root/'manifest.json')
    folders = dict(basic=args.root/'basic-v1',specialist=args.root/args.specialist)
    reports = {backend:read(folder/'report.json') for backend,folder in folders.items()}
    for backend,report in reports.items():
        if report['identity']['manifest_sha256'] != sha(args.root/'manifest.json'):
            raise ValueError('Model inputs used a different benchmark')
        if report['identity']['runner_sha256'] != sha(folders[backend]/'runner-snapshot.py'):
            raise ValueError('Inference implementation snapshot changed')
    rows = []
    for case in manifest['cases']:
        for name in ('audio', 'labels', 'sustain_labels'):
            if sha(case[name]) != case[name+'_sha256']:
                raise ValueError('Benchmark evidence changed')
        if case['unobservable_reference_offsets'] or case['unobservable_sustain_offsets']:
            raise ValueError('Reference tail exceeds available audio; report separately before scoring')
        scope = lambda notes:[n for n in notes if 0 <= n['onset'] < case['score_seconds']]
        references = dict(key=scope(read(case['labels'])), sustain=scope(read(case['sustain_labels'])))
        row = dict(name=case['name'], audio_sha256=case['audio_sha256'],
                   score_seconds=case['score_seconds'], models={})
        for backend,report in reports.items():
            folder = folders[backend]
            path = folder/(case['name']+'-notes.json')
            evidence = next(r for r in report['cases'] if r['name'] == case['name'])
            if sha(path) != evidence['notes_sha256'] or evidence['audio_sha256'] != case['audio_sha256']:
                raise ValueError('Predictions differ from completed inference')
            predicted = scope(read(path))
            if any(not np.isfinite([n['onset'],n['offset'],n['pitch']]).all()
                   or n['offset'] <= n['onset'] for n in predicted):
                raise ValueError('Invalid returned intervals must be audited before scoring')
            row['models'][backend] = dict(evidence=evidence,
                metrics={semantics:{name:metrics(ref,predicted,strict=strict)
                    for name,strict in [('standard',False),('strict',True)]}
                    for semantics,ref in references.items()})
        rows.append(row)
        print(json.dumps(dict(name=row['name'],results={k:v['metrics']['sustain']['standard']
                                                     for k,v in row['models'].items()})),flush=True)
    summary = {}
    for backend in reports:
        summary[backend] = {}
        for semantics in ('key','sustain'):
            summary[backend][semantics] = {}
            for tolerance in ('standard','strict'):
                values = [r['models'][backend]['metrics'][semantics][tolerance] for r in rows]
                summary[backend][semantics][tolerance] = dict(
                    macro={metric:float(np.mean([v[metric]['f1'] for v in values])) for metric in ('onsets','intervals')},
                    pooled={metric:counts(*(sum(v[metric][key] for v in values) for key in ('matched','reference','predicted')))
                            for metric in ('onsets','intervals')},
                    long_reference=sum(v['long_reference'] for v in values),
                    long_matched=sum(v['long_matched'] for v in values),
                    predicted_long=sum(v['predicted_long'] for v in values))
    result = dict(manifest_sha256=sha(args.root/'manifest.json'), protocol_sha256=manifest['protocol_sha256'],
                  pedal_protocol_sha256=manifest['pedal_protocol_sha256'], runner_sha256=sha(__file__),
                  models={k:r['identity'] for k,r in reports.items()}, rows=rows, summary=summary,
                  default_enabled=False, optional_application_backend=True,
                  notice='Public natural-piano benchmark; MIDI pedal sustain and key release are separate, neither is a chart LN target.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
