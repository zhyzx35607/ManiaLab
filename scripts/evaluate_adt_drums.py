"""Score fixed ADT outputs on physical drum labels and authored selections."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.adt_token_audit import GROUPS, group

CLASSES = tuple(GROUPS)
LABELS = dict(KD='kick', SD='snare', HH='hi-hat', TT='tom', CY='cymbal', OT='other')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def match_times(reference, predicted, tolerance):
    """Maximum-cardinality same-class matching, then minimum absolute error.

    Retain original indices, duplicate predictions, and out-of-audio proposals.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not tolerance > 0 or not all(np.isfinite(t) for t in [*reference, *predicted]):
        raise ValueError('Expected finite clocks and positive tolerance')
    pairs = []
    if reference and predicted:
        delta = np.abs(np.asarray(reference)[:, None] - np.asarray(predicted)[None, :])
        allowed = delta <= tolerance
        costs = np.concatenate((np.where(allowed, delta/tolerance/(len(reference)+1), 1e6),
                                np.ones((len(reference), len(reference)))), axis=1)
        rows, cols = linear_sum_assignment(costs)
        pairs = [[int(i), int(j)] for i, j in zip(rows, cols)
                 if j < len(predicted) and allowed[i, j]]
    ri, pi = {i for i, _ in pairs}, {j for _, j in pairs}
    errors = [predicted[j]-reference[i] for i, j in pairs]
    n, p, m = len(reference), len(predicted), len(pairs)
    return dict(reference=n, predicted=p, matched=m, missing=n-m, extra=p-m,
                precision=m/p if p else 0., recall=m/n if n else 0.,
                f1=2*m/(n+p) if n+p else 0., pairs=pairs,
                unmatched_reference=[i for i in range(n) if i not in ri],
                unmatched_predicted=[i for i in range(p) if i not in pi],
                signed_errors_ms=errors,
                mean_absolute_error_ms=sum(map(abs, errors))/m if m else None)


def aggregate(items):
    n, p, m = (sum(r[k] for r in items) for k in ('reference', 'predicted', 'matched'))
    return dict(reference=n, predicted=p, matched=m, missing=n-m, extra=p-m,
                precision=m/p if p else 0., recall=m/n if n else 0.,
                f1=2*m/(n+p) if n+p else 0.)


def score(reference, predicted, tolerance):
    result = {}
    for kind in (*CLASSES, 'unknown'):
        truth = [e for e in reference if e['kind'] == kind]
        proposed = [e for e in predicted if e['kind'] == kind]
        result[kind] = match_times([e['time_ms'] for e in truth],
                                  [e['time_ms'] for e in proposed], tolerance)
        result[kind].update(reference_events=truth, predicted_events=proposed)
    return result


def boundary_counts(scores, width=50.):
    """Diagnose selected-pair edge misses; equally optimal pairs may differ."""
    result = {}
    for kind, row in scores.items():
        near = {i for i, e in enumerate(row['reference_events'])
                if min(e['time_ms'] % 2560, 2560-e['time_ms'] % 2560) <= width}
        missing = set(row['unmatched_reference'])
        result[kind] = dict(near_reference=len(near), near_missing=len(near & missing),
                           interior_reference=row['reference']-len(near),
                           interior_missing=len(missing-near))
    return result


def load_labels(row):
    path = ROOT/row['annotation']
    if sha256(path) != row['annotation_sha256']:
        raise ValueError('Original drum labels changed')
    events = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or fields[1] not in LABELS:
            raise ValueError('Unexpected physical drum annotation: '+line)
        events.append(dict(time_ms=float(fields[0])*1000, kind=LABELS[fields[1]], line=i+1))
    if [{k: e[k] for k in ('time_ms', 'kind')} for e in events] != row['events']:
        raise ValueError('Raw physical labels differ from original manifest')
    return events


def as_events(rows):
    return [dict(**e, time_ms=e['onset']*1000) for e in rows]


def author_scores(reference, predicted, duration):
    result = {}
    for name, start, end in [('all', None, None), ('fit_region', 0, 162340),
                             ('calibration_region', 162340, 188286),
                             ('later_sections', 188286, duration)]:
        result[name] = {}
        for clock, shift in [('raw', 0), ('chart_lead_22_5ms', 22.5)]:
            human = [e for e in reference if start is None or start <= e['time_ms'] < end]
            proposed = [dict(**{k: v for k, v in e.items() if k != 'time_ms'},
                             time_ms=e['time_ms']-shift) for e in predicted
                        if start is None or start <= e['time_ms']-shift < end]
            values = score(human, proposed, 40)
            for label, kinds in [('lane4_hh_tt', {'hi-hat', 'tom'}),
                                  ('lane4_hh_tt_cy', {'hi-hat', 'tom', 'cymbal'})]:
                # A physical union collapses simultaneous subclass predictions,
                # but only in this named diagnostic. Primary scores keep repeats.
                times = sorted({e['time_ms'] for e in proposed if e['kind'] in kinds})
                values[label] = match_times([e['time_ms'] for e in human if e['kind'] == 'hi-hat'], times, 40)
                values[label]['notice'] = 'Union diagnostic only; lane 4 is not an exhaustive physical label.'
            result[name][clock] = values
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--inference', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--partial', action='store_true', help='Explicit incomplete snapshot; never a full-corpus result')
    args = cli.parse_args()
    source, output = args.inference.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new local evidence directory')
    identity = read(source/'identity.json')
    progress = read(source/'progress.json')
    manifest_path = ROOT/'test-data/physical-drums/dataset-v2/manifest.json'
    if sha256(manifest_path) != identity['manifest_sha256']:
        raise ValueError('Input manifest changed')
    records = {r['id']: r for r in read(manifest_path)['records'] if r['dataset'] == 'MDB'}
    if [r['id'] for r in progress] != [r['id'] for r in identity['inputs'][:len(progress)]]:
        raise ValueError('Recording sequence differs')
    complete = (source/'report.json').exists()
    if complete:
        finished = read(source/'report.json')
        if finished['records'] != progress or not finished['frozen_model_unchanged'] or len(progress) != 24:
            raise ValueError('Final inference proof differs')
    if not complete and not args.partial:
        raise ValueError('Inference unfinished; a partial snapshot must be explicit')
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output/'scoring-snapshot.py')
    save(output/'progress-snapshot.json', progress)
    results, sakura = [], None
    for row in progress:
        folder = source/row['id']
        for name, digest in row['artifacts_sha256'].items():
            if sha256(folder/name) != digest:
                raise ValueError('Saved inference changed: '+row['id']+'/'+name)
        variants = {v: as_events(read(folder/(v+'-events.json'))) for v in ('official', 'audited')}
        for events in variants.values():
            if any(e['kind'] != group(e['custom_pitch']) for e in events):
                raise ValueError('Wrong custom-class mapping')
        if row['dataset'] == 'MDB':
            source_row = records[row['id']]
            if source_row['audio_sha256'] != row['audio_sha256'] or sha256(ROOT/source_row['audio']) != row['audio_sha256']:
                raise ValueError('Wrong physical drum recording')
            truth = load_labels(source_row)
            scores = {v: {str(t): score(truth, events, t) for t in (20, 40, 50)} for v, events in variants.items()}
            result = dict(id=row['id'], previous_split=row['previous_split'], scores=scores,
                          annotation_sha256=source_row['annotation_sha256'],
                          boundaries_50ms={v: boundary_counts(s['40']) for v, s in scores.items()},
                          boundary_notice='Edge attribution uses the selected optimal matching; alternative equal optima may change which label is missed.',
                          inference_summary=row)
            results.append(result)
        else:
            from manialab.domain import parse_osu
            from scripts.research_source_attack_classes import drum_event_counts
            path = ROOT/'test-data/sakura-benchmark/reference.osu'
            if sha256(path) != read(path.parent/'manifest.json')['charts']['reference']['chart_sha256']:
                raise ValueError('Author reference changed')
            truth = [dict(time_ms=t, kind=kind) for t, kind in drum_event_counts(parse_osu(path.read_text(encoding='utf-8-sig'))['notes'])]
            result = dict(id='sakura', reference_sha256=sha256(path), inference_summary=row,
                          scores={v: author_scores(truth, events, row['duration_seconds']*1000) for v, events in variants.items()})
            sakura = result
        save(output/(row['id']+'.json'), result)
    summaries = {v: {split: {str(t): {kind: aggregate([r['scores'][v][str(t)][kind] for r in results
                    if split == 'all_development' or r['previous_split'] == split]) for kind in (*CLASSES, 'unknown')}
                    for t in (20, 40, 50)} for split in ('all_development', 'train', 'validation', 'test')}
                 for v in ('official', 'audited')}
    # Compare only exact recording IDs available in both saved runs; no mixed cohorts.
    baseline = ROOT/'test-data/physical-drums/evaluation-solo-negatives-v1'
    old = read(baseline/'report.json')
    if sha256(baseline/'events.json') != old['events_sha256']:
        raise ValueError('Saved baseline predictions changed')
    old_dataset = ROOT/'test-data/physical-drums/dataset-solo-negatives-v1/manifest.json'
    if sha256(old_dataset) != old['dataset_sha256']:
        raise ValueError('Baseline dataset identity changed')
    old_rows = {r['id']: r for r in read(old_dataset)['records']}
    old_events = read(baseline/'events.json')
    common = []
    for r in results:
        if r['id'] not in old_events:
            continue
        if any(records[r['id']][k] != old_rows[r['id']][k] for k in ('audio_sha256', 'annotation_sha256', 'split')):
            raise ValueError('Baseline used different audio/labels/split')
        measured = score(load_labels(records[r['id']]), old_events[r['id']], 40)
        old_record = next(x for x in old['records'] if x['id'] == r['id'])
        if any(measured[k]['matched'] != old_record['scores']['40'][k]['matched'] for k in ('kick', 'snare', 'hi-hat')):
            raise ValueError('Baseline matching was not reproduced')
        common.append(dict(id=r['id'], previous_split=r['previous_split'], baseline=measured,
                           official=r['scores']['official']['40'], audited=r['scores']['audited']['40']))
    comparison = {split: dict(ids=[r['id'] for r in common if r['previous_split'] == split],
                    scores={v: {kind: aggregate([r[v][kind] for r in common if r['previous_split'] == split])
                                for kind in ('kick', 'snare', 'hi-hat')} for v in ('baseline', 'official', 'audited')})
                  for split in ('validation', 'test')}
    report = dict(complete_corpus=complete, mdb_recordings=len(results), expected_mdb_recordings=23,
                  inference_identity_sha256=sha256(source/'identity.json'),
                  inference_report_sha256=sha256(source/'report.json') if complete else None,
                  summaries=summaries, baseline_comparison_40ms=comparison,
                  source_inference=str(source), baseline_report_sha256=sha256(baseline/'report.json'),
                  script_sha256=sha256(__file__), application_enabled=False, direct_listening_performed=False,
                  notice='All MDB recordings are development regressions. Sakura measures authored selection, not exhaustive audio truth. Raw repeated and padding predictions remain in primary denominators. No LN durations are inferred.',
                  artifacts_sha256={p.name: sha256(p) for p in output.iterdir()})
    save(output/'report.json', report)
    print(json.dumps(dict(complete_corpus=complete, mdb_recordings=len(results),
                         audited_40ms=summaries['audited']['all_development']['40'])), flush=True)


if __name__ == '__main__':
    main()
