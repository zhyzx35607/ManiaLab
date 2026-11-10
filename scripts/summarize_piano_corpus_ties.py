"""Verify the local imported-tie audit and publish aggregate evidence only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'Identity differs: {path.name}')


def valid(array):
    return ((array['duration_div'] > 0) & (array['duration_beat'] > 0) &
            (array['pitch'] >= 21) & (array['pitch'] <= 108))


def keys(array):
    return Counter((str(row['id']), int(row['onset_div']), int(row['pitch'])) for row in array)


def verify_delta(before, after, summary):
    old, new = keys(before), keys(after)
    counts = dict(before_events=len(before), after_events=len(after), added_events=sum((new - old).values()),
                  removed_events=sum((old - new).values()), net_added_events=len(after) - len(before),
                  before_duplicate_keys=len(before) - len(old), after_duplicate_keys=len(after) - len(new))
    old_duration, new_duration = {}, {}
    for array, target in ((before, old_duration), (after, new_duration)):
        for row in array:
            key = (str(row['id']), int(row['onset_div']), int(row['pitch']))
            target.setdefault(key, []).append(int(row['duration_div']))
    counts['existing_event_duration_changes'] = sum(
        sorted(old_duration[key]) != sorted(new_duration[key]) for key in old.keys() & new.keys())
    if counts != summary:
        raise ValueError('Event counts do not match saved arrays')


def edge_set(values):
    return {(tuple(a), tuple(b)) for a, b in values}


def verify_ties(detail, summary, included=None):
    geometry = {tuple(row['key']): row for row in detail['written_geometry']}
    next_edges, previous_edges = edge_set(detail['before_next_edges']), edge_set(detail['before_prev_edges'])
    old_all = next_edges | previous_edges
    new_all = edge_set(detail['after_next_edges']) | edge_set(detail['after_prev_edges'])
    if included is None:
        included = set(geometry)
    old = {edge for edge in old_all if edge[1] in included}
    new = {edge for edge in new_all if edge[1] in included}
    same_bar = lambda edge: edge[0][0] == edge[1][0] and geometry[edge[0]]['measure'] == geometry[edge[1]]['measure']
    removed = old - new
    counts = dict(imported_edges=len(old), reciprocal_imported_edges=sum(edge[1] in included for edge in next_edges & previous_edges),
                  one_direction_only_imported_edges=sum(edge[1] in included for edge in next_edges ^ previous_edges),
                  retained_imported_edges=len(old & new), removed_imported_edges=len(removed), added_observed_edges=len(new - old),
                  removed_same_measure=sum(same_bar(edge) for edge in removed),
                  removed_cross_measure=sum(not same_bar(edge) for edge in removed),
                  imported_same_measure=sum(same_bar(edge) for edge in old),
                  imported_cross_measure=sum(not same_bar(edge) for edge in old),
                  imported_both_endpoints_in_window=sum(edge[0] in included for edge in old),
                  imported_entering_window=sum(edge[0] not in included for edge in old),
                  imported_leaving_window=sum(edge[0] in included and edge[1] not in included for edge in old_all))
    if counts != summary:
        raise ValueError('Tie counts do not match saved observed edges')


def initial_comparison(initial, completed):
    identity = read(initial / 'identity.json')
    verify(initial / 'runner-snapshot.py', identity['runner_sha256'])
    verify(initial / 'protocol-snapshot.md', identity['protocol_sha256'])
    if identity['runner_sha256'] != '7579cd1e5a2a46ed6d88890ca79cbb85b3adb7a52f74a748c78662bf538d317b':
        raise ValueError('Initial run snapshot is not the expected slow audit')
    artifacts = {path.name: sha256(path) for path in initial.iterdir() if path.is_file()}
    rows = read(initial / 'progress.json')
    names = []
    arrays_checked = 0
    for row in rows:
        old = read(initial / (row['name'] + '-detail.json'))
        new = read(completed / (row['name'] + '-detail.json'))
        old['summary'].pop('elapsed_seconds')
        new['summary'].pop('elapsed_seconds')
        if old != new:
            raise ValueError(f'Accelerated audit changed initial result: {row["name"]}')
        old_array = initial / (row['name'] + '-arrays.npz')
        if old_array.exists():
            with np.load(old_array, allow_pickle=False) as a, np.load(completed / old_array.name, allow_pickle=False) as b:
                if set(a.files) != set(b.files) or any(not np.array_equal(a[key], b[key]) for key in a.files):
                    raise ValueError('Accelerated audit changed an initial array')
            arrays_checked += 1
        names.append(row['name'])
    for name, digest in artifacts.items():
        verify(initial / name, digest)
    return dict(initial_runner_sha256=identity['runner_sha256'], initial_completed_records=len(rows),
                compared_names=names, detail_equal_except_elapsed=True, arrays_equal_cases=arrays_checked,
                initial_artifacts_unchanged=True, initial_artifacts_sha256=artifacts)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT / 'test-data/piano-svsep')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous summaries')
    folder, dataset = args.root / 'corpus-ties-v2', args.root / 'dataset-v2'
    report = read(folder / 'report.json')
    identity = report['identity']
    verify(folder / 'runner-snapshot.py', identity['runner_sha256'])
    verify(folder / 'protocol-snapshot.md', identity['protocol_sha256'])
    verify(ROOT / 'docs/piano-corpus-ties-protocol.md', identity['base_protocol_sha256'])
    verify(dataset / 'manifest.json', identity['manifest_sha256'])
    verify(dataset / 'runner-snapshot.py', identity['prepare_runner_sha256'])
    for name, digest in report['artifacts_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unexpected artifact path')
        verify(folder / name, digest)
    manifest = read(dataset / 'manifest.json')
    selected = {row['name']: row for row in manifest['cases'] if row['status'] == 'ready' and row['split'] in ('train', 'validation')}
    if set(selected) != {row['name'] for row in report['cases']}:
        raise ValueError('Audit cohort does not match train and validation')
    public = []
    for row in report['cases']:
        source = selected[row['name']]
        verify(Path(source['path']), source['sha256'])
        verify(dataset / source['cache'], source['cache_sha256'])
        detail = read(folder / (row['name'] + '-detail.json'))
        if detail['summary'] != row:
            raise ValueError('Summary differs from detailed audit')
        if 'full_ties' in row:
            verify_ties(detail, row['full_ties'])
            with np.load(folder / (row['name'] + '-arrays.npz'), allow_pickle=False) as arrays:
                raw, changed = arrays['raw'], arrays['processed']
            verify_delta(raw, changed, row['full_unfiltered_events'])
            verify_delta(raw[valid(raw)], changed[valid(changed)], row['full_valid_events'])
            if row['window_status'] == 'exact_cache_match':
                bounds = detail['cached_measure_intervals']
                in_window = lambda value: any(start <= value < end for start, end in bounds)
                a = raw[np.array([in_window(int(value)) for value in raw['onset_div']]) & valid(raw)]
                b = changed[np.array([in_window(int(value)) for value in changed['onset_div']]) & valid(changed)]
                verify_delta(a, b, row['window_valid_events'])
                included = {tuple(key) for key, onset in detail['global_onsets'] if in_window(onset)}
                verify_ties(detail, row['window_ties'], included)
                with np.load(dataset / source['cache'], allow_pickle=False) as cached:
                    if keys(b) != keys(cached['notes']):
                        raise ValueError('Processed window event keys differ from original cache')
                if not all(row['cache_reproduction_checks'].values()):
                    raise ValueError('False exact-cache assertion')
        safe_fields = ('name', 'split', 'source_sha256', 'cache_sha256', 'status', 'window_status', 'full_ties',
                       'full_unfiltered_events', 'full_valid_events', 'window_ties', 'window_valid_events',
                       'returned_tie_id_count', 'written_note_count', 'window_written_note_count', 'window_measures',
                       'cache_reproduction_checks', 'error_stage')
        public.append({key: row[key] for key in safe_fields if key in row} |
                      dict(warning_count=sum(row['warnings'].values()),
                           error_type=row['error'].split(':', 1)[0] if 'error' in row else None))
    totals = {}
    for split in ('train', 'validation'):
        rows = [row for row in report['cases'] if row['split'] == split]
        subtotal = dict(selected=len(rows), full_complete=sum('full_ties' in row for row in rows),
                        exact_windows=sum(row['window_status'] == 'exact_cache_match' for row in rows),
                        errors=sum('error' in row for row in rows))
        for field in ('full_ties', 'full_unfiltered_events', 'full_valid_events', 'window_ties', 'window_valid_events'):
            counter = Counter()
            for row in rows:
                if field in row and (not field.startswith('window') or row['window_status'] == 'exact_cache_match'):
                    counter.update(row[field])
            subtotal[field] = dict(counter)
        totals[split] = subtotal
    if totals != report['totals']:
        raise ValueError('Aggregate statistics were not reproduced')
    for flag in ['test_arrays_read', 'test_scores_read', 'old_cache_changed', 'model_loaded', 'weights_changed',
                 'enabled_in_application', 'musical_acceptance']:
        if report[flag] is not False:
            raise ValueError(f'Unexpected audit scope: {flag}')
    result = dict(date='2026-11-09', summarizer_sha256=sha256(__file__), report_sha256=sha256(folder / 'report.json'),
                  identity=identity, verification=dict(artifacts=len(report['artifacts_sha256']), source_files=len(selected),
                    train_validation_cache_files=len(selected), saved_edge_counts_recomputed=True,
                    saved_event_arrays_recomputed=True, cached_window_event_keys_equal=True),
                  initial_run_comparison=initial_comparison(args.root / 'corpus-ties-v1', folder),
                  selected=len(selected), completed=report['completed'], errors=report['errors'], totals=totals,
                  affected_scores=dict(full_same_measure_tie_loss=sum(row.get('full_ties', {}).get('removed_same_measure', 0) > 0 for row in public),
                    window_same_measure_tie_loss=sum(row.get('window_ties', {}).get('removed_same_measure', 0) > 0 for row in public),
                    window_added_events=sum(row.get('window_valid_events', {}).get('added_events', 0) > 0 for row in public)),
                  cases=public, elapsed_seconds=report['elapsed_seconds'], limitations=report['limitations'] + [
                    'Test-read flags refer only to this audit runner, not to all work in the wider conversation.',
                    'Failed audit cases remain in the selected denominator and are excluded from completed statistics.',
                    'Raw note/event IDs, array contents and individual warning messages remain local.'],
                  test_arrays_read_by_audit=False, test_scores_read_by_audit=False, old_cache_changed=False,
                  model_loaded=False, weights_changed=False, enabled_in_application=False, musical_acceptance=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), sha256=sha256(args.output), selected=result['selected'],
                         completed=result['completed'], errors=result['errors'], verification=result['verification'])), flush=True)


if __name__ == '__main__':
    main()
