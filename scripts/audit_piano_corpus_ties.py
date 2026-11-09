"""Audit imported tie loss in frozen train/validation scores without changing caches."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256

MANIFEST_SHA = 'b8e8e2046fd02acf02dfff1d38969ab4a5de8130f1d32e4a75eaf9659c0da7e1'
PREPARE_SHA = 'ae2547f445d59c005b9827ecc470ffe5242d42c91031cdef5c4bfbd1a57db1a1'
SOURCE_REVISION = '1462e7c28d7adaae033150883a7cb00238ee364a'
CORPUS_REVISION = 'aa732b635bc75c6583d86c6ac488e77b08b1b37e'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'Identity differs: {path.name}')


def git_identity(path, revision):
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(path), 'diff', 'HEAD', '--name-only'], text=True).strip()
    if actual != revision or changes:
        raise ValueError(f'Repository identity differs: {path.name}')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def scalar(value):
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError('Expected a scalar score attribute')
        return value.item()
    return value.item() if isinstance(value, np.generic) else value


def score_snapshot(score, partitura):
    rows, next_edges, prev_edges = [], set(), set()
    indices = {}
    for part_index, part in enumerate(score.parts):
        for note_index, note in enumerate(part.iter_all(partitura.score.Note)):
            indices[id(note)] = (part_index, note_index)
    for part_index, part in enumerate(score.parts):
        divisions = list(map(scalar, part._quarter_durations))
        if len(divisions) != 1 or divisions[0] <= 0 or int(divisions[0]) != divisions[0]:
            raise ValueError('Expected one positive integral divisions value per selected part')
        measure_number_map = part.measure_number_map
        for note_index, note in enumerate(part.iter_all(partitura.score.Note)):
            key = (part_index, note_index)
            rows.append(dict(key=list(key), part_id=str(part.id), note_id=note.id,
                             start_tick=int(note.start.t), end_tick=int(note.end.t),
                             pitch=int(note.midi_pitch), staff=scalar(note.staff), voice=scalar(note.voice),
                             measure=scalar(measure_number_map(note.start.t)), divisions=int(divisions[0])))
            if note.tie_next is not None:
                if id(note.tie_next) not in indices:
                    raise ValueError('Imported tie_next endpoint is outside written notes')
                next_edges.add((key, indices[id(note.tie_next)]))
            if note.tie_prev is not None:
                if id(note.tie_prev) not in indices:
                    raise ValueError('Imported tie_prev endpoint is outside written notes')
                prev_edges.add((indices[id(note.tie_prev)], key))
    return rows, next_edges, prev_edges


def keys(array):
    return Counter((str(note['id']), int(note['onset_div']), int(note['pitch'])) for note in array)


def expanded(counter):
    return [list(key) for key, count in sorted(counter.items()) for _ in range(count)]


def valid(array):
    return ((array['duration_div'] > 0) & (array['duration_beat'] > 0) &
            (array['pitch'] >= 21) & (array['pitch'] <= 108))


def window_mask(times, bounds):
    result = np.zeros(len(times), dtype=bool)
    for start, end in bounds:
        result |= (times >= start) & (times < end)
    return result


def event_delta(before, after):
    old, new = keys(before), keys(after)
    before_duration, after_duration = {}, {}
    for array, dest in ((before, before_duration), (after, after_duration)):
        for note in array:
            key = (str(note['id']), int(note['onset_div']), int(note['pitch']))
            dest.setdefault(key, []).append(int(note['duration_div']))
    changed = [list(key) for key in sorted(old.keys() & new.keys())
               if sorted(before_duration[key]) != sorted(after_duration[key])]
    summary = dict(before_events=len(before), after_events=len(after), added_events=sum((new - old).values()),
                   removed_events=sum((old - new).values()), net_added_events=len(after) - len(before),
                   existing_event_duration_changes=len(changed), before_duplicate_keys=len(before) - len(old),
                   after_duplicate_keys=len(after) - len(new))
    detail = dict(added_keys=expanded(new - old), removed_keys=expanded(old - new), duration_changed_keys=changed)
    return summary, detail


def tie_summary(before, after, next_before, prev_before, geometry, included=None):
    if included is None:
        included = set(geometry)
    selected = lambda edges: {edge for edge in edges if edge[1] in included}
    original, current = selected(before), selected(after)
    removed, added = original - current, current - original
    cross = lambda edge: (edge[0][0] != edge[1][0] or
                          geometry[edge[0]]['measure'] != geometry[edge[1]]['measure'])
    return dict(imported_edges=len(original), reciprocal_imported_edges=len(selected(next_before & prev_before)),
                one_direction_only_imported_edges=len(selected(next_before ^ prev_before)),
                retained_imported_edges=len(original & current), removed_imported_edges=len(removed),
                added_observed_edges=len(added), removed_same_measure=sum(not cross(edge) for edge in removed),
                removed_cross_measure=sum(cross(edge) for edge in removed),
                imported_same_measure=sum(not cross(edge) for edge in original),
                imported_cross_measure=sum(cross(edge) for edge in original),
                imported_both_endpoints_in_window=sum(edge[0] in included for edge in original),
                imported_entering_window=sum(edge[0] not in included for edge in original),
                imported_leaving_window=sum(edge[0] in included and edge[1] not in included for edge in before))


def load_cache(path, row):
    verify(path, row['cache_sha256'])
    with np.load(path, allow_pickle=False) as saved:
        notes, measures, bounds = saved['notes'], saved['measures'], saved['measure_bounds']
    if len(np.unique(measures)) != 16 or bounds.shape != (len(notes), 2):
        raise ValueError('Expected 16 fully identified cached measures')
    if len(set(notes['id'])) != len(notes):
        raise ValueError('Cached note IDs are not unique')
    intervals = []
    for measure in np.unique(measures):
        one = np.unique(bounds[measures == measure], axis=0)
        if len(one) != 1:
            raise ValueError('Cached measure has inconsistent bounds')
        intervals.append(one[0])
    if not np.all((notes['onset_div'] >= bounds[:, 0]) & (notes['onset_div'] < bounds[:, 1])):
        raise ValueError('Cached onset is outside its measure')
    return notes, measures, bounds, np.asarray(intervals)


def reproduce_cache(score, after_array, cached, sanitize):
    expected_notes, expected_measures, expected_bounds, intervals = cached
    array = sanitize(after_array.copy())
    clock = score[np.array([part._quarter_durations[0] for part in score]).argmax()]
    measures = np.asarray(clock.measure_number_map(array['onset_div'])).astype(int)
    bounds = np.asarray(clock.measure_map(array['onset_div']))
    first = np.sort(np.unique(measures))[:16]
    selected = np.isin(measures, first) & valid(array)
    array, measures, bounds = array[selected], measures[selected], bounds[selected]
    order = np.lexsort((array['pitch'], array['onset_div']))
    array, measures, bounds = array[order], measures[order], bounds[order]
    checks = dict(notes=np.array_equal(array, expected_notes), measures=np.array_equal(measures, expected_measures),
                  bounds=np.array_equal(bounds, expected_bounds), first_measure_ids=np.array_equal(first, np.unique(expected_measures)),
                  interval_mask_matches_selector=np.array_equal(
                      window_mask(after_array['onset_div'], intervals) & valid(after_array), selected))
    if not all(checks.values()):
        raise ValueError(f'Original cache could not be reproduced exactly: {checks}')
    return checks


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=ROOT / 'test-data/piano-svsep/source')
    cli.add_argument('--corpus', type=Path, default=ROOT / 'test-data/piano-svsep/corpus')
    cli.add_argument('--dataset', type=Path, default=ROOT / 'test-data/piano-svsep/dataset-v2')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous audit results')
    manifest_path = args.dataset / 'manifest.json'
    verify(manifest_path, MANIFEST_SHA)
    verify(args.dataset / 'runner-snapshot.py', PREPARE_SHA)
    manifest = read(manifest_path)
    rows = [row for row in manifest['cases'] if row['status'] == 'ready' and row['split'] in ('train', 'validation')]
    if Counter(row['split'] for row in rows) != {'train': 157, 'validation': 16}:
        raise ValueError('Selected dataset cohort differs')
    git_identity(args.source, SOURCE_REVISION)
    git_identity(args.corpus, CORPUS_REVISION)
    selected_paths = {}
    for row in rows:
        path = Path(row['path']).resolve()
        if not path.is_relative_to((args.corpus / 'scores').resolve()) or path.suffix != '.musicxml':
            raise ValueError('Selected source is outside the expected corpus')
        if Path(row['cache']).name != row['cache']:
            raise ValueError('Unexpected cache path')
        verify(path, row['sha256'])
        verify(args.dataset / row['cache'], row['cache_sha256'])
        selected_paths[row['name']] = path
    sys.path.insert(0, str(args.source.resolve()))
    import partitura
    import partitura.io.importmusicxml as importer
    import partitura.utils.music as music_utils
    from piano_svsep.utils import remove_ties_acros_barlines, sanitize_staff_voices
    protocol = ROOT / 'docs/piano-corpus-ties-fast-protocol.md'
    identity = dict(source_revision=SOURCE_REVISION, corpus_revision=CORPUS_REVISION,
                    manifest_sha256=MANIFEST_SHA, prepare_runner_sha256=PREPARE_SHA,
                    runner_sha256=sha256(__file__), protocol_sha256=sha256(protocol),
                    base_protocol_sha256=sha256(ROOT / 'docs/piano-corpus-ties-protocol.md'),
                    preprocessing_sha256=sha256(args.source / 'piano_svsep/utils/vocsep.py'),
                    importer_sha256=sha256(importer.__file__), array_code_sha256=sha256(music_utils.__file__),
                    partitura_version=importlib.metadata.version('partitura'), numpy_version=np.__version__,
                    scope='Ready train and validation only; already-imported tie pointers, not XML declaration truth',
                    selected_counts=dict(Counter(row['split'] for row in rows)))
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / 'runner-snapshot.py')
    shutil.copyfile(protocol, args.output / 'protocol-snapshot.md')
    save = lambda name, data: (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    save('identity.json', identity)
    results, started = [], time.perf_counter()
    for index, row in enumerate(rows):
        began = time.perf_counter()
        path = selected_paths[row['name']]
        record = dict(name=row['name'], split=row['split'], source_sha256=row['sha256'],
                      cache_sha256=row['cache_sha256'], status='error', window_status='not_checked')
        detail = {}
        caught = []
        stage = 'load_cache'
        try:
            cached = load_cache(args.dataset / row['cache'], row)
            stage = 'import_scores'
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                reference, processed = partitura.load_score(str(path)), partitura.load_score(str(path))
                before = score_snapshot(reference, partitura)
                if before != score_snapshot(processed, partitura):
                    raise ValueError('Independent imports differ')
                stage = 'raw_note_array'
                raw = reference.note_array(include_time_signature=True, include_grace_notes=True, include_staff=True)
                if before != score_snapshot(reference, partitura):
                    raise ValueError('Raw note_array mutated source geometry or ties')
                stage = 'upstream_preprocessing'
                returned = remove_ties_acros_barlines(processed)
                after = score_snapshot(processed, partitura)
                if before[0] != after[0]:
                    raise ValueError('Preprocessing changed written geometry')
                stage = 'processed_note_array'
                changed = processed.note_array(include_time_signature=True, include_grace_notes=True, include_staff=True)
                if after != score_snapshot(processed, partitura) or before != score_snapshot(reference, partitura):
                    raise ValueError('A reference or processed score changed during note_array creation')
            original_edges, changed_edges = before[1] | before[2], after[1] | after[2]
            geometry = {tuple(note['key']): note for note in before[0]}
            record['full_ties'] = tie_summary(original_edges, changed_edges, before[1], before[2], geometry)
            record['full_unfiltered_events'], full_detail = event_delta(raw, changed)
            record['full_valid_events'], valid_detail = event_delta(raw[valid(raw)], changed[valid(changed)])
            record['returned_tie_id_count'] = len(returned[0])
            record['written_note_count'] = len(before[0])
            detail.update(geometry_sha256=digest(before[0]), written_geometry=before[0],
                          before_next_edges=sorted(before[1]), before_prev_edges=sorted(before[2]),
                          after_next_edges=sorted(after[1]), after_prev_edges=sorted(after[2]),
                          returned_tie_ids=returned.tolist(), full_event_delta=full_detail,
                          full_valid_event_delta=valid_detail)
            np.savez_compressed(args.output / (row['name'] + '-arrays.npz'), raw=raw, processed=changed)
            record['status'] = 'full_complete'
            stage = 'cache_reproduction'
            record['cache_reproduction_checks'] = reproduce_cache(processed, changed, cached, sanitize_staff_voices)
            stage = 'window_statistics'
            intervals = cached[3]
            raw_window = raw[window_mask(raw['onset_div'], intervals) & valid(raw)]
            changed_window = changed[window_mask(changed['onset_div'], intervals) & valid(changed)]
            record['window_valid_events'], window_detail = event_delta(raw_window, changed_window)
            raw_divisions, changed_divisions = np.unique(raw['divs_pq']), np.unique(changed['divs_pq'])
            if len(raw_divisions) != 1 or not np.array_equal(raw_divisions, changed_divisions):
                raise ValueError('No common, stable integer score clock')
            common = int(raw_divisions[0])
            global_onsets = {}
            for key, note in geometry.items():
                if common % note['divisions']:
                    raise ValueError('Part clock cannot be scaled exactly')
                global_onsets[key] = note['start_tick'] * (common // note['divisions'])
            included = {key for key, onset in global_onsets.items() if any(start <= onset < end for start, end in intervals)}
            record['window_ties'] = tie_summary(original_edges, changed_edges, before[1], before[2], geometry, included)
            record['window_written_note_count'] = len(included)
            record['window_measures'] = len(intervals)
            record['window_status'] = 'exact_cache_match'
            detail.update(window_event_delta=window_detail, cached_measure_intervals=intervals.tolist(),
                          common_divisions=common, global_onsets=[[list(key), value] for key, value in global_onsets.items()])
            record['status'] = 'complete'
        except Exception as exc:
            record.update(error_stage=stage, error=f'{type(exc).__name__}: {exc}')
            if record['status'] == 'full_complete':
                record['window_status'] = 'unavailable'
        record['warnings'] = dict(Counter(str(item.message) for item in caught))
        record['elapsed_seconds'] = time.perf_counter() - began
        verify(path, row['sha256'])
        verify(args.dataset / row['cache'], row['cache_sha256'])
        detail['summary'] = record
        save(row['name'] + '-detail.json', detail)
        results.append(record)
        save('progress.json', results)
        print(json.dumps(dict(index=index + 1, total=len(rows), name=row['name'], split=row['split'],
                              status=record['status'], window_status=record['window_status'],
                              full_ties=record.get('full_ties'), window_events=record.get('window_valid_events'),
                              error=record.get('error')), ensure_ascii=False), flush=True)
    verify(manifest_path, MANIFEST_SHA)
    verify(args.dataset / 'runner-snapshot.py', PREPARE_SHA)
    verify(__file__, identity['runner_sha256'])
    verify(protocol, identity['protocol_sha256'])
    verify(Path(importer.__file__), identity['importer_sha256'])
    verify(Path(music_utils.__file__), identity['array_code_sha256'])
    git_identity(args.source, SOURCE_REVISION)
    git_identity(args.corpus, CORPUS_REVISION)
    for row in rows:
        verify(selected_paths[row['name']], row['sha256'])
        verify(args.dataset / row['cache'], row['cache_sha256'])
    totals = {}
    for split in ('train', 'validation'):
        cohort = [row for row in results if row['split'] == split]
        summary = dict(selected=len(cohort), full_complete=sum('full_ties' in row for row in cohort),
                       exact_windows=sum(row['window_status'] == 'exact_cache_match' for row in cohort),
                       errors=sum('error' in row for row in cohort))
        for field in ('full_ties', 'full_unfiltered_events', 'full_valid_events', 'window_ties', 'window_valid_events'):
            counter = Counter()
            for row in cohort:
                if field in row and (not field.startswith('window') or row['window_status'] == 'exact_cache_match'):
                    counter.update(row[field])
            summary[field] = dict(counter)
        totals[split] = summary
    report = dict(identity=identity, cases=results, totals=totals, elapsed_seconds=time.perf_counter() - started,
                  completed=sum(row['status'] == 'complete' for row in results), errors=sum('error' in row for row in results),
                  test_arrays_read=False, test_scores_read=False, selected_training_cache_read=True,
                  old_cache_changed=False, model_loaded=False, weights_changed=False, upstream_unchanged=True,
                  enabled_in_application=False, musical_acceptance=False,
                  limitations=['Imported tie pointers are not complete XML truth; import omissions are not measured.',
                               'Cross-bar deletion is distinguished from erroneous same-bar deletion.',
                               'Window tie statistics include written segments; event statistics apply original cache validity filters.',
                               'Score-event counts are neither audible onset counts nor physical release labels.'])
    report['artifacts_sha256'] = {path.name: sha256(path) for path in args.output.iterdir()
                                  if path.name not in ('report.json', 'progress.json')}
    save('report.json', report)
    print(json.dumps(dict(completed=report['completed'], errors=report['errors'], totals=totals,
                         report_sha256=sha256(args.output / 'report.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
