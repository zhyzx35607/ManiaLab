"""Verify saved MusicXML source representations and publish aggregate diagnostics."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Identity differs: {path.name}')


def verify_source(source):
    written = {row['note']['id']: row for row in source['written']}
    if len(written) != len(source['written']):
        raise ValueError('Duplicate source structural ID')
    verified = set(source['verified_note_ids'])
    issues = {identifier: reasons for identifier, reasons in source['issues']}
    if set(issues) != set(written) - verified or not verified <= written.keys():
        raise ValueError('Verification mask and issues disagree')
    excluded = {identifier for identifier, _ in source['excluded']}
    if excluded & written.keys() or len(excluded) != len(source['excluded']):
        raise ValueError('Excluded source paths overlap accepted notes')
    edges = {tuple(edge) for edge in source['ties']}
    if len(edges) != len(source['ties']):
        raise ValueError('Duplicate resolved source tie')
    successors, predecessors = {}, {}
    for a, b in edges:
        if a not in verified or b not in verified or a in successors or b in predecessors:
            raise ValueError('Unverified or branching resolved tie')
        first, second = written[a]['note'], written[b]['note']
        if first['part'] != second['part'] or first['pitch'] != second['pitch'] or first['end_tick'] != second['start_tick']:
            raise ValueError('Resolved source tie violates time/part/pitch')
        successors[a], predecessors[b] = b, a
    for identifier, row in written.items():
        if identifier in verified:
            if row['problems'] or row.get('playback_conditions'):
                raise ValueError('Conditional or problematic source note was verified')
            if ('start' in row['sound_ties']) != (identifier in successors):
                raise ValueError('Verified source start declaration lacks exact outgoing edge')
            if ('stop' in row['sound_ties']) != (identifier in predecessors):
                raise ValueError('Verified source stop declaration lacks exact incoming edge')
    representation = source['representation']
    ownership = Counter(identifier for event in representation['events'] for identifier in event['written_ids'])
    if ownership != Counter(written.keys()):
        raise ValueError('Source segments are not owned exactly once')
    views = {view['note']['id']: view for view in representation['notation']}
    if len(views) != len(representation['notation']) or set(views) != set(written):
        raise ValueError('Notation views do not cover source notes')
    for event in representation['events']:
        members = event['written_ids']
        first, last = written[members[0]]['note'], written[members[-1]]['note']
        if event['id'] != members[0] or event['start_tick'] != first['start_tick'] or event['end_tick'] != last['end_tick']:
            raise ValueError('Sustain event changed its source interval or identity')
        if event['part'] != first['part'] or event['pitch'] != first['pitch']:
            raise ValueError('Sustain event changed source part/pitch')
        if any((a, b) not in edges for a, b in zip(members, members[1:])):
            raise ValueError('Sustain includes an unconfirmed source edge')
        eligible = all(identifier in verified for identifier in members)
        if event['supervision_eligible'] != eligible:
            raise ValueError('Sustain supervision does not match verification mask')
        for index, identifier in enumerate(members):
            view = views[identifier]
            expected_role = ('attack' if index == 0 else 'continuation') if eligible else 'unknown'
            if view['event_id'] != event['id'] or view['role'] != expected_role or view['note'] != written[identifier]['note']:
                raise ValueError('Written view lost source fields or mask semantics')
    if len(representation['events']) != len(written) - len(edges):
        raise ValueError('Source event count is inconsistent with tie forest')
    counts = dict(written=len(written), ties=len(edges), score_events=len(representation['events']),
                  verified=len(verified), unknown=len(issues), excluded=len(excluded),
                  continuations=sum(view['role'] == 'continuation' for view in views.values()),
                  unknown_reasons=dict(Counter(reason for reasons in issues.values() for reason in reasons)))
    if sum(view['role'] == 'unknown' for view in views.values()) != len(issues):
        raise ValueError('Unknown source count differs from written views')
    return counts


def initial_run(root):
    folder = root / 'musicxml-events-v1'
    identity = read(folder / 'identity.json')
    for name, expected in identity['code_sha256'].items():
        extension = '.md' if name == 'protocol' else '.py'
        verify(folder / (name + '-snapshot' + extension), expected)
    progress = read(folder / 'progress.json')
    for row in progress:
        detail = read(folder / row['detail_file'])
        if detail['summary'] != {key: value for key, value in row.items() if key != 'detail_file'}:
            raise ValueError('Initial partial run summary differs')
    mismatch_path = folder / 'beethoven_piano_sonatas_08-1-detail.json'
    source = read(mismatch_path)['source']
    grouped = defaultdict(list)
    for measure in source['measures']:
        grouped[measure['ordinal']].append(measure)
    mismatch = None
    for ordinal, measures in sorted(grouped.items()):
        starts = {measure['start_tick'] for measure in measures}
        ends = {measure['end_tick'] for measure in measures}
        if len(starts) > 1 or len(ends) > 1:
            difference = max(ends) - min(ends)
            mismatch = dict(name='beethoven_piano_sonatas_08-1', first_mismatch_measure_ordinal=ordinal,
                            labels=sorted({measure['label'] for measure in measures}),
                            common_start=len(starts) == 1, end_difference_ticks=difference,
                            ticks_per_quarter=source['ticks_per_quarter'],
                            end_difference_quarters=str(Fraction(difference, source['ticks_per_quarter'])))
            break
    return dict(identity=identity, recorded_files=len(progress), complete_records=sum(row['status'] == 'complete' for row in progress),
                error_records=sum(row['status'] == 'error' for row in progress), full_report_exists=(folder / 'report.json').exists(),
                progress_sha256=sha256(folder / 'progress.json'), clock_diagnostic=mismatch,
                used_as_repaired_results=False)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT / 'test-data/piano-svsep')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous public summaries')
    folder = args.root / 'musicxml-events-v2'
    report = read(folder / 'report.json')
    identity = report['identity']
    for name, expected in identity['code_sha256'].items():
        extension = '.md' if name == 'protocol' else '.py'
        verify(folder / (name + '-snapshot' + extension), expected)
    for name, expected in report['artifacts_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unexpected artifact path')
        verify(folder / name, expected)
    manifest_path = args.root / 'dataset-v2/manifest.json'
    verify(manifest_path, identity['dataset_manifest_sha256'])
    manifest = read(manifest_path)
    selected = {row['name']: row for row in manifest['cases'] if row['status'] == 'ready' and row['split'] in ('train', 'validation')}
    if set(selected) != {row['name'] for row in report['cases']}:
        raise ValueError('Source check selection differs from train/validation cohort')
    controls = read(args.root / 'bar-identity-v1/report.json')
    by_name = {case['name']: case for case in controls['cases']}
    for control in report['controls']:
        verify(args.root / 'bar-identity-v1' / (control['name'] + '.musicxml'), control['input_sha256'])
        original = by_name[control['name']]
        expected = (4, 4) if control['name'] == 'parallel_unison_ties' else (original['expected_raw_events'], 0)
        if (control['events'], control['unknown']) != expected:
            raise ValueError('Saved source control result differs from protocol')
    cases = []
    for row in report['cases']:
        origin = selected[row['name']]
        verify(Path(origin['path']), origin['sha256'])
        if origin['sha256'] != row['input_sha256']:
            raise ValueError('Per-file source identity differs')
        detail = read(folder / row['detail_file'])
        if detail['summary'] != {key: value for key, value in row.items() if key != 'detail_file'}:
            raise ValueError('Detailed source summary differs')
        if 'source' in detail:
            counts = verify_source(detail['source'])
            if any(row[key] != value for key, value in counts.items()):
                raise ValueError('Source representation counts were not reproduced')
        if row['status'] == 'complete':
            differences = detail['geometry_differences']
            if row['geometry_differences'] != len(differences) or row['geometry_equal'] != (not differences):
                raise ValueError('Geometry comparison record is inconsistent')
            if row['difference_reasons'] != dict(Counter(item['reason'] for item in differences)):
                raise ValueError('Geometry difference reasons are inconsistent')
        cases.append({key: value for key, value in row.items() if key != 'detail_file'})
    totals = {split: {field: sum(row.get(field, 0) for row in cases if row['split'] == split)
                     for field in ('written', 'ties', 'score_events', 'verified', 'unknown', 'excluded', 'continuations')}
              for split in ('train', 'validation')}
    if totals != report['totals']:
        raise ValueError('Aggregate source counts differ')
    if sum(row['status'] == 'complete' for row in cases) != report['completed'] or sum(row['status'] == 'error' for row in cases) != report['errors']:
        raise ValueError('Source completion counts differ')
    for flag in ('test_sources_read_by_runner', 'model_loaded', 'training_cache_read', 'training_cache_changed',
                 'enabled_in_application', 'musical_acceptance'):
        if report[flag] is not False:
            raise ValueError(f'Unexpected source-check scope: {flag}')
    errors = Counter(row['error'] for row in cases if 'error' in row)
    reasons = Counter()
    for row in cases:
        reasons.update(row.get('unknown_reasons', {}))
    result = dict(date='2026-11-09', summarizer_sha256=sha256(__file__), report_sha256=sha256(folder / 'report.json'),
                  identity=identity, verification=dict(artifacts=len(report['artifacts_sha256']), source_files=len(selected),
                    executed_snapshots_verified=True, saved_source_counts_recomputed=True, source_event_invariants=True,
                    recorded_geometry_comparisons_consistent=True, geometry_imports_rerun=False, old_cache_arrays_read=False),
                  initial_partial_run=initial_run(args.root), controls=report['controls'], selected=len(selected),
                  completed=report['completed'], errors=report['errors'], geometry_equal_files=report['geometry_equal_files'],
                  coverage={split: dict(selected=sum(row['split'] == split for row in cases),
                    completed=sum(row['split'] == split and row['status'] == 'complete' for row in cases)) for split in ('train', 'validation')},
                  totals=totals, unknown_reasons=dict(reasons), rejection_reasons=dict(errors), cases=cases,
                  elapsed_seconds=report['elapsed_seconds'], test_sources_read_by_runner=False,
                  development_test_format_peek=report['development_test_format_peek'], model_loaded=False,
                  training_cache_read=False, training_cache_changed=False, enabled_in_application=False, musical_acceptance=False,
                  limitations=report['limitations'] + [
                    'Geometry equality verifies the saved independent comparison; this summarizer does not rerun imports.',
                    'Earlier imported-tie audit and this source adapter have different accepted cohorts and cannot be compared as paired totals.',
                    'Per-note source representations, IDs, declarations and warnings remain in the ignored local research directory.'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), sha256=sha256(args.output), verification=result['verification'],
                         completed=result['completed'], errors=result['errors'], coverage=result['coverage'])), flush=True)


if __name__ == '__main__':
    main()
