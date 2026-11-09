"""Check score-event semantics against seven frozen, self-authored XML controls.

The tiny XML reader is deliberately fixture-specific, not a production importer.
Authored edge identities come from the frozen manifest, never pitch matching.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256
from scripts.piano_score_events import WrittenNote, score_events

SOURCE_REPORT_SHA256 = 'be6b8d92d91653ae369d2a95ae090752bb6ea3a1ec3f183f7c3726b54558d1d9'
EXPECTED = {
    'within_measure': ([(0, 960)], 1),
    'within_measure_chain': ([(0, 1440)], 2),
    'across_measure': ([(1440, 2400)], 1),
    'repeated_pitch': ([(0, 480), (480, 960)], 0),
    'cross_staff': ([(0, 960)], 1),
    'single_note': ([(0, 480)], 0),
    'parallel_unison_ties': ([(0, 960), (0, 960)], 2),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'Frozen identity changed: {path.name}')


def source_written(path):
    """Read only the exact, simple notation subset used by the frozen controls."""
    tree = ET.parse(path).getroot()
    if tree.tag != 'score-partwise' or len(tree.findall('part')) != 1:
        raise ValueError('Expected one self-authored part')
    part = tree.find('part')
    if part.get('id') != 'P1':
        raise ValueError('Unexpected authored part ID')
    result, declarations = [], {}
    for index, measure in enumerate(part.findall('measure')):
        if measure.get('number') != str(index + 1):
            raise ValueError('Unexpected fixture measure sequence')
        cursor = index * 1920
        for child in measure:
            if child.tag == 'attributes':
                if index != 0 or child.findtext('divisions') != '480':
                    raise ValueError('Only fixed fixture divisions are supported')
                if (child.findtext('time/beats'), child.findtext('time/beat-type')) != ('4', '4'):
                    raise ValueError('Only fixed fixture 4/4 is supported')
                continue
            if child.tag == 'backup':
                cursor -= int(child.findtext('duration'))
                if cursor < index * 1920:
                    raise ValueError('Backup escapes current fixture measure')
                continue
            if child.tag != 'note':
                raise ValueError(f'Unsupported fixture element: {child.tag}')
            allowed = {'pitch', 'duration', 'tie', 'voice', 'type', 'staff', 'notations', 'rest'}
            if any(element.tag not in allowed for element in child):
                raise ValueError('Unsupported note element in fixture')
            duration = int(child.findtext('duration'))
            if duration <= 0 or cursor + duration > (index + 1) * 1920:
                raise ValueError('Invalid fixture duration')
            if child.find('rest') is not None:
                cursor += duration
                continue
            if (child.findtext('pitch/step'), child.findtext('pitch/octave')) != ('C', '4'):
                raise ValueError('Only authored C4 controls are supported')
            if child.find('pitch/alter') is not None:
                raise ValueError('Altered pitch was not authored in these controls')
            identifier = child.get('id')
            if not identifier or identifier in declarations:
                raise ValueError('Expected unique explicit fixture IDs')
            ties = [element.get('type') for element in child.findall('tie')]
            tied = [element.get('type') for element in child.findall('notations/tied')]
            if sorted(ties) != sorted(tied) or len(ties) != len(set(ties)):
                raise ValueError('Sound and notation declarations disagree')
            if not set(ties) <= {'start', 'stop'}:
                raise ValueError('Unsupported fixture tie declaration')
            declarations[identifier] = set(ties)
            result.append(WrittenNote(identifier, part.get('id'), 60, cursor, cursor + duration,
                                      index + 1, int(child.findtext('staff')), child.findtext('voice')))
            cursor += duration
        if cursor != (index + 1) * 1920:
            raise ValueError('Fixture measure is not complete')
    return tuple(result), declarations


def case_check(folder, case):
    name = case['name']
    path = folder / (name + '.musicxml')
    verify(path, case['input_sha256'])
    written, declarations = source_written(path)
    ties = [tuple(pair) for pair in case['expected_ties']]
    outgoing, incoming = Counter(a for a, _ in ties), Counter(b for _, b in ties)
    if any(count != 1 for count in list(outgoing.values()) + list(incoming.values())):
        raise ValueError('Authored manifest branches')
    ids = {note.id for note in written}
    if not set(outgoing) | set(incoming) <= ids:
        raise ValueError('Authored endpoint missing from XML')
    for identifier, declared in declarations.items():
        expected = ({'start'} if identifier in outgoing else set()) | ({'stop'} if identifier in incoming else set())
        if declared != expected:
            raise ValueError('Source declarations disagree with authored connection manifest')
    source_before = tuple(asdict(note) for note in written)
    result = score_events(written, ties, verified_note_ids=ids)
    expected_intervals, expected_continuations = EXPECTED[name]
    checks = {
        'event_count': len(result.events) == len(expected_intervals),
        'event_intervals': sorted((event.start_tick, event.end_tick) for event in result.events) == sorted(expected_intervals),
        'all_known_eligible': all(event.supervision_eligible for event in result.events),
        'all_written_fields_preserved': {view.note.id: asdict(view.note) for view in result.notation} ==
                                       {note.id: asdict(note) for note in written},
        'one_owner_per_written_note': Counter(identifier for event in result.events for identifier in event.written_ids) ==
                                      Counter(note.id for note in written),
        'continuation_count': sum(view.role == 'continuation' for view in result.notation) == expected_continuations,
        'attack_count': sum(view.role == 'attack' for view in result.notation) == len(expected_intervals),
        'roles_follow_source_declarations': all(view.role == ('continuation' if 'stop' in declarations[view.note.id] else 'attack')
                                                for view in result.notation),
        'source_not_mutated': tuple(asdict(note) for note in written) == source_before,
    }
    by_id = {note.id: note for note in written}
    ownership = {identifier: event.id for event in result.events for identifier in event.written_ids}
    checks['declared_edge_identity_preserved'] = all(ownership[a] == ownership[b] for a, b in ties)
    checks['chain_coverage'] = all(
        event.start_tick == by_id[event.written_ids[0]].start_tick and
        event.end_tick == by_id[event.written_ids[-1]].end_tick and
        all(by_id[a].end_tick == by_id[b].start_tick and (a, b) in ties
            for a, b in zip(event.written_ids, event.written_ids[1:]))
        for event in result.events)
    unknown = score_events(written, ties, verified_note_ids=[])
    checks['all_unknown_masked'] = all(not event.supervision_eligible for event in unknown.events) and all(
        view.role == 'unknown' for view in unknown.notation)
    removed = written[-1].id
    partial = score_events(written, ties, verified_note_ids=ids - {removed})
    expected_unknown = set(next(event.written_ids for event in result.events if removed in event.written_ids))
    checks['one_unknown_masks_only_its_event'] = all(
        event.supervision_eligible == (removed not in event.written_ids) for event in partial.events) and all(
        (view.role == 'unknown') == (view.note.id in expected_unknown) for view in partial.notation)
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    detail = dict(input_sha256=case['input_sha256'], source_written=[asdict(note) for note in written],
                  authored_edges=ties, declarations={k: sorted(v) for k, v in declarations.items()},
                  known=asdict(result), all_unknown=asdict(unknown), one_unknown=asdict(partial),
                  withheld_note_id=removed, checks=checks)
    summary = dict(name=name, written_count=len(written), tie_count=len(ties), event_count=len(result.events),
                   attack_count=sum(v.role == 'attack' for v in result.notation),
                   continuation_count=sum(v.role == 'continuation' for v in result.notation), checks=checks)
    return detail, summary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=ROOT / 'test-data/piano-svsep/bar-identity-v1')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous results')
    source_report_path = args.source / 'report.json'
    verify(source_report_path, SOURCE_REPORT_SHA256)
    source_report = read(source_report_path)
    if [case['name'] for case in source_report['cases']] != list(EXPECTED):
        raise ValueError('Unexpected source control manifest')
    for name, digest in source_report['artifacts_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unexpected source artifact path')
        verify(args.source / name, digest)
    module = ROOT / 'scripts/piano_score_events.py'
    protocol = ROOT / 'docs/piano-score-events-protocol.md'
    identity = dict(runner_sha256=sha256(__file__), interface_sha256=sha256(module),
                    protocol_sha256=sha256(protocol), source_report_sha256=SOURCE_REPORT_SHA256,
                    source_artifacts_verified=len(source_report['artifacts_sha256']),
                    python_version=sys.version, source='Seven self-authored XML controls and authored tie manifest')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / 'runner-snapshot.py')
    shutil.copyfile(module, args.output / 'interface-snapshot.py')
    shutil.copyfile(protocol, args.output / 'protocol-snapshot.md')
    save = lambda name, data: (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    save('identity.json', identity)
    start = time.perf_counter()
    results = []
    for case in source_report['cases']:
        detail, summary = case_check(args.source, case)
        save(case['name'] + '-detail.json', detail)
        results.append(summary)
        print(json.dumps(summary), flush=True)
    verify(module, identity['interface_sha256'])
    verify(__file__, identity['runner_sha256'])
    verify(protocol, identity['protocol_sha256'])
    verify(source_report_path, SOURCE_REPORT_SHA256)
    for name, digest in source_report['artifacts_sha256'].items():
        verify(args.source / name, digest)
    report = dict(identity=identity, cases=results, completed=len(results), errors=0,
                  elapsed_seconds=time.perf_counter() - start,
                  old_audit_unchanged=True, model_loaded=False, training_cache_read=False,
                  training_cache_changed=False, weights_changed=False, enabled_in_application=False,
                  musical_acceptance=False,
                  limitations=['Only authored tie edges were supplied; arbitrary XML tie ambiguity remains unresolved.',
                               'Notated ends do not measure physical or acoustic release.',
                               'This is a data representation check, not model or musical acceptance.'])
    report['artifacts_sha256'] = {path.name: sha256(path) for path in args.output.iterdir() if path.name != 'report.json'}
    save('report.json', report)
    print(json.dumps(dict(completed=len(results), report_sha256=sha256(args.output / 'report.json'),
                         event_counts=[result['event_count'] for result in results])), flush=True)


if __name__ == '__main__':
    main()
