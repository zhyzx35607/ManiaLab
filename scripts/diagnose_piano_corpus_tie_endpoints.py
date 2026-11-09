"""Classify imported tie endpoints outside the earlier exact-Note audit domain."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import warnings
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256

AUDIT_SHA = '1fbdeb843e237f97396ae7258ec349ca07f9015274f5a296b729fa11f408bb8d'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Identity differs: {path.name}')


def source_declarations(path):
    root = ET.parse(path).getroot()
    counts, special = Counter(), []
    for pi, part in enumerate(root.findall('part'), 1):
        for mi, measure in enumerate(part.findall('measure'), 1):
            for ni, note in enumerate(measure.findall('note'), 1):
                counts['written_notes'] += 1
                counts['explicit_note_ids'] += int(note.get('id') is not None)
                sound = [tie.get('type') for tie in note.findall('tie')]
                if not sound:
                    continue
                kind = 'grace' if note.find('grace') is not None else 'cue' if note.find('cue') is not None else 'rest' if note.find('rest') is not None else 'ordinary'
                counts['sound_tie_notes'] += 1
                counts['sound_tie_' + kind + '_notes'] += 1
                if kind != 'ordinary':
                    special.append(dict(structural_path=f'part[{pi}]/measure[{mi}]/note[{ni}]',
                                        part_id=part.get('id'), measure_label=measure.get('number'), kind=kind,
                                        xml_id=note.get('id'), sound_ties=sound,
                                        notation_ties=[dict(tie.attrib) for tie in note.findall('notations/tied')],
                                        pitch=dict(step=note.findtext('pitch/step'), octave=note.findtext('pitch/octave'),
                                                   alter=note.findtext('pitch/alter')),
                                        duration=note.findtext('duration'), staff=note.findtext('staff'), voice=note.findtext('voice'),
                                        grace_attributes=None if note.find('grace') is None else dict(note.find('grace').attrib)))
    return dict(counts), special


def describe(note):
    return dict(type=type(note).__name__, id=getattr(note, 'id', None),
                start=None if note.start is None else int(note.start.t),
                end=None if note.end is None else int(note.end.t),
                pitch=None if not hasattr(note, 'midi_pitch') else int(note.midi_pitch),
                staff=getattr(note, 'staff', None), voice=getattr(note, 'voice', None))


def endpoints(score, partitura):
    rows, unique_nodes, unique_edges = [], {}, set()
    all_registered = {id(item) for part in score.parts for item in part.iter_all()}
    for pi, part in enumerate(score.parts):
        ordinary = list(part.iter_all(partitura.score.Note))
        exact_ids = {id(note) for note in ordinary}
        note_ids = {id(note) for note in part.iter_all(partitura.score.Note, include_subclasses=True)}
        generic_ids = {id(note) for note in part.iter_all(partitura.score.GenericNote, include_subclasses=True)}
        for note in ordinary:
            for direction in ('tie_prev', 'tie_next'):
                other = getattr(note, direction)
                if other is None or id(other) in exact_ids:
                    continue
                source, target = (other, note) if direction == 'tie_prev' else (note, other)
                unique_edges.add((id(source), id(target)))
                meta = dict(part_index=pi, direction=direction, from_note=describe(note), other=describe(other),
                            registered_note_subclass_same_part=id(other) in note_ids,
                            registered_generic_note_same_part=id(other) in generic_ids,
                            registered_anywhere=id(other) in all_registered)
                rows.append(meta)
                unique_nodes[id(other)] = meta
    if not rows:
        raise ValueError('Earlier outside-Note condition was not reproduced')
    nodes = list(unique_nodes.values())
    return rows, dict(pointer_references=len(rows), unique_relations=len(unique_edges), unique_outside_endpoints=len(nodes),
                      endpoint_types=dict(Counter(node['other']['type'] for node in nodes)),
                      registered_note_subclass_endpoints=sum(node['registered_note_subclass_same_part'] for node in nodes),
                      registered_generic_note_endpoints=sum(node['registered_generic_note_same_part'] for node in nodes),
                      registered_anywhere_endpoints=sum(node['registered_anywhere'] for node in nodes),
                      zero_duration_endpoints=sum(node['other']['start'] is not None and
                                                  node['other']['start'] == node['other']['end'] for node in nodes))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT / 'test-data/piano-svsep')
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--summary', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists() or args.summary.exists():
        raise ValueError('Preserve previous diagnostic outputs')
    audit_path = args.root / 'corpus-ties-v2/report.json'
    verify(audit_path, AUDIT_SHA)
    audit = read(audit_path)
    manifest_path = args.root / 'dataset-v2/manifest.json'
    verify(manifest_path, audit['identity']['manifest_sha256'])
    failed = {row['name'] for row in audit['cases'] if 'error' in row}
    selected = [row for row in read(manifest_path)['cases'] if row['name'] in failed]
    if len(selected) != 22 or any(row['split'] not in ('train', 'validation') for row in selected):
        raise ValueError('Unexpected diagnostic cohort')
    import partitura
    import partitura.io.importmusicxml as importer
    import partitura.score as score_code
    protocol = ROOT / 'docs/piano-corpus-tie-endpoints-protocol.md'
    identity = dict(audit_report_sha256=AUDIT_SHA, manifest_sha256=sha256(manifest_path),
                    runner_sha256=sha256(__file__), protocol_sha256=sha256(protocol),
                    partitura_version=importlib.metadata.version('partitura'),
                    importer_sha256=sha256(importer.__file__), score_code_sha256=sha256(score_code.__file__))
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / 'runner-snapshot.py')
    shutil.copyfile(protocol, args.output / 'protocol-snapshot.md')
    save = lambda name, data: (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    save('identity.json', identity)
    rows, started = [], time.perf_counter()
    for source in selected:
        path = Path(source['path'])
        verify(path, source['sha256'])
        result = dict(name=source['name'], split=source['split'], source_sha256=source['sha256'], status='error')
        detail, caught = {}, []
        try:
            counts, declarations = source_declarations(path)
            result['source_declaration_counts'] = counts
            detail['special_source_declarations'] = declarations
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                score = partitura.load_score(str(path))
                relations, summary = endpoints(score, partitura)
            detail['outside_note_pointer_records'] = relations
            result.update(endpoint_summary=summary, status='complete')
        except Exception as exc:
            result.update(error_type=type(exc).__name__)
            detail['error'] = str(exc)
        result['warning_count'] = len(caught)
        detail.update(summary=result, warnings=dict(Counter(str(item.message) for item in caught)))
        save(source['name'] + '-detail.json', detail)
        verify(path, source['sha256'])
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    verify(audit_path, AUDIT_SHA)
    verify(manifest_path, identity['manifest_sha256'])
    verify(__file__, identity['runner_sha256'])
    verify(protocol, identity['protocol_sha256'])
    endpoint_totals, declaration_totals, types = Counter(), Counter(), Counter()
    for row in rows:
        declaration_totals.update(row.get('source_declaration_counts', {}))
        if 'endpoint_summary' in row:
            types.update(row['endpoint_summary']['endpoint_types'])
            endpoint_totals.update({key: value for key, value in row['endpoint_summary'].items() if key != 'endpoint_types'})
    report = dict(identity=identity, cases=rows, completed=sum(row['status'] == 'complete' for row in rows),
                  errors=sum(row['status'] != 'complete' for row in rows), elapsed_seconds=time.perf_counter() - started,
                  endpoint_totals=dict(endpoint_totals), endpoint_types=dict(types),
                  source_declaration_totals=dict(declaration_totals),
                  old_audit_changed=False, old_cache_read=False, old_cache_changed=False,
                  test_scores_read_by_diagnostic=False, upstream_preprocessing_called=False,
                  model_loaded=False, musical_acceptance=False,
                  limitations=['Only the 22 earlier audit failures were diagnosed; original audit counts remain frozen.',
                               'Special XML declarations and imported object counts are not asserted to be a one-to-one mapping.',
                               'Registered GraceNote endpoints do not establish correct tie pairing or acoustic timing.'])
    report['artifacts_sha256'] = {path.name: sha256(path) for path in args.output.iterdir() if path.name != 'report.json'}
    save('report.json', report)
    public = dict(report, report_sha256=sha256(args.output / 'report.json'))
    public.pop('artifacts_sha256')
    public['local_artifacts_verified'] = len(report['artifacts_sha256'])
    for name, expected in report['artifacts_sha256'].items():
        verify(args.output / name, expected)
    args.summary.write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(completed=report['completed'], errors=report['errors'], endpoint_totals=dict(endpoint_totals),
                         endpoint_types=dict(types), source_declaration_totals=dict(declaration_totals),
                         report_sha256=public['report_sha256'], public_sha256=sha256(args.summary)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
