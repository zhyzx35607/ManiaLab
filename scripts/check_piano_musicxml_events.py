"""Verify source score events and independent imported geometry on train/validation."""
import argparse
from collections import Counter
from dataclasses import asdict
from io import BytesIO
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
from scripts.piano_musicxml_events import read_musicxml


def imported_geometry(path, source, partitura):
    tree = ET.parse(path)
    identifiers = {}
    for pi, part in enumerate(tree.getroot().findall('part'), 1):
        for mi, measure in enumerate(part.findall('measure'), 1):
            for ni, note in enumerate(measure.findall('note'), 1):
                identifier = f'ml-p{pi}-m{mi}-n{ni}'
                identifiers[identifier] = f'part[{pi}]/measure[{mi}]/note[{ni}]'
                note.set('id', identifier)
                note.attrib.pop('ID', None)
    score = partitura.load_musicxml(BytesIO(ET.tostring(tree.getroot(), encoding='utf-8')))
    imported, differences = {}, []
    for part in score.parts:
        clock = part.quarter_map
        zero = float(clock(0))
        for note in part.notes:
            if note.id not in identifiers:
                differences.append(dict(reason='unknown_imported_id', id=note.id))
                continue
            key = identifiers[note.id]
            if key in imported:
                differences.append(dict(reason='duplicate_imported_id', id=key))
            imported[key] = dict(part=part.id, pitch=note.midi_pitch,
                start=float(clock(note.start.t)) - zero, end=float(clock(note.end.t)) - zero,
                staff=note.staff, voice=str(note.voice))
    expected = {row.note.id for row in source.written}
    exclusions = {identifier for identifier, _ in source.excluded}
    for key in sorted(set(imported) - expected - exclusions):
        differences.append(dict(reason='extra_imported_note', id=key))
    for row in source.written:
        note = row.note
        if note.id not in imported:
            differences.append(dict(reason='missing_imported_note', id=note.id))
            continue
        actual = imported[note.id]
        desired = dict(part=note.part, pitch=note.pitch, start=note.start_tick/source.ticks_per_quarter,
            end=note.end_tick/source.ticks_per_quarter, staff=note.staff, voice=note.voice or '1')
        fields = [field for field in desired if (abs(desired[field] - actual[field]) > 1e-8
            if field in ('start', 'end') else desired[field] != actual[field])]
        if fields:
            differences.append(dict(reason='geometry_mismatch', id=note.id, fields=fields,
                                    expected=desired, actual=actual))
    return differences, len(imported)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset', type=Path, default=ROOT/'test-data/piano-svsep/dataset-v2')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous source checks')
    import partitura
    import partitura.io.importmusicxml as importer
    manifest_path = args.dataset/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = [row for row in manifest['cases'] if row['status'] == 'ready' and row['split'] in ('train', 'validation')]
    if Counter(row['split'] for row in rows) != {'train': 157, 'validation': 16}:
        raise ValueError('Unexpected frozen split selection')
    args.output.mkdir(parents=True)
    save = lambda name, value: (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    sources = {'runner': Path(__file__), 'adapter': ROOT/'scripts/piano_musicxml_events.py',
        'representation': ROOT/'scripts/piano_score_events.py', 'protocol': ROOT/'docs/piano-musicxml-events-protocol.md'}
    identity = dict(dataset_manifest_sha256=sha256(manifest_path), corpus_revision=manifest['corpus_revision'],
        code_sha256={name: sha256(path) for name, path in sources.items()},
        partitura_version=partitura.__version__, partitura_import_sha256=sha256(importer.__file__),
        input_scope='ready train/validation, complete source files, no old note arrays read')
    for name, path in sources.items():
        shutil.copyfile(path, args.output/(name+'-snapshot'+path.suffix))
    save('identity.json', identity)
    started, controls, results = time.monotonic(), [], []
    control_path = ROOT/'test-data/piano-svsep/bar-identity-v1'
    control_report = json.loads((control_path/'report.json').read_text(encoding='utf-8'))
    for case in control_report['cases']:
        path = control_path/(case['name']+'.musicxml')
        if sha256(path) != case['input_sha256']:
            raise ValueError('Control source changed')
        source = read_musicxml(path)
        ambiguous = case['name'] == 'parallel_unison_ties'
        count = len(source.representation.events)
        unknown = sum(view.role == 'unknown' for view in source.representation.notation)
        assert (count, unknown) == ((4, 4) if ambiguous else (case['expected_raw_events'], 0))
        controls.append(dict(name=case['name'], events=count, unknown=unknown, input_sha256=case['input_sha256']))
    for index, row in enumerate(rows):
        path = Path(row['path'])
        if sha256(path) != row['sha256']:
            raise ValueError('Frozen source changed: '+row['name'])
        record = dict(name=row['name'], split=row['split'], input_sha256=row['sha256'], status='error')
        detail = {}
        try:
            source = read_musicxml(path)
            detail['source'] = asdict(source)
            record.update(written=len(source.written), ties=len(source.ties),
                score_events=len(source.representation.events), verified=len(source.verified_note_ids),
                unknown=len(source.issues), excluded=len(source.excluded),
                continuations=sum(view.role == 'continuation' for view in source.representation.notation),
                unknown_reasons=dict(Counter(reason for _, reasons in source.issues for reason in reasons)))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                differences, imported_count = imported_geometry(path, source, partitura)
            detail['geometry_differences'] = differences
            detail['warnings'] = dict(Counter(str(w.message) for w in caught))
            record.update(status='complete', geometry_equal=not differences, imported_count=imported_count,
                geometry_differences=len(differences), difference_reasons=dict(Counter(d['reason'] for d in differences)))
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        if sha256(path) != row['sha256']:
            raise ValueError('Source mutated during check')
        detail['summary'] = record
        filename = row['name']+'-detail.json'
        save(filename, detail)
        record['detail_file'] = filename
        results.append(record)
        save('progress.json', results)
        print(json.dumps(dict(index=index+1, total=len(rows), **record)), flush=True)
    if any(sha256(path) != identity['code_sha256'][name] for name, path in sources.items()):
        raise ValueError('Code changed while running')
    if sha256(manifest_path) != identity['dataset_manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    totals = {split: {field: sum(row.get(field, 0) for row in results if row['split'] == split)
        for field in ('written', 'ties', 'score_events', 'verified', 'unknown', 'excluded', 'continuations')}
        for split in ('train', 'validation')}
    report = dict(identity=identity, controls=controls, cases=results, totals=totals,
        completed=sum(row['status']=='complete' for row in results), errors=sum(row['status']=='error' for row in results),
        geometry_equal_files=sum(row.get('geometry_equal', False) for row in results),
        elapsed_seconds=round(time.monotonic()-started, 3), test_sources_read_by_runner=False,
        development_test_format_peek='K279-1 first 180 lines viewed before split recognized; not scored/trained',
        model_loaded=False, training_cache_read=False, training_cache_changed=False,
        enabled_in_application=False, musical_acceptance=False,
        limitations=['Complete-file score notation, not the old 16-measure cache scope.',
            'Resolved score tie is not an acoustic release or cross-note melody identity.',
            'Ambiguous, incomplete or unsupported ties are masked; exclusions and errors are retained.'])
    report['artifacts_sha256'] = {path.name:sha256(path) for path in args.output.iterdir()
        if path.name not in ('progress.json', 'report.json')}
    save('report.json', report)
    print(json.dumps(dict(completed=report['completed'], errors=report['errors'],
        geometry_equal_files=report['geometry_equal_files'], totals=totals, elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
