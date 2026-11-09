"""Explore inactive-frame onset cohorts and one fixed, non-model ablation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_piano_acoustics import counts, pairs

SOURCE_REPORT_SHA256 = '6b8e3a1e3109b8e39ddd906700e36978820038b3b92f83a468843bd832957d88'
BENCHMARK_SHA256 = 'f1e7bcffc17e9aa5399aa2e923939748e3c9cc2a29e8e1b0a766e2fbc20e3c6b'
SCORING_SHA256 = '1f3b4289ce25defac4af9613f5b3a75206d17319434fa475df8c22e552fb929f'
CASE_NAMES = ('bach', 'bartok', 'beethoven', 'brahms', 'chopin', 'haydn',
              'liszt', 'mozart', 'rachmaninoff', 'ravel', 'skryabin')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort(events, matched_indices, indices):
    matched = sum(index in matched_indices for index in indices)
    return dict(events=len(indices), matched=matched, unmatched=len(indices)-matched,
                match_fraction=matched/len(indices) if indices else None)


def analyze(reference, events):
    """Compare original candidates to one fixed removal, preserving identities."""
    scoped = [(i, event) for i, event in enumerate(events) if 0 <= event['onset'] < 60]
    original_indices = [i for i, _ in scoped]
    original = [event for _, event in scoped]
    retained_positions = [i for i, event in enumerate(original)
                          if event['release_reason'] != 'frame_inactive']
    retained = [original[i] for i in retained_positions]
    first = pairs(reference, original, strict=False, offsets=False)
    second_local = pairs(reference, retained, strict=False, offsets=False)
    second = [(r, retained_positions[p]) for r, p in second_local]
    first_prediction = {p for _, p in first}
    second_prediction = {p for _, p in second}
    first_reference = {r for r, _ in first}
    second_reference = {r for r, _ in second}
    inactive = [i for i, event in enumerate(original) if event['release_reason'] == 'frame_inactive']
    other = retained_positions
    before = counts(len(first), len(reference), len(original))
    after = counts(len(second), len(reference), len(retained))
    if after['matched'] > before['matched']:
        raise ValueError('Removing candidates unexpectedly increased maximum matching cardinality')
    cohorts = {name: cohort(original, first_prediction, indices)
               for name, indices in (('frame_inactive', inactive), ('other', other))}
    losses = dict(removed_previously_matched=len(set(inactive) & first_prediction),
        net_matched_loss=len(first)-len(second),
        reference_identity_losses=len(first_reference-second_reference),
        reference_identity_gains=len(second_reference-first_reference),
        retained_newly_matched=len(second_prediction-first_prediction))
    summary = dict(events_70s=len(events), events_scored=len(original), reference=len(reference),
        cohorts=cohorts, original=before, without_frame_inactive=after,
        delta={key: after[key]-before[key] for key in ('matched', 'predicted', 'precision', 'recall', 'f1')},
        true_onset_loss=losses, physical_fields_preserved=True)
    detail = dict(summary=summary, original_event_indices=original_indices,
        retained_event_indices=[original_indices[i] for i in retained_positions],
        removed_event_indices=[original_indices[i] for i in inactive],
        original_pairs=[dict(reference_index=r, event_index=original_indices[p]) for r, p in first],
        retained_pairs=[dict(reference_index=r, event_index=original_indices[p]) for r, p in second],
        removed_previously_matched_event_indices=[original_indices[p] for p in inactive if p in first_prediction],
        lost_reference_indices=sorted(first_reference-second_reference),
        gained_reference_indices=sorted(second_reference-first_reference),
        retained_newly_matched_event_indices=[original_indices[p] for p in sorted(second_prediction-first_prediction)])
    return summary, detail


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior results; choose a new output folder')
    source_dir = ROOT/'test-data/piano-release-semantics-v1'
    source_path = source_dir/'report.json'
    manifest_path = ROOT/'test-data/smd/selected-v1/benchmark/manifest.json'
    scoring = ROOT/'scripts/evaluate_piano_acoustics.py'
    if (sha(source_path) != SOURCE_REPORT_SHA256 or sha(manifest_path) != BENCHMARK_SHA256
            or sha(scoring) != SCORING_SHA256):
        raise ValueError('Frozen research inputs or matching implementation changed')
    source, manifest = read(source_path), read(manifest_path)
    sources = {row['name']: row for row in source['cases']}
    if tuple(case['name'] for case in manifest['cases']) != CASE_NAMES:
        raise ValueError('Unexpected benchmark coverage or order')
    code = dict(runner=Path(__file__), scoring=scoring,
                protocol=ROOT/'docs/piano-inactive-onset-protocol.md')
    identity = dict(source_report_sha256=SOURCE_REPORT_SHA256,
        benchmark_manifest_sha256=BENCHMARK_SHA256,
        code_sha256={key: sha(path) for key, path in code.items()},
        reference_semantics='key MIDI onsets only; offsets excluded from matching',
        score_onsets_seconds=[0, 60], input_seconds=70, onset_tolerance_seconds=.05,
        matching='fixed same-pitch one-to-one; existing maximum-cardinality onset matcher',
        ablation='remove only release_reason == frame_inactive',
        exploratory=True, blind_test=False, threshold_search=False,
        model_loaded=False, decoder_run=False, application_enabled=False,
        prior_observation='12/16 delayed-tail changes lacked same-pitch reference onset within 50ms')
    args.output.mkdir(parents=True)
    save = lambda name, value: (args.output/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    for key, path in code.items():
        shutil.copyfile(path, args.output/(key+'-snapshot'+path.suffix))
    save('identity.json', identity)
    rows = []
    input_hashes = {}
    for case in manifest['cases']:
        name = case['name']
        if (case['start_seconds'], case['score_seconds'], case['duration_seconds']) != (0., 60., 70.):
            raise ValueError('Benchmark window changed')
        event_path = source_dir/(name+'-events.json')
        labels_path = ROOT/case['labels']
        if (sha(event_path) != source['artifacts_sha256'][event_path.name]
                or sha(labels_path) != case['labels_sha256']):
            raise ValueError('Frozen events or onset labels changed')
        events = read(event_path)['events']
        if len(events) != sources[name]['events']:
            raise ValueError('Original event count changed')
        all_reference = read(labels_path)
        reference = [note for note in all_reference if 0 <= note['onset'] < 60]
        summary, detail = analyze(reference, events)
        detail['reference_original_indices'] = [i for i, note in enumerate(all_reference) if 0 <= note['onset'] < 60]
        row = dict(name=name, events_sha256=sha(event_path), labels_sha256=sha(labels_path), **summary)
        save(name+'-matches.json', detail)
        input_hashes[str(event_path)] = row['events_sha256']
        input_hashes[str(labels_path)] = row['labels_sha256']
        rows.append(row)
        print(json.dumps(row), flush=True)
    sakura_path = source_dir/'sakura-nova-piano-events.json'
    if sha(sakura_path) != source['artifacts_sha256'][sakura_path.name]:
        raise ValueError('Frozen Sakura candidates changed')
    sakura_events = read(sakura_path)['events']
    reason_counts = Counter(event['release_reason'] for event in sakura_events)
    sakura = dict(name='sakura-nova-piano', events=len(sakura_events),
        frame_inactive=reason_counts['frame_inactive'],
        other=len(sakura_events)-reason_counts['frame_inactive'],
        release_reason_counts=dict(reason_counts), scored=False, reference_read=False,
        events_sha256=sha(sakura_path))
    input_hashes[str(sakura_path)] = sakura['events_sha256']
    save('sakura-cohorts.json', sakura)
    pooled = {kind: counts(*(sum(row[kind][field] for row in rows)
                           for field in ('matched', 'reference', 'predicted')))
              for kind in ('original', 'without_frame_inactive')}
    cohort_totals = {}
    for kind in ('frame_inactive', 'other'):
        total = {field: sum(row['cohorts'][kind][field] for row in rows)
                 for field in ('events', 'matched', 'unmatched')}
        total['match_fraction'] = total['matched']/total['events'] if total['events'] else None
        cohort_totals[kind] = total
    summary = dict(pooled=pooled, cohorts=cohort_totals,
        macro={kind: {field: sum(row[kind][field] for row in rows)/len(rows)
                      for field in ('precision', 'recall', 'f1')}
               for kind in ('original', 'without_frame_inactive')},
        true_onset_loss={field: sum(row['true_onset_loss'][field] for row in rows)
                         for field in rows[0]['true_onset_loss']},
        delta={field: pooled['without_frame_inactive'][field]-pooled['original'][field]
               for field in ('matched', 'predicted', 'precision', 'recall', 'f1')})
    if (sha(source_path) != SOURCE_REPORT_SHA256 or sha(manifest_path) != BENCHMARK_SHA256
            or any(sha(path) != digest for path, digest in input_hashes.items())
            or any(sha(path) != identity['code_sha256'][key] for key, path in code.items())):
        raise ValueError('Research inputs or code changed during matching')
    report = dict(identity=identity, cases=rows, summary=summary, sakura=sakura,
        input_sha256=input_hashes, model_rerun=False, application_enabled=False,
        musical_acceptance=False,
        limitations=['Exploratory ablation informed by previous inspection, not a blind test.',
                     'Unmatched candidates are defined by one fixed onset/pitch tolerance.',
                     'Higher aggregate precision cannot justify deleting real short onsets.',
                     'Sakura has cohort counts only and no reference scoring.'])
    report['artifacts_sha256'] = {path.name: sha(path) for path in args.output.iterdir()}
    save('report.json', report)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
