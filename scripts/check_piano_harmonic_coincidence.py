"""Count harmonic pitch coincidences in frozen real-piano onset matches.

No audio inference, rematching, filtering or application edits are performed.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256

EXPECTED_MATCH_REPORT = '2caa746775a9b53d238b0f0b73fce83a27a3765fcdf0bf0f86268906d699c2fb'
EXPECTED_EVENT_REPORT = '6b8e3a1e3109b8e39ddd906700e36978820038b3b92f83a468843bd832957d88'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def harmonic_partners(events):
    """Use only pitches/onsets; return all qualifying lower attack partners."""
    ordered = sorted(range(len(events)), key=lambda i: (events[i]['onset'], i))
    times = [events[i]['onset'] for i in ordered]
    result = []
    for event in events:
        partners = []
        for position in range(bisect_left(times, event['onset']-.030), bisect_right(times, event['onset']+.030)):
            index = ordered[position]; lower = events[index]
            if lower['pitch'] >= event['pitch']:
                continue
            for harmonic in range(2, 9):
                cents = 100*(event['pitch']-lower['pitch'])-1200*math.log2(harmonic)
                if abs(cents) <= 35:
                    partners.append(dict(event_index=index, harmonic=harmonic, difference_cents=cents))
        result.append(partners)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new local test-data directory')
    previous = ROOT/'test-data/piano-inactive-onsets-v1'
    source = ROOT/'test-data/piano-release-semantics-v1'
    if sha256(previous/'report.json') != EXPECTED_MATCH_REPORT or sha256(source/'report.json') != EXPECTED_EVENT_REPORT:
        raise ValueError('Frozen reports changed')
    matched_report, source_report = read(previous/'report.json'), read(source/'report.json')
    for folder, report in ((previous, matched_report), (source, source_report)):
        for filename, digest in report['artifacts_sha256'].items():
            if sha256(folder/filename) != digest:
                raise ValueError('Frozen input changed: '+filename)
    args.output.mkdir(parents=True)
    protocol = ROOT/'docs/piano-harmonic-coincidence-protocol.md'
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    shutil.copyfile(protocol, args.output/'protocol-snapshot.md')
    rows = []; totals = Counter(); inputs = {}
    match_paths = sorted(previous.glob('*-matches.json'))
    if len(match_paths) != 11:
        raise ValueError('Expected eleven fixed recordings')
    for path in match_paths:
        name = path.name.removesuffix('-matches.json')
        events_path = source/(name+'-events.json')
        inputs[name] = dict(events_sha256=sha256(events_path), matches_sha256=sha256(path))
        events = read(events_path)['events']; matches = read(path)
        ids = matches['original_event_indices']
        if ids != [i for i, e in enumerate(events) if 0 <= e['onset'] < 60]:
            raise ValueError('Onset scoring window changed')
        paired = {r['event_index'] for r in matches['original_pairs']}
        if len(paired) != len(matches['original_pairs']):
            raise ValueError('Duplicate onset match')
        partners = harmonic_partners(events); counts = Counter(); details = []
        for index in ids:
            event = events[index]
            cohort = 'inactive' if event['release_reason'] == 'frame_inactive' else 'other'
            outcome = 'matched' if index in paired else 'unmatched'
            relation = 'coincident' if partners[index] else 'no_coincidence'
            counts[cohort+'_'+outcome+'_'+relation] += 1
            details.append(dict(event_index=index, matched=index in paired,
                release_reason=event['release_reason'], lower_partners=partners[index]))
        row = dict(name=name, targets=len(ids), counts=dict(counts))
        rows.append(row); totals.update(counts)
        (args.output/(name+'.json')).write_text(json.dumps(dict(summary=row, events=details), indent=2), encoding='utf-8')
    if sum(totals.values()) != 5005 or sum(n for k, n in totals.items() if k.startswith('inactive_')) != 47:
        raise ValueError('Frozen cohort changed')
    if sum(n for k, n in totals.items() if '_matched_' in k) != 4766:
        raise ValueError('Frozen matches changed')
    report = dict(identity=dict(matches_report_sha256=EXPECTED_MATCH_REPORT,
        event_report_sha256=EXPECTED_EVENT_REPORT, runner_sha256=sha256(Path(__file__)), protocol_sha256=sha256(protocol)),
        input_hashes=inputs, cases=rows, counts=dict(totals),
        parameters=dict(onset_distance_seconds=.030, harmonics=list(range(2,9)), max_cents=35),
        onset_matching_rerun=False, model_rerun=False, notes_changed=False,
        application_enabled=False, musical_acceptance=False,
        artifacts_sha256={p.name:sha256(p) for p in args.output.iterdir()})
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['counts']), flush=True)


if __name__ == '__main__':
    main()
