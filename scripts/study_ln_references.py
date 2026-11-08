"""Measure explicitly selected charts without treating authored notes as audio truth.

The selection contains pinned SHA256 hashes and research groups. Source charts
and paired audio stay in test-data; the report contains descriptors and a small
number of locations to inspect. This script does not train a model or alter a
reference chart. Different difficulties of a song are development comparisons,
never independent validation songs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, validate
from manialab.ln_patterns import analyze_ln_patterns
from manialab.timing import BeatClock


def beat_duration(start, end, clock):
    """Integrate beats across red-line changes, rather than using head BPM."""
    boundaries = [start, *(t for t in clock.offsets if start < t < end), end]
    return sum((b - a) / clock(a) for a, b in zip(boundaries, boundaries[1:]))


def compare_intervals(left, right, tolerance=3):
    """One-to-one matching of same-time heads; lane positions are ignored."""
    left = sorted(left, key=lambda n: (n.start, n.end))
    right = sorted(right, key=lambda n: (n.start, n.end))
    starts = [n.start for n in right]
    used, counts, examples = set(), Counter(), []
    for note in left:
        nearby = [i for i in range(bisect_left(starts, note.start-tolerance),
                                   bisect_right(starts, note.start+tolerance)) if i not in used]
        if not nearby:
            counts['unmatched_left'] += 1
            continue
        index = min(nearby, key=lambda i: (right[i].hold != note.hold,
                                          abs(right[i].end-note.end),
                                          abs(right[i].start-note.start)))
        other = right[index]
        used.add(index)
        if note.hold != other.hold:
            kind = 'ln_to_tap' if note.hold else 'tap_to_ln'
        elif abs(note.end-other.end) <= tolerance:
            kind = 'same_interval'
        else:
            kind = 'longer_tail' if other.end > note.end else 'shorter_tail'
        counts[kind] += 1
        if kind not in ('same_interval',) and len(examples) < 8:
            examples.append(dict(kind=kind, head_ms=note.start,
                                 left_end_ms=note.end, right_end_ms=other.end))
    counts['unmatched_right'] = len(right)-len(used)
    return dict(counts=dict(counts), examples=examples, head_tolerance_ms=tolerance,
                notice='Same-time musical layers are ambiguous; matching is descriptive, not proof of mapper intent.')


def describe(chart):
    notes, keys = chart['notes'], chart['keys']
    check = validate(notes, keys)
    if not check['valid']:
        raise ValueError('Reference contains conflicting objects: '+str(check['issues'][:3]))
    clock = BeatClock(chart['timing'])
    holds = [n for n in notes if n.hold]
    lengths = np.array([beat_duration(n.start, n.end, clock) for n in holds])
    taps = sorted(n.start for n in notes if not n.hold)
    candidates = []
    for n, length in zip(holds, lengths):
        if length < 2:
            continue
        count = bisect_left(taps, n.end)-bisect_right(taps, n.start)
        candidates.append(dict(start_ms=n.start, end_ms=n.end, lane=n.lane+1,
                               beats=round(float(length), 3), taps_during=count))
    # Bounded examples from separate song sections; don't fill the list with
    # several simultaneous holds from the same moment.
    examples = {}
    for row in sorted(candidates, key=lambda r: -(r['beats']+min(8, r['taps_during']))):
        examples.setdefault(row['start_ms']//15000, row)
    interactions = analyze_ln_patterns(notes, keys, timing_points=chart['timing'])
    return dict(structure=check, patterns={k:v for k,v in interactions.items() if k != 'locations'},
        ln_duration_beats=dict(zip(('p10','p50','p90','max'),
            [round(float(x), 3) for x in np.quantile(lengths, [.1,.5,.9,1])])) if len(lengths) else {},
        ln_duration_counts={label:int(np.sum((lengths >= a)&(lengths < b)))
            for label,a,b in [('under_half',0,.5-1e-2),('half_to_one',.5-1e-2,1-1e-2),
                              ('one_to_two',1-1e-2,2-1e-2),('two_to_four',2-1e-2,4-1e-2),
                              ('four_or_more',4-1e-2,float('inf'))]},
        short_ln_under_240ms=sum(n.end-n.start < 240 for n in holds),
        inspection_examples=sorted(list(examples.values())[:6],key=lambda r:r['start_ms']))


def run(selection, corpus, output):
    records = json.loads(selection.read_text(encoding='utf-8'))
    rows, charts, groups, fingerprints = [], {}, defaultdict(list), {}
    for record in records:
        bid = record['beatmap_id']
        raw = (corpus/'maps'/f'{bid}.osu').read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError(f'Pinned chart hash mismatch: {bid}')
        chart = parse_osu(raw.decode('utf-8-sig'))
        if chart['keys'] != record['keys']:
            raise ValueError(f'Keycount mismatch: {bid}')
        fingerprint = hashlib.sha256(json.dumps(sorted((n.start,n.end,n.lane)
            for n in chart['notes'])).encode()).hexdigest()
        duplicate = fingerprints.get(fingerprint)
        fingerprints.setdefault(fingerprint, bid)
        metadata = chart['metadata']
        row = {k:record.get(k) for k in ('beatmap_id','beatmapset_id','sha256','group','language_id','status')}
        row.update(keys=chart['keys'], title=metadata.get('TitleUnicode') or metadata.get('Title'),
            artist=metadata.get('ArtistUnicode') or metadata.get('Artist'), creator=metadata.get('Creator'),
            version=metadata.get('Version'), duplicate_layout_of=duplicate, **describe(chart))
        rows.append(row)
        charts[bid] = chart
        # Pack-level grouping is unsafe: one pack can contain many songs.
        groups[(record['beatmapset_id'],chart['general'].get('AudioFilename'),chart['keys'])].append(bid)
    comparisons = []
    for members in groups.values():
        members.sort(key=lambda bid:len(charts[bid]['notes']))
        for left, right in zip(members,members[1:]):
            comparisons.append(dict(left=left,right=right,**compare_intervals(charts[left]['notes'],charts[right]['notes'])))
    report = dict(protocol=__doc__, selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
        chart_count=len(rows), key_counts=dict(Counter(r['keys'] for r in rows)),
        rows=rows, same_audio_difficulty_comparisons=comparisons,
        limits=['Language is catalog metadata, not automatic lyric verification.',
                'Selected development references, not a representative sample or a blind test.',
                'LN duration and occupancy describe charts, not audio transcription accuracy.',
                '7K/9K hand statistics approximate the shared center key.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(charts=len(rows),comparisons=len(comparisons),keys=report['key_counts'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', type=Path)
    parser.add_argument('--corpus', type=Path, default=Path('test-data/corpus'))
    parser.add_argument('--output', type=Path, default=Path('test-data/chinese-ln/structure-report.json'))
    args = parser.parse_args()
    run(args.selection,args.corpus,args.output)
