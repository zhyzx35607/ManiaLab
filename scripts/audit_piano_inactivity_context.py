"""Audit saved context decisions and check all eligible SMD candidate gaps.

No replay, inference, note edits, threshold fitting or model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.domain import parse_osu, UNOBSERVED_RELEASES
from scripts.check_expression_current_replay import read, save, sha


def sampled_low_intervals(frame, onset):
    """Separate frame loop to check extraction rather than importing it."""
    spans, first = [], None
    for i in range(len(frame)+1):
        low = i < len(frame) and bool(np.all(frame[i] <= .1)) and bool(np.all(onset[i] <= .3))
        if low and first is None:first = i
        if not low and first is not None:
            if i-1-first >= 8:spans.append([first*10, (i-1)*10])
            first = None
    return spans


def candidate_gaps(notes, intervals, stop):
    """Raw interval union; no expressiveness or strong-detection filtering."""
    merged = []
    for n in sorted(notes, key=lambda x: (x['start'], x['end'])):
        if merged and n['start'] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], n['end'])
        else:merged.append([n['start'], n['end']])
    gaps = []
    for previous, following in zip(merged, merged[1:]):
        left, right = previous[1], following[0]
        if right >= stop or right-left < 80:continue
        supported = [[max(a, left), min(b, right)] for a, b in intervals if min(b, right)-max(a, left) >= 80]
        if supported:gaps.append(dict(candidate_gap=[left, right], inactive_intervals=supported))
    return gaps


def coverage(labels, a, b, key):
    boundaries = {float(a), float(b)}
    active = [(max(a, n['onset']*1000), min(b, n[key]*1000)) for n in labels
              if n['onset']*1000 < b and n[key]*1000 > a]
    for left, right in active:boundaries.update((left, right))
    points = sorted(boundaries)
    return sum(right-left for left, right in zip(points, points[1:])
               if any(x <= left and y >= right for x, y in active))


def run(source, output):
    source, output = source.resolve(), output.resolve()
    assert not output.exists(), 'Preserve earlier audit receipts'
    report, plan = read(source/'report.json'), read(source/'execution-plan.json')
    assert sha(source/'execution-plan.json') == report['execution_plan_sha256']
    checked = {}
    def verify(path, expected):
        assert sha(path) == expected, str(path)
        checked[str(path.relative_to(ROOT))] = expected
    for path, digest in plan['inputs'].items():verify(ROOT/path, digest)
    for path, digest in plan['sources'].items():verify(source/'source'/path, digest)
    for row in report['cases']:
        for name, digest in row['artifacts'].items():verify(source/f"{row['keys']}k"/name, digest)
    for row in report['smd']:
        for name, digest in row['artifacts'].items():verify(source/'smd'/row['name']/name, digest)
    verify(source/'reference-activity-audit.json', report['reference_audit_sha256'])
    verify(source/'review.json', report['review_sha256'])
    baseline = ROOT/'test-data/articulation-interval-gaps-v3'
    analysis = read(ROOT/'test-data/sakura-tsumugi-expression-onset-v3/analysis.json')
    raw = {n['id']: n for n in analysis['candidates']}
    caches = [ROOT/path for path in plan['inputs'] if path.endswith('piano-specialist-activations.npz')]
    assert len(caches) == 1
    with np.load(caches[0], allow_pickle=False) as a:
        frame, onset = a['frame_output'], a['reg_onset_output']
    independent_intervals = sampled_low_intervals(frame, onset)
    assert independent_intervals == read(source/'piano-inactive-intervals.json')
    cases = []
    for row in report['cases']:
        keys = row['keys'];folder = source/f'{keys}k'
        before = {n['id']: n for n in read(baseline/f'{keys}k/candidates.json')}
        after = {n['id']: n for n in read(folder/'candidates.json')}
        assert before.keys() == after.keys()
        for identity, n in after.items():
            p = before[identity]
            # Raw/evidence ends may accompany the expressive end; no attack changes.
            assert all(n[k] == p[k] for k in n if k not in ('end', 'evidence_end'))
            assert n['end'] >= p['end']
            if n['end'] != p['end']:assert n['voice'] == 'piano' and n['release_reason'] not in UNOBSERVED_RELEASES
        selected = read(folder/'selected.json');placed = read(folder/'placed.json')
        assert all(n == after[n['id']] for n in selected)
        pbyid = {n['id']: n for n in placed}
        assert len(pbyid) == len(placed) == len(selected)
        for n in selected:assert all(n[k] == pbyid[n['id']][k] for k in n if k != 'lane')
        parsed = parse_osu((folder/'chart.osu').read_text('utf-8'))
        assert sorted((n.start, n.end, n.lane) for n in parsed['notes']) == sorted((n['start'], n['end'], n['lane']) for n in placed)
        for gap in read(folder/'gaps.json'):
            a, b = gap['inactive_start_ms'], gap['inactive_end_ms']
            assert b-a >= 80 and a % 10 == b % 10 == 0
            assert not any(n['voice'] == 'piano' and n['start'] < b and n['end'] > a for n in raw.values())
            assert np.all(frame[a//10:b//10+1] <= .1) and np.all(onset[a//10:b//10+1] <= .3)
        prior = read(baseline/f'{keys}k/selected.json');old = {n['id']: n for n in prior};new = {n['id']: n for n in selected}
        changes = dict(removed=[old[k] for k in sorted(old.keys()-new.keys())],
            added=[new[k] for k in sorted(new.keys()-old.keys())],
            changed=[dict(old=old[k], new=new[k]) for k in sorted(old.keys() & new.keys()) if old[k] != new[k]])
        save(output/f'{keys}k-selection-changes.json', changes)
        cases.append(dict(keys=keys, removed_ids=len(changes['removed']), added_ids=len(changes['added']),
            same_id_changes=len(changes['changed']),
            changed_voices=dict(Counter(x['old']['voice'] for x in changes['changed'])),
            removed_voices=dict(Counter(n['voice'] for n in changes['removed'])),
            added_voices=dict(Counter(n['voice'] for n in changes['added']))))
    smd_root = ROOT/'test-data/smd/selected-v1/benchmark';manifest = read(smd_root/'manifest.json')
    smd = []
    for case in manifest['cases']:
        name = case['name'];folder = source/'smd'/name
        with np.load(smd_root/'specialist-v2'/f'{name}-activations.npz', allow_pickle=False) as a:
            spans = sampled_low_intervals(a['frame_output'], a['reg_onset_output'])
        assert spans == read(folder/'inactive_intervals.json')
        notes = read(folder/'candidates.json');old, new = read(folder/'old.json'), read(folder/'new.json')
        assert old == new
        gaps = candidate_gaps(notes, spans, case['score_seconds']*1000)
        path = ROOT/case['sustain_labels'];verify(path, case['sustain_labels_sha256']);labels = read(path)
        for gap in gaps:
            gap['measurements'] = [dict(start_ms=a, end_ms=b, duration_ms=b-a,
                key_activity_ms=coverage(labels, a, b, 'key_offset'),
                pedal_activity_ms=coverage(labels, a, b, 'offset')) for a, b in gap['inactive_intervals']]
        smd.append(dict(name=name, all_eligible_candidate_gaps=gaps))
    save(output/'all-smd-gaps.json', smd)
    reviews = read(source/'review.json');audio_count = 0
    rate, pcm = wavfile.read(ROOT/'test-data/sakura-benchmark/raw-pcm-audit-v1/piano.wav')
    for i, review in enumerate(reviews):
        folder = source/'review'/str(i)
        for name, digest in review['artifacts'].items():verify(folder/name, digest)
        for mode, metadata in review['audio'].items():
            sr, clip = wavfile.read(folder/f'{mode}.wav')
            assert sr == rate == metadata['sample_rate'] and len(clip) == metadata['samples']
            if mode == 'original':
                start = metadata['first_sample'];expected = np.round(pcm[start:start+len(clip)]*.7*metadata['gain']*32767).astype(np.int16)
                assert np.array_equal(clip[:, 0], expected) and np.array_equal(clip[:, 1], expected)
            audio_count += 1
    gaps = [m for c in smd for gap in c['all_eligible_candidate_gaps'] for m in gap['measurements']]
    summary = dict(files_verified=len(checked), source_report_sha256=sha(source/'report.json'),
        cases=cases, checked_audio_files=audio_count, independent_interval_extraction=True,
        all_smd_candidate_gaps=sum(len(c['all_eligible_candidate_gaps']) for c in smd),
        all_smd_low_intervals=len(gaps), with_key_activity=sum(g['key_activity_ms'] > 0 for g in gaps),
        with_pedal_activity=sum(g['pedal_activity_ms'] > 0 for g in gaps),
        full_key_activity=sum(abs(g['key_activity_ms']-g['duration_ms']) < 1e-6 for g in gaps),
        full_pedal_activity=sum(abs(g['pedal_activity_ms']-g['duration_ms']) < 1e-6 for g in gaps),
        musical_acceptance=False, artifacts={p.name:sha(p) for p in output.glob('*.json')})
    save(output/'report.json', summary)
    print(summary)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args();run(args.source, args.output)
