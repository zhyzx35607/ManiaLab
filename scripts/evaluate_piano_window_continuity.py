"""Compare the fixed overlap repair on every saved SMD window, then score."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.piano import OUTPUTS, decode
from scripts.check_expression_current_replay import read, save, sha
from scripts.evaluate_piano_acoustics import metrics, pairs, counts
from manialab.piano_continuity import continue_overlap


def run(source, output, require_release_regression=False):
    source, output = source.resolve(), output.resolve()
    assert not output.exists() and output.is_relative_to(ROOT / 'test-data')
    collection = read(source / 'report.json')
    assert sha(source / 'plan.json') == collection['plan_sha256']
    plan = read(source / 'plan.json')
    for name, digest in plan['inputs'].items():
        assert sha(ROOT / name) == digest
    sources = {p:sha(ROOT / p) for p in ('scripts/evaluate_piano_window_continuity.py',
        'manialab/piano_continuity.py', 'manialab/piano.py')}
    save(output / 'plan.json', dict(collection_sha256=sha(source / 'report.json'), sources=sources,
        labels_read_after_all_decodes=True, default_enabled=False,
        require_release_regression=require_release_regression))
    for name in sources:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    manifest = read(ROOT / 'test-data/smd/selected-v1/benchmark/manifest.json')
    prepared = []
    for case in manifest['cases']:
        rows = sorted([r for r in collection['windows'] if r['name'] == case['name']], key=lambda r:r['start_sample'])
        assert [r['start_seconds'] for r in rows] == list(range(0, 65, 5))
        raw = []
        for row in rows:
            assert sha(source / row['path']) == row['sha256']
            with np.load(source / row['path'], allow_pickle=False) as z:
                raw.append({k:z[k] for k in OUTPUTS})
        repaired = [raw[0]]
        trace = []
        for i in range(1, len(raw)):
            fixed, changes = continue_overlap(raw[i-1], raw[i], owned_left=0 if i == 1 else 250,
                require_release_regression=require_release_regression)
            repaired.append(fixed)
            trace.extend([dict(window_start_seconds=rows[i]['start_seconds'], **c) for c in changes])
        predictions = {}
        for name, windows in (('baseline', raw), ('candidate', repaired)):
            activation = {k:np.concatenate([w[k][0 if i==0 else 250:1000 if i==len(windows)-1 else 750]
                                            for i,w in enumerate(windows)])[:7000] for k in OUTPUTS}
            predictions[name] = decode(activation)
            save(output / case['name'] / (name + '.json'), predictions[name])
            np.savez_compressed(output / case['name'] / (name + '-activations.npz'), **activation)
        before, after = predictions['baseline'], predictions['candidate']
        assert len(before) == len(after)
        assert all(all(a[k] == b[k] for k in ('onset', 'pitch', 'velocity', 'onset_score')) for a,b in zip(before,after))
        save(output / case['name'] / 'trace.json', trace)
        changes = [dict(index=i, before=a, after=b) for i,(a,b) in enumerate(zip(before, after)) if a != b]
        save(output / case['name'] / 'changes.json', changes)
        prepared.append((case, predictions, trace, changes))
    result = []
    for case, predictions, trace, changes in prepared:
        scope = lambda events:[n for n in events if 0 <= n['onset'] < case['score_seconds']]
        row = dict(name=case['name'], proposals=len(trace), changed_events=len(changes), scores={}, paired_changes={})
        for semantics, field in (('key','labels'), ('pedal','sustain_labels')):
            assert sha(ROOT / case[field]) == case[field+'_sha256']
            ref = scope(read(ROOT / case[field]))
            row['scores'][semantics] = {variant:{mode:metrics(ref,scope(notes),strict=strict)
                for mode,strict in (('standard',False),('strict',True))} for variant,notes in predictions.items()}
            indices = [j for j,n in enumerate(predictions['baseline']) if 0 <= n['onset'] < case['score_seconds']]
            pairing = {indices[j]:i for i,j in pairs(ref,[predictions['baseline'][j] for j in indices])}
            detailed = []
            for change in changes:
                index = change['index']
                if not 0 <= change['before']['onset'] < case['score_seconds']:
                    continue
                label = ref[pairing[index]] if index in pairing else None
                a = abs(change['before']['offset']-label['offset'])*1000 if label else None
                b = abs(change['after']['offset']-label['offset'])*1000 if label else None
                detailed.append(dict(**change, reference=label, before_error_ms=a, after_error_ms=b,
                    outcome='unmatched_onset' if label is None else 'improved' if b<a else 'worsened' if b>a else 'unchanged'))
            save(output / case['name'] / (semantics+'-changes.json'), detailed)
            row['paired_changes'][semantics] = dict(Counter(r['outcome'] for r in detailed))
        result.append(row)
        print(case['name'], row['proposals'], row['changed_events'], row['paired_changes'], flush=True)
    totals = {}
    for semantics in ('key','pedal'):
        totals[semantics] = {}
        for variant in ('baseline','candidate'):
            totals[semantics][variant] = {}
            for mode in ('standard','strict'):
                totals[semantics][variant][mode] = {}
                for kind in ('onsets','intervals'):
                    values = [r['scores'][semantics][variant][mode][kind] for r in result]
                    totals[semantics][variant][mode][kind] = counts(*(sum(v[k] for v in values)
                        for k in ('matched','reference','predicted')))
    assert sources == {p:sha(ROOT / p) for p in sources}
    save(output / 'report.json', dict(plan_sha256=sha(output/'plan.json'), cases=result, totals=totals,
        all_onsets_pitch_velocity_preserved=True, default_enabled=False, musical_acceptance=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--require-release-regression', action='store_true')
    args = cli.parse_args()
    run(args.source, args.output, args.require_release_regression)
