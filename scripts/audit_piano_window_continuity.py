"""Audit changed frame ownership and cardinality scores without repair helpers."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.piano import OUTPUTS, decode
from scripts.check_expression_current_replay import read, save, sha


def cardinality(reference, predicted, strict, offsets):
    total = 0
    for pitch in {n['pitch'] for n in reference}:
        a = [n for n in reference if n['pitch'] == pitch]
        b = [n for n in predicted if n['pitch'] == pitch]
        if not b:
            continue
        allowed = np.zeros((len(a), len(b)), bool)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                good = abs(x['onset'] - y['onset']) <= (.04 if strict else .05)
                if offsets:
                    good &= abs(x['offset'] - y['offset']) <= (.08 if strict else max(.05, .2 * (x['offset'] - x['onset'])))
                allowed[i, j] = good
        total += int(np.count_nonzero(maximum_bipartite_matching(csr_matrix(allowed), perm_type='column') >= 0))
    return total


def run(raw_root, source, output):
    raw_root, source, output = raw_root.resolve(), source.resolve(), output.resolve()
    assert not output.exists()
    raw_report, report = read(raw_root/'report.json'), read(source/'report.json')
    assert sha(source/'plan.json') == report['plan_sha256']
    plan = read(source/'plan.json')
    assert sha(raw_root/'report.json') == plan['collection_sha256']
    assert sha(raw_root/'plan.json') == raw_report['plan_sha256']
    for name, digest in read(raw_root/'plan.json')['inputs'].items():
        assert sha(ROOT/name) == digest
    for name, digest in plan['sources'].items():
        assert sha(source/'source'/name) == digest
    manifest = read(ROOT/'test-data/smd/selected-v1/benchmark/manifest.json')
    checked_arrays = checked_scores = proposed = 0
    for case in manifest['cases']:
        name = case['name']
        raws = {}
        baseline = {k:np.empty((7000,88),np.float32) for k in OUTPUTS}
        coverage = np.zeros(7000,int)
        for row in [r for r in raw_report['windows'] if r['name']==name]:
            path = raw_root/row['path']
            assert sha(path)==row['sha256']
            start = int(row['start_seconds']*100)
            with np.load(path,allow_pickle=False) as z:
                raw = {k:z[k] for k in OUTPUTS}
            raws[start] = raw
            a,b = row['owned_global_frames']
            coverage[a:b] += 1
            for k in OUTPUTS:
                baseline[k][a:b] = raw[k][a-start:b-start]
        assert np.all(coverage==1)
        expected = {k:v.copy() for k,v in baseline.items()}
        trace = read(source/name/'trace.json')
        for change in trace:
            start = int(change['window_start_seconds']*100)
            old, new = raws[start-500], raws[start]
            pitch = change['pitch']-21
            anchor, fade = change['anchor_frame'],change['outgoing_fade_frame']
            assert (0 if start==500 else 250)<=anchor<500
            assert np.all(old['frame_output'][anchor:fade,pitch]>.1)
            assert old['frame_output'][fade,pitch]<=.1 and fade+4<1000
            assert new['frame_output'][250,pitch]<=.1
            for key, field in (('frame_output','frame_slice'),('reg_offset_output','offset_slice')):
                a,b = change[field]
                expected[key][start+a:start+b,pitch] = old[key][a+500:b+500,pitch]
        proposed += len(trace)
        row = next(r for r in report['cases'] if r['name']==name)
        for variant, values in (('baseline',baseline),('candidate',expected)):
            with np.load(source/name/(variant+'-activations.npz'),allow_pickle=False) as z:
                for k in OUTPUTS:
                    np.testing.assert_array_equal(values[k],z[k])
                    checked_arrays += 1
            predictions = read(source/name/(variant+'.json'))
            assert decode(values)==predictions
            selected = [n for n in predictions if 0<=n['onset']<case['score_seconds']]
            for semantics, field in (('key','labels'),('pedal','sustain_labels')):
                assert sha(ROOT/case[field])==case[field+'_sha256']
                labels = [n for n in read(ROOT/case[field]) if 0<=n['onset']<case['score_seconds']]
                for mode, strict in (('standard',False),('strict',True)):
                    for kind, offsets in (('onsets',False),('intervals',True)):
                        actual = row['scores'][semantics][variant][mode][kind]
                        assert actual['matched']==cardinality(labels,selected,strict,offsets)
                        assert actual['reference']==len(labels) and actual['predicted']==len(selected)
                        checked_scores += 1
    save(output/'report.json',dict(source_report_sha256=sha(source/'report.json'),
        raw_report_sha256=sha(raw_root/'report.json'), case_count=len(manifest['cases']),
        arrays_reconstructed=checked_arrays, cardinality_scores_checked=checked_scores,
        proposals=proposed, neural_inference=False, musical_acceptance=False))
    print(read(output/'report.json'))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--raw',type=Path,required=True)
    cli.add_argument('--source',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    run(args.raw,args.source,args.output)
