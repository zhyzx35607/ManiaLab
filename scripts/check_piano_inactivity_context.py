"""Compare source inactivity context without retraining or enabling it in-app."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import manialab.expression as expression
from manialab.expression_articulation import articulation_policy, attack_runs
from manialab.arrange import parameters, assign_lanes
from manialab.domain import Note, note_from, validate, export_osu, parse_osu
from manialab.piano import OUTPUTS, decode, MODEL_SHA256
from manialab.timing import chart_clock_notes, align_candidates, grid_report
from manialab.audition import render_audition
from scripts.check_articulation_interval_gaps import expression_pair, attack_sets, SMD
from scripts.check_expression_current_replay import ANALYSIS, ANALYSIS_SHA, read, save, sha, endpoints, summary, whole_capacity
from scripts.check_expression_duplicates import coordinates
from scripts.piano_inactivity_context import inactive_intervals, activity_policy, gap_support

BASE = ROOT/'test-data/articulation-interval-gaps-v3'
BASE_SHA = '1fcf7a60dbf820c21a49efc866a56a035821706dc7a6ba609e02eb4d2fd8d279'
PCM = ROOT/'test-data/sakura-benchmark/raw-pcm-audit-v1/piano.wav'


def compare(source, settings, intervals):
    full = [note_from(n) for n in source['candidates'] if n['voice'] == 'piano']
    additions = []
    def policy(local, *, observed_until_ms):
        result, gaps = activity_policy(local, full, intervals, observed_until_ms=observed_until_ms)
        additions.extend(gaps)
        return result
    with patch.object(expression, 'articulation_policy', policy):
        result = expression_pair(source, settings, articulation_policy)
    return result, additions


def measured_coverage(rows, begin, end, endpoint):
    stop, total = begin, 0.
    for left, right in sorted((max(begin, r['onset']*1000), min(end, r[endpoint]*1000))
                              for r in rows if r['onset']*1000 < end and r[endpoint]*1000 > begin):
        total += max(0., right-max(left, stop))
        stop = max(stop, right)
    return total


def plot_gap(pcm, rate, arrays, rows, intervals, gap, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    left, right = gap
    begin, end = max(0, left-1500), min(len(pcm)/rate*1000, right+1500)
    a, b = round(begin*rate/1000), round(end*rate/1000)
    lo, hi = max(0, int(begin//10)), min(len(arrays['frame_output']), int(np.ceil(end/10)))
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
    axes[0].plot((a+np.arange(b-a)[::22])/rate, pcm[a:b:22], lw=.6)
    axes[0].set_ylabel('Original PCM')
    axes[0].set_title('Nova piano / candidate gap and full-source activity (no release edits)', loc='left')
    axes[1].imshow(arrays['frame_output'][lo:hi].T, origin='lower', aspect='auto',
        extent=(lo/100, hi/100, 20.5, 108.5), vmin=0, vmax=1, cmap='magma')
    axes[1].set_ylabel('Frame pitch')
    times = np.arange(lo, hi)/100
    axes[2].plot(times, arrays['frame_output'][lo:hi].max(axis=1), label='Max frame / all 88')
    axes[2].plot(times, arrays['reg_onset_output'][lo:hi].max(axis=1), label='Max onset / all 88')
    axes[2].axhline(.1, lw=.5, ls='--');axes[2].axhline(.3, lw=.5, ls='--')
    axes[2].legend(fontsize=8);axes[2].set_ylabel('Activation')
    for n in rows:
        if n.end >= begin and n.start <= end:
            axes[3].plot([n.start/1000, n.end/1000], [n.pitch]*2,
                color='#f97316' if n.release_reason == 'reattack' else '#0369a1', lw=2)
    axes[3].set_ylabel('Raw candidates');axes[3].set_xlabel('Original audio time (s)')
    for ax in axes:
        ax.axvspan(left/1000, right/1000, color='gray', alpha=.15)
        for x, y in intervals:
            if x < end and y > begin:
                ax.axvspan(max(begin, x)/1000, min(end, y)/1000, color='green', alpha=.13)
        ax.set_xlim(begin/1000, end/1000)
    fig.tight_layout();fig.savefig(path, dpi=120);plt.close(fig)


def run(output, dependencies):
    if dependencies:
        sys.path.insert(0, str(dependencies.resolve()))
    assert not output.exists(), 'Preserve earlier and partial runs'
    assert sha(ANALYSIS) == ANALYSIS_SHA and sha(BASE/'report.json') == BASE_SHA
    baseline = read(BASE/'report.json');frozen = read(BASE/'execution-plan.json')
    assert sha(BASE/'execution-plan.json') == baseline['execution_plan_sha256']
    inputs = {p: sha(p) for p in (ANALYSIS, BASE/'report.json', BASE/'execution-plan.json', PCM)}
    for row in baseline['cases']:
        for name, digest in row['artifacts'].items():
            path = BASE/f"{row['keys']}k"/name
            assert sha(path) == digest;inputs[path] = digest
    caches = list((ROOT/'test-data/sakura-benchmark/piano-full-pipeline-v1').rglob('piano-specialist-activations.npz'))
    assert len(caches) == 1
    cache = caches[0];inputs[cache] = sha(cache)
    rate, pcm = wavfile.read(PCM)
    with np.load(cache, allow_pickle=False) as saved:
        import json
        identity = json.loads(str(saved['identity']))
        arrays = {key: saved[key] for key in OUTPUTS}
    assert pcm.dtype == np.float32 and pcm.ndim == 1
    assert identity['sample_rate'] == rate and identity['samples'] == len(pcm)
    assert identity['model_sha256'] == MODEL_SHA256
    assert identity['pcm_sha256'] == hashlib.sha256(np.asarray(pcm, dtype='<f4').tobytes()).hexdigest()
    analysis = read(ANALYSIS);raw = {n['id']: note_from(n) for n in analysis['candidates'] if n['voice'] == 'piano'}
    events = decode(arrays);reclassified = 0
    assert len(raw) == len(events)
    for index, e in enumerate(events):
        n = raw[f'piano:specialist:{index}']
        assert (n.start, n.end, n.pitch, n.strength) == (round(e['onset']*1000), round(e['offset']*1000), e['pitch'], e['onset_score'])
        if n.release_reason != e['release_reason']:
            assert (n.release_reason, e['release_reason']) == ('frame_decay', 'frame_inactive');reclassified += 1
    assert reclassified == 89
    manifest = read(SMD/'manifest.json');smd_report = read(SMD/'specialist-v2/report.json')
    assert sha(SMD/'manifest.json') == smd_report['identity']['manifest_sha256']
    for p in (SMD/'manifest.json', SMD/'specialist-v2/report.json'):
        inputs[p] = sha(p)
    for case in manifest['cases']:
        name = case['name'];expected = next(c for c in smd_report['cases'] if c['name'] == name)
        for suffix in ('notes.json', 'activations.npz'):
            p = SMD/'specialist-v2'/f'{name}-{suffix}';inputs[p] = sha(p)
            if suffix == 'notes.json':assert sha(p) == expected['notes_sha256']
    sources = ('scripts/piano_inactivity_context.py', 'scripts/check_piano_inactivity_context.py',
        'scripts/check_articulation_interval_gaps.py', 'scripts/check_expression_current_replay.py',
        'scripts/check_expression_duplicates.py', 'manialab/piano.py', 'manialab/domain.py',
        'manialab/expression.py', 'manialab/expression_articulation.py', 'manialab/expression_activity.py',
        'manialab/arrange.py', 'manialab/layout.py', 'manialab/timing.py', 'manialab/audition.py')
    source_hashes = {s: sha(ROOT/s) for s in sources}
    save(output/'execution-plan.json', dict(sources=source_hashes,
        inputs={p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()},
        generation_clock=frozen['generation_clock'], settings=frozen['settings'],
        frame_threshold=.1, onset_threshold=.3, minimum_interval_ms=80,
        label_access='Only after every expression replay and export completes',
        neural_inference=False, reference_chart_read=False))
    for s in sources:
        dest = output/'source'/s;dest.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/s, dest)
    intervals = inactive_intervals(arrays['frame_output'], arrays['reg_onset_output'])
    save(output/'piano-inactive-intervals.json', intervals)
    rows, clock, review_selected = [], frozen['generation_clock'], {}
    for keys in range(6, 11):
        settings = parameters(frozen['settings'] | dict(keys=keys));folder = output/f'{keys}k'
        (plan, old, new, metadata, old_trace, trace, changes), gaps = compare(analysis, settings, intervals)
        assert plan == read(BASE/f'{keys}k/plan.json')
        old_aligned, _ = align_candidates(chart_clock_notes(old, clock['audio_shift_ms']), clock, confirmed=True)
        assert [n.data() for n in old_aligned] == read(BASE/f'{keys}k/candidates.json')
        # The unchanged selector/layout were already rerun in the immediately preceding baseline.
        previous = [note_from(n) for n in read(BASE/f'{keys}k/selected.json')]
        prior_selection = read(BASE/f'{keys}k/selection.json')
        old_placed = [note_from(n) for n in read(BASE/f'{keys}k/placed.json')]
        assert old_trace == read(BASE/f'{keys}k/expression_trace.json')
        aligned, _ = align_candidates(chart_clock_notes(new, clock['audio_shift_ms']), clock, confirmed=True)
        original_aligned = [n.data() for n in aligned];expression_trace = copy.deepcopy(trace)
        selected, selection = expression.select_expression(aligned, metadata, trace, settings, 60000/clock['bpm'])
        assert [n.data() for n in aligned] == original_aligned
        lookup = {n.id: n for n in aligned};assert all(n == lookup[n.id] for n in selected)
        placed = assign_lanes(selected, settings, beat_ms=60000/clock['bpm'])
        assert endpoints(placed) == endpoints(selected)
        capacity, structural, grid = whole_capacity(selected, settings), validate(placed, keys), grid_report(placed, clock)
        assert capacity['feasible'] and structural['valid'] and grid['off_grid'] == 0
        exported = export_osu(placed, keys, ROOT/'vendor/mapperatorinator/template.osu', title='Piano activity context',
            audio='audio.mp3', bpm=clock['bpm'], offset=clock['offset'], version=f'{keys}K activity research',
            grid_division=clock['division'], bookmarks=analysis['structure']['boundaries'][1:-1])
        parsed = parse_osu(exported);assert coordinates(parsed['notes']) == coordinates(placed) and parsed['keys'] == keys
        before, after = attack_sets(previous, metadata), attack_sets(selected, metadata)
        data = dict(plan=plan, changes=changes, gaps=gaps, expression_trace=expression_trace,
            candidates=original_aligned, selected=[n.data() for n in selected], selection=selection, placed=[n.data() for n in placed])
        for name, value in data.items():save(folder/f'{name}.json', value)
        with (folder/'chart.osu').open('x', encoding='utf-8') as f:f.write(exported)
        rows.append(dict(keys=keys, old=summary(previous, prior_selection), new=summary(selected, selection),
            expression_changes=len(changes), added_boundaries=len(gaps),
            removed_attacks=dict(Counter(v for _, v, _ in before-after)), added_attacks=dict(Counter(v for _, v, _ in after-before)),
            capacity=capacity, structural=structural, grid=grid,
            artifacts={name: sha(folder/name) for name in [*(f'{n}.json' for n in data), 'chart.osu']}))
        if keys == 10:review_selected = dict(old=old_placed, new=placed)
        print(f'{keys}K: {len(gaps)} new boundaries; {len(changes)} expressions; {len(selected)} objects / {sum(n.hold for n in selected)} LNs', flush=True)
    smd = []
    for case in manifest['cases']:
        name = case['name']
        with np.load(SMD/'specialist-v2'/f'{name}-activations.npz', allow_pickle=False) as data:
            pred = {key: data[key] for key in OUTPUTS}
        events = decode(pred);archived = read(SMD/'specialist-v2'/f'{name}-notes.json')
        signature = lambda n: (n['onset'], n['offset'], n['pitch'])
        assert sorted(map(signature, events)) == sorted(map(signature, archived))
        notes = [Note(round(n['onset']*1000), round(n['offset']*1000), pitch=n['pitch'],
            strength=n['onset_score'], voice='piano', source='pitch', id=f'p:{i}', release_reason=n['release_reason']) for i, n in enumerate(events)]
        source = dict(duration=case['duration_seconds'], candidates=[n.data() for n in notes],
            structure=dict(segments=[dict(start_ms=0, end_ms=round(case['duration_seconds']*1000))]))
        quiet = inactive_intervals(pred['frame_output'], pred['reg_onset_output'])
        (_, old, new, _, _, trace, changes), gaps = compare(source, parameters(), quiet)
        assert {n.id for n in new} == {n.id for n in notes if n.strength >= .35}
        data = dict(candidates=source['candidates'], old=[n.data() for n in old], new=[n.data() for n in new],
            trace=trace, changes=changes, gaps=gaps, inactive_intervals=quiet)
        for field, value in data.items():save(output/'smd'/name/f'{field}.json', value)
        stop = case['score_seconds']*1000
        smd.append(dict(name=name, candidates=len(notes), strong_candidates=len(new),
            boundaries=len(gaps), scored_boundaries=sum(g['before_attack_ms'] < stop for g in gaps),
            changed=len(changes), scored_changes=sum(c['start'] < stop for c in changes),
            old_ln=sum(n.hold and n.start < stop for n in old), new_ln=sum(n.hold and n.start < stop for n in new),
            artifacts={f'{field}.json': sha(output/'smd'/name/f'{field}.json') for field in data}))
        print(f'SMD {name}: {len(gaps)} new boundaries; {len(changes)} expressions', flush=True)
    # Every candidate/expression decision above is now frozen; only here read note labels.
    audits = []
    for case in manifest['cases']:
        name = case['name'];path = ROOT/case['sustain_labels']
        assert sha(path) == case['sustain_labels_sha256'];inputs[path] = sha(path)
        labels = read(path);local = []
        for gap in read(output/'smd'/name/'gaps.json'):
            if gap['before_attack_ms'] >= case['score_seconds']*1000:continue
            a, b = gap['inactive_start_ms'], gap['inactive_end_ms']
            local.append(gap | dict(duration_ms=b-a,
                pedal_activity_ms=measured_coverage(labels, a, b, 'offset'),
                key_activity_ms=measured_coverage(labels, a, b, 'key_offset')))
        audits.append(dict(name=name, label_sha256=sha(path), gaps=local))
    save(output/'reference-activity-audit.json', audits)
    # Review all six pre-existing candidate gaps, including the rejected active one.
    strong = [n for n in raw.values() if n.strength >= .35];runs, _ = attack_runs(strong)
    candidates = []
    for groups in runs:
        last_end = 0
        for group, following in zip(groups, groups[1:]):
            last_end = max(last_end, max(n.end for n in group));next_start = following[0].start
            if next_start-last_end >= 80:candidates.append((last_end, next_start))
    reviews = []
    for index, (left, right) in enumerate(candidates):
        folder = output/'review'/str(index);folder.mkdir(parents=True)
        plot_gap(pcm, rate, arrays, list(raw.values()), intervals, (left, right), folder/'evidence.png')
        begin, end = max(0, left-1500), min(len(pcm)/rate*1000, right+1500)
        files = ['evidence.png'];audio = {}
        for mode in ('original', 'old', 'new'):
            clicks = [] if mode == 'original' else [(t, kind) for n in review_selected[mode] if n.voice == 'piano'
                for t, kind in [(n.start, 'head')]+([(n.end, 'tail')] if n.hold else [])]
            stream, metadata = render_audition(PCM, begin, end-begin, clicks)
            name = mode+'.wav';(folder/name).write_bytes(stream.getvalue())
            decoded_rate, decoded = wavfile.read(folder/name)
            assert decoded_rate == metadata['sample_rate'] and len(decoded) == metadata['samples']
            files.append(name);audio[mode] = metadata
        rms = lambda a, b: float(np.sqrt(np.mean(np.asarray(pcm[round(a*rate/1000):round(b*rate/1000)], np.float64)**2)))
        reviews.append(dict(candidate_gap=[left, right], model_low_interval=gap_support(intervals, left, right),
            gap_rms=rms(left, right), preceding_second_rms=rms(left-1000, left), audio=audio,
            artifacts={name: sha(folder/name) for name in files}))
    save(output/'review.json', reviews)
    assert all(sha(p) == digest for p, digest in inputs.items())
    assert source_hashes == {s: sha(ROOT/s) for s in sources}
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
    save(output/'report.json', dict(execution_plan_sha256=sha(output/'execution-plan.json'),
        cases=rows, smd=smd, reference_audit_sha256=sha(output/'reference-activity-audit.json'),
        review_sha256=sha(output/'review.json'), neural_inference=False, source_pcm_verified=True,
        original_notes_unchanged=True, historical_release_metadata_differences=89,
        default_enabled=False, musical_acceptance=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--plot-dependencies', type=Path)
    args = cli.parse_args()
    run(args.output, args.plot_dependencies)
