"""Review current Sakura other attacks against their original Nova waveform.

Replay only: no inference, reference labels, timing fit or candidate changes.
Audio clips keep the original clock, including the intentional chart advance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.audition import render_audition
from scripts.check_expression_current_replay import ANALYSIS, ANALYSIS_SHA, read, save, sha

BASE = ROOT / 'test-data/drum-group-fallback-v3'
BASE_SHA = 'b3c822d35b91f87d9be1f51709c08b5c4dc361d4de01092c496c09961437c7e5'
FULL = ROOT / 'test-data/sakura-tsumugi-fullsong-v1'


def plot(pcm, rate, begin, end, rows, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    first, last = round(begin*rate/1000), round(end*rate/1000)
    wave = pcm[first:last]
    fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 1, 1]})
    axes[0].plot((first+np.arange(len(wave))[::22])/rate, wave[::22], lw=.5)
    axes[0].set_ylabel('Original PCM')
    axes[0].set_title('Nova other stem / current 10K retained attacks', loc='left')
    f, t, z = stft(wave, rate, nperseg=2048, noverlap=2048-128,
                   boundary=None, padded=False)
    valid = (f >= 80) & (f <= 5000)
    power = np.maximum(np.abs(z[valid]), 1e-10)
    db = 20*np.log10(power/max(float(power.max()), 1e-10))
    axes[1].pcolormesh(t+first/rate, f[valid], db, shading='auto',
                       cmap='magma', vmin=-60, vmax=0)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Whole stem: harmonics are not separate instrument evidence', fontsize=10, loc='left')
    for row in rows:
        raw, chart = row['candidate_start_ms']/1000, row['chart_start_ms']/1000
        axes[2].plot([raw, chart], [row['pitch']]*2, color='#94a3b8', lw=1)
    axes[2].scatter([r['candidate_start_ms']/1000 for r in rows], [r['pitch'] for r in rows],
                    marker='|', s=65, label='Raw candidate', color='#0284c7')
    axes[2].scatter([r['chart_start_ms']/1000 for r in rows], [r['pitch'] for r in rows],
                    marker='x', s=20, label='Exported chart (includes 22.5 ms advance)', color='#dc2626')
    axes[2].set_ylabel('MIDI pitch')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[3].scatter([r['candidate_start_ms']/1000 for r in rows],
                    [r['quantization_delta_ms'] for r in rows], s=18)
    axes[3].axhline(0, color='#64748b', lw=.7)
    for value in (-30, 30):
        axes[3].axhline(value, color='#d97706', lw=.7, ls='--')
    axes[3].set_ylabel('Snap delta (ms)')
    axes[3].set_xlabel('Original audio time (s); positive delta means snapping later')
    for axis in axes:
        axis.set_xlim(first/rate, last/rate)
        axis.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def run(output, dependencies):
    if dependencies:
        sys.path.insert(0, str(dependencies.resolve()))
    import numpy as np
    from scipy.io import wavfile
    assert not output.exists(), 'Preserve previous/partial reviews'
    assert sha(BASE/'report.json') == BASE_SHA and sha(ANALYSIS) == ANALYSIS_SHA
    report = read(BASE/'report.json')
    assert sha(BASE/'execution-plan.json') == report['execution_plan_sha256']
    plan = read(BASE/'execution-plan.json')
    case = next(c for c in report['cases'] if c['keys'] == 10)
    path = BASE/'10k/placed.json'
    assert sha(path) == case['artifacts']['placed.json']
    raw = {n['id']: n for n in read(ANALYSIS)['candidates']}
    placed = read(path)
    sources = read(FULL/'execution-plan.json')
    source = next(s for s in sources['sources'] if s['voice'] == 'other')
    audio = Path(source['audio'])
    assert sha(audio) == source['audio_sha256']
    rate, pcm = wavfile.read(audio)
    assert rate == 22050 and pcm.dtype == np.float32 and pcm.ndim == 1
    assert len(pcm) == source['samples'] and np.isfinite(pcm).all()
    clock = plan['generation_clock']
    assert clock == dict(bpm=74., offset=178., division=4, audio_shift_ms=22.5)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in
              (BASE/'report.json', BASE/'execution-plan.json', path, ANALYSIS,
               FULL/'execution-plan.json', audio, ROOT/'manialab/audition.py')}
    rows, per_voice = [], {}
    for voice in sorted({n['voice'] for n in placed}):
        values = [n['start']-n['evidence_start'] for n in placed if n['voice'] == voice]
        per_voice[voice] = dict(retained=len(values), median_ms=float(np.median(values)),
            mean_ms=float(np.mean(values)), delayed_over_30ms=sum(v > 30 for v in values),
            advanced_over_30ms=sum(v < -30 for v in values))
    for note in placed:
        if note['voice'] != 'other':
            continue
        candidate = raw[note['id']]
        assert note['pitch'] == candidate['pitch'] and note['instrument_id'] == candidate['instrument_id']
        rows.append(dict(id=note['id'], pitch=note['pitch'], instrument_id=note['instrument_id'],
            candidate_start_ms=candidate['start'], prequantized_chart_ms=note['evidence_start'],
            chart_start_ms=note['start'], quantization_delta_ms=note['start']-note['evidence_start'],
            chart_minus_audio_ms=note['start']-candidate['start']))
    assert len(rows) == 186 and sum(abs(r['quantization_delta_ms']) > 30 for r in rows) == 65
    bins = defaultdict(list)
    for row in rows:
        bins[row['candidate_start_ms']//8000*8000].append(row)
    ranked = sorted(bins, key=lambda start: (
        -sum(abs(r['quantization_delta_ms']) > 30 for r in bins[start]),
        -sum(abs(r['quantization_delta_ms']) for r in bins[start] if abs(r['quantization_delta_ms']) > 30), start))
    runner_sha = sha(__file__)
    save(output/'plan.json', dict(inputs=inputs, runner_sha256=runner_sha,
        clock=clock, selected_windows_ms=ranked[:3],
        audition=dict(music_volume=.7, click_volume=.3,
            same_sample_and_kind_share_click=True, pre_crop_click_tails_carried=False),
        selection='8s bins from audio zero; descending count >30ms, descending absolute sum of those, ascending start',
        neural_inference=False, source_or_notes_modified=False, reference_chart_read=False))
    shutil.copyfile(__file__, output/'runner-snapshot.py')
    save(output/'events.json', rows)
    clips = []
    for begin in ranked[:3]:
        end = min(begin+8000, len(pcm)*1000/rate)
        # Keep the clicked events even when a move crosses a clip boundary.
        visible = [r for r in rows if any(begin <= r[k] < end for k in
                                         ('candidate_start_ms', 'chart_start_ms'))]
        plot(pcm, rate, begin, end, visible, output/f'{begin}.png')
        clip = dict(begin_ms=begin, end_ms=end, counted_candidates=len(bins[begin]),
            moved_over_30ms=sum(abs(r['quantization_delta_ms']) > 30 for r in bins[begin]),
            classes=dict(Counter(r['instrument_id'].rsplit(':', 1)[-1] for r in bins[begin])), audio={})
        for mode, key in (('source', None), ('raw-heads', 'candidate_start_ms'), ('chart-heads', 'chart_start_ms')):
            events = [] if key is None else [(r[key], 'head') for r in rows]
            buffer, metadata = render_audition(audio, begin, end-begin, events, .7, .3)
            dest = output/f'{begin}-{mode}.wav'
            with dest.open('xb') as stream:
                stream.write(buffer.getvalue())
            checked_rate, checked = wavfile.read(dest)
            assert checked_rate == rate and checked.shape == (round((end-begin)*rate/1000), 2)
            clip['audio'][mode] = metadata | dict(file=dest.name, sha256=sha(dest))
        clips.append(clip)
    assert inputs == {p: sha(ROOT/p) for p in inputs} and sha(__file__) == runner_sha
    save(output/'report.json', dict(per_voice=per_voice, clips=clips,
        plan_sha256=sha(output/'plan.json'), events_sha256=sha(output/'events.json'),
        figures={f'{b}.png': sha(output/f'{b}.png') for b in ranked[:3]},
        neural_inference=False, musical_acceptance=False, listened=False,
        limits=['Selected high-movement windows are diagnostic, not a representative score.',
            'The frozen manual chart clock is not independently verified audio timing.',
            'Snap deltas exclude chart advance; chart-head audio includes that advance.',
            'All model classes share one Nova other stem; spectrum/RMS cannot verify each class.',
            'STFT windows smear attacks; click previews do not evaluate note pitch, duration or LN quality.',
            'Playback gains and integer sample positions are recorded; crop edges can create transients.']))
    print(json.dumps(dict(per_voice=per_voice, windows=ranked[:3]), ensure_ascii=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--plot-dependencies', type=Path)
    args = cli.parse_args()
    run(args.output, args.plot_dependencies)
