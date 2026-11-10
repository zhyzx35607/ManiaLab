"""Plot both arrangements on the unchanged Nova other waveform and clock."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.audition import render_audition
from scripts.check_expression_current_replay import ANALYSIS, ANALYSIS_SHA, read, save, sha
from scripts.check_expression_coherence import BASE, BASE_SHA, WINDOWS


def run(study, output, dependencies):
    study, output = study.resolve(), output.resolve()
    if dependencies:
        sys.path.insert(0, str(dependencies.resolve()))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.io import wavfile
    from scipy.signal import stft

    assert not output.exists(), 'Preserve previous and partial reviews'
    assert sha(BASE/'report.json') == BASE_SHA and sha(ANALYSIS) == ANALYSIS_SHA
    reports = [read(folder/'report.json') for folder in (BASE, study)]
    inputs = {ANALYSIS: ANALYSIS_SHA, Path(__file__): sha(__file__),
              ROOT/'manialab/audition.py': sha(ROOT/'manialab/audition.py')}
    notes = []
    for folder, report in zip((BASE, study), reports):
        plan = folder/'execution-plan.json'
        assert sha(plan) == report['execution_plan_sha256']
        case = next(c for c in report['cases'] if c['keys'] == 10)
        path = folder/'10k/placed.json'
        assert sha(path) == case['artifacts']['placed.json']
        inputs.update({folder/'report.json': sha(folder/'report.json'), plan: sha(plan), path: sha(path)})
        notes.append([n for n in read(path) if n['voice'] == 'other'])
    full = ROOT/'test-data/sakura-tsumugi-fullsong-v1/execution-plan.json'
    source = next(s for s in read(full)['sources'] if s['voice'] == 'other')
    audio = Path(source['audio'])
    assert sha(audio) == source['audio_sha256']
    inputs.update({full: sha(full), audio: source['audio_sha256']})
    rate, pcm = wavfile.read(audio)
    assert rate == 22050 and pcm.dtype == np.float32 and pcm.ndim == 1
    assert len(pcm) == source['samples'] and np.isfinite(pcm).all()
    raw = [n for n in read(ANALYSIS)['candidates'] if n['voice'] == 'other']
    changed = {n['id'] for n in notes[0]} ^ {n['id'] for n in notes[1]}
    changed_bins = sorted({n['start']//8000*8000 for n in raw if n['id'] in changed})
    windows = list(dict.fromkeys([*WINDOWS, *((begin, min(begin+8000, round(len(pcm)*1000/rate)))
                                             for begin in changed_bins)]))
    identities = sorted({n['instrument_id'] for n in raw})
    colors = {identity: plt.get_cmap('tab20')(i) for i, identity in enumerate(identities)}
    save(output/'plan.json', dict(inputs={str(path.relative_to(ROOT)): value for path, value in inputs.items()},
        windows_ms=windows, original_windows_ms=WINDOWS,
        added_window_rule='Every 8-second bin containing a changed other candidate start, without ranking.',
        changed_candidate_ids=sorted(changed), voices=['other'], chart_time_shift_ms=0,
        music_volume=.7, click_volume=.3, neural_inference=False, listened=False,
        warning='One residual stem; colored classes are predictions, not isolated instruments.'))
    shutil.copyfile(__file__, output/'runner.py')
    clips = []
    for begin, end in windows:
        first, last = round(begin*rate/1000), round(end*rate/1000)
        wave = pcm[first:last]
        fig, axes = plt.subplots(5, 1, figsize=(16, 11), sharex=True,
            gridspec_kw={'height_ratios': [1, 2, 2, 2, 2]})
        axes[0].plot((first+np.arange(len(wave))[::22])/rate, wave[::22], lw=.5)
        axes[0].set_ylabel('Original PCM')
        axes[0].set_title('Fixed Nova other stem / chord-evidence baseline vs group-coherence experiment', loc='left')
        f, t, z = stft(wave, rate, nperseg=4096, noverlap=4096-128, boundary=None, padded=False)
        valid = (f >= 80) & (f <= 5000)
        power = np.maximum(np.abs(z[valid]), 1e-10)
        axes[1].pcolormesh(t+first/rate, f[valid], 20*np.log10(power/max(float(power.max()), 1e-10)),
            shading='auto', cmap='magma', vmin=-60, vmax=0)
        axes[1].set_yscale('log'); axes[1].set_ylabel('Hz / whole stem')
        visible_classes = set()
        for axis, row, label in zip(axes[2:], (raw, *notes), ('Raw detected intervals', 'Baseline chart', 'Coherence chart')):
            visible = [n for n in row if n['start'] < end and n['end'] >= begin]
            for note in visible:
                color = colors[note['instrument_id']]
                visible_classes.add(note['instrument_id'])
                axis.plot([note['start']/1000, note['end']/1000], [note['pitch']]*2, c=color, lw=2, alpha=.8)
                axis.scatter([note['start']/1000], [note['pitch']], c=[color], s=14, marker='o' if note['end']>note['start'] else 'x')
            axis.set_ylim(25, 110); axis.set_ylabel(label+'\nMIDI pitch')
        for axis in axes:
            axis.set_xlim(begin/1000, end/1000); axis.grid(alpha=.15)
        axes[-1].set_xlabel('Seconds on original playback clock; chart advance is NOT undone')
        handles = [Line2D([0], [0], color=colors[key], label=key.rsplit(':', 1)[-1]) for key in sorted(visible_classes)]
        fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, 1))
        figure = output/f'{begin}.png'
        fig.savefig(figure, dpi=120); plt.close(fig)
        clip = dict(start_ms=begin, end_ms=end, figure=figure.name, figure_sha256=sha(figure), audio={})
        for name, events in [('source', []), *[(name, [(n[k], kind) for n in row
                for k, kind in [('start', 'head'), ('end', 'tail')] if k == 'start' or n['end']>n['start']])
                for name, row in zip(('baseline', 'coherence'), notes)]]:
            data, metadata = render_audition(audio, begin, end-begin, events, .7, .3)
            path = output/f'{begin}-{name}.wav'
            with path.open('xb') as stream:
                stream.write(data.getvalue())
            checked_rate, checked = wavfile.read(path)
            assert checked_rate == rate and checked.shape == (round((end-begin)*rate/1000), 2)
            clip['audio'][name] = metadata | dict(file=path.name, sha256=sha(path))
        clips.append(clip)
    assert inputs == {path: sha(path) for path in inputs}
    save(output/'report.json', dict(plan_sha256=sha(output/'plan.json'), clips=clips,
        neural_inference=False, listened=False, musical_acceptance=False,
        limits=['STFT smears attacks and harmonics do not verify independent instruments.',
            'Click previews encode selected heads/releases, not pitch or playing quality.',
            'The original manual chart clock has not been independently validated.']))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--study', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--plot-dependencies', type=Path)
    args = cli.parse_args()
    run(args.study, args.output, args.plot_dependencies)
