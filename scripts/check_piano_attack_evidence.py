"""Inspect four fixed short piano candidates against frozen PCM and activations.

This produces evidence plots, never a note filter or accuracy estimate.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256

CENTERS = (215.881, 216.500)
EVENT_INDICES = (1549, 1458, 1502, 1409)
EXPECTED_AUDIO = '8eb46bbdf283028e5f5d4c13169741e734cc57f0bfd2fe3f19dd590f38ad2722'
EXPECTED_ANALYSIS = 'ef410a5feb747f2503d956c0b2300847b9bcf09acda2ea895785071ba6ba9cc7'
EXPECTED_SEMANTICS = '6b8e3a1e3109b8e39ddd906700e36978820038b3b92f83a468843bd832957d88'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--plot-dependencies', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence output')
    sys.path.insert(0, str(args.plot_dependencies.resolve()))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from scipy.signal import stft

    bench = ROOT/'test-data/sakura-benchmark'
    audio_path = bench/'raw-pcm-audit-v1/piano.wav'
    analysis_path = bench/'piano-full-pipeline-v1/specialist-analysis.json'
    semantics_path = ROOT/'test-data/piano-release-semantics-v1/report.json'
    events_path = semantics_path.parent/'sakura-nova-piano-events.json'
    activation_path = ROOT/'test-data/piano-onnx/portable-v1/sakura-nova-piano-activations.npz'
    for path, expected in ((audio_path, EXPECTED_AUDIO), (analysis_path, EXPECTED_ANALYSIS),
                           (semantics_path, EXPECTED_SEMANTICS)):
        if sha256(path) != expected:
            raise ValueError('Frozen input changed: '+str(path))
    semantics = read(semantics_path)
    row = next(v for v in semantics['cases'] if v['name'] == 'sakura-nova-piano')
    if (sha256(events_path) != semantics['artifacts_sha256'][events_path.name]
            or sha256(activation_path) != row['activations_sha256']):
        raise ValueError('Frozen events/activations changed')
    events = read(events_path)['events']
    old = {n['id']: n for n in read(analysis_path)['candidates'] if n['voice'] == 'piano'}
    if len(events) != 1557 or len(old) != 1557:
        raise ValueError('Unexpected candidate count')
    for i, event in enumerate(events):
        previous = old[f'piano:specialist:{i}']
        assert (round(event['onset']*1000), round(event['offset']*1000), event['pitch']) == (
            previous['start'], previous['end'], previous['pitch'])
    scoped = [i for i, e in enumerate(events)
              if 215.853 <= e['onset'] < 227.853 and e['pitch'] >= 90
              and e['offset']-e['onset'] <= .015]
    if set(scoped) != set(EVENT_INDICES):
        raise ValueError('Four predeclared candidates changed')
    rate, samples = wavfile.read(audio_path)
    if rate != 22050 or samples.dtype != np.float32 or samples.shape != (6258672,):
        raise ValueError('Expected original float32 mono PCM')
    if not np.isfinite(samples).all():
        raise ValueError('Nonfinite PCM')
    with np.load(activation_path, allow_pickle=False) as data:
        onset = data['reg_onset_output'].copy()
        frame = data['frame_output'].copy()
    if onset.shape != frame.shape or onset.shape != (28384, 88):
        raise ValueError('Unexpected 100 FPS activation clock')
    if abs(len(samples)/rate-len(onset)/100) > .011:
        raise ValueError('Activation/PCM duration mismatch')
    if not np.isfinite(onset).all() or not np.isfinite(frame).all():
        raise ValueError('Nonfinite activation')
    args.output.mkdir(parents=True)
    paths = dict(audio=audio_path, analysis=analysis_path, semantics=semantics_path,
                 events=events_path, activations=activation_path, runner=Path(__file__),
                 protocol=ROOT/'docs/piano-attack-evidence-protocol.md')
    identity = {key: sha256(path) for key, path in paths.items()}
    for key in ('runner', 'protocol'):
        shutil.copyfile(paths[key], args.output/(key+'-snapshot'+paths[key].suffix))
    frequency = lambda pitch: 440*2**((pitch-69)/12)
    records, cards = [], []
    for center in CENTERS:
        begin, end = center-.250, center+.450
        left, right = round((begin-.250)*rate), round((end+.250)*rate)
        signal = samples[left:right]
        chosen = [i for i in EVENT_INDICES if begin <= events[i]['onset'] < end]
        neighbors = sorted({i for target in chosen for i, e in enumerate(events)
                            if abs(e['onset']-events[target]['onset']) <= .030
                            and e['pitch'] < events[target]['pitch']} - set(chosen))
        pitches = sorted({events[i]['pitch'] for i in chosen+neighbors})
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True,
                                 gridspec_kw={'height_ratios': [.7, 1.5, 1.5, 1, 1]})
        axes[0].plot(left/rate+np.arange(len(signal))/rate, signal, lw=.4)
        axes[0].set_ylabel('PCM')
        axes[0].set_title(f'Original Nova piano | {begin:.3f}-{end:.3f}s | no snapping or chart shift')
        cache = {}
        for panel, size in zip(axes[1:3], (512, 2048)):
            f, t, z = stft(signal, rate, nperseg=size, noverlap=size-44,
                           window='hann', boundary=None, padded=False)
            t = t+left/rate
            assert abs(t[0]-(left+size/2)/rate) < 1e-9
            assert np.allclose(np.diff(t), 44/rate, rtol=0, atol=1e-12)
            take = (t >= begin) & (t <= end)
            t, z = t[take], z[:, take]
            magnitude = np.abs(z).astype(np.float64)
            cache[f'times_{size}'] = t
            cache[f'frequencies_{size}'] = f
            cache[f'magnitude_{size}'] = magnitude
            valid = f > 0
            db = 20*np.log10(np.maximum(magnitude[valid], 1e-12)/max(float(magnitude.max()), 1e-12))
            panel.pcolormesh(t, 69+12*np.log2(f[valid]/440), db,
                            shading='auto', cmap='magma', vmin=-75, vmax=0)
            panel.set_ylim(min(pitches)-2, max(pitches)+3)
            panel.set_ylabel(f'{size} FFT\nfrequency as MIDI')
            panel.set_title(f'Hann window {size/rate*1000:.2f}ms; hop 2.00ms is not onset resolution', fontsize=9, loc='left')
            for i in chosen:
                e = events[i]
                panel.scatter([e['onset']], [e['pitch']], color='cyan', marker='x', s=38)
        f, t, magnitude = (cache[k+'_2048'] for k in ('frequencies', 'times', 'magnitude'))
        bands, band_bins = {}, {}
        for pitch in pitches:
            fundamental = frequency(pitch)
            half = max(fundamental*(2**(35/1200)-1), rate/2048/2)
            bins = np.flatnonzero(np.abs(f-fundamental) <= half)
            if not len(bins):
                raise ValueError('Empty spectral band')
            band_bins[pitch] = bins
            bands[pitch] = np.sqrt(np.square(magnitude[bins]).sum(axis=0))
            cache[f'band_{pitch}'] = bands[pitch]
            axes[3].plot(t, 20*np.log10(np.maximum(bands[pitch], 1e-12)/max(float(bands[pitch].max()), 1e-12)),
                         label=f'MIDI {pitch}')
        axes[3].set_ylim(-65, 3)
        axes[3].set_ylabel('Band envelope dB\n(each normalized)')
        axes[3].legend(loc='lower right', ncol=4, fontsize=8)
        times = np.arange(len(onset))/100
        visible = (times >= begin) & (times <= end)
        for index in chosen:
            event = events[index]
            pitch = event['pitch']
            line, = axes[4].plot(times[visible], onset[visible, pitch-21], label=f'{pitch} onset')
            axes[4].plot(times[visible], frame[visible, pitch-21], ls='--', color=line.get_color(), label=f'{pitch} frame')
            for panel in axes[1:]:
                panel.axvline(event['onset'], color=line.get_color(), alpha=.25, lw=.8)
            candidates = []
            for lower in neighbors:
                parent = events[lower]
                if parent['pitch'] >= pitch or abs(parent['onset']-event['onset']) > .030:
                    continue
                for harmonic in range(2, 9):
                    cents = 1200*np.log2(frequency(pitch)/(frequency(parent['pitch'])*harmonic))
                    if abs(cents) > 35:
                        continue
                    a, b = bands[pitch], bands[parent['pitch']]
                    correlated = (t >= event['onset']-.15) & (t <= event['onset']+.35)
                    correlation = (float(np.corrcoef(a[correlated], b[correlated])[0, 1])
                                   if np.std(a[correlated]) > 1e-12 and np.std(b[correlated]) > 1e-12 else None)
                    candidates.append(dict(event_index=lower, pitch=parent['pitch'], harmonic=harmonic,
                        frequency_difference_cents=float(cents), onset_difference_ms=(event['onset']-parent['onset'])*1000,
                        envelope_correlation=correlation,
                        fft_bin_overlap=len(set(band_bins[pitch]) & set(band_bins[parent['pitch']]))))
            onset_local = (times >= event['onset']) & (times <= event['onset']+.05)
            records.append(dict(event_index=index, pitch=pitch, onset_seconds=event['onset'],
                output_interval_ms=(event['offset']-event['onset'])*1000,
                old_release_reason=old[f'piano:specialist:{index}']['release_reason'],
                release_reason=event['release_reason'], release_observed=event['release_observed'],
                frame_max_first_50ms=float(frame[onset_local, pitch-21].max()),
                possible_lower_harmonics=candidates, independent_attack_established=False))
        axes[4].set_ylim(0, 1.05)
        axes[4].set_ylabel('Saved model outputs')
        axes[4].set_xlabel('Original audio time / seconds')
        axes[4].legend(loc='upper right', ncol=3, fontsize=8)
        for panel in axes:
            panel.set_xlim(begin, end)
        fig.tight_layout()
        stem = f'attack-{round(center*1000)}'
        fig.savefig(args.output/(stem+'.png'), dpi=130)
        plt.close(fig)
        np.savez_compressed(args.output/(stem+'.npz'), **cache,
                            activation_times=times[visible], onset=onset[visible], frame=frame[visible])
        cards.append(f'<h2>{begin:.3f}–{end:.3f} seconds</h2><img src="{stem}.png" style="width:100%">')
    if {r['event_index'] for r in records} != set(EVENT_INDICES):
        raise ValueError('Missing or duplicated evidence candidates')
    html = '<!doctype html><meta charset="utf-8"><title>Piano attack evidence</title><main style="max-width:1400px;margin:auto;font-family:system-ui">'
    html += '<h1>原钢琴轨：极短高音证据</h1><p>泛音接近和频带相关性不能证明误检。原音符没有修改；这不是直接试听或音乐验收。</p>'
    html += '<p>原12秒回放：<a href="../piano-audio-review-v2/index.html">已有原声、候选合成与双声道对照</a></p>'
    (args.output/'index.html').write_text(html+''.join(cards)+'</main>', encoding='utf-8')
    if any(sha256(path) != identity[key] for key, path in paths.items()):
        raise ValueError('Input changed during analysis')
    report = dict(identity=identity, centers_seconds=CENTERS, stft_windows=[512, 2048], hop=44,
        band_cents=35, events=records, model_rerun=False, notes_changed=False,
        direct_listening_performed=False, application_enabled=False, musical_acceptance=False,
        artifacts_sha256={p.name: sha256(p) for p in args.output.iterdir()})
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['events']), flush=True)


if __name__ == '__main__':
    main()
