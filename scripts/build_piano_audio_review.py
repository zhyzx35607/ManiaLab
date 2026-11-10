"""Build source/synthesis audio and spectrogram checks from frozen piano evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256

CLIPS = ((6664, 18664), (215853, 227853), (241798, 253798))
UNOBSERVED = {'duration_limit', 'input_boundary', 'forced_segment_boundary', 'unresolved_or_postprocessed'}


def synthesize(notes, begin, count, rate, np):
    """No invented/retriggered attacks at a crop boundary; no added release tail."""
    result = np.zeros(count, dtype=np.float64)
    clock = begin/1000 + np.arange(count)/rate
    for note in notes:
        onset, offset = note['start']/1000, note['end']/1000
        selected = (clock >= onset) & (clock < offset)
        if not selected.any():
            continue
        t = clock[selected] - onset
        duration = offset - onset
        envelope = np.minimum(t/.005, 1) * np.minimum((duration-t)/.02, 1) * np.exp(-t/1.8)
        frequency = 440 * 2**((note['pitch']-69)/12)
        wave = sum(weight*np.sin(2*np.pi*frequency*harmonic*t)
                   for harmonic, weight in ((1, 1), (2, .35), (3, .15)) if frequency*harmonic < rate/2)
        result[selected] += max(.1, min(1., note['strength'])) * envelope * wave
    return result


def figure(signal, notes, rate, begin, end, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.signal import stft
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    low_pitch = min(33, min((note['pitch']-2 for note in notes), default=33))
    high_pitch = max(98, max((note['pitch']+2 for note in notes), default=98))
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={'height_ratios': [.8, 2.5, 2]})
    times = begin/1000 + np.arange(len(signal))/rate
    axes[0].plot(times[::22], signal[::22], lw=.5, color='#2563eb')
    axes[0].set_ylabel('PCM amplitude')
    axes[0].set_title(f'Original Nova piano / frozen raw candidates | {begin/1000:.3f}-{end/1000:.3f} s', loc='left')
    frequency, time, spectrum = stft(signal, rate, window='hann', nperseg=4096,
        noverlap=4096-128, boundary=None, padded=False)
    valid = frequency > 0
    pitches = 69+12*np.log2(frequency[valid]/440)
    magnitude = np.maximum(np.abs(spectrum[valid]), 1e-9)
    db = 20*np.log10(magnitude/max(1e-9, float(magnitude.max())))
    axes[1].pcolormesh(time+begin/1000, pitches, db, shading='auto', cmap='magma', vmin=-65, vmax=0)
    axes[1].set_ylabel('Spectrum: MIDI-equivalent frequency')
    axes[1].set_ylim(low_pitch, high_pitch)
    axes[1].set_title('Audio evidence: bright harmonics are not automatically separate notes', loc='left', fontsize=10)
    for note in notes:
        left, right = max(begin, note['start'])/1000, min(end, note['end'])/1000
        uncertain = note.get('release_reason') in UNOBSERVED
        color = '#dc2626' if uncertain else '#0891b2'
        axes[2].add_patch(Rectangle((left, note['pitch']-.35), right-left, .7,
            facecolor=color, edgecolor=color, alpha=.65))
        if begin <= note['start'] < end:
            axes[2].plot(note['start']/1000, note['pitch'], '|', color='#111827', ms=6)
    axes[2].set_ylim(low_pitch, high_pitch)
    axes[2].set_xlim(begin/1000, end/1000)
    axes[2].set_yticks(range((low_pitch//12+1)*12, high_pitch, 12))
    axes[2].set_ylabel('Detected MIDI pitch')
    axes[2].set_xlabel('Original audio time (seconds); no chart shift or snapping')
    axes[2].set_title('Raw intervals: cyan = detected endpoint; red = unobserved endpoint (not an LN release)', loc='left', fontsize=10)
    axes[2].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--plot-dependencies', type=Path, help='Optional isolated plotting packages')
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous audio review artifacts')
    if args.plot_dependencies:
        sys.path.insert(0, str(args.plot_dependencies.resolve()))
    import numpy as np
    import matplotlib
    from scipy.io import wavfile
    root = ROOT/'test-data/sakura-benchmark'
    source_path = root/'raw-pcm-audit-v1/piano.wav'
    analysis_path = root/'piano-full-pipeline-v1/specialist-analysis.json'
    audit_path = root/'raw-pcm-audit-v1/report.json'
    frozen_path = ROOT/'test-data/piano-svsep/probe-v3/report.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    expected = next(row['raw_float_sha256'] for row in audit['rows'] if row['role']=='piano')
    if sha256(source_path) != expected or sha256(analysis_path) != frozen['identity']['analysis_sha256']:
        raise ValueError('Frozen audio/analysis identity changed')
    analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
    notes = [note for note in analysis['candidates'] if note['voice']=='piano']
    if len(notes) != 1557 or len({note['id'] for note in notes}) != 1557:
        raise ValueError('Expected the same 1557 frozen piano candidates')
    rate, samples = wavfile.read(source_path)
    if rate != 22050 or samples.dtype != np.float32 or samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError('Expected original float mono PCM at 22050 Hz')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output/'runner-snapshot.py')
    protocol = ROOT/'docs/piano-audio-review-protocol.md'
    shutil.copyfile(protocol, args.output/'protocol-snapshot.md')
    summaries, cards = [], []
    for begin, end in CLIPS:
        name = f'clip-{begin}'
        left, right = round(begin*rate/1000), round(end*rate/1000)
        signal = samples[left:right].astype(np.float64)
        chosen = [note for note in notes if note['start'] < end and note['end'] > begin]
        synthetic = synthesize(chosen, left/rate*1000, len(signal), rate, np)
        rms = lambda value: float(np.sqrt(np.mean(value**2)))
        synth_gain = rms(signal)/max(rms(synthetic), 1e-12)
        synthetic *= synth_gain
        common_gain = min(1., .95/max(float(np.abs(signal).max()), float(np.abs(synthetic).max()), 1e-12))
        stereo = np.stack([signal, synthetic], axis=1)*common_gain
        for mode, data in [('source', signal*common_gain), ('synthesis', synthetic*common_gain), ('stereo', stereo)]:
            target = args.output/f'{name}-{mode}.wav'
            wavfile.write(target, rate, np.rint(data*32767).astype(np.int16))
            check_rate, actual = wavfile.read(target)
            assert check_rate == rate and actual.shape == data.shape and len(actual) == right-left
            assert np.max(np.abs(actual.astype(np.float64)/32767-data)) <= 1/32767
        figure(signal, chosen, rate, left/rate*1000, right/rate*1000, args.output/f'{name}.png')
        row = dict(name=name, begin_ms=begin, end_ms=end, first_sample=left, stop_sample=right,
            sample_rate=rate, samples=len(signal), candidates_overlapping=len(chosen),
            attacks_in_clip=sum(begin <= n['start'] < end for n in chosen),
            carried_from_before=sum(n['start'] < begin for n in chosen),
            continue_after_clip=sum(n['end'] > end for n in chosen),
            unobserved_ends=sum(n.get('release_reason') in UNOBSERVED for n in chosen),
            release_reasons=dict(Counter(n.get('release_reason') for n in chosen)),
            synthesis_rms_gain=synth_gain, common_playback_gain=common_gain,
            source_rms=rms(signal), synthesis_rms_before_common_gain=rms(synthetic))
        summaries.append(row)
        (args.output/f'{name}-notes.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding='utf-8')
        cards.append(f'''<section><h2>{begin/1000:.3f}–{end/1000:.3f} 秒</h2>
<p>{row['attacks_in_clip']} 个检测起音；{row['carried_from_before']} 个音从片段前持续进入。</p>
<label>回放 <select data-name="{name}"><option value="source">原Nova钢琴轨</option><option value="synthesis">候选音符合成</option><option value="stereo">左原轨 / 右合成</option></select></label>
<audio controls preload="metadata" src="{name}-source.wav"></audio>
<img src="{name}.png" alt="波形、频谱及原始候选音符区间"><p class="muted">合成音是简易谐波音色。红色音尾未被模型观察到，不能当作LN放手。</p></section>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>樱花草 · 钢琴采音对照</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;color:#192536;background:#f3f6fa}section{background:white;padding:20px;border-radius:12px;margin:24px 0}h1{font-size:28px}h2{font-size:21px}audio,img{display:block;width:100%;margin-top:14px}select{font:inherit;padding:6px}.muted{color:#546577}</style>
<h1>樱花草 · 原钢琴轨与采音结果</h1><p>三个固定片段，共用原始音频时钟。切换回放保留当前位置；一次只播放一个片段。此页不需要osu!。</p>
<p class="muted">这轮重放已有1557个候选，没有重新推理、对齐人工谱或修改音符。频谱可辅助核查，但谐波不等于独立音符；合成回放不等于真实钢琴音色。人工试听尚未完成。</p>''' + '\n'.join(cards) + '''
<script>document.querySelectorAll('select').forEach(s=>s.addEventListener('change',()=>{const a=s.closest('section').querySelector('audio'), t=a.currentTime, playing=!a.paused;a.pause();a.src=s.dataset.name+'-'+s.value+'.wav';a.addEventListener('loadedmetadata',()=>{a.currentTime=Math.min(t,a.duration||0);if(playing)a.play().catch(()=>{});},{once:true});a.load();}));document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause();});}));</script></html>'''
    (args.output/'index.html').write_text(document, encoding='utf-8')
    assert sha256(source_path) == expected and sha256(analysis_path) == frozen['identity']['analysis_sha256']
    result = dict(identity=dict(runner_sha256=sha256(__file__), protocol_sha256=sha256(protocol),
        source_audio_sha256=expected, analysis_sha256=sha256(analysis_path),
        audit_sha256=sha256(audit_path), frozen_report_sha256=sha256(frozen_path),
        matplotlib=matplotlib.__version__, numpy=np.__version__), clips=summaries,
        reused_candidates=1557, model_rerun=False, source_candidates_changed=False,
        reference_chart_read=False, chart_shift_ms=0, grid_snap=False,
        model_listened_to_audio=False, human_listening_completed=False, musical_acceptance=False)
    result['artifacts_sha256'] = {p.name:sha256(p) for p in args.output.iterdir()}
    (args.output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(clips=summaries, report_sha256=sha256(args.output/'report.json'))), flush=True)


if __name__ == '__main__':
    main()
