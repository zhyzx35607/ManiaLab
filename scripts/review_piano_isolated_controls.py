"""Render every fixed control with original PCM and frozen model evidence.

This is a descriptive review, not a threshold search or a transcription model.
All time axes use the input sample clock. No estimated release is a true label.
"""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_provenance import sha256


def read(path):
    return path.read_text(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-dependencies', type=Path, required=True)
    args = parser.parse_args()
    dataset, run, output = args.dataset.resolve(), args.run.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Preserve old evidence; use a new local test-data folder')
    manifest = json.loads(read(dataset/'manifest.json'))
    report = json.loads(read(run/'report.json'))
    if (sha256(dataset/'manifest.json') != report['identity']['manifest_sha256']
            or report['real_network_inference_completed'] != len(manifest['cases'])
            or len(manifest['cases']) != 83):
        raise ValueError('Incomplete or different fixed run')
    for folder, hashes in ((dataset, manifest['artifacts_sha256']),
                           (run, report['artifacts_sha256'])):
        for name, digest in hashes.items():
            if sha256(folder/name) != digest:
                raise ValueError('Evidence changed: '+str(folder/name))
    sys.path.insert(0, str(args.plot_dependencies.resolve()))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from scipy.signal import stft

    output.mkdir(parents=True)
    shutil.copyfile(__file__, output/'review-snapshot.py')
    rows, cards, links = [], [], []
    for case in manifest['cases']:
        name = case['name']
        result = json.loads(read(run/(name+'-result.json')))
        rate, samples = wavfile.read(dataset/case['file'])
        if rate != 16000 or samples.dtype != np.float32 or samples.shape != (160000,):
            raise ValueError('Unexpected input clock or PCM')
        with np.load(run/(name+'-activations.npz'), allow_pickle=False) as arrays:
            onset, frame = arrays['reg_onset_output'], arrays['frame_output']
        if onset.shape != (1000, 88) or frame.shape != onset.shape:
            raise ValueError('Unexpected activation clock')
        expected, events, score = result['expected'], result['events'], result['score']
        pitches = sorted({e['pitch'] for e in [*expected, *events]})
        # Absolute magnitude scales are shared across all cases, including gain4.
        frequency, times, spectrum = stft(samples.astype(float), fs=rate,
            window='hann', nperseg=2048, noverlap=1888, nfft=2048,
            boundary=None, padded=False)
        decibels = 20*np.log10(np.maximum(np.abs(spectrum), 1e-8))
        figure, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [1, 2, 1, 1, 1.3]})
        block = samples.reshape(-1, 160)
        t = (np.arange(len(block))+.5)*.01
        axes[0].fill_between(t, block.min(axis=1), block.max(axis=1), color='#3355aa')
        axes[0].set(ylim=(-.4, .4), ylabel='PCM min/max')
        axes[1].pcolormesh(times, frequency, decibels, shading='auto',
                          vmin=-100, vmax=-20, cmap='magma')
        axes[1].set(yscale='log', ylim=(40, 7000), ylabel='Hz / -100 to -20 dB')
        activation_times = np.arange(1000)/100
        colors = {pitch: plt.get_cmap('tab10')(index % 10) for index, pitch in enumerate(pitches)}
        for pitch in pitches:
            axes[2].plot(activation_times, onset[:, pitch-21], label=str(pitch), color=colors[pitch])
            axes[3].plot(activation_times, frame[:, pitch-21], label=str(pitch), color=colors[pitch])
        axes[2].axhline(.3, color='gray', ls=':', lw=.7)
        axes[3].axhline(.1, color='gray', ls=':', lw=.7)
        axes[2].set(ylim=(-.03, 1.03), ylabel='Onset')
        axes[3].set(ylim=(-.03, 1.03), ylabel='Frame')
        if pitches:
            axes[2].legend(ncol=min(8, len(pitches)), fontsize=7, loc='upper right')
        extra = set(score['extra_pred_indices'])
        for index, event in enumerate(events):
            axes[4].plot([event['onset'], event['offset']], [event['pitch']]*2,
                         color=colors[event['pitch']], alpha=.7, lw=2)
            axes[4].scatter([event['onset']], [event['pitch']],
                            color='red' if index in extra else colors[event['pitch']],
                            marker='x' if index in extra else 'o', s=28)
        for event in expected:
            axes[4].scatter([event['anchor_seconds']], [event['pitch']],
                            marker='|', color='black', s=180)
            for axis in axes[:4]:
                axis.axvline(event['anchor_seconds'], color='black', ls='--', lw=.7, alpha=.5)
        axes[4].set(ylabel='MIDI pitch', xlabel='Input physical time (seconds)', xlim=(0, 10))
        if pitches:
            axes[4].set(ylim=(min(pitches)-1, max(pitches)+1), yticks=pitches)
        figure.suptitle(f'{name}: expected {score["expected"]}, matched {score["matched"]}, '
                       f'extra {score["extra"]}\nEnvelope anchors = black marks; '
                       'red x = unmatched prediction; decoded tails are not release truth', fontsize=11)
        figure.tight_layout()
        figure.savefig(output/(name+'.png'), dpi=130)
        plt.close(figure)
        shutil.copyfile(dataset/case['file'], output/(name+'.wav'))
        shutil.copyfile(run/(name+'-result.json'), output/(name+'.json'))
        rows.append(dict(name=name, gain=case['gain'], condition=case['condition'],
                         expected=score['expected'], matched=score['matched'], missing=score['missing'],
                         extra=score['extra'], image=name+'.png', audio=name+'.wav', result=name+'.json'))
        safe = html.escape(name)
        links.append(f'<tr><td><a href="#{safe}">{safe}</a></td><td>{score["expected"]}</td>'
                     f'<td>{score["matched"]}</td><td>{score["missing"]}</td><td>{score["extra"]}</td></tr>')
        event_lines = '\n'.join(f'{e["pitch"]:3d} {e["onset"]:.6f} – {e["offset"]:.6f} '
                                f'{e["release_reason"]} onset={e["onset_score"]:.3f}' for e in events)
        cards.append(f'<section id="{safe}"><h2>{safe}</h2>'
                     f'<audio controls preload="none" src="{safe}.wav"></audio> '
                     f'<a href="{safe}.json">完整预测与匹配</a>'
                     f'<img loading="lazy" alt="{safe} 原声与模型对照" src="{safe}.png">'
                     f'<details><summary>全部预测</summary><pre>{html.escape(event_lines) or "无预测"}</pre>'
                     '</details></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>钢琴独立声源对照</title>
<style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:24px;background:#fafafa;color:#18202c}
table{border-collapse:collapse}th,td{padding:7px 14px;border-bottom:1px solid #ddd;text-align:left}
section{margin:32px 0;padding:16px;background:white;border:1px solid #ddd}img{width:100%}
pre{overflow:auto}audio{vertical-align:middle}a{color:#164aa0}</style>
<h1>83组独立钢琴声源对照</h1><p>所有输入都列出，不只挑成功案例。声音包络锚点不是按键真值；
黑色标记表示加入的声源，红叉表示未匹配预测。持续线仅为模型输出，未评价音尾或 LN。</p>
<p>音频保持模型实际输入音量，不单独归一化。波形使用每10ms的最小/最大值；频谱使用128ms Hann窗、
10ms步长与统一绝对色标，窗间存在重叠，不能把步长解释成独立时间分辨率。
激活图仅显示加入或检测到的音高。点击播放可人工核查，但本报告未完成直接试听。</p>
<table><thead><tr><th>输入</th><th>预期</th><th>匹配</th><th>缺失</th><th>额外</th></tr></thead><tbody>
'''+''.join(links)+'</tbody></table>'+''.join(cards)+'</html>'
    (output/'index.html').write_text(page, encoding='utf-8')
    summary = dict(source_manifest_sha256=sha256(dataset/'manifest.json'),
                   source_report_sha256=sha256(run/'report.json'), runner_sha256=sha256(Path(__file__)),
                   matplotlib_version=matplotlib.__version__, cases=rows,
                   direct_listening_performed=False, application_changed=False,
                   artifacts_sha256={p.name: sha256(p) for p in output.iterdir()})
    (output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(cases=len(rows), html=str(output/'index.html'))), flush=True)


if __name__ == '__main__':
    main()
