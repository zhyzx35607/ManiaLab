"""Render fixed early errors against original audio, without listening claims."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.evaluate_adt_drums import read, save, CLASSES


def first_errors(scores, kinds):
    choices = []
    for kind in kinds:
        row = scores[kind]
        for label, indices, field in [('missing', row['unmatched_reference'], 'reference_events'),
                                      ('extra', row['unmatched_predicted'], 'predicted_events')]:
            if indices:
                event = min((row[field][i] for i in indices), key=lambda e: e['time_ms'])
                choices.append(dict(label=kind+'-'+label, anchor=event['time_ms']/1000))
    return choices


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--evaluation', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    evaluation, output = args.evaluation.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use new local review directory')
    report = read(evaluation/'report.json')
    for name, digest in report['artifacts_sha256'].items():
        if sha256(evaluation/name) != digest:
            raise ValueError('Evaluation changed')
    source = Path(report['source_inference'])
    identity = read(source/'identity.json')
    if sha256(source/'identity.json') != report['inference_identity_sha256']:
        raise ValueError('Inference identity changed')
    sys.path.insert(0, str(ROOT/'test-data/plotlibs'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from scipy.signal import stft
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output/'review-snapshot.py')
    cards, examples = [], []
    for summary in read(evaluation/'progress-snapshot.json'):
        name = summary['id']
        result = read(evaluation/(name+'.json'))
        source_row = next(r for r in identity['inputs'] if r['id'] == name)
        path = Path(source_row['audio'])
        if sha256(path) != source_row['audio_sha256']:
            raise ValueError('Original audio changed')
        rate, raw = wavfile.read(path)
        if raw.dtype.kind == 'i':
            raw = raw.astype(np.float64)/(2**(raw.dtype.itemsize*8-1))
        elif raw.dtype.kind == 'u':
            raw = (raw.astype(np.float64)-128)/128
        raw = raw.astype(np.float32)
        mono = raw.mean(axis=1) if raw.ndim == 2 else raw
        if summary['dataset'] == 'Nova':
            selected = result['scores']['audited']['all']['raw']
            official = result['scores']['official']['all']['raw']
            choices = first_errors(selected, ['hi-hat'])
        else:
            selected = result['scores']['audited']['40']
            official = result['scores']['official']['40']
            choices = first_errors(selected, CLASSES)
        windows = read(source/name/'windows.json')
        for label, condition in [('first-duplicate', lambda w: any(i['kind'] == 'duplicate_onset_class' for i in w['audit']['issues'])),
                                  ('first-unterminated-window', lambda w: not w['audit']['terminated'])]:
            match = next((w for w in windows if condition(w)), None)
            if match:
                choices.append(dict(label=label, anchor=match['start_sample']/24000,
                                    window_index=match['index']))
        references = [e for k in CLASSES for e in selected[k]['reference_events']]
        strict = [e for k in CLASSES for e in selected[k]['predicted_events']]
        compatible = [e for k in CLASSES for e in official[k]['predicted_events']]
        for item in choices:
            anchor = item['anchor']
            start = max(0., anchor-1.)
            end = min(len(mono)/rate, anchor+3.)
            if end <= start:
                continue
            a, b = int(start*rate), int(end*rate)
            start, end = a/rate, b/rate
            clip = mono[a:b]
            stem = name+'-'+item['label']
            wavfile.write(output/(stem+'.wav'), rate, raw[a:b])
            fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 2, 1.3, 1.3, 1.3]})
            block = max(1, rate//1000)
            samples = clip[:len(clip)//block*block].reshape(-1, block)
            t = start+(np.arange(len(samples))+.5)*block/rate
            axes[0].fill_between(t, samples.min(axis=1), samples.max(axis=1), color='#3455a5')
            axes[0].set(ylabel='Original PCM')
            f, t, z = stft(clip.astype(float), fs=rate, nperseg=1024,
                           noverlap=1024-max(1, rate//200), boundary=None, padded=False)
            axes[1].pcolormesh(t+start, f, 20*np.log10(np.maximum(abs(z), 1e-8)),
                               shading='auto', cmap='magma', vmin=-100, vmax=-20)
            axes[1].set(yscale='log', ylim=(30, min(15000, rate/2)), ylabel='Hz / dB')
            for ax, events, label, color in zip(axes[2:], [references, compatible, strict],
                    ['Author selection' if name == 'sakura' else 'Physical annotations',
                     'Official decode + unique', 'Strict triples (repeats kept)'], ['#222222', '#2d6cbb', '#13815b']):
                counts = Counter((e['time_ms']/1000, e['kind']) for e in events if start <= e['time_ms']/1000 < end)
                for (time, kind), count in counts.items():
                    at = CLASSES.index(kind)
                    ax.vlines(time, at-.3, at+.3, color=color, lw=1.2)
                    if count > 1:
                        ax.text(time, at+.3, str(count), fontsize=8, color='#b02b2b')
                ax.set(yticks=range(len(CLASSES)), yticklabels=CLASSES,
                       ylim=(-.5, len(CLASSES)-.5), ylabel=label)
            for ax in axes:
                ax.axvline(anchor, color='#b02b2b', ls='--', lw=.8)
                for i in range(int(start/2.56), int(end/2.56)+1):
                    if start <= i*2.56 <= end:
                        ax.axvline(i*2.56, color='#888888', ls=':', lw=.8)
            axes[-1].set(xlim=(start, end), xlabel='Original audio clock (seconds)')
            fig.suptitle(f'{stem} / {anchor:.6f}s\nOriginal audio; no gain, snapping, time correction or LN tails; counts show duplicates')
            fig.tight_layout()
            fig.savefig(output/(stem+'.png'), dpi=120)
            plt.close(fig)
            detail = dict(recording=name, **item, start=start, end=end, source_audio_sha256=source_row['audio_sha256'],
                          original_rate=rate, clip_peak=float(abs(clip).max()),
                          image=stem+'.png', audio=stem+'.wav', direct_listening_performed=False)
            examples.append(detail)
            cards.append(f'<section><h2>{html.escape(stem)}</h2><p>{start:.3f}–{end:.3f} 秒；红虚线为检查位置，灰虚线为模型分窗。</p>'
                         f'<audio controls preload="none" src="{stem}.wav"></audio>'
                         f'<img loading="lazy" src="{stem}.png" alt="原始声音与鼓起音类别对照"></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>细分鼓采音原声检查</title><style>body{font:16px system-ui;max-width:1250px;margin:auto;padding:24px;line-height:1.6}img{width:100%}section{border-top:1px solid #ccc;margin-top:24px}audio{max-width:100%}</style>
<h1>细分鼓采音：固定最早遗漏与额外事件</h1><p>MDB逐类固定选择最早未匹配参考和最早额外预测；《樱花草》选择第4列与模型hi-hat最早差异。另显示最早重复与最早未正常结束的窗口。
三条时间线依次为参考、官方解码、保留重复的连续三元组；数字表示同一时刻同类音的数量。图中均用原声音时钟，不做拍线吸附。
《樱花草》参考是作者选择，第4列含tom等，其他三类没有作者参考；未匹配不直接证明声音不存在。
频谱1024采样窗、约5ms步进；原始PCM不归一化、保留超过1的float峰值，浏览器播放链可能限幅。页面音频供人工试听，本轮仅做图形审查，未完成亲耳验收。</p>'''
    if not report['complete_corpus']:
        page += '<p><strong>这是明确标记的部分结果，不能代表全部23段MDB。</strong></p>'
    (output/'index.html').write_text(page+''.join(cards)+'</html>', encoding='utf-8')
    save(output/'report.json', dict(evaluation_report_sha256=sha256(evaluation/'report.json'),
         complete_corpus=report['complete_corpus'], examples=examples, direct_listening_performed=False,
         artifacts_sha256={p.name: sha256(p) for p in output.iterdir()}))
    print(json.dumps(dict(examples=len(examples), page=str(output/'index.html'))), flush=True)


if __name__ == '__main__':
    main()
