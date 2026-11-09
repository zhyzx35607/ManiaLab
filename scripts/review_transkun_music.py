"""Render deterministic original-audio excerpts for the full music comparison."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.evaluate_piano_acoustics import pairs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def examples(plan, result, old):
    new = result['trace']['piano_events']
    if plan['kind']=='smd':
        reference = [n for n in read(ROOT/plan['labels']) if 0 <= n['onset'] < plan['score_seconds']]
        comparison = result['comparison']['50ms']
        chosen = []
        for label, field in [('new-miss','new_missing_ref_indices'), ('recovered','recovered_ref_indices')]:
            indices = comparison[field]
            if indices:
                index = min(indices,key=lambda i:(reference[i]['onset'],reference[i]['pitch']))
                chosen.append((label,reference[index]['onset'],reference[index]['pitch']))
        scored_new = [n for n in new if 0 <= n['onset'] < plan['score_seconds']]
        matched = {j for _,j in comparison['transkun_matches']}
        unmatched = [n for j,n in enumerate(scored_new) if j not in matched]
        if unmatched:
            first = min(unmatched,key=lambda n:(n['onset'],n['pitch']))
            chosen.append(('unmatched-prediction',first['onset'],first['pitch']))
        if not chosen and reference:
            chosen.append(('first-reference',reference[0]['onset'],reference[0]['pitch']))
        return chosen, reference
    matches = pairs(old,new)
    chosen = []
    for label, events, matched in [('kong-only',old,{i for i,_ in matches}),
                                   ('transkun-only',new,{j for _,j in matches})]:
        unmatched = sorted((n for i,n in enumerate(events) if i not in matched),
                           key=lambda n:(n['onset'],n['pitch']))
        chosen.extend((label+f'-{i+1}',n['onset'],n['pitch']) for i,n in enumerate(unmatched[:3]))
    return chosen, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot-dependencies',type=Path,default=ROOT/'test-data/plotlibs')
    args=parser.parse_args()
    run,output=args.run.resolve(),args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new local research directory')
    report=read(run/'report.json')
    for name,digest in report['artifacts_sha256'].items():
        if sha256(run/name)!=digest:
            raise ValueError('Inference evidence changed')
    sys.path.insert(0,str(args.plot_dependencies.resolve()))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from scipy.signal import stft
    output.mkdir(parents=True)
    shutil.copyfile(__file__,output/'review-snapshot.py')
    rows,cards,tables=[],[],[]
    for plan in report['identity']['plans']:
        name=plan['name']
        result_path=run/(name+'-result.json')
        result=read(result_path)
        if sha256(plan['baseline_notes'])!=plan['baseline_sha256']:
            raise ValueError('Baseline evidence changed')
        old=read(plan['baseline_notes'])
        new=result['trace']['piano_events']
        pcm=np.load(run/(name+'-16k.npy'),allow_pickle=False)
        choices,reference=examples(plan,result,old)
        shutil.copyfile(result_path,output/(name+'-result.json'))
        safe=html.escape(name)
        if plan['kind']=='smd':
            comparison=result['comparison']['50ms']
            cells=''.join(f'<td>{comparison[k]["f1"]:.4f}</td>' for k in ('kong','transkun'))
            cells+=f'<td>{len(comparison["new_missing_ref_indices"])}</td><td>{len(comparison["recovered_ref_indices"])}</td>'
        else:
            cells='<td colspan="4">无物理音高/音尾真值，仅比较模型差异</td>'
        tables.append(f'<tr><td><a href="{safe}-result.json">{safe} 完整结果</a></td>{cells}</tr>')
        for label,anchor,pitch in choices:
            if not 0 <= anchor < len(pcm)/16000:
                rows.append(dict(name=name,label=label,anchor=anchor,pitch=pitch,
                                 outside_input=True,image=None,audio=None))
                cards.append(f'<section><h2>{safe} / {html.escape(label)}</h2><p>'
                    f'候选 {anchor:.6f} 秒、音高 {pitch} 位于输入区间之外；没有对应可回放原声。'
                    f'该候选仍保留在完整结果中。</p></section>')
                continue
            start=max(0.,anchor-1.5)
            end=min(len(pcm)/16000,anchor+2.5)
            a,b=round(start*16000),round(end*16000)
            if b-a<2048:
                raise ValueError('Unexpected out-of-input example; inspect full result before rendering')
            clip=pcm[a:b]
            stem=name+'-'+label
            wavfile.write(output/(stem+'.wav'),16000,clip)
            fig,axes=plt.subplots(5,1,figsize=(12,10),sharex=True,
                                   gridspec_kw={'height_ratios':[1,2,1.2,1.2,1.2]})
            block=clip[:len(clip)//160*160].reshape(-1,160)
            times=start+(np.arange(len(block))+.5)*.01
            axes[0].fill_between(times,block.min(axis=1),block.max(axis=1),color='#3455a5')
            axes[0].set(ylabel='PCM')
            f,t,z=stft(clip.astype(float),fs=16000,nperseg=2048,noverlap=1888,boundary=None,padded=False)
            axes[1].pcolormesh(t+start,f,20*np.log10(np.maximum(abs(z),1e-8)),
                               shading='auto',vmin=-100,vmax=-20,cmap='magma')
            axes[1].set(yscale='log',ylim=(40,7000),ylabel='Hz / dB')
            visible=[n for n in [*reference,*old,*new] if n['offset']>=start and n['onset']<=end]
            lo=min([pitch]+[n['pitch'] for n in visible])-1
            hi=max([pitch]+[n['pitch'] for n in visible])+1
            for ax,notes,title,color in zip(axes[2:],[reference,old,new],
                    ['MIDI key (not LN)','Kong sustain','Transkun key'],['#222222','#466fd5','#09845c']):
                for n in notes:
                    if n['offset']<start or n['onset']>end:
                        continue
                    ax.plot([max(start,n['onset']),min(end,n['offset'])],[n['pitch']]*2,color=color,lw=2,alpha=.6)
                    if start<=n['onset']<=end:
                        ax.scatter([n['onset']],[n['pitch']],color=color,s=12)
                ax.set(ylim=(lo,hi),ylabel=title)
                if not notes:
                    ax.text(.02,.5,'No physical pitch/release labels',transform=ax.transAxes)
            for ax in axes:
                ax.axvline(anchor,color='#b62e34',ls='--',lw=.8)
            axes[-1].set(xlim=(start,end),xlabel='Original physical time (seconds)')
            fig.suptitle(f'{name} / {label} / pitch {pitch} at {anchor:.6f}s\n'
                         '128 ms spectrum window, 10 ms hop; decoded endpoints are not LN decisions')
            fig.tight_layout()
            fig.savefig(output/(stem+'.png'),dpi=130)
            plt.close(fig)
            rows.append(dict(name=name,label=label,anchor=anchor,pitch=pitch,start=start,end=end,
                             image=stem+'.png',audio=stem+'.wav'))
            title=html.escape(stem)
            cards.append(f'<section><h2>{title}</h2><p>原时钟 {start:.3f}–{end:.3f} 秒，关注音高 {pitch}。</p>'
                f'<audio controls preload="none" src="{title}.wav"></audio>'
                f'<img loading="lazy" src="{title}.png" alt="{title} 原声频谱与双模型输出"></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实钢琴原声对照</title><style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:24px;color:#18202c}
img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
section{padding:16px;border:1px solid #ddd;margin-top:24px}h2{overflow-wrap:anywhere}audio{max-width:100%}</style>
<h1>真实钢琴与 Nova 原声核查</h1><p>完整12份结果都可下载。SMD图例固定取每首时间最早的新增漏采、找回起音、
未匹配预测；无差异时取第一个参考音。Nova固定取每方向最早3个不一致候选，不将人工谱省略当误采。
图例是定位线索，不能替代逐曲汇总。起音F1使用同音高50ms一对一匹配。</p>
<p>音频保持原始幅度；音尾语义不同，均不等于LN放手。本页未完成直接试听验收。</p>
<table><tr><th>录音</th><th>Kong起音F1</th><th>Transkun起音F1</th><th>新漏</th><th>找回</th></tr>
'''+''.join(tables)+'</table>'+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    saved=dict(source_report_sha256=sha256(run/'report.json'),examples=rows,direct_listening_performed=False,
               artifacts_sha256={p.name:sha256(p) for p in output.iterdir()})
    (output/'report.json').write_text(json.dumps(saved,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(examples=len(rows),page=str(output/'index.html'))),flush=True)


if __name__=='__main__':
    main()
