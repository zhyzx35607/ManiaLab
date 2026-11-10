"""Render original-audio evidence for all validation recordings and epochs."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.transkun_runtime import sha256
from scripts.piano_match_coverage import changes
from scripts.evaluate_piano_acoustics import pairs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def choose_examples(reference, before, after, sustain_reference=None, tail_changes=None):
    delta=changes(reference,before,after)
    choices=[]
    for label,field in [('new-miss','new_missing'),('recovered','recovered')]:
        indices=delta['selected'][field]
        if indices:
            i=min(indices,key=lambda j:(reference[j]['onset'],reference[j]['pitch']))
            choices.append((label,reference[i]['onset'],reference[i]['pitch'],
                            i in delta['unavoidable'][field]))
    unmatched=[n for i,n in enumerate(after) if i not in delta['after']['predicted_selected']]
    if unmatched:
        first=min(unmatched,key=lambda n:(n['onset'],n['pitch']))
        choices.append(('unmatched-prediction',first['onset'],first['pitch'],None))
    paired_before={i for i,_ in pairs(before,after)}
    removed=[n for i,n in enumerate(before)
             if i not in delta['before']['predicted_selected'] and i not in paired_before]
    if removed:
        first=min(removed,key=lambda n:(n['onset'],n['pitch']))
        choices.append(('removed-unmatched-prediction',first['onset'],first['pitch'],None))
    if sustain_reference is not None:
        old_intervals={i for i,_ in pairs(sustain_reference,before,offsets=True)}
        new_intervals={i for i,_ in pairs(sustain_reference,after,offsets=True)}
        for label,indices in [('lost-sustain-interval',old_intervals-new_intervals),
                              ('gained-sustain-interval',new_intervals-old_intervals)]:
            if indices:
                i=min(indices,key=lambda j:(sustain_reference[j]['onset'],sustain_reference[j]['pitch']))
                choices.append((label,sustain_reference[i]['offset'],sustain_reference[i]['pitch'],None))
    if tail_changes:
        regressions=[row for row in tail_changes if row['exceeds_80ms']]
        if regressions:
            worst=max(regressions,key=lambda row:(row['error_increase_seconds'],-row['reference_index']))
            choices.append(('largest-tail-regression',worst['reference_end'],worst['pitch'],None))
    if not choices:
        first=min(reference,key=lambda n:(n['onset'],n['pitch']))
        choices.append(('first-reference',first['onset'],first['pitch'],None))
    return choices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cache,run,output=args.cache.resolve(),args.run.resolve(),args.output.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'test-data').resolve()):
        raise ValueError('Use a new local research directory')
    report=read(run/'report.json')
    source=read(cache/'report.json')
    if report['identity']['cache_report_sha256']!=sha256(cache/'report.json'):
        raise ValueError('Wrong source cache')
    code=report['identity']['code']
    scoring=code.get('scoring',code.get('evaluate_piano_acoustics'))
    if scoring!=sha256(ROOT/'scripts/evaluate_piano_acoustics.py'):
        raise ValueError('Current scoring differs from the training snapshot')
    for folder,data in [(cache,source),(run,report)]:
        for name,digest in data['artifacts_sha256'].items():
            if sha256(folder/name)!=digest:
                raise ValueError('Evidence changed: '+name)
    sys.path.insert(0,str(ROOT/'test-data/plotlibs'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from scipy.signal import stft
    output.mkdir(parents=True)
    shutil.copyfile(__file__,output/'review-snapshot.py')
    cards,table,examples=[],[],[]
    for recording in [r for r in source['records'] if r['split']=='validation']:
        name=recording['id']
        labels=read(cache/(name+'-labels.json'))
        reference=labels['notes']
        reference_scored=[n for n in reference if 0<=n['onset']<170]
        sustain_scored=[n for n in labels['sustain_notes'] if 0<=n['onset']<170]
        old=read(run/(name+'-epoch0-events.json'))['notes']
        old_scored=[n for n in old if 0<=n['onset']<170]
        pcm=np.load(cache/(name+'-16k.npy'),allow_pickle=False)
        if pcm.shape!=(2880000,):
            raise ValueError('Unexpected physical input clock')
        for epoch in (0,1,2):
            row=next(r for r in report['epochs'][epoch]['validation'] if r['id']==name)
            score=row['metrics']['key']['50ms']['onsets']
            events_path=run/(name+f'-epoch{epoch}-events.json')
            shutil.copyfile(events_path,output/events_path.name)
            table.append(f'<tr><td>{html.escape(name)}</td><td>{epoch}</td><td>{score["matched"]}</td>'
                f'<td>{score["reference"]-score["matched"]}</td><td>{score["predicted"]-score["matched"]}</td>'
                f'<td>{score["f1"]:.6f}</td><td><a href="{events_path.name}">完整结果</a></td></tr>')
            if epoch==0:
                continue
            new=read(events_path)['notes']
            new_scored=[n for n in new if 0<=n['onset']<170]
            tail_changes=None
            if 'piano_joint_selection' in code:
                comparison_path=run/(name+f'-epoch{epoch}-comparison.json')
                if comparison_path.name not in report['artifacts_sha256']:
                    raise ValueError('Joint-head comparison is not part of the verified run')
                tail_changes=read(comparison_path)['50ms']['tail_changes']
            for label,anchor,pitch,unambiguous in choose_examples(reference_scored,old_scored,new_scored,sustain_scored,tail_changes):
                stem=f'{name}-epoch{epoch}-{label}'
                begin,end=max(0.,anchor-1.5),min(180.,anchor+2.5)
                a,b=round(begin*16000),round(end*16000)
                begin,end=a/16000,b/16000
                clip=pcm[a:b]
                wavfile.write(output/(stem+'.wav'),16000,clip)
                fig,axes=plt.subplots(6,1,figsize=(12,12),sharex=True,
                                     gridspec_kw={'height_ratios':[1,2,1.2,1.2,1.2,1.2]})
                blocks=clip[:len(clip)//160*160].reshape(-1,160)
                t=begin+(np.arange(len(blocks))+.5)*.01
                axes[0].fill_between(t,blocks.min(axis=1),blocks.max(axis=1),color='#3455a5')
                axes[0].set(ylabel='PCM')
                f,t,z=stft(clip.astype(float),fs=16000,nperseg=2048,noverlap=1888,boundary=None,padded=False)
                axes[1].pcolormesh(t+begin,f,20*np.log10(np.maximum(abs(z),1e-8)),
                                   shading='auto',vmin=-100,vmax=-20,cmap='magma')
                axes[1].set(yscale='log',ylim=(20,7000),ylabel='Hz / dB')
                displayed=[reference,labels['sustain_notes'],old,new]
                visible=[n for ns in displayed for n in ns if n['offset']>=begin and n['onset']<=end]
                lower=min([pitch]+[n['pitch'] for n in visible])-1
                upper=max([pitch]+[n['pitch'] for n in visible])+1
                for ax,notes,title,color in zip(axes[2:],displayed,
                        ['MIDI key (not LN)','MIDI pedal (not LN)','Original model tail',f'Epoch {epoch} model tail'],
                        ['#222222','#8355a3','#466fd5','#09845c']):
                    for n in notes:
                        if n['offset']<begin or n['onset']>end:
                            continue
                        ax.plot([max(begin,n['onset']),min(end,n['offset'])],[n['pitch']]*2,color=color,lw=2,alpha=.6)
                        if begin<=n['onset']<=end:
                            ax.scatter([n['onset']],[n['pitch']],color=color,s=12)
                    ax.set(ylim=(lower,upper),ylabel=title)
                for ax in axes:
                    ax.axvline(anchor,color='#b62e34',ls='--',lw=.8)
                    if begin<170<end:
                        ax.axvspan(170,end,color='#777777',alpha=.12)
                        ax.axvline(170,color='#777777',ls=':',lw=.8)
                axes[-1].set(xlim=(begin,end),xlabel='Original physical time (seconds)')
                fig.suptitle(f'{name} / epoch {epoch} / {label} / pitch {pitch} at {anchor:.6f}s\n'
                             '128 ms spectrum window, 10 ms hop; selected matching difference; not LN acceptance')
                fig.tight_layout()
                fig.savefig(output/(stem+'.png'),dpi=130)
                plt.close(fig)
                examples.append(dict(recording=name,epoch=epoch,label=label,anchor=anchor,pitch=pitch,
                    audio=stem+'.wav',image=stem+'.png',start=begin,end=end,
                    anchor_kind='sustain-offset' if 'sustain-interval' in label or label=='largest-tail-regression' else 'onset',
                    unavoidable_coverage_change=unambiguous))
                note=('全部最大基数配对均支持这处覆盖变化。' if unambiguous is True else
                      '存在其他最大基数配对：此处仅表示选定配对的归属变化。' if unambiguous is False else
                      '按当前确定性配对选取的检查位置，不单独判定声学正误。')
                cards.append(f'<section><h2>{stem}</h2><p>{note}</p><audio controls preload="none" src="{stem}.wav"></audio>'
                             f'<img loading="lazy" src="{stem}.png" alt="原声与模型差异"></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>声学头适配原声对照</title><style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:24px;color:#18202c}
img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
section{padding:16px;border:1px solid #ddd;margin-top:24px}audio{max-width:100%}</style>
<h1>声学头适配：全部验证录音、全部epoch</h1><p>输入前180秒，评分前170秒。每首每轮固定取最早新漏、找回、
未匹配预测、消失的未匹配预测及踏板区间配对变化，无检查位置则首参考音；不挑最好一轮。
踏板区间变化以参考音尾为图中锚点。此处在首版图页之后补充了被删除的额外预测与音尾变化类别，
不参与训练和选轮。双头实验另展示误差增量最大的可比音尾退步，门槛为严格超过80ms；
该项只覆盖前后起音配对均无歧义的参考。播放模型实际输入的16kHz单声道片段，
不额外归一化并保留物理时钟。128ms频谱窗用于检查声音结构，不提供逐采样的起音证明。按键参考和踏板延音语义不同，
均不是LN目标。图中完整参考保留至输入末端，170秒后的灰色区域不参与起音评分。
找回/新漏位置来自确定性最大匹配，另标记其覆盖变化是否在全部最大基数配对下均成立；
此保守核查不使用次级最小时间误差。本页没有完成直接试听验收。</p><table><tr><th>录音</th><th>epoch</th><th>匹配</th><th>漏采</th>
<th>额外</th><th>F1</th><th>原始预测</th></tr>'''+''.join(table)+'</table>'+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    result=dict(source_report_sha256=sha256(run/'report.json'),cache_report_sha256=sha256(cache/'report.json'),
        helpers_sha256={name:sha256(ROOT/'scripts'/name) for name in
                        ('evaluate_piano_acoustics.py','piano_match_coverage.py')},
        examples=examples,direct_listening_performed=False,
        artifacts_sha256={p.name:sha256(p) for p in output.iterdir()})
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(examples=len(examples),page=str(output/'index.html'))),flush=True)


if __name__=='__main__':
    main()
