"""Publish aggregate encoder-expression evidence without audio or note labels."""
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    root=Path('test-data/corpus-expanded');read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    path=root/'mapper-release-dataset-v1/manifest.json';dataset=read(path)
    result=dict(date=str(date.today()),enabled_in_application=False,musical_acceptance=False,
        dataset=dict(sha256=sha(path),charts=len(dataset['records']),files=dataset['files'],exclusions=dataset['exclusions'],
            splits=dict(Counter(r['split'] for r in dataset['records'])),
            keys=dict(Counter(str(r['keys']) for r in dataset['records'])),projection=dataset['projection']),
        geometry=read(root/'mapper-embeddings-v1/geometry.json'),
        integrity=read(root/'mapper-release-dataset-v1/integrity-check.json'),training={},comparisons={})
    for name in ('mapper-release-model-v1','mapper-release-rhythm-model-v1'):
        result['training'][name]=read(root/name/'report.json')
    for split in ('validation','test'):
        data=read(root/f'mapper-release-{split}-v1/report.json');rows=data['rows']
        variants=('embedding','rhythm_control','pretrained','next_anchor')
        data['macro_f1']={v:sum(r['scores'][v]['complete_holds']['f1'] for r in rows)/len(rows) for v in variants}
        data['macro_strict_f1']={v:sum(r['scores'][v]['strict_holds']['f1'] for r in rows)/len(rows) for v in variants}
        data['long_holds']={v:{k:sum(r['scores'][v]['long_holds'][k] for r in rows) for k in ('reference','predicted','matched')} for v in variants}
        data['short_holds']={v:{k:sum(r['scores'][v]['short_holds'][k] for r in rows) for k in ('reference','predicted','matched')} for v in variants}
        data['versus_control']=dict(Counter('win' if r['scores']['embedding']['complete_holds']['f1']>r['scores']['rhythm_control']['complete_holds']['f1']
            else 'loss' if r['scores']['embedding']['complete_holds']['f1']<r['scores']['rhythm_control']['complete_holds']['f1'] else 'tie' for r in rows))
        result['comparisons'][split]=data
    result['limits']=['Author attacks/counts/Timing/LN ratio supplied; not an automatic transcription evaluation.',
        'Mixed-audio contextual embeddings do not supply instrument identity or explicit physical note boundaries.',
        'These development songs have been used previously. PCA uses only this experiment training split.',
        'Comparisons are paired within this experiment; windows differ from the preceding stem experiment.',
        'Authored LN similarity and legal occupancy do not prove musical quality.']
    result['decision']='Reject default use: held-out generation regressed versus matched rhythm control and the community model.'
    Path('docs/mapper-embedding-release-research.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Mapper音频表示与放手规划','',f"{result['date']}，研究权重未接入应用，未达到整体交付标准。",'',
        '采用固定Mapper mania编码器的真实音频路径：16.376秒窗口、8.192秒步长、保留中心区。卷积前端脉冲核验时间间隔为16ms，但注意力混合完整窗口，不能称每个向量都是独立音符。输入仅为核验过的原混音准备WAV，没有作者音符、Timing或书签进入编码器。','',
        f"共{result['dataset']['charts']}份配对谱，训练/验证/测试为{result['dataset']['splits']}。32维PCA只用训练歌曲拟合；同样本纯节奏对照的标签、权重、起音与78维节奏前缀逐项一致。独立歌曲组与缓存身份核验通过。",'',
        '放手模型仍接收人工起音/物件数、Timing和整曲LN条件。这是表达实验，不能当作自动采音通过；也没有学习成品LN的乐器归属。两种模型均从社区模型初始化，24轮按验证损失选轮次，允许epoch 0。','',
        '|版本|选择轮次|验证损失|','|---|---:|---:|']
    for name,report in result['training'].items():
        best=next(r for r in report['history'] if r['epoch']==report['selected_epoch'])
        lines.append(f"|{name}|{report['selected_epoch']}|{best['validation_loss']:.5f}|")
    lines+=['','实际生成固定中央最多16秒、24拍前后文，分层流规划、偏置0。所有决定先落盘再评分；超过音频缓存的候选明确拒绝，不作边缘夹取。逐曲排除及原因见机器报告。','',
        '|划分|片段数|音频表示F1|同样本节奏|社区原模型|改善/相同/退步|','|---|---:|---:|---:|---:|---:|']
    for split,r in result['comparisons'].items():
        f=r['macro_f1'];c=r['versus_control']
        lines.append(f"|{split}|{len(r['rows'])}|{f['embedding']:.4f}|{f['rhythm_control']:.4f}|{f['pretrained']:.4f}|{c.get('win',0)}/{c.get('tie',0)}/{c.get('loss',0)}|")
    lines+=['','|划分|音频表示严格F1|节奏严格F1|音频表示≥2秒匹配/参考|预测≥2秒|','|---|---:|---:|---:|---:|']
    for split,r in result['comparisons'].items():
        a=r['long_holds']['embedding'];s=r['macro_strict_f1']
        lines.append(f"|{split}|{s['embedding']:.4f}|{s['rhythm_control']:.4f}|{a['matched']}/{a['reference']}|{a['predicted']}|")
    lines+=['','结论：不启用。验证平均值略增，但测试相对同样本节奏模型和社区模型均退步，长LN没有新增匹配。两秒长LN在测试片段中仅1条，证据不足。小模型记住音频表示也不等于学会音乐表达。','',
        '这是开发对照。平均F1或训练损失不能代替逐声部试听、全曲持续表达、自动Timing及6–10K手感验收。原音频、谱面、逐事件数据和权重仅在test-data中；公开机器报告保留逐曲统计及身份哈希。','',
        '[事前协议](mapper-embedding-release-protocol.md)；[完整统计](mapper-embedding-release-research.json)。','']
    Path('docs/mapper-embedding-release-research.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({s:{k:r[k] for k in ('macro_f1','macro_strict_f1','long_holds','versus_control','exclusions')} for s,r in result['comparisons'].items()}),flush=True)


if __name__=='__main__':main()
