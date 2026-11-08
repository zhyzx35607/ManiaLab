"""Aggregate paired-stem expression comparisons without publishing note data."""
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    root=Path('test-data/corpus-expanded');read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    path=root/'paired-stem-release-dataset-v1/manifest.json'
    dataset=read(path)
    result=dict(date=str(date.today()), enabled_in_application=False, musical_acceptance=False,
        dataset=dict(sha256=sha(path), charts=len(dataset['records']), files=dataset['files'],exclusions=dataset['exclusions'],
            split_counts=dict(Counter(r['split'] for r in dataset['records'])),
            key_counts=dict(Counter(str(r['keys']) for r in dataset['records'])),
            groups={s:len({r['group'] for r in dataset['records'] if r['split']==s}) for s in ('train','validation','test')},
            old_split_disagreements=sum(r['split']!=r['old_split'] for r in dataset['records']),
            source='Hybrid Demucs four stems; original NPY scale; author attacks and Timing supplied'),
        training={},comparisons={})
    for folder in ('paired-stem-release-model-v1','paired-stem-rhythm-control-model-v1','paired-stem-frozen-rhythm-model-v1'):
        path=root/folder/'report.json'
        if path.exists():result['training'][folder]=read(path)
    identity=root/'paired-stem-frozen-rhythm-model-v1/initial-equivalence.json'
    if identity.exists():result['frozen_rhythm_equivalence']=read(identity)
    integrity=root/'paired-stem-release-dataset-v1/integrity-check.json'
    if integrity.exists():result['dataset_integrity']=read(integrity)
    for mode in ('flat','hierarchical'):
        for split in ('validation','test'):
            path=root/f'paired-stem-{mode}-{split}-v1/report.json'
            if not path.exists():continue
            raw=read(path);rows=raw['rows'];variants=('acoustic','rhythm_control','pretrained','next_anchor')
            raw['macro_f1']={name:sum(r['scores'][name]['complete_holds']['f1'] for r in rows)/len(rows) for name in variants}
            raw['macro_strict_f1']={name:sum(r['scores'][name]['strict_holds']['f1'] for r in rows)/len(rows) for name in variants}
            raw['long_holds']={name:{key:sum(r['scores'][name]['long_holds'][key] for r in rows)
                                   for key in ('reference','predicted','matched')} for name in variants}
            raw['versus_matched_control']=dict(Counter('win' if r['scores']['acoustic']['complete_holds']['f1']>r['scores']['rhythm_control']['complete_holds']['f1']
                else 'loss' if r['scores']['acoustic']['complete_holds']['f1']<r['scores']['rhythm_control']['complete_holds']['f1'] else 'tie' for r in rows))
            result['comparisons'][mode+'-'+split]=raw
    result['limits']=['All original attacks/counts and author Timing are supplied; this is not transcription.',
        'Four separated stems do not identify individual instruments or assign finished LN ownership.',
        'Previous project experiments have used these songs; comparisons are development diagnostics.',
        'A higher complete-hold score against one author is not musical acceptance.',
        'No learned weights were enabled in the application.']
    Path('docs/paired-stem-release-research.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    d=result['dataset'];s=d['split_counts']
    lines=['# 多乐曲声轨与LN表达对照','',f"{result['date']}。仍未达到完整交付标准；权重未启用。",'',
        f"本次实际准备{d['charts']}份配对谱：{s.get('train',0)}份拟合、{s.get('validation',0)}份验证、{s.get('test',0)}份测试，分别属于{d['groups']['train']}/{d['groups']['validation']}/{d['groups']['test']}个歌曲组。原音频、分离元数据及原始声轨身份均核验，裁剪保留绝对时间。排除项目及理由见机器报告。",'',
        '输入仍是人工起音和物件数、已知Timing及LN比例；这一步学习声音变化与表达之间的关系，不能当作解决自动采音。音频是此前已有的公开Hybrid Demucs四声部缓存，非Nova六声部。使用原幅度NPY，不使用试听归一化WAV。', '',
        '声学版本在78维节奏特征上追加60维声轨能量、频谱、持续与尾部变化观测。对照使用完全相同的起音组、标签、权重、歌曲划分和社区初始化，仅去掉新增60维；双方均24轮，通过验证选择轮次。旧音频划分与本次社区歌曲分组不一致的条目已重分配，测试不进入拟合。', '',
        '## 实际生成', '',
        '每谱在可用音频上下文中固定中央最多32秒，前文24拍，后文24拍。偏置为0，分层端点权重固定0.25，不按测试曲调参。起点位于评分段的所有LN均计入，尾部越界不会从分母消失。', '',
        '|划分与解码|声学F1|同样本节奏对照|社区原模型|下一起音|声学改善/相同/退步|',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in result['comparisons'].items():
        f=r['macro_f1'];c=r['versus_matched_control']
        lines.append(f"|{name}|{f['acoustic']:.4f}|{f['rhythm_control']:.4f}|{f['pretrained']:.4f}|{f['next_anchor']:.4f}|{c.get('win',0)}/{c.get('tie',0)}/{c.get('loss',0)}|")
    if not result['comparisons']:lines+=['','实际生成比较尚未完成。']
    lines+=['','|划分与解码|声学≥2秒匹配/参考|声学预测≥2秒|同样本节奏匹配|','|---|---:|---:|---:|']
    for name,r in result['comparisons'].items():
        a=r['long_holds']['acoustic'];b=r['long_holds']['rhythm_control']
        lines.append(f"|{name}|{a['matched']}/{a['reference']}|{a['predicted']}|{b['matched']}|")
    if result['comparisons']:
        lines+=['','本轮声学适配没有通过：在固定测试片段上明显落后于同样本节奏对照和社区原模型，权重拒绝启用。验证片段长LN有局部匹配，不足以抵消整体退步；测试只有1条两秒长LN，长条泛化证据尤其有限。']
    if 'frozen_rhythm_equivalence' in result:
        row=result['frozen_rhythm_equivalence']
        lines+=['','## 冻结节奏网络的后续诊断','',
            '为区分声学特征与小样本遗忘，保留原78维节奏权重及后续网络，仅训练新增声学输入，运行24轮；验证首次允许选择未适配的epoch 0。最终选择epoch 0，说明这组声学增量没有带来验证损失收益。', '',
            f"已核验全部原参数逐项不变、新声学权重全零，并对{row['validation_options']}个验证候选比较得分，最大误差{row['max_score_error']:.8g}。因此没有重复生成相同参数的对照来充当新成果。这里没有证明所有声学建模都无效，只否定当前粗能量/频谱增量的启用依据。"]
    lines+=['','每首曲子的严格10/20ms起止结果、短LN、结构验证及模型哈希均保留在机器报告。不能用平均值隐藏单曲退步，或把增加LN数量当作长条理解。歌曲素材已在此前工程中使用过，不称全新盲测。', '',
        '声学来源和分段身份解决了数据配对问题，仍未学到成品LN的具体乐器归属；误采音、缺少声部层数和乐句一致性也不能由这个有条件的实验自行修复。', '',
        '协议：[声轨学习](paired-stem-release-protocol.md)、[匹配对照](paired-stem-comparison-protocol.md)。[完整报告](paired-stem-release-research.json)。音频、谱面、逐事件数据及权重仅在本地忽略目录。','']
    Path('docs/paired-stem-release-research.md').write_text('\n'.join(lines),encoding='utf-8')
    print('docs/paired-stem-release-research.md')


if __name__=='__main__':main()
