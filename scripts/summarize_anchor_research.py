"""Publish aggregate anchor-planning results; never copy event labels or audio."""
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from scripts.evaluate_anchor_gestures import describe, long_comparison
from scripts.probe_mapper_adapter import sha


def main():
    root = Path('test-data/sakura-benchmark'); corpus = Path('test-data/corpus-expanded')
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    dataset = read(corpus/'anchor-dataset-v1/manifest.json')
    human = parse_osu((root/'finished.osu').read_text(encoding='utf-8-sig'))['notes']
    result = dict(date=str(date.today()), enabled_in_application=False, musical_acceptance=False,
        community_dataset=dict(manifest_sha256=sha(corpus/'anchor-dataset-v1/manifest.json'),
            included_charts=len(dataset['records']), exclusions=dataset['exclusions'],
            key_counts=dict(Counter(str(r['keys']) for r in dataset['records'])),
            splits={split:dict(charts=sum(r['split']==split for r in dataset['records']),
                song_groups=len({r['group'] for r in dataset['records'] if r['split']==split}),
                key_counts=dict(Counter(str(r['keys']) for r in dataset['records'] if r['split']==split)),
                **dataset['files'][split]) for split in ('train', 'validation', 'test')}),
        community_training=read(corpus/'anchor-model-v1/report.json'),
        source_training={}, sakura={}, community_comparisons={},
        limits=['All new models remain disabled in the application; no Release.',
            'Known author Timing, detected development sources and whole-chart LN condition in Sakura experiments.',
            'Same-song repeated phrases; no independent acoustic validation.',
            'Community comparison inputs are authored attacks/counts, not detected audio.',
            'Simultaneous-group supervision does not identify which source owns a finished LN.',
            'Onset-only endpoint candidates omit genuine releases without a nearby attack.',
            'Web playback/seek checks do not constitute musical listening acceptance.'])
    for folder in ('anchor-model-v2', 'anchor-transfer-model-v1', 'anchor-acoustic-model-v1'):
        if (root/folder/'report.json').exists(): result['source_training'][folder]=read(root/folder/'report.json')
    for folder in ('anchor-evaluation-v1', 'anchor-capped-v1', 'anchor-transfer-evaluation-v1',
                   'anchor-transfer-capped-v1', 'anchor-acoustic-evaluation-v1', 'anchor-acoustic-capped-v1'):
        if not (root/folder/'report.json').exists(): continue
        report = read(root/folder/'report.json')
        generated = parse_osu((root/folder/'planned.osu').read_text(encoding='utf-8-sig'))['notes']
        report['whole_generated_summary'] = describe(generated)
        report['whole_long_comparison'] = long_comparison(human, generated, 0, 283840)
        report['later_long_comparison'] = long_comparison(human, generated, 188286, 283840)
        report['chart_sha256'] = sha(root/folder/'planned.osu')
        result['sakura'][folder] = report
    for split in ('validation', 'test'):
        report = read(corpus/f'anchor-{split}-v1/report.json')
        report['macro_complete_hold_f1'] = {name:sum(r['scores'][name]['complete_hold_spans']['f1'] for r in report['rows'])/len(report['rows'])
            for name in ('learned', 'next_anchor')}
        report['wins_ties_losses'] = dict(Counter('better' if r['scores']['learned']['complete_hold_spans']['f1'] > r['scores']['next_anchor']['complete_hold_spans']['f1']
            else 'worse' if r['scores']['learned']['complete_hold_spans']['f1'] < r['scores']['next_anchor']['complete_hold_spans']['f1'] else 'tie' for r in report['rows']))
        result['community_comparisons'][split] = report
    diagnosis = corpus/'anchor-candidate-diagnosis-v1.json'
    if diagnosis.exists(): result['candidate_diagnosis'] = read(diagnosis)
    layout = root/'anchor-source-layout-v1/report.json'
    if layout.exists(): result['source_layout_correction'] = read(layout)
    for folder in result['sakura']:
        ui = root/folder/'review-check.json'
        if ui.exists(): result['sakura'][folder]['browser_check'] = read(ui)
    output = Path('docs/anchor-gesture-research.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    names = {'anchor-evaluation-v1':'单曲节奏学习', 'anchor-capped-v1':'单曲学习＋LN 上限',
        'anchor-transfer-evaluation-v1':'社区预训练＋单曲适配', 'anchor-transfer-capped-v1':'社区适配＋LN 上限',
        'anchor-acoustic-evaluation-v1':'社区初始化＋声学特征', 'anchor-acoustic-capped-v1':'声学特征＋LN 上限'}
    lines = ['# 起音锚点 LN 规划研究', '', f"{result['date']}。**已有局部改善，未达到交付标准，全部权重仍不默认启用。**", '',
        '本轮固定已检测的起音和物件数，学习短键／LN 终点分布，再统一检查全曲占轨容量，最后交给既有定轨器。原始输入音尾、成品轨道、书签和绿线不参与特征。候选终点来自八拍内已有起音，不能把没有候选的作者放手计成已解决。', '',
        '## 《樱花草》开发结果', '',
        '已知作者 Timing；输入为冻结的 2291 物件检测草稿。前段拟合、中段校准、后段比较，重复乐句相关性仍存在。下面的完整 LN F1 同时要求起点与终点匹配；单个作者只是其中一种解释，分数不是玩家满意率。', '',
        '|实验|全曲 LN 数|后段完整 LN F1|215.853 秒片段匹配 / F1|241.798 秒片段匹配 / F1|后段 ≥2秒 LN 匹配|',
        '|---|---:|---:|---:|---:|---:|']
    for folder, row in result['sakura'].items():
        score = row['scores']; a=score['probe_215853']['variants']['learned']['comparison']['complete_hold_spans']
        b=score['probe_241798']['variants']['learned']['comparison']['complete_hold_spans']
        later=score['later_sections']['variants']['learned']['comparison']['complete_hold_spans']
        long=row['later_long_comparison']
        lines.append(f"|{names[folder]}|{row['validation']['ln']}|{later['f1']:.4f}|{a['matched']}/{a['reference']} / {a['f1']:.4f}|{b['matched']}/{b['reference']} / {b['f1']:.4f}|{long['matched']}/{long['reference']}|")
    lines += ['', '同一检测草稿的“拉到下一起音”对照，后段完整 LN F1 为 0.2423。此前自由 Mapper 实验在第一个片段为 9/57、F1 0.1818；两条管线的输出数量、上下文和表达限制不同，这项比较只说明该开发片段的变化，不证明全面超越上游。', '',
        '各新实验保持全部 2291 个输入起音和物件，导出通过无重叠校验。这不等于修复采音：作者成品有 2806 个物件；后段检测输入 778 个、作者 974 个。输入漏选和层数差异仍在。未限制版本也可能生成过多 LN；可选上限保留全部短键位置，不剪尾，也不强迫填满配额。', '',
        '按本轮候选条件，全曲 961 条人工 LN 中只有 879 条两端存在候选，考虑输入同时起音数量后的必要条件上限为 833；后段对应上限为 300/355。这是表达能力限制，不是达成成绩。', '',
        '## 社区语料与歌曲隔离比较', '',
        f"既有有效快照中排除 5 份仅用于定轨的条目和 1 份缺少有效红线的条目（1255026），剩余 {len(dataset['records'])} 份。沿用歌曲／谱包／音频哈希连通分组，6422 份拟合、735 份验证、744 份测试；实际拟合 76032 组起音、1564534 个候选。不是把 7901 份都放入训练集。", '',
        '纯节奏模型为 70 特征、64/32 隐藏层，24 轮中按验证损失选第9轮。每首歌每键数共享约48行采样预算，每谱至少1行、最多32行，因此难度特别多的歌曲组可能超过48行；含不同 LN 比例，并按键数及歌曲组加权。额外固定每个键数、每个比较划分各两首，共20个32秒片段，优先长条为主与混合谱，缺少时明确回退。不给模型成品轨道和音尾，但人工起音数量、Timing及全曲LN条件是已知输入。', '',
        '|划分|学习模型平均完整 LN F1|下一起音对照|改善／相同／退步|', '|---|---:|---:|---:|']
    for split, row in result['community_comparisons'].items():
        m=row['macro_complete_hold_f1']; c=row['wins_ties_losses']
        lines.append(f"|{split}|{m['learned']:.4f}|{m['next_anchor']:.4f}|{c.get('better',0)}/{c.get('tie',0)}/{c.get('worse',0)}|")
    lines += ['', '通用表现不稳定，验证平均退步，不能启用。部分参考谱有大量小于100ms的LN，另一些尾部附近没有新起音；本轮候选无法充分表达它们。这不是多训练几轮就能补齐的问题。后续需要保留独立放手事件，并把声部持续与乐句节奏证据纳入规划。', '',
        '## 声学增量与验证边界', '',
        '声学版本额外使用40个 Nova 能量及冻结 YourMT3 观测特征，显式区分强制截断。它没有把物理音尾当作 LN 真值，也没有用 Reference 的音尾作输入。只在同曲已有划分中适配，不能称已在数千首歌曲的分离音轨上训练。', '',
        '## 声部列号误作演奏位置的修正', '',
        '检查谱图发现，旧调用把检测草稿的声部标记列传作 `import`，继承了原轨道位置偏好和短连打保留规则。声部身份不等于成品演奏位置。新增调用保留声部连续性、双手负担和 LN 代价，关闭这种虚假的位置跟随和成品短连打继承；旧成品重定轨流程保持原语义。', '',
        '在声学＋上限版本的同一批2291个固定事件上，左右手物件由1427/864变为1154/1137，快速同轨计数由392降到17，紧迫放手再按仍为0；全部起止、LN数、最长LN及占轨峰值保持一致，导出无冲突。这是调用语义和结构诊断，不能把更平均的数量或更少连打直接当作音乐质量验收。', '',
        '前表保存的是修正前的生成报告；起止指标不受这次重新定轨影响。独立的新对照保留在 `anchor-source-layout-v1`。后续评估入口默认使用修正后的调用，`--legacy-layout` 可以重现原调用方式。', '',
        '保留所有失败与退步。原始音频、逐音符标签、权重、生成谱及对照页都在忽略目录。浏览器检查覆盖播放、点击定位、声轨切换和窄屏显示；尚无人耳或玩家验收，不发布 Release。', '',
        '协议见 [基础实验](anchor-gesture-protocol.md)、[数量上限](anchor-gesture-ln-ceiling.md)、[声学特征](anchor-acoustic-protocol.md)；完整统计、逐曲退步、哈希见 [机器报告](anchor-gesture-research.json)。', '',
        '```powershell', '.venv/Scripts/python.exe scripts/train_anchor_gestures.py prepare --source corpus --output <新数据目录>',
        '.venv-audio/Scripts/python.exe scripts/train_anchor_gestures.py fit --dataset <数据目录> --output <模型目录>',
        '.venv/Scripts/python.exe scripts/evaluate_corpus_anchors.py evaluate --selection test-data/corpus-expanded/anchor-comparison-plan-v1.json --model <模型目录>/model.json --output <新比较目录> --split test',
        '.venv/Scripts/python.exe scripts/evaluate_anchor_gestures.py --model <单曲模型目录>/model.json --output <新比较目录> --review', '```', '']
    Path('docs/anchor-gesture-research.md').write_text('\n'.join(lines), encoding='utf-8')
    print(str(output), flush=True)


if __name__ == '__main__': main()
