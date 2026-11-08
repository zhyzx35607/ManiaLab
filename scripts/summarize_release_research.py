"""Publish aggregate independent-release results, excluding audio and labels."""
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    root = Path('test-data/sakura-benchmark'); corpus = Path('test-data/corpus-expanded')
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    result = dict(date=str(date.today()), enabled_in_application=False, musical_acceptance=False,
                  sakura={}, community={}, training={}, limits=[
        'Known author Timing and fixed detected attacks/counts in Sakura; this does not test automatic timing or transcription.',
        'The same-song calibration support contains only 19 onset groups; repeated phrases can correlate across sections.',
        'Source attribution inside simultaneous groups is still ambiguous.',
        'Community inputs are authored attacks and counts with tails/columns removed; no paired audio entered this community fit.',
        'The twenty comparison excerpts are development comparisons already inspected in earlier experiments.',
        'Optimum objective value and legal occupancy do not certify musical quality.'
    ])
    failure = corpus/'release-prepare-v1-aborted.json'
    if failure.exists(): result['aborted_preparation'] = read(failure)
    path = corpus/'release-dataset-v2/manifest.json'
    if path.exists():
        data = read(path)
        result['dataset'] = dict(manifest_sha256=sha(path), charts=len(data['records']),
            splits=dict(Counter(r['split'] for r in data['records'])),
            keys=dict(Counter(str(r['keys']) for r in data['records'])), exclusions=data['exclusions'], files=data['files'])
    folders = ['release-evaluation-v1', 'release-flow-v2', 'release-hierarchical-v2',
               'release-acoustic-single-flow-v2', 'release-acoustic-single-hierarchical-v2',
               'release-transfer-flow-v2', 'release-transfer-hierarchical-v2',
               'release-acoustic-transfer-flow-v2', 'release-acoustic-transfer-hierarchical-v2']
    for folder in folders:
        path = root/folder/'report.json'
        if not path.exists(): continue
        raw = read(path)
        result['sakura'][folder] = dict(report_sha256=sha(path), identity=raw['identity'],
            calibration=raw['calibration'], planner=raw['planner'], validation=raw['validation'],
            attacks_preserved=raw['input_attack_counts_preserved'],
            scores={k:v['variants']['learned']['head_scoped'] for k,v in raw['scores'].items()})
        check = root/folder/'review-check.json'
        if check.exists(): result['sakura'][folder]['browser_check'] = read(check)
    controlled = root/'release-decoder-comparison-v2/report.json'
    if controlled.exists(): result['same_probability_decoder_comparison'] = read(controlled)
    for folder in ['release-model-v2', 'release-acoustic-single-model-v2',
                   'release-transfer-model-v2', 'release-acoustic-transfer-model-v2']:
        path = root/folder/'report.json'
        if path.exists(): result['training'][folder] = read(path)
    path = corpus/'release-model-v2/report.json'
    if path.exists(): result['training']['community-release-model-v2'] = read(path)
    for planner in ('flow', 'hierarchical'):
        for split in ('validation', 'test'):
            path = corpus/f'release-{planner}-{split}-v2/report.json'
            if not path.exists(): continue
            raw = read(path)
            rows = raw['rows']
            def metric(row, name): return row['head_scoped'][name]['complete_holds']['f1']
            raw['macro_f1'] = {name:sum(metric(row,name) for row in rows)/len(rows)
                               for name in ('learned', 'next_anchor')}
            raw['wins_ties_losses'] = dict(Counter('win' if metric(r,'learned') > metric(r,'next_anchor')
                else 'loss' if metric(r,'learned') < metric(r,'next_anchor') else 'tie' for r in rows))
            result['community'][planner+'-'+split] = raw
    out = Path('docs/release-planning-research.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    names = {'release-evaluation-v1':'候选v1＋原求解（失败）', 'release-flow-v2':'单曲＋平坦流规划',
        'release-hierarchical-v2':'单曲＋分层流规划', 'release-acoustic-single-flow-v2':'单曲声学适配＋平坦',
        'release-acoustic-single-hierarchical-v2':'单曲声学适配＋分层',
        'release-transfer-flow-v2':'社区初始化适配＋平坦', 'release-transfer-hierarchical-v2':'社区初始化适配＋分层',
        'release-acoustic-transfer-flow-v2':'社区初始化声学适配＋平坦',
        'release-acoustic-transfer-hierarchical-v2':'社区初始化声学适配＋分层'}
    lines = ['# 独立放手与全曲规划研究', '', f"{result['date']}。**仍未达到交付标准，未启用新权重或发布 Release。**", '',
        '原起音锚点版本遗漏无新起音处的放手和部分短LN。本轮扩展到已知红线与输入起音的独立候选，16拍视野，优先保留真实输入起音。候选覆盖不等于模型知道应该选择哪个位置。', '',
        '## 修复两个确定的实验问题', '',
        '社区谱2571106含1e-61毫秒的红线周期，累计节拍约1e59，后续正常时差在浮点数中消失。中止v1数据准备，保留日志；v2校验单调性、0.1ms精度、毫秒往返及候选视野，无法表示的谱记录排除原因。原谱不修改，也不猜测替换BPM。', '',
        '单曲v1整曲在60秒限时内只给出5条LN及973个未确定事件，MIP gap为2.1088；相同模型的小片段能得到明显不同的可行解。新增网络流形式，保留原起音计数与全曲占轨约束，求到整数最优再接受。它关闭原来的放手开启成本，不能声称与带该成本的旧目标完全相同。五组随机小样本的平坦目标与原整数求解结果逐项一致。', '',
        '另加分层目标：先对总LN数量计误差，再对端点分布计误差（固定权重0.25），避免总持条概率被多个尾部分散后全部投成短键。它不强制填满LN配额，不支持原全曲硬上限，且保持实验开关。', '',
        '## 《樱花草》实际生成', '',
        '统一使用2291个检测草稿物件、作者Timing及全曲LN比例。前段拟合、中段校准、后段比较；该校准范围在16拍上下文要求下只有19组起音，明显偏少。下表保留尾部越出评分段的LN，因此两个片段的参考分母为59/62，不能直接与旧报告57/60混用。', '',
        '|版本|全曲LN|校准偏置|后段完整LN F1|片段一匹配|片段二匹配|后段≥2s匹配|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for folder, r in result['sakura'].items():
        score = r['scores']; a=score['probe_215853']['complete_holds']; b=score['probe_241798']['complete_holds']
        later=score['later_sections']; long=later['long_holds']
        lines.append(f"|{names[folder]}|{r['validation']['ln']}|{r['calibration']['ln_bias']}|{later['complete_holds']['f1']:.4f}|{a['matched']}/{a['reference']}|{b['matched']}/{b['reference']}|{long['matched']}/{long['reference']}|")
    if 'same_probability_decoder_comparison' in result:
        r=result['same_probability_decoder_comparison']; a=r['runs']['flat']; b=r['runs']['hierarchical']
        lines += ['', f"额外固定同一模型及偏置{r['ln_bias']}，平坦／分层规划分别生成{a['planner']['holds']}/{b['planner']['holds']}个LN，后段完整LN F1为{a['later']['complete_holds']['f1']:.4f}/{b['later']['complete_holds']['f1']:.4f}。这项对照没有重新校准偏置；上表各版本则按既定校准范围独立选偏置。"]
    lines += ['', '候选v2的单曲训练数据和v1逐特征相同，重新训练所得标准化参数与各层权重也逐项相同；修复主要排除其他谱的异常时钟。单曲声学适配是额外80轮，不与从零80轮称同预算。尚未形成可靠的长条延续或声部归属，不能把更多LN视作改善。', '', '## 社区训练与比较', '']
    if 'dataset' in result:
        r=result['dataset']; s=r['splits']
        lines += [f"v2有效{r['charts']}份谱；{s.get('train',0)}份拟合、{s.get('validation',0)}份验证、{s.get('test',0)}份测试。排除原因与数据哈希见机器报告。训练只使用拟合部分，验证选轮次，测试不进入拟合。", '']
    if result['community']:
        lines += ['|对照|学习平均F1|下一起音对照|改善/相同/退步|', '|---|---:|---:|---:|']
        for name,r in result['community'].items():
            c=r['wins_ties_losses']
            lines.append(f"|{name}|{r['macro_f1']['learned']:.4f}|{r['macro_f1']['next_anchor']:.4f}|{c.get('win',0)}/{c.get('tie',0)}/{c.get('loss',0)}|")
    else: lines += ['新社区生成比较尚未完成，不把正在准备或训练的任务算作结果。']
    lines += ['', '社区任务的输入是人工起音数量与Timing，不是音频采音验收。已有75份音频配对的库存检查发现26份旧划分与本轮歌曲分组不一致，多曲声学训练必须重新统一分组；库存检查本身没有进行音频训练。', '',
        '新试听页检查了播放、定位、声轨切换和窄屏，并查看谱图；这些界面检查没有代替人耳或演奏验收。来源归属、漏选、长LN以及独立歌曲上的表达仍需继续解决。', '',
        '协议：[独立候选](release-candidate-protocol.md)、[流规划](gesture-flow-protocol.md)、[声学适配](release-acoustic-protocol.md)。完整结果与哈希：[机器报告](release-planning-research.json)。音频、原谱、逐事件输出及权重均留在忽略目录。', '']
    Path('docs/release-planning-research.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out)


if __name__ == '__main__': main()
