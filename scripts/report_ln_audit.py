"""Render LN evidence and conditioning limitations from saved evaluations."""
import json
from pathlib import Path
from statistics import median

root=Path(__file__).resolve().parents[1]
report=json.loads((root/'docs/ln-audit.json').read_text(encoding='utf-8'))
style=json.loads((root/'docs/style-audit.json').read_text(encoding='utf-8'))
if report['source_fingerprint']!=style['source_fingerprint']:
    raise ValueError('Style and default audit fingerprints differ')
lines=['# LN 起止与人工参考的差距','',
    'alpha.4 的持续音候选更多，但现有检测音尾仍不能等同于谱师选择的 LN 收尾。以下完整保留默认参数和参考风格条件下的结果，不用长条数量代替音乐性。','',
    '一条唯一 `(起点,尾端)` 区间只计一次，忽略轨道和同起止和弦的重复；匹配使用最大一对一配对。起点容差 20ms，尾端容差 80ms；另记录起点接近且区间交并比至少 0.7 的重叠匹配。人工谱面表达编排选择，不是持续声音的完整标签。','',
    '## 已知人工网格：默认参数与参考风格','',
    '两种条件都使用人工 BPM / Offset，**不是自动端到端结果**。默认条件为密度 7、LN 目标 55%、最短 LN 240ms。参考风格条件额外调用程序现有的 `.osu` 偏好导入功能，提取密度、LN 比例、最短长度和占轨规模；不复制音符或轨道。','',
    '| 集合 | 可生成曲数 | 默认 快速 / 深度 LN F1 | 参考风格 快速 / 深度 LN F1 | 人工 LN 不足 240ms |',
    '|---|---:|---:|---:|---:|']
def fmt(value):return '—' if value is None else f'{value:.3f}'
for group,results in report['groups'].items():
    conditioned={}
    for mode in ('basic','deep'):
        values=[r['modes'][mode]['ln']['endpoints']['f1'] for r in style['songs']
                if r['group']==group and mode in r['modes'] and r['modes'][mode]['ln']['endpoints']['f1'] is not None]
        conditioned[mode]=median(values) if values else None
    reference=[r['reference_lengths'] for r in report['songs'] if r['group']==group]
    short=sum(r['under_240ms'] for r in reference);total=sum(r['unique_intervals'] for r in reference)
    b,d=(results[m]['reference'] for m in ('basic','deep'))
    lines.append(f"| {group} | {d['generated']} | {fmt(b['endpoints_median_f1'])} / {fmt(d['endpoints_median_f1'])} | {fmt(conditioned['basic'])} / {fmt(conditioned['deep'])} | {short}/{total} |")
lines += ['', 'F1 很低，且深度模式并非所有集合都更高。短 LN 比例能解释一部分默认参数的差异；即使提供参考风格，仍有起音选择、持续检测及收尾的明显缺口，不能归因于参数一种因素。', '',
    '## 较长 LN 单独检查', '',
    '下面只统计人工参考中 ≥800ms / ≥2000ms 的唯一区间，分母为这些人工长条数，分子为生成结果中的一对一匹配数。仍是已知人工网格、默认深度参数；没有把变速跳过歌曲计为已生成。', '',
    '| 集合 | ≥800ms 起止匹配 | ≥800ms 重叠匹配 | ≥2000ms 起止匹配 | ≥2000ms 重叠匹配 |',
    '|---|---:|---:|---:|---:|']
for group in report['groups']:
    rows=[r['modes']['deep']['reference'] for r in report['songs'] if r['group']==group and not r['modes']['deep']['reference'].get('skipped')]
    counts=[]
    for threshold in ('800','2000'):
        local=[r['long_reference'][threshold] for r in rows]
        n=sum(r['reference'] for r in local)
        counts.extend(f"{sum(r[k] for r in local)}/{n}" for k in ('endpoints_matched','overlap_matched'))
    lines.append('| '+group+' | '+' | '.join(counts)+' |')
lines += ['', '## 候选是否曾有足够持续证据', '',
    '在全曲密度、LN 数量和占轨选择前，逐条检查人工 LN 附近是否存在起音候选，以及任一候选是否持续到参考尾端附近。候选可重复支撑不同参考区间，所以这是乐观的可覆盖上界，**不是实际生成成绩或检测准确率**。即使声音有足够持续，仍未解决哪一段该拉长条和何时收尾。', '',
    '| 集合 | 参考 LN 区间 | 深度无近邻起音 | 深度持续不足 | 深度持续足够 | 已有接近音尾的候选 |',
    '|---|---:|---:|---:|---:|---:|']
for group,results in report['groups'].items():
    s=results['deep']['available_support']
    lines.append('| '+group+' | '+' | '.join(str(s[k]) for k in ('reference_intervals','no_attack','insufficient_duration','sufficient_duration','existing_endpoint_nearby'))+' |')
lines += ['', '逐歌曲结果和完全自动路径另见 [ln-audit.json](ln-audit.json)，参考风格条件见 [style-audit.json](style-audit.json)。自动路径被阻止的歌曲不被人工 Timing 替代。时长阈值和容差是本轮诊断约定，不是 osu! 的质量标准。', '',
    '这些结果支持下一阶段分别学习“选哪条持续证据”和“在持续区间内何时释放”，并增加人工听辨、同歌同密度对照及玩家评价。当前不能宣称 LN 音乐表达已经达到成熟谱师或全面超越 Mapperatorinator。','']
(root/'docs/ln-audit.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote docs/ln-audit.md')
