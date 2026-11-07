"""Summarize frozen preference experiments, including rejected candidates."""
from pathlib import Path
import hashlib
import json
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'test-data/overnight'


def read(name):
    return json.loads((WORK/(name+'.json')).read_text(encoding='utf-8'))


def rows(name):
    value = read(name)
    return value if isinstance(value,list) else value.get('rows',value.get('cross_validation',[]))


def summarize(items):
    gains=[r['results']['probe']['match']['f1']-r['results']['baseline']['match']['f1'] for r in items]
    return dict(songs=len(gains),mean_f1_gain=round(mean(gains),6),worst_f1_gain=round(min(gains),6),
                improved_at_least_005=sum(g>=.005 for g in gains),positive_gain=sum(g>0 for g in gains))


def main():
    models={}
    for key,prefix in [('phase_sensitive','onset-preference'),('phase_invariant','onset-invariant')]:
        model=read(prefix+'-probe')
        sets={name:rows(prefix+suffix) for name,suffix in [
            ('development_cv','-probe'),('extension1_reference','-test1'),
            ('extension2_reference','-test2'),('extension2_automatic','-auto-test2')]}
        models[key]=dict(features=len(model['features']),training_ids=model['training_ids'],
            model_sha256=hashlib.sha256((WORK/(prefix+'-probe.json')).read_bytes()).hexdigest(),
            summaries={k:summarize(v) for k,v in sets.items()},results=sets)
    phases={}
    phase_rows=read('onset-phase-sensitivity')
    for phase in (0,.25,.5,.75):
        gains=[next(s['match']['f1'] for s in row['shifts'] if s['beat_shift']==phase)-row['baseline']['f1'] for row in phase_rows]
        phases[str(phase)]=dict(songs=len(gains),mean_f1_gain=round(mean(gains),6),worst_f1_gain=round(min(gains),6))
    models['phase_sensitive']['phase_relabel_test']=phases
    models['phase_sensitive']['decision']='Rejected: relabeling the beat without changing actual grid/note timestamps causes substantial regression.'
    protocol=read('onset-invariant-protocol')
    known=models['phase_invariant']['summaries']['extension2_reference']
    auto=models['phase_invariant']['summaries']['extension2_automatic']
    thresholds=protocol['thresholds']
    gate=dict(mean_known_grid_f1_gain=known['mean_f1_gain']>=thresholds['mean_known_grid_f1_gain'],
        fraction_gain_at_least_005=known['improved_at_least_005']/known['songs']>=thresholds['fraction_gain_at_least_005'],
        max_single_loss=known['worst_f1_gain']>=-thresholds['max_single_loss'],
        min_stable_auto_songs=auto['songs']>=thresholds['min_stable_auto_songs'],
        mean_auto_f1_gain=auto['mean_f1_gain']>=thresholds['mean_auto_f1_gain'],
        max_single_auto_loss=auto['worst_f1_gain']>=-thresholds['max_single_auto_loss'])
    models['phase_invariant'].update(protocol=protocol,gate=gate,
        decision='Rejected: did not pass all pre-recorded quality gates; no application integration.' if not all(gate.values()) else 'Quality gates passed; application integration still requires separate validation.')
    release=read('ln-release-probe');cv=release['cross_validation']
    gains=[r['probe']['endpoints']['f1']-r['baseline']['endpoints']['f1'] for r in cv if r['baseline']['endpoints']['f1'] is not None]
    release_summary=dict(songs=len(gains),mean_endpoint_f1_gain=round(mean(gains),6),worst_endpoint_f1_gain=round(min(gains),6),
        improved_at_least_005=sum(g>=.005 for g in gains),
        long_overlap_matches={t:{k:sum(r[k]['long_reference'][t]['overlap_matched'] for r in cv) for k in ('baseline','probe')} for t in ('800','2000')})
    report=dict(notice='Research diagnostics only. No onset-preference or learned-release weights are enabled in alpha.4.',
        onset=models,ln_release=dict(summary=release_summary,training_ids=release['training_ids'],results=cv,
        model_sha256=hashlib.sha256((WORK/'ln-release-probe.json').read_bytes()).hexdigest(),
        decision='Rejected on development cross-validation: small overall gain substantially damages long-LN overlap recall. No independent performance claim.'),
        reproducibility=dict(onset_script_sha256=hashlib.sha256((ROOT/'scripts/research_onset_preference.py').read_bytes()).hexdigest(),
            ln_script_sha256=hashlib.sha256((ROOT/'scripts/research_ln_releases.py').read_bytes()).hexdigest()))
    (ROOT/'docs/preference-research.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 未接入发布版的学习实验','',
        'alpha.4 启用的是轨道偏好模型。下面的起音取舍与 LN 收尾模型都没有进入应用；保留失败结果，避免只展示有利的均值。人工谱面的选择不是完整声音标注，这些成绩也不是玩家评价。','',
        '## 起音取舍','',
        '用 21 个恒定网格开发歌曲组训练歌曲均衡的 L2 logistic 模型，正则 0.025、强度 0.35，5 折按歌曲分组。输入来自声部分离后的候选；只调整候选强度，仍受原有密度 / LN 约束。追加两批歌曲分别报告，用户个人谱面不参加训练。','',
        '| 模型 / 条件 | 可评价歌曲 | 平均起音 F1 增益 | 最差单曲变化 | 提升至少 0.005 |','|---|---:|---:|---:|---:|']
    labels=dict(development_cv='开发集交叉验证',extension1_reference='新增一：人工网格',extension2_reference='新增二：人工网格',extension2_automatic='新增二：自动网格')
    for key,label in [('phase_sensitive','含拍标签，32 特征'),('phase_invariant','去拍标签，28 特征')]:
        for name,s in models[key]['summaries'].items():
            lines.append(f"| {label} / {labels[name]} | {s['songs']} | {s['mean_f1_gain']:+.4f} | {s['worst_f1_gain']:+.4f} | {s['improved_at_least_005']}/{s['songs']} |")
    lines+=['',
        '含拍标签模型在原始人工网格上表现较好，但保持所有实际网格和音符时间不变，只把强拍标签移动 1/4 拍后，平均 F1 变化为 '+f"{phases['0.25']['mean_f1_gain']:+.4f}"+'、最差单曲 '+f"{phases['0.25']['worst_f1_gain']:+.4f}"+'。这说明它依赖尚未可靠确定的强弱拍位置，故否决接入。','',
        '去拍标签版本移除强弱拍和拍速单位，邻接间隔使用真实秒数。查看第二批结果前固定门槛：人工网格平均增益至少 0.015、至少 60% 歌曲提升 0.005、单曲下降不超过 0.05；自动路径至少 5 首、均值不退步、单曲下降不超过 0.05。实际人工网格平均增益 0.01138，9/17 曲提升 0.005，前两项未通过。未调低门槛，也没有作为实验开关塞进发布版。','',
        '第二批 23 首中，17 首能用恒定人工网格评价，8 首通过自动 Timing；不是 23 首全部自动成功。第一批已经用于诊断相位问题，因此去拍标签版本的第一批成绩属于开发观察。','',
        '## LN 收尾','',
        '另做了一个收尾排序探针：在已选择 LN 的现有持续区间内提出更早的网格尾端，读取音长、剩余支撑、声部起音 / 放音邻近关系等 20 项特征。保持起音和 LN 数量、不延长证据，歌曲均衡训练，只在开发歌曲上交叉验证。','',
        f"21 首的平均 LN 起止 F1 仅提升 {release_summary['mean_endpoint_f1_gain']:.5f}；≥800ms 人工 LN 的重叠匹配由 79 降至 54，≥2000ms 由 14 降至 0。模型偏向缩短长条，损害本项目重点，故在开发阶段否决，没有继续宣称新歌曲收益。",'',
        '这个失败说明：短 LN 较多的人工取舍标签会主导整体指标，单独优化总体 F1 可能破坏长 LN。下一轮需要分长度 / 声部平衡的真实持续标注、长条保留约束与玩家偏好，而不是盲目裁短模型音尾。','',
        '完整逐曲统计、冻结模型哈希和门槛见 [preference-research.json](preference-research.json)。复现实验的脚本为 `scripts/research_onset_preference.py` 和 `scripts/research_ln_releases.py`；音源和训练中间文件不再分发。','']
    (ROOT/'docs/preference-research.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(onset_invariant_gate=gate,ln_release=release_summary),indent=2))


if __name__=='__main__':
    main()
