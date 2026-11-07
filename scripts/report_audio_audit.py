"""Publish aggregate audit evidence without redistributing songs or charts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def summarize(folders):
    root=Path(__file__).resolve().parents[1]
    base_model=json.loads((root/'assets/model-manifest.json').read_text(encoding='utf-8'))['basic_pitch']
    if hashlib.sha256((root/'assets/basic-pitch.onnx').read_bytes()).hexdigest()!=base_model['sha256']:
        raise ValueError('Basic Pitch file differs from its pinned manifest')
    deep_models=json.loads((root/'assets/deep-models.json').read_text(encoding='utf-8'))['models']
    expected_deep={r['filename']:r['sha256'] for r in deep_models}
    rows, fingerprints, identities = [], {}, set()
    for folder in folders:
        for path in sorted(folder.glob('*/audit.json')):
            item = json.loads(path.read_text(encoding='utf-8'))
            identity = (folder.name, item['beatmap_id'])
            if identity in identities:
                raise ValueError(f'Duplicate audit: {identity}')
            identities.add(identity)
            if set(item['modes']) != {'basic', 'deep'}:
                raise ValueError(f'Incomplete audio comparison: {path}')
            fingerprint = item['source_fingerprint']
            fingerprints[fingerprint['sha256']] = fingerprint
            analysis=json.loads((path.parent/'deep/analysis.json').read_text(encoding='utf-8'))
            separation=analysis['separation']
            if (separation['model_sha256']!=expected_deep['hdemucs_high_trained.pt'] or
                    separation['beat_evidence']['model_sha256']!=expected_deep['beat-this-final0.ckpt']):
                raise ValueError(f'Audio snapshot uses unexpected separation/beat weights: {path}')
            automatic = json.loads((path.parent/'auto-audit.json').read_text(encoding='utf-8'))
            if (automatic['source_fingerprint'] != fingerprint['sha256'] or
                    automatic['audio_shift_ms'] != item['audio_shift_ms'] or
                    automatic['beatmap_id'] != item['beatmap_id']):
                raise ValueError(f'Automatic comparison does not match the source audit: {path}')
            modes = {}
            for name, result in item['modes'].items():
                tempo, generated = result['tempo'], result['generation']
                modes[name] = dict(
                    tempo={k: tempo.get(k) for k in ('bpm', 'offset', 'division', 'stable', 'exact_bpm', 'reliability')},
                    timing_comparison=result.get('timing_comparison'),
                    raw_attack_match=result['raw_attack_match'],
                    mapping_attack_match=result['mapping_attack_match'],
                    candidates=result['candidates'], voices=result['candidate_voices'],
                    phrase_evidence=result['phrase_evidence'],
                    generation=generated,
                    automatic_generation=automatic['modes'][name],
                )
            rows.append(dict(group=folder.name, beatmap_id=item['beatmap_id'], title=item['title'],
                             keys=item['keys'], audio_shift_ms=item['audio_shift_ms'],
                             reference=item['reference'], modes=modes))
    if not rows:
        raise ValueError('No completed audits found')
    if len(fingerprints) != 1:
        raise ValueError('Source/model fingerprints differ; rerun a consistent audit before publishing')
    shifts = {r['audio_shift_ms'] for r in rows}
    if len(shifts) != 1:
        raise ValueError('Mapping clock settings differ; summarize them in separate reports')
    shift = next(iter(shifts))
    groups = {}
    for group in dict.fromkeys(r['group'] for r in rows):
        subset = [r for r in rows if r['group'] == group]
        summary = dict(songs=len(subset), keys=sorted({r['keys'] for r in subset}), modes={})
        for mode in ('basic', 'deep'):
            results = [r['modes'][mode] for r in subset]
            comparisons = [m['timing_comparison'] for m in results if m['timing_comparison']]
            generated = [m['generation'] for m in results if 'validation' in m['generation']]
            auto = [m['automatic_generation'] for m in results]
            auto_generated = [m for m in auto if m['generated']]
            summary['modes'][mode] = dict(
                stable_auto_timing=sum(m['tempo']['stable'] is True for m in results),
                comparable_constant_timing=len(comparisons),
                median_mapping_phase_error_ms=round(median(abs(c['mapping_phase_error_ms']) for c in comparisons), 3) if comparisons else None,
                median_raw_candidate_f1_20ms=round(median(m['raw_attack_match']['20']['f1'] for m in results), 4),
                median_mapping_candidate_f1_20ms=round(median(m['mapping_attack_match']['20']['f1'] for m in results), 4),
                generated_on_reference_grid=len(generated),
                median_generated_attack_f1_20ms=round(median(g['attack_match']['f1'] for g in generated), 4) if generated else None,
                invalid_charts=sum(not g['validation']['valid'] for g in generated),
                off_grid_endpoints=sum(g['grid']['off_grid'] for g in generated),
                nonoptimal_attack_searches=sum(not g['selection']['attack_optimal'] for g in generated),
                ln_maximum_unproven=sum(not g['selection']['maximum_proven'] for g in generated),
                automatic_generation=dict(
                    generated=len(auto_generated), blocked=len(auto)-len(auto_generated),
                    median_attack_f1_20ms=round(median(g['attack_match']['f1'] for g in auto_generated), 4) if auto_generated else None,
                    invalid_charts=sum(not g['validation']['valid'] for g in auto_generated),
                    off_grid_endpoints=sum(g['grid']['off_grid'] for g in auto_generated),
                ),
            )
        groups[group] = summary
    return dict(
        purpose='Development comparison with human mapping choices; not exhaustive transcription accuracy or player evaluation',
        audio_shift_ms=shift,
        protocol=dict(
            clock=f'Raw audio candidates and an explicit {shift:g}ms mapping shift are reported separately',
            matching='Unique onset timestamps, greedy one-to-one match within 20/40ms; chord size and lanes are not scored',
            generation='Uses known human reference BPM/Offset, default preferences and 1/4 beat grid; changing reference tempo or phase is skipped',
            automatic_generation='Uses automatic BPM/Offset and division without reference assistance; unstable timing stops generation and remains in the denominator',
            leakage='Development sets have been examined during iteration; this is not an unseen benchmark',
            runtime='Analysis caches are reused; these run times must not be presented as cold inference speed',
        ),
        source_fingerprint=next(iter(fingerprints.values())),
        audio_models=dict(basic_pitch=base_model,separation_and_beats=deep_models,
            verification='Every deep snapshot records matching separator/beat SHA256; current Basic Pitch file matches the pinned manifest. Basic snapshots did not separately record a model checksum.'),
        groups=groups, songs=rows,
    )


def markdown(report):
    lines = ['# 真实音频审计 · alpha.4', '',
        '这里比较的是程序检测 / 选择的起音与人工谱面的取舍。人工谱面没有标出所有可听声音，因此这些指标**不是完整采音准确率**，也不能证明游玩手感。', '',
        f"共 {len(report['songs'])} 首，按本地来源目录分组。开发中查看过这些结果，不称为独立盲测。用户另行提供的《有点甜》单独记录，不公开音频。", '',
        f"候选匹配使用唯一的起音时间，以 20ms 容差一对一匹配，不评价和弦数量、轨道或音尾。原始采音与提前 {report['audio_shift_ms']:g}ms 后的制谱时间分别记录。下面是**制谱时间**下的候选匹配 F1 中位数：", '',
        '| 集合 | 歌曲 | 快速采音 | 四声部采音 |', '|---|---:|---:|---:|']
    for name, group in report['groups'].items():
        b, d = group['modes']['basic'], group['modes']['deep']
        lines.append(f"| {name} | {group['songs']} | {b['median_mapping_candidate_f1_20ms']:.3f} | {d['median_mapping_candidate_f1_20ms']:.3f} |")
    lines += ['', '四声部候选通常找回更多人工起音，但候选数量也明显增多，精度下降；不能声称全面优于直接混音检测。程序仍提供快速采音对照。', '',
        '下面对固定网格歌曲生成完整候选。**这里使用人工谱面的已知 BPM / Offset**，用于单独比较采音取舍；不是端到端自动生成成绩。含变速或红线相位重置的歌曲跳过，不把多段 Timing 压成一条红线。', '',
        '| 集合 | 可生成歌曲 | 快速 / 深度起音 F1 中位数 | 深度自动 Timing 稳定 | 可比较相位误差中位数 |',
        '|---|---:|---:|---:|---:|']
    for name, group in report['groups'].items():
        b, d = group['modes']['basic'], group['modes']['deep']
        lines.append(f"| {name} | {d['generated_on_reference_grid']} | {b['median_generated_attack_f1_20ms']:.3f} / {d['median_generated_attack_f1_20ms']:.3f} | {d['stable_auto_timing']}/{group['songs']} | {d['median_mapping_phase_error_ms']:.2f}ms（{d['comparable_constant_timing']} 首） |")
    lines += ['', '相位误差只比较恒定网格且拍速相同或为半速 / 倍速的歌曲，表示相对人工红线的差异；它不是物理波形真值。自动 Timing 不稳定的结果仍完整保留在 JSON，包括真实小数被取整的候选和检测错误。', '',
        '**自动路径单独统计**：下表完全使用检测出的 BPM、Offset 和细分，没有提供人工红线；不稳定就停止生成。F1 只统计成功生成的子集，不能拿它与上表不同歌曲范围的中位数直接比较。被拦住的歌曲仍计入总数，自动通过也不等于音乐正确。', '',
        '| 集合 | 快速自动生成 / 总数 | 快速起音 F1 | 深度自动生成 / 总数 | 深度起音 F1 |',
        '|---|---:|---:|---:|---:|']
    for name, group in report['groups'].items():
        b, d = (group['modes'][m]['automatic_generation'] for m in ('basic', 'deep'))
        score = lambda x: '—' if x is None else f'{x:.3f}'
        lines.append(f"| {name} | {b['generated']}/{group['songs']} | {score(b['median_attack_f1_20ms'])} | {d['generated']}/{group['songs']} | {score(d['median_attack_f1_20ms'])} |")
    lines += ['',
        '每个实际生成结果都检查轨道冲突，并回读导出网格。所有记录、LN 长度分布、求解器是否证实最优和逐歌曲失败情况见 [audio-audit.json](audio-audit.json)。求解超时时会使用已找到且通过约束检查的方案，不能把它称为最优解。', '',
        '源码与权重指纹：`'+report['source_fingerprint']['sha256']+'`。审计复用了模型缓存，不将本轮运行秒数当作首次完整分析耗时。', '',
        '复跑：在本地准备合法配对音源 / `.osu`，执行 `scripts/evaluate_audio.py`；完成同一源码与模型的所有集合后运行 `scripts/report_audio_audit.py`。原始素材和研究缓存不随报告发布。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders', nargs='+', type=Path, default=[Path('test-data')/n for n in ('audio-audit', 'audio-holdout', 'user-audio-audit')])
    parser.add_argument('--output', type=Path, default=Path('docs/audio-audit.json'))
    args = parser.parse_args()
    report = summarize(args.folders)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output.with_suffix('.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps(report['groups'], ensure_ascii=False, indent=2))
