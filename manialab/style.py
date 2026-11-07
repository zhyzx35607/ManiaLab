"""Describe a reference chart; numerical style is not a judgement of its quality."""
from collections import Counter
import math
import re

import numpy as np

from .domain import validate


def describe_chart(parsed):
    notes = parsed['notes']
    keys = parsed['keys']
    if not 6 <= keys <= 10:
        raise ValueError('风格参考需要原生 6–10K mania 谱面')
    report = validate(notes, keys)
    if not report['valid']:
        raise ValueError('参考谱面存在轨道冲突：'+report['issues'][0])
    if len(notes) < 20:
        raise ValueError('参考谱面至少需要 20 个物件，元数据占位图不能作为风格参考')
    starts = np.sort(np.array([n.start for n in notes]))
    first, last = int(starts[0]), max(n.end for n in notes)
    lengths = [n.end-n.start for n in notes if n.hold]
    # Half-second sampling, excluding empty windows; report both the observed
    # peak and a robust bound so an isolated burst does not set every passage.
    checkpoints = np.arange(first, last+1, 500)
    counts = np.searchsorted(starts, checkpoints+1000, side='left')-np.searchsorted(starts, checkpoints, side='left')
    active = counts[counts > 0]
    chords = list(Counter(n.start for n in notes).values())
    density = max(1, min(24, math.ceil(float(np.percentile(active, 95)))))
    minimum = min(2000, max(100, int(math.floor(float(np.percentile(lengths,10))/20)*20))) if lengths else 240
    preferences = dict(density=density, ln_ratio=round(report['ln']/len(notes), 3), min_ln=minimum,
                       chord_size=min(keys, max(1, math.ceil(float(np.percentile(chords,98))))),
                       max_held=min(keys, max(1, report['max_held'])))
    metadata = parsed['metadata']
    label = ' '.join(str(metadata.get(k,'')) for k in ('Title','Version','Tags'))
    converted = bool(re.search(r'inverse|conversion|convert|转谱|转换|倒置', label, re.I))
    warnings = []
    if converted:
        warnings.append('标题或标签包含转换 / Inverse 标记：只参考数量分布，不把长条长度当作音乐持续音真值。')
    if int(active.max()) > 24:
        warnings.append('参考图局部密度超过当前 24 物件/秒上限，建议值已限制到支持范围。')
    return dict(title=metadata.get('Title','Untitled'), creator=metadata.get('Creator','Unknown'),
                version=metadata.get('Version',''), keys=keys, objects=len(notes),
                ln_ratio=report['ln_ratio'], max_held=report['max_held'],
                duration_ms=last-first, average_nps=round(len(notes)/max(1,(last-first)/1000),2),
                active_nps95=round(float(np.percentile(active,95)),2), peak_nps=int(active.max()),
                ln_duration_ms={str(q):round(float(np.percentile(lengths,q))) for q in (10,50,90,100)} if lengths else {},
                preferences=preferences, warnings=warnings,
                notice='只提取密度、LN 比例及占轨规模，不复制旋律、音符时间或轨道，也不代表难度或制谱质量。')
