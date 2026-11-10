"""Explainable arrangement options grounded in the current song's detections.

Corpus statistics calibrate density ranges, not note timestamps or difficulty
stars. Recommendations are proposals: generating still goes through the same
audio evidence, timing confirmation and occupancy constraints.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .arrange import parameters

MODES = {
    'melody': ('旋律与放手', '较少声部，突出持续音、乐句和清晰收尾', .42, .80, .18, .90),
    'hybrid': ('均衡 Hybrid', '旋律持续与节奏短键配合，按段落调整密度', .65, .62, .55, .65),
    'layers': ('密集多声部', '保留更多伴奏层次，增加短键与 LN 的交错', .88, .45, .75, .45),
}
EVIDENCE_THRESHOLD = .35


def plan_evidence(analysis, plan):
    """Use the same evidence pool for the proposal and its actual generation."""
    candidates=analysis['candidates'];phrases=analysis.get('phrase_candidates',[])
    if plan.get('source')!='local-audio-recommendation':return candidates,phrases,0
    selected=[n for n in candidates if n.get('strength',1)>=EVIDENCE_THRESHOLD]
    return selected,[n for n in phrases if n.get('strength',1)>=EVIDENCE_THRESHOLD],len(candidates)-len(selected)


def evidence_signature(analysis):
    content={k:analysis.get(k) for k in ('revision','duration','candidates','phrase_candidates','separation','structure')}
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_plan(plan, analysis, keys, bpm):
    if plan.get('source') != 'local-audio-recommendation':
        return
    if plan.get('keys') != keys:
        raise ValueError('目标键数已改变，请重新选择本地编排建议，或清除分段计划')
    if not math.isclose(plan.get('bpm', 0), bpm, abs_tol=.001):
        raise ValueError('BPM 已改变，请重新选择本地编排建议，或清除分段计划')
    if plan.get('evidence_signature') != evidence_signature(analysis):
        raise ValueError('采音证据已改变，请重新选择本地编排建议，或清除分段计划')


def load_profiles(resources):
    path=Path(resources)/'assets/arrangement-profiles.json'
    if not path.exists():return None
    data=json.loads(path.read_text(encoding='utf-8'))
    return data if data.get('revision')==1 else None


def adjusted_segments(plan, current):
    """Respect later global slider edits without flattening quiet sections."""
    if plan.get('source')!='local-audio-recommendation':
        return plan.get('segments',[])
    base=plan['preferences'];result=[]
    for segment in plan.get('segments',[]):
        local=segment['preferences'].copy()
        for key in ('density','ln_ratio','ln_coordination'):
            if key not in local or key not in base:continue
            if key=='density':
                local[key]=int(np.clip(math.ceil(local[key]*current[key]/max(1,base[key])),1,24))
            elif key=='ln_ratio':
                # An evidence-empty section must not receive LN quota from a
                # global edit. The selector still needs actual sustained notes.
                local[key]=round(float(np.clip(local[key]*current[key]/max(.01,base[key]),0,1)),3)
            else:
                local[key]=round(float(np.clip(local[key]+current[key]-base[key],0,1)),3)
        result.append(dict(segment,preferences=local))
    return result


def summarize(notes, duration_ms, minimum):
    # Multiple pitches on a strong attack are useful layers, but are not that
    # many independent rhythmic attacks. Keep both counts explicitly.
    notes=[n for n in notes if n.get('strength',1)>=EVIDENCE_THRESHOLD]
    if not notes:return dict(candidates=0,attacks_per_second=0.,layer_nps95=0.,sustain_fraction=0.,voices={})
    starts=np.sort([n['start'] for n in notes])
    bins={round(n['start']/45) for n in notes}
    checkpoints=np.arange(float(starts[0]),float(starts[-1])+1,500)
    counts=np.searchsorted(starts,checkpoints+1000)-np.searchsorted(starts,checkpoints)
    return dict(candidates=len(notes),attacks_per_second=round(len(bins)/max(1,duration_ms/1000),3),
        layer_nps95=round(float(np.percentile(counts,95)),2),
        sustain_fraction=round(sum(n['end']-n['start']>=minimum for n in notes)/len(notes),4),
        voices=dict(Counter(n.get('voice','mix') for n in notes)))


def pattern_opportunities(notes, beat):
    """Identify bounded gestures from detected intervals, without extending them."""
    notes=sorted((n for n in notes if n.get('strength',1)>=EVIDENCE_THRESHOLD),key=lambda n:n['start'])
    if not notes:return []
    result=[]
    def add(name,reason):
        if not any(r['name']==name for r in result):result.append(dict(name=name,reason=reason))
    holds=[n for n in notes if n['end']-n['start']>=max(100,beat*.25)]
    taps=[n for n in notes if n['end']-n['start']<max(100,beat*.25)]
    if any(n['end']-n['start']>=beat*2 for n in holds):
        add('持续音锚点','存在至少两拍的检测持续区间，可保留长按并在其他轨安排伴奏')
    tap_times=np.array([n['start'] for n in taps])
    if any(np.any((tap_times>n['start'])&(tap_times<n['end'])) for n in holds):
        add('长按中的短键','检测到持续音内部的其他起音，可比较分手配合与同手 Hybrid')
    ending={}
    for n in holds:
        ending.setdefault(round(n['end']/40),[]).append(n)
    groups=[g for g in ending.values() if len(g)>1]
    if groups:
        add('共同收尾','多个检测持续区间在约 40ms 内结束；只采用拍线量化后本来同尾的组')
    if any(max(n['start'] for n in g)-min(n['start'] for n in g)>beat*.25 for g in groups):
        add('分次进入、共同放手','存在不同起音而尾端接近的持续层，可保留起音差异组织放手')
    streams=sorted({(n.get('voice','mix'),n.get('instrument_id','')) for n in notes})
    for voice,instrument_id in streams:
        row=[n for n in notes if n.get('voice','mix')==voice
             and n.get('instrument_id','')==instrument_id and n.get('source')=='pitch']
        if len(row)<4:continue
        times=np.array([n['start'] for n in row]);gaps=np.diff(times)
        intervals=gaps[gaps>40]
        if len(intervals)>=4 and np.median(abs(intervals-np.median(intervals)))<max(20,beat*.06):
            add('重复节奏动机','同一检测声部的起音间隔较稳定，适合保持连贯布局并按乐句变化')
        # Simultaneous chord pitches are not a melodic ascent. Only consider
        # isolated attacks when suggesting a direction-following gesture.
        isolated=[n for i,n in enumerate(row) if (i==0 or n['start']-row[i-1]['start']>40)
                  and (i==len(row)-1 or row[i+1]['start']-n['start']>40)]
        pitches=np.array([n.get('pitch',60) for n in isolated]);movement=np.diff(pitches)
        if len(movement)>=3 and any(np.all(x>0) or np.all(x<0) for x in
                                    (movement[i:i+3] for i in range(len(movement)-2))):
            add('旋律方向','检测到连续上行或下行音高；可提高轨道连贯，避免无理由跳轨')
        for left,right in zip(row,row[1:]):
            gap=right['start']-left['end']
            if min(left['end']-left['start'],right['end']-right['start'])>=max(100,beat*.25) and 0<gap<=min(150,beat*.3):
                add('反键接续候选','有两条持续音之间的短正间隙；可试同轨接续，不填长间隙')
            if left['end']-left['start']>=beat*.25 and right['end']-right['start']<beat*.25 and 0<gap<=beat*.3:
                add('放手后短键','检测到 LN 尾端附近的新短音，可比较分轨与尾接短键')
            if left['end']-left['start']<beat*.25 and right['end']-right['start']>=beat*.25 and 0<right['start']-left['start']<=beat*.3:
                add('短键进入长按','短音后有持续音，可组织 Shield 或分轨进入，保留真实起止')
    starts={}
    for n in holds:starts.setdefault(round(n['start']/40),[]).append(n)
    if any(max(n['end'] for n in g)-min(n['end'] for n in g)>=beat*.5 for g in starts.values() if len(g)>1):
        add('共同按下、分次放手','检测到同时开始但长度不同的持续层，可按释放顺序安排双手')
    if holds and len(taps)<len(holds)*.2:
        add('持续层之间的协调','短音较少，可用不同进入与放手时刻表达层次，减少无依据的添音')
    if not holds:
        add('短音节奏','当前区间没有足够持续证据，采用交替或和弦短键')
    return result


def recommend(analysis, current, resources, bpm=None):
    if not analysis or not analysis.get('candidates'):
        raise ValueError('请先分析音频，编排建议需要本曲的起音与持续音证据')
    prefs=parameters(current);keys=prefs['keys']
    bpm=float(bpm if bpm is not None else analysis.get('tempo',{}).get('bpm',120))
    if not math.isfinite(bpm) or not 20<=bpm<=400:
        raise ValueError('建议使用的 BPM 必须在 20–400 之间')
    beat=60000/bpm
    minimum=int(np.clip(math.floor(beat*.25/20)*20,100,300))
    profile=load_profiles(resources)
    calibration=(profile or {}).get('keys',{}).get(str(keys),{})
    corpus_bands=calibration.get('active_nps95',{})
    notes=analysis['candidates'];length=round(analysis['duration']*1000)
    summary=summarize(notes,length,minimum)
    if not summary['candidates']:
        raise ValueError('没有足够强的检测候选，请先检查采音；暂不提供自动编排建议')
    evidence_ceiling=max(1,summary['layer_nps95'])
    # New analyses have acoustic-change sections. Legacy projects keep their
    # old windows until the user runs structure detection; neither implies
    # semantic verse/chorus recognition.
    sections=analysis.get('segments') or [dict(start_ms=0,end_ms=length)]
    measured=[]
    for s in sections:
        a,b=s['start_ms'],s['end_ms']
        local_notes=[n for n in notes if a<=n['start']<b]
        measured.append((a,b,summarize(local_notes,b-a,minimum),pattern_opportunities(local_notes,beat)))
    options=[]
    for index,(name,(label,description,fraction,ratio,coordination,release)) in enumerate(MODES.items()):
        reference_cap=float(corpus_bands.get(('10','50','75')[index],(5,9,14)[index]))
        if name=='melody':
            # Low-density LN is not calibrated against a median dominated by
            # ranked high-speed tap charts. These are draft workload limits,
            # not official difficulty predictions.
            ln_reference=calibration.get('styles',{}).get('ln_majority',{})
            if ln_reference.get('songs',0)>=5:
                reference_cap=min(reference_cap,float(ln_reference['active_nps95']['25']))
            reference_cap=min(reference_cap,float(keys))
        density=int(np.clip(math.ceil(min(evidence_ceiling*fraction,reference_cap)),1,24))
        ln_ratio=round(min(ratio,summary['sustain_fraction']+.10),2)
        settings=dict(density=density,ln_ratio=ln_ratio,min_ln=minimum,
            chord_size=min(keys,(2,3,5)[index]),max_held=min(keys-1,(2,3,5)[index]),
            phrase_ln=(.8,.6,.4)[index],focus='pitched' if name=='melody' else 'balanced',
            ln_coordination=coordination,ln_release=release,ln_inverse=0.)
        if name=='melody':
            settings.update(vocals_weight=1.5,bass_weight=.8,drums_weight=.55,other_weight=.7,guitar_weight=1.,piano_weight=1.2)
        else:
            settings.update({v+'_weight':1. for v in ('vocals','bass','drums','other','guitar','piano')})
        segments=[]
        descriptions=[]
        for a,b,local,opportunities in measured:
            local_density=int(np.clip(math.ceil(min(reference_cap,local['layer_nps95']*fraction)),1,24))
            sustainable=local['sustain_fraction']
            local_ratio=round(min(ratio,sustainable+.10),2) if local['candidates'] else 0.
            if sustainable>=.6 and local['attacks_per_second']<=5:
                design='持续层与分组放手'
                local_coordination=max(0,coordination-.1)
            elif local['attacks_per_second']>=7:
                design='交替短键与伴奏层次'
                local_coordination=coordination
            elif sustainable>=.35:
                design='LN 与短键 Hybrid'
                local_coordination=coordination
            elif local['candidates']:
                design='清晰起音，少量 LN'
                local_coordination=max(0,coordination-.15)
            else:
                design='缺少可靠证据，保留空白'
                local_coordination=0.
            local_settings=dict(density=local_density,ln_ratio=local_ratio,ln_coordination=round(local_coordination,2))
            segments.append(dict(start_ms=a,end_ms=b,preferences=local_settings))
            descriptions.append(dict(start_ms=a,end_ms=b,design=design,evidence=local,density=local_density,
                                     ln_ratio=local_ratio,opportunities=opportunities))
        options.append(dict(id=name,title=label,description=description,preferences=settings,segments=segments,
            section_reasons=descriptions,
            explanation=f'{label}：根据本曲可靠检测层次与持续区间安排。全局密度上限 {density}/秒，局部按证据调整；不是 osu! 星级。'))
    best='melody' if summary['sustain_fraction']>=.55 and summary['attacks_per_second']<6 else 'hybrid'
    signature=evidence_signature(analysis)
    context=hashlib.sha256(json.dumps([signature,keys,bpm,(profile or {}).get('manifest_sha256')]).encode()).hexdigest()
    return dict(revision=1,evidence_signature=signature,context_signature=context,keys=keys,bpm=bpm,
        suggested=best,summary=summary,options=options,
        reference=dict(available=bool(calibration),charts=calibration.get('charts',0),
            songs=calibration.get('songs',0),ln_charts=calibration.get('styles',{}).get('ln_majority',{}).get('charts',0),
            manifest_sha256=(profile or {}).get('manifest_sha256')),
        notice=('新分段模式以逐段声部方案决定长短，LN 数值是表达倾向，不是强制比例。只取强度≥0.35的检测，仍需试听误检。'
                if analysis.get('structure') else '方案只取强度≥0.35的检测，描述编排负担而非官方星级。仍需试听误检；不会补音或为凑 LN 比例拉长。清除方案后可自行取舍较弱声音。'))
