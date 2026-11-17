"""Evidence identity and preference helpers shared by recommendation flows."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

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
