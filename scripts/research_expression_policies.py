"""Non-default source-activity and expression proposals for fixed comparisons."""
from manialab.arrange import parameters
from manialab.domain import note_from
from manialab.expression import propose_expression
from manialab.expression_articulation import source_activity


def propose(analysis,settings,*,articulated=False,independent=False):
    plan=propose_expression(analysis,settings)
    p=parameters(settings);previous=None
    for section in plan['sections']:
        a,b=section['start_ms'],section['end_ms']
        local=[note_from(n) for n in analysis['candidates'] if a<=n['start']<b and n.get('strength',0)>=.35]
        scores={}
        for voice,policy in section['voices'].items():
            policy['activity']=source_activity([n for n in local if n.voice==voice],a,b)
            if p.get(voice+'_weight',1.)>0:scores[voice]=policy['activity']['activity_coverage']
        melodic={v:s for v,s in scores.items() if v!='drums' and s>.02}
        if melodic.get('vocals',0)>=.18:main='vocals'
        elif previous in melodic and melodic[previous]>=max(melodic.values(),default=0)*.85:main=previous
        else:main=max(melodic,key=melodic.get) if melodic else max(scores,key=scores.get) if scores else None
        section['main_voice']=main;previous=main
        for voice,policy in section['voices'].items():
            if scores.get(voice,0)<=.02:mode='omit'
            elif voice=='drums' or p['ln_ratio']==0:mode='onsets'
            elif articulated and voice in ('piano','guitar') and p['ln_ratio']<.75:mode='articulation'
            elif independent or voice==main or p['ln_ratio']>=.75:mode='sustain'
            else:mode='onsets'
            policy.update(mode=mode,role='primary' if voice==main else 'support')
        section['reason']='研究方案：可靠音尾活动与主次取舍；连续起音与各声部持续表达单独比较，未作为默认建议。'
    plan.update(revision=2,source='research',research_policy=dict(articulated=articulated,independent=independent))
    return plan
