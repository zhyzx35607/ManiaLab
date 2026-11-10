import copy
import json
from pathlib import Path

import pytest

from manialab.arrange import parameters
from manialab.recommendations import adjusted_segments, pattern_opportunities, recommend, validate_plan, plan_evidence
from manialab.server import create_app
from manialab.storage import Projects

ROOT=Path(__file__).parents[1]


def evidence(dense=False):
    notes=[]
    for time in range(1000,15000,125 if dense else 1000):
        for pitch in ((48,60,72) if dense else (60,)):
            notes.append(dict(id=f'{time}:{pitch}',start=time,end=time+(200 if dense else 800),
                pitch=pitch,source='pitch',voice='vocals',strength=.8,lane=-1))
    return dict(revision=8,duration=30,candidates=notes,phrase_candidates=[],
        tempo=dict(bpm=120,offset=0,stable=True,division=4),
        segments=[dict(start_ms=0,end_ms=15000),dict(start_ms=15000,end_ms=30000)])


@pytest.mark.parametrize('keys',range(6,11))
def test_recommendations_respond_to_sound_and_preserve_inputs(tmp_path,keys):
    sparse,dense=evidence(),evidence(True)
    original=copy.deepcopy(sparse)
    a=recommend(sparse,parameters({'keys':keys}),tmp_path)
    b=recommend(dense,parameters({'keys':keys}),tmp_path)
    assert sparse==original
    assert a['suggested']=='melody'
    assert b['summary']['attacks_per_second']>a['summary']['attacks_per_second']
    assert b['options'][2]['preferences']['density']>a['options'][2]['preferences']['density']
    assert [o['preferences']['density'] for o in b['options']]==sorted(o['preferences']['density'] for o in b['options'])
    for option in a['options']+b['options']:
        assert parameters({'keys':keys}|option['preferences'])['keys']==keys
        for segment in option['segments']:
            parameters({'keys':keys}|segment['preferences'])
        assert option['segments'][1]['preferences']['ln_ratio']==0
        assert option['section_reasons'][1]['design']=='缺少可靠证据，保留空白'
    assert not a['reference']['available']


def test_corpus_calibration_uses_current_key_only(tmp_path):
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/arrangement-profiles.json').write_text(json.dumps(dict(revision=1,
        manifest_sha256='fixture',keys={'10':dict(charts=100,songs=50,active_nps95={'10':2,'50':4,'75':6})})))
    a=recommend(evidence(True),parameters(),tmp_path)
    assert [o['preferences']['density'] for o in a['options']]==[2,4,6]
    assert a['reference']['charts']==100
    assert not recommend(evidence(True),parameters({'keys':6}),tmp_path)['reference']['available']


def test_stale_audio_recommendation_is_rejected_and_current_plan_is_saved(tmp_path):
    store=Projects(tmp_path/'projects');pid,_=store.create()
    state=dict(id=pid,title='Recommendations',mode='audio',settings=parameters(),active=-1,variants=[])
    state['analysis']=evidence();store.save(state)
    app=create_app(ROOT,tmp_path);client=app.test_client();h={'X-ManiaLab-Token':app.config['TOKEN']}
    path=f"/api/projects/{state['id']}"
    result=client.post(path+'/recommendations',json={'settings':{'keys':8}},headers=h)
    assert result.status_code==200
    signature=result.json['evidence_signature']
    context=result.json['context_signature']
    denied=client.post(path+'/apply-recommendation',json={'choice':'melody','evidence_signature':'stale'},headers=h)
    assert denied.status_code==400
    original=copy.deepcopy(state['analysis'])
    chosen=client.post(path+'/apply-recommendation',json={'choice':'melody','settings':{'keys':8},
        'evidence_signature':signature,'context_signature':context},headers=h)
    assert chosen.status_code==200,chosen.json
    assert chosen.json['analysis']==original
    assert chosen.json['settings']['keys']==8
    assert chosen.json['plan']['source']=='local-audio-recommendation'
    assert len(chosen.json['plan']['segments'])==2
    assert store.load(state['id'])['plan']['choice']=='melody'
    stale=client.post(path+'/apply-recommendation',json={'choice':'melody','settings':{'keys':10},
        'evidence_signature':signature,'context_signature':context},headers=h)
    assert stale.status_code==400 and '键数' in stale.json['error']
    app.extensions['executor'].shutdown()


def test_empty_or_invalid_tempo_cannot_create_a_plan(tmp_path):
    with pytest.raises(ValueError):recommend({},parameters(),tmp_path)
    for bpm in (0,float('nan'),1000):
        with pytest.raises(ValueError):recommend(evidence(),parameters(),tmp_path,bpm)


def test_global_edits_scale_local_plan_without_filling_silence():
    plan=dict(source='local-audio-recommendation',preferences=dict(density=6,ln_ratio=.5,ln_coordination=.5),
        segments=[dict(start_ms=0,end_ms=10000,preferences=dict(density=4,ln_ratio=.4,ln_coordination=.4)),
                  dict(start_ms=10000,end_ms=20000,preferences=dict(density=1,ln_ratio=0,ln_coordination=0))])
    original=copy.deepcopy(plan)
    segments=adjusted_segments(plan,parameters({'density':12,'ln_ratio':.75,'ln_coordination':.8}))
    assert segments[0]['preferences']==dict(density=8,ln_ratio=.6,ln_coordination=.7)
    assert segments[1]['preferences']['ln_ratio']==0
    assert plan==original


def test_motifs_require_supported_intervals_and_preserve_them():
    notes=[dict(start=0,end=2000,pitch=60,voice='piano',source='pitch'),
           dict(start=500,end=2010,pitch=64,voice='piano',source='pitch'),
           dict(start=1000,end=1000,pitch=48,voice='drums',source='transient')]
    original=copy.deepcopy(notes)
    names={r['name'] for r in pattern_opportunities(notes,500)}
    assert {'持续音锚点','共同收尾','分次进入、共同放手','长按中的短键'}<=names
    assert '反键接续候选' not in names
    assert notes==original
    chord=[dict(start=1000,end=1200,pitch=p,source='pitch',voice='piano') for p in (48,55,60,64)]
    assert '旋律方向' not in {r['name'] for r in pattern_opportunities(chord,500)}


def test_other_instrument_interleaving_does_not_invent_a_melodic_line_or_rhythm():
    notes=[dict(start=i*500,end=i*500+100,pitch=60+i,source='pitch',voice='other',
                instrument_id=f'yourmt3:program:{40 if i%2 else 0}') for i in range(6)]
    names={r['name'] for r in pattern_opportunities(notes,500)}
    assert not {'旋律方向','重复节奏动机'} & names
    same_instrument=[dict(n,instrument_id='yourmt3:program:40') for n in notes]
    assert {'旋律方向','重复节奏动机'} <= {r['name'] for r in pattern_opportunities(same_instrument,500)}


def test_changed_tempo_keys_and_evidence_require_reselection():
    analysis=evidence()
    plan=dict(source='local-audio-recommendation',keys=10,bpm=120,
        evidence_signature=recommend(analysis,parameters(),ROOT)['evidence_signature'])
    validate_plan(plan,analysis,10,120)
    with pytest.raises(ValueError,match='BPM'):validate_plan(plan,analysis,10,240)
    with pytest.raises(ValueError,match='键数'):validate_plan(plan,analysis,8,120)
    analysis['candidates'][0]['end']+=125
    with pytest.raises(ValueError,match='采音'):validate_plan(plan,analysis,10,120)


def test_recommendation_and_generation_use_same_strong_evidence_pool(tmp_path):
    analysis=evidence();weak=dict(analysis['candidates'][0],id='weak',start=22000,end=23000,strength=.2)
    analysis['candidates'].append(weak);original=copy.deepcopy(analysis)
    proposal=recommend(analysis,parameters(),tmp_path)
    assert proposal['options'][0]['section_reasons'][1]['design']=='缺少可靠证据，保留空白'
    notes,phrases,omitted=plan_evidence(analysis,dict(source='local-audio-recommendation'))
    assert omitted==1 and all(n['id']!='weak' for n in notes) and analysis==original
    assert plan_evidence(analysis,{})[0]==analysis['candidates']
    with pytest.raises(ValueError,match='足够强'):recommend(dict(analysis,candidates=[weak]),parameters(),tmp_path)
