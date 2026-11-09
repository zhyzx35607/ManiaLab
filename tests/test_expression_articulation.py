import copy

import pytest

from manialab.arrange import parameters
from manialab.domain import Note
from manialab.expression import propose_expression,expression_evidence
from manialab.expression_articulation import attack_runs,articulation_policy,source_activity
from tests.test_structure_expression import analysis
from scripts.research_expression_policies import propose


def piano(at,end,pitch=60,id=None,reason='release_regression'):
    return Note(at,end,pitch=pitch,id=id or str(at),source='pitch',voice='piano',release_reason=reason)


def test_repeated_decaying_attacks_keep_rhythm_and_supported_final_sustain():
    notes=[piano(t,t+1200,id=str(i)) for i,t in enumerate((1000,1250,1500,3000,3250,3500))]
    a=analysis(notes);original=copy.deepcopy(a);settings=parameters()
    plan=propose(a,settings,articulated=True)
    assert plan['sections'][0]['voices']['piano']['mode']=='articulation'
    result,_,trace=expression_evidence(a,plan,settings)
    assert [n.start for n in result]==[n.start for n in notes]
    assert [n.hold for n in result]==[False,False,True,False,False,True]
    assert [n.end for n in result if n.hold]==[2700,4700]
    assert a==original
    assert trace['0']['articulation']['run']==trace['2']['articulation']['run']
    assert trace['3']['articulation']['run']!=trace['0']['articulation']['run']


def test_chord_clusters_do_not_chain_away_a_distinct_attack_or_spread_layers():
    notes=[piano(t,t+1100,pitch=p,id=f'{t}:{p}') for t in (1000,1500,2000) for p in (48,60,64)]
    rules=articulation_policy(notes)
    assert all(rules[n.id]['tap']==(n.start<2000) for n in notes)
    groups,_=attack_runs([piano(1000,1100),piano(1025,1125),piano(1050,1150)])
    assert [[n.start for n in group] for group in groups[0]]==[[1000,1025],[1050]]


def test_separated_sustains_and_explicit_manual_mode_keep_original_intervals():
    notes=[piano(t,t+700) for t in (1000,2000,3000)]
    assert not any(r['tap'] for r in articulation_policy(notes).values())
    a=analysis([piano(t,t+1500) for t in (1000,1250,1500)])
    settings=parameters();plan=propose_expression(a,settings)
    plan['sections'][0]['voices']['piano']['mode']='sustain'
    result,_,_=expression_evidence(a,plan,settings)
    assert [n.end for n in result]==[2500,2750,3000]
    for ratio in (0,1):
        plan=propose_expression(a,parameters({'ln_ratio':ratio}))
        assert plan['sections'][0]['voices']['piano']['mode']==('onsets' if ratio==0 else 'sustain')


@pytest.mark.parametrize('reason',['duration_limit','input_boundary','forced_segment_boundary','unresolved_or_postprocessed'])
def test_unobserved_tails_cannot_change_source_priority_or_trigger_articulation(reason):
    rows=[piano(t,t+9000,reason=reason) for t in (1000,1250,1500)]
    assert not any(v['tap'] for v in articulation_policy(rows).values())
    activity=source_activity(rows,0,12000)
    assert activity['activity_coverage']==activity['attack_coverage']
    assert activity['observed_interval_coverage']==0
    a=analysis([piano(1000,11000,reason=reason),
        Note(2000,5000,voice='other',source='pitch',id='observed',release_reason='explicit_model_note_off')])
    assert propose(a,parameters())['sections'][0]['main_voice']=='other'


def test_section_activity_counts_exact_clipped_union_and_short_attacks():
    rows=[piano(990,4000),piano(990,4000,pitch=64,id='upper')]
    values=source_activity(rows,0,1000)
    assert values['activity_coverage']==.01 and values['attack_coverage']==.01
    assert source_activity([piano(0,1000)],0,1000)['activity_coverage']==1.


def test_articulation_does_not_make_group_tail_from_missing_release():
    rows=[piano(1000,2500),piano(1250,2750),piano(1500,10000,reason='input_boundary')]
    a=analysis(rows);settings=parameters();plan=propose(a,settings,articulated=True)
    result,_,trace=expression_evidence(a,plan,settings)
    assert all(not n.hold for n in result)
    assert trace['1500']['reason']=='unobserved_release'
    assert a['candidates'][2]['end']==10000


def test_saved_articulation_policy_roundtrips_and_exports_through_project_api(tmp_path):
    import io
    import json
    from pathlib import Path
    from manialab.server import create_app
    from manialab.storage import Projects
    from manialab.domain import parse_osu
    from tests.test_server import wait_job
    root=Path(__file__).parents[1]
    app=create_app(root,tmp_path);client=app.test_client()
    headers={'X-ManiaLab-Token':app.config['TOKEN']}
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];url=f'/api/projects/{pid}'
    state['analysis']=analysis([piano(t,t+1250) for t in (1000,1250,1500)])
    state['analysis']['revision']=10
    state['analysis']['tempo'].update(stable=True,division=4)
    original=json.loads(json.dumps(state['analysis']['candidates']))
    Projects(tmp_path/'projects').save(state)
    try:
        response=client.post(url+'/expression',json={'action':'automatic'},headers=headers)
        assert response.status_code==200
        plan=response.json['expression']
        assert plan['sections'][0]['voices']['piano']['mode']=='sustain'
        plan['sections'][0]['voices']['piano']['mode']='articulation'
        assert client.post(url+'/expression',json={'expression':plan},headers=headers).status_code==200
        payload=dict(settings={'keys':10,'density':12},bpm=120,offset=0,division=4,audio_shift_ms=0,timing_confirmed=True)
        assert wait_job(client,client.post(url+'/generate',json=payload,headers=headers).json)['status']=='done'
        current=client.get(url).json
        assert current['expression']['source']=='manual'
        assert current['analysis']['candidates']==original
        parsed=parse_osu(client.get(url+'/export/osu').data.decode())
        assert [(n.start,n.end) for n in parsed['notes']]==[(1000,1000),(1250,1250),(1500,2750)]
        # Existing saved sustain intent remains independently selectable.
        plan['sections'][0]['voices']['piano']['mode']='sustain'
        assert client.post(url+'/expression',json={'expression':plan},headers=headers).status_code==200
        assert wait_job(client,client.post(url+'/generate',json=payload,headers=headers).json)['status']=='done'
        parsed=parse_osu(client.get(url+'/export/osu').data.decode())
        assert [(n.start,n.end) for n in parsed['notes']]==[(1000,2250),(1250,2500),(1500,2750)]
    finally:
        app.extensions['executor'].shutdown()
