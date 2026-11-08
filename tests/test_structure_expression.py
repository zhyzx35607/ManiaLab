import copy
import numpy as np
import pytest

from manialab.arrange import parameters, assign_lanes
from manialab.domain import Note, validate, export_osu, sections, values
from manialab.expression import propose_expression, expression_evidence, select_expression, validate_expression
from manialab.structure import detect_structure, edit_boundaries, reference_marks


def analysis(notes, length=12000):
    a = dict(duration=length/1000, tempo=dict(bpm=120, offset=0), candidates=[n.data() for n in notes])
    a['structure'] = edit_boundaries({}, [0, length], a)
    return a


def test_sections_follow_acoustic_change_not_fifteen_second_windows():
    sr=8000
    t=np.arange(sr*48)/sr
    y=np.sin(2*np.pi*np.where(t<20,220,1760)*t).astype(np.float32)*.2
    a=dict(duration=48,tempo=dict(bpm=120,offset=0),candidates=[])
    s=detect_structure(a,y,sr)
    assert any(abs(t-20000)<=1000 for t in s['boundaries'])
    assert s['boundaries'][0]==0 and s['boundaries'][-1]==48000
    assert 15000 not in s['boundaries']
    silent=detect_structure(a,np.zeros_like(y),sr)
    assert silent['boundaries']==[0,48000]


def test_bookmarks_are_separate_from_sv_and_kiai():
    m=reference_marks('[Editor]\nBookmarks: 1000, 2000\n[TimingPoints]\n0,500,4,2,1,70,1,0\n1200,-50,4,2,1,60,0,0\n2000,-100,4,2,1,70,0,1')
    assert m['bookmarks']==[1000,2000]
    assert len(m['green_lines'])==2 and m['green_lines'][1]['kiai']


def test_boundary_edits_reject_invalid_or_tiny_segments():
    a=analysis([])
    for v in ([float('nan')],['1000'],[-1],[1000,1001],[13000]):
        with pytest.raises(ValueError):edit_boundaries({},v,a)


def test_main_voice_sustains_do_not_compete_for_global_ln_quota():
    notes=[Note(t,t+650,pitch=60+i%3,strength=.6,source='pitch',voice='vocals',id=f'v{i}') for i,t in enumerate(range(1000,10000,1000))]
    notes += [Note(t,t,strength=.95,source='transient',voice='drums',id=f'd{t}') for t in range(1000,10000,500)]
    a=analysis(notes);original=copy.deepcopy(a)
    p=parameters({'density':3,'ln_ratio':.2,'chord_size':2})
    plan=propose_expression(a,p)
    evidence,meta,trace=expression_evidence(a,plan,p)
    selected,report=select_expression(evidence,meta,trace,p,500)
    vocals=[n for n in selected if n.voice=='vocals']
    assert len(vocals)==9 and all(n.end-n.start==650 for n in vocals)
    assert a==original
    output=assign_lanes(selected,p)
    assert validate(output,10)['valid']
    assert report['actual_ln_ratio']>.2


@pytest.mark.parametrize('keys',range(6,11))
def test_complete_groups_respect_future_holds_and_rolling_density(keys):
    notes=[Note(t,t+1400,pitch=48+i%12,strength=.7,source='pitch',voice=v,id=f'{v}{i}')
           for v in ('vocals','piano','bass') for i,t in enumerate(range(0,8000,250))]
    a=analysis(notes);p=parameters({'keys':keys,'ln_ratio':1,'density':9,'max_held':3,'chord_size':3})
    evidence,meta,trace=expression_evidence(a,propose_expression(a,p),p)
    selected,report=select_expression(evidence,meta,trace,p,500)
    assert selected
    assert all(n.end-n.start==1400 for n in selected)
    for n in selected:
        assert sum(n.start-1000<x.start<=n.start for x in selected)<=9
    assert validate(assign_lanes(selected,p),keys)['valid']
    assert len(report['decisions'])==len(notes)


def test_policy_is_stable_within_section_and_changes_only_explicitly():
    a=analysis([Note(t,t+700,source='pitch',voice='piano',id=str(t)) for t in (1000,2000,7000,8000)])
    a['structure']=edit_boundaries(a['structure'],[6000],a)
    plan=propose_expression(a,parameters())
    plan['sections'][1]['voices']['piano']['mode']='onsets'
    evidence,_,trace=expression_evidence(a,plan,parameters())
    assert [n.hold for n in evidence]==[True,True,False,False]
    assert trace['7000']['reason']=='section_onsets'
    a['candidates'][0]['end']+=1
    with pytest.raises(ValueError,match='改变'):validate_expression(plan,a)


def test_explicit_local_mute_and_quiet_section_hold_cap_apply_to_planned_groups():
    notes=[Note(1000,5000,pitch=p,source='pitch',voice='piano',id=str(p)) for p in (60,64)]
    notes.append(Note(8000,8500,source='pitch',voice='piano',id='later'))
    a=analysis(notes);p=parameters({'ln_ratio':1})
    evidence,meta,trace=expression_evidence(a,propose_expression(a,p),p)
    segments=[dict(start_ms=2000,end_ms=4000,preferences={'max_held':1}),
              dict(start_ms=7000,end_ms=10000,preferences={'piano_weight':0})]
    selected,report=select_expression(evidence,meta,trace,p,500,segments)
    assert 'later' not in {n.id for n in selected}
    assert sum(n.start<=3000<n.end for n in selected)<=1
    assert next(r for r in report['decisions'] if r['id']=='later')['reason']=='voice_not_selected'


def test_phrase_mode_represents_internal_pitch_without_double_counting():
    notes=[Note(1000,1400,source='pitch',voice='vocals',id='a'),Note(1400,2000,source='pitch',voice='vocals',id='b')]
    a=analysis(notes)
    a['phrase_candidates']=[dict(notes[0].data(),end=2000,phrase=True,support=['a','b'],support_coverage=1)]
    plan=propose_expression(a,parameters());plan['sections'][0]['voices']['vocals']['mode']='phrase'
    evidence,meta,trace=expression_evidence(a,plan,parameters())
    assert len(evidence)==1 and evidence[0].end==2000
    assert trace['b']['represented_by']=='a'


def test_export_marks_use_bookmarks_without_invented_green_lines():
    from pathlib import Path
    text=export_osu([Note(1000,1000,0)],10,Path(__file__).parents[1]/'vendor/mapperatorinator/template.osu',bookmarks=[1000,3000])
    assert values(sections(text)['Editor'])['Bookmarks']=='1000,3000'
    assert len([l for l in sections(text)['TimingPoints'] if l.strip()])==1


def test_rejected_phrase_does_not_claim_to_represent_audible_interior_notes():
    notes=[Note(1000,1400,source='pitch',voice='vocals',id='a',strength=.2),
           Note(1400,2000,source='pitch',voice='vocals',id='b',strength=.8)]
    a=analysis(notes)
    a['phrase_candidates']=[dict(notes[0].data(),end=2000,phrase=True,support=['a','b'],support_coverage=1)]
    p=parameters();plan=propose_expression(a,p)
    plan['sections'][0]['voices']['vocals']['mode']='phrase'
    evidence,_,trace=expression_evidence(a,plan,p)
    assert [n.id for n in evidence]==['b']
    assert trace['a']['reason']=='weak_detection'
    # A valid phrase can still be muted by a later local arrangement policy.
    a['candidates'][0]['strength']=.8
    plan=propose_expression(a,p);plan['sections'][0]['voices']['vocals']['mode']='phrase'
    evidence,meta,trace=expression_evidence(a,plan,p)
    selected,report=select_expression(evidence,meta,trace,p,500,
        [dict(start_ms=0,end_ms=3000,preferences={'vocals_weight':0})])
    assert not selected
    assert next(r for r in report['decisions'] if r['id']=='b')['reason']=='phrase_not_selected'


def test_structure_api_preserves_audio_evidence_and_exports_saved_plan(tmp_path):
    import io
    from pathlib import Path
    from manialab.server import create_app
    from manialab.storage import Projects
    from tests.test_server import wait_job
    root=Path(__file__).parents[1];app=create_app(root,tmp_path);client=app.test_client()
    h={'X-ManiaLab-Token':app.config['TOKEN']}
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=h).json
    pid=state['id'];url=f'/api/projects/{pid}'
    notes=[Note(t,t+750,source='pitch',voice='vocals',id=str(t)) for t in range(1000,11000,1000)]
    a=analysis(notes);a['revision']=8;a['tempo'].update(stable=True,division=4)
    import json
    original=json.loads(json.dumps(a['candidates']))
    state['analysis']=a;state['reference_marks']={'bookmarks':[6000],'green_lines':[{'time_ms':3000}]}
    Projects(tmp_path/'projects').save(state)
    try:
        job=client.post(url+'/structure',json={'action':'bookmarks','audio_shift_ms':22.5},headers=h).json
        assert wait_job(client,job)['status']=='done'
        current=client.get(url).json
        assert current['analysis']['structure']['boundaries']==[0,6022,12000]
        assert current['analysis']['candidates']==original
        plan=current['expression'];plan['sections'][1]['voices']['vocals']['mode']='onsets'
        saved=client.post(url+'/expression',json={'expression':plan},headers=h)
        assert saved.status_code==200 and saved.json['expression']['source']=='manual'
        payload=dict(settings={'keys':10},bpm=120,offset=0,division=4,audio_shift_ms=0,timing_confirmed=True)
        assert wait_job(client,client.post(url+'/generate',json=payload,headers=h).json)['status']=='done'
        current=client.get(url).json;v=current['variants'][-1]
        assert all(n['end']>n['start'] for n in v['notes'] if n['start']<6000)
        assert all(n['end']==n['start'] for n in v['notes'] if n['start']>6022)
        text=client.get(url+'/export/osu').data.decode()
        assert values(sections(text)['Editor'])['Bookmarks']=='6022'
        assert len(v['selection']['decisions'])==10
        assert wait_job(client,client.post(url+'/structure',json={'action':'edit','boundaries':[5000]},headers=h).json)['status']=='done'
        assert client.post(url+'/expression',json={'expression':plan},headers=h).status_code==400
        # The saved variant keeps its own section snapshot after later edits.
        assert values(sections(client.get(url+'/export/osu').data.decode())['Editor'])['Bookmarks']=='6022'
    finally:app.extensions['executor'].shutdown()
