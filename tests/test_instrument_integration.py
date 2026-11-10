"""Real application routing and API controls; no neural model is loaded."""
from dataclasses import replace
import io
from pathlib import Path
import time

import numpy as np
import pytest

from manialab import audio, server
from manialab.arrange import parameters
from manialab.domain import Note
from manialab.storage import Projects
from tests.test_server import wait_job

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('specialist',[False,True])
def test_application_routes_only_selected_bass_other_and_keeps_energy_filter(tmp_path,monkeypatch,specialist):
    mixture=np.full(audio.SR,.2,dtype=np.float32)
    roles=['bass','vocals','other','guitar']
    for i,voice in enumerate(roles):
        stem=np.full(audio.SR,.1*(i+1),dtype=np.float32)
        if voice=='bass':stem[int(.45*audio.SR):int(.6*audio.SR)]=0
        np.save(tmp_path/(voice+'.npy'),stem)
    separation=dict(source='provided',voices=roles,audio_sha256='audio',set_sha256='stems',cached=True)
    class Sources:
        def separate(self,*args):return tmp_path,separation
    special_calls=[]
    class Specialist:
        def analyze(self,stem,rate,cache,identity,voice,**kwargs):
            special_calls.append((voice,cache.name,kwargs))
            assert rate==audio.SR and kwargs['allow_predicted_releases'] is True
            return [Note(start,700,pitch=40,source='pitch',voice=voice,id=f'{voice}:tsumugi:{start}',
                release_reason='model_predicted_offset') for start in (100,500)],dict(cached=True,source_semantics='unit_weight_not_confidence')
    generic_calls=[];refinements=[];fusions=[];bleed_inputs=[]
    def infer(stem,*args):
        generic_calls.append(float(stem[0]))
        return np.zeros((10,88)),np.zeros((10,88)),np.arange(10)/10
    def refine(notes,*args):refinements.append(notes);return notes,{}
    def fuse(notes,*args):fusions.append(notes);return [replace(n,start=n.start+1) for n in notes],[],0
    def bleed(notes,*args):bleed_inputs.append([n.voice for n in notes]);return notes,[]
    monkeypatch.setattr(audio,'decode',lambda *args:mixture)
    monkeypatch.setattr(audio,'infer',infer)
    monkeypatch.setattr(audio,'pitched_notes',lambda *args,**kwargs:[Note(100,700,pitch=60,source='pitch',id='generic')])
    monkeypatch.setattr(audio,'refine_releases',refine)
    monkeypatch.setattr(audio,'fuse_attacks',fuse)
    monkeypatch.setattr(audio,'reject_source_bleed',bleed)
    monkeypatch.setattr('manialab.articulation.steady_bass_onsets',lambda *args:[])
    monkeypatch.setattr('manialab.voices.vocal_phrases',lambda *args:[])
    monkeypatch.setattr('manialab.structure.detect_structure',lambda *args:dict(segments=[dict(start_ms=0,end_ms=1000)]))
    result=audio.analyze_deep(Path('mix'),tmp_path,Path('unused.onnx'),parameters(),Sources(),
        instrument_detector=Specialist() if specialist else None,allow_predicted_releases=specialist)
    assert len(generic_calls)==len(refinements)==len(fusions)==(2 if specialist else 4)
    assert result['instrument_engine']==('tsumugi' if specialist else 'general')
    assert result['allow_predicted_releases'] is specialist
    if specialist:
        assert [(v,cache) for v,cache,_ in special_calls]==[('bass','instrument-bass'),('other','instrument-other')]
        assert set(bleed_inputs[0])=={'vocals','guitar'}
        notes=[n for n in result['candidates'] if n['voice'] in ('bass','other')]
        assert [(n['voice'],n['start'],n['end']) for n in notes]==[('bass',100,700),('other',100,700),('other',500,700)]
        assert all(n['id'].startswith(n['voice']+':tsumugi:') for n in notes)
        bass=next(v for v in result['voices'] if v['voice']=='bass')
        assert bass['detector']=='tsumugi' and bass['rejected_low_energy_pitch']==1
        assert not bass['cross_source_bleed_checked']
        assert bass['release_refinement']['source_energy_filter']['rejected_ids']==['bass:tsumugi:500']
        assert result['source_bleed_policy']['excluded_unit_weight_voices']==['bass','other']
    else:
        assert not special_calls and set(bleed_inputs[0])==set(roles)
        assert result['source_bleed_policy']['excluded_unit_weight_voices']==[]
        assert all(v['cross_source_bleed_checked'] for v in result['voices'])


def test_low_level_entry_rejects_release_without_specialist_before_decoding(tmp_path,monkeypatch):
    monkeypatch.setattr(audio,'decode',lambda *args:pytest.fail('Invalid option reached audio decoder'))
    for value in (True,'true',1,None):
        with pytest.raises(ValueError,match='模型音尾'):
            audio.analyze_deep(Path('unused'),tmp_path,Path('unused'),parameters(),None,allow_predicted_releases=value)


def create_project(tmp_path,monkeypatch):
    monkeypatch.setattr(server.InstrumentDetector,'status',lambda self:dict(available=True,default=False))
    app=server.create_app(ROOT,tmp_path)
    client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    return app,client,headers,state


def test_api_requires_explicit_valid_mode_and_persists_job_options(tmp_path,monkeypatch):
    calls=[]
    def analyze(*args,**kwargs):
        calls.append(kwargs)
        return dict(candidates=[],duration=1,tempo=dict(bpm=120,offset=0),
            instrument_engine='tsumugi' if kwargs['instrument_detector'] else 'general',
            allow_predicted_releases=kwargs['allow_predicted_releases'])
    monkeypatch.setattr(server,'analyze_deep',analyze)
    app,client,headers,state=create_project(tmp_path,monkeypatch)
    url=f'/api/projects/{state["id"]}/analyze'
    try:
        assert client.get('/api/status').json['instruments']['default'] is False
        for payload in ([],{'instrument_engine':'bad'},{'instrument_engine':None},
            {'instrument_engine':'tsumugi','engine':'basic'},
            {'instrument_engine':'general','allow_predicted_releases':True},
            *[dict(instrument_engine='tsumugi',engine='deep',allow_predicted_releases=v) for v in ('true',1,None,[])],
            {'instrument_engine':'tsumugi','engine':'provided'}):
            response=client.post(url,json=payload,headers=headers)
            assert response.status_code==400,(payload,response.json)
        assert not calls
        for mode in ('deep','deep6'):
            response=client.post(url,json=dict(engine=mode,instrument_engine='tsumugi'),headers=headers)
            assert response.status_code==200,response.json
            job=wait_job(client,response.json)
            assert job['status']=='done' and job['identity']['allow_predicted_releases'] is False
            assert isinstance(calls[-1]['instrument_detector'],server.InstrumentDetector)
            assert calls[-1]['allow_predicted_releases'] is False
        response=client.post(url,json=dict(engine='deep',instrument_engine='tsumugi',allow_predicted_releases=True),headers=headers)
        assert wait_job(client,response.json)['status']=='done'
        saved=client.get(f'/api/projects/{state["id"]}').json
        assert saved['analysis_options']['instrument_engine']=='tsumugi'
        assert saved['analysis_options']['allow_predicted_releases'] is True
        response=client.post(url,json={'engine':'deep'},headers=headers)
        assert wait_job(client,response.json)['status']=='done'
        assert calls[-1]['instrument_detector'] is None and calls[-1]['allow_predicted_releases'] is False
        saved=client.get(f'/api/projects/{state["id"]}').json
        saved['provided_stems']={'source':'provided','voices':['piano']}
        Projects(tmp_path/'projects').save(saved)
        response=client.post(url,json=dict(engine='provided',instrument_engine='tsumugi'),headers=headers)
        assert response.status_code==400 and '没有贝斯' in response.json['error']
        saved['provided_stems']['voices']=['bass'];Projects(tmp_path/'projects').save(saved)
        response=client.post(url,json=dict(engine='provided',instrument_engine='tsumugi'),headers=headers)
        assert wait_job(client,response.json)['status']=='done'
        monkeypatch.setattr(server.InstrumentDetector,'status',lambda self:dict(available=False))
        response=client.post(url,json=dict(engine='deep',instrument_engine='tsumugi'),headers=headers)
        assert response.status_code==400 and '未包含' in response.json['error']
    finally:app.extensions['executor'].shutdown()


@pytest.mark.parametrize('failure',[ValueError('worker failed'),InterruptedError('cancelled')])
def test_failed_or_cancelled_specialist_keeps_saved_analysis_and_options(tmp_path,monkeypatch,failure):
    def analyze(*args,**kwargs):raise failure
    monkeypatch.setattr(server,'analyze_deep',analyze)
    app,client,headers,state=create_project(tmp_path,monkeypatch)
    state['analysis']={'sentinel':'old analysis'}
    state['analysis_options']={'instrument_engine':'general','allow_predicted_releases':False}
    Projects(tmp_path/'projects').save(state)
    try:
        response=client.post(f'/api/projects/{state["id"]}/analyze',json=dict(engine='deep',instrument_engine='tsumugi'),headers=headers)
        assert response.status_code==200
        deadline=time.monotonic()+5
        while True:
            job=client.get('/api/jobs/'+response.json['id']).json
            if job['status'] in ('error','cancelled'):break
            assert time.monotonic()<deadline
            time.sleep(.01)
        assert job['status']==('cancelled' if isinstance(failure,InterruptedError) else 'error')
        assert client.get(f'/api/projects/{state["id"]}').json==state
    finally:app.extensions['executor'].shutdown()
