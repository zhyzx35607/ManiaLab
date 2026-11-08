import io
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import zipfile
import pytest

from manialab.server import create_app
from manialab.domain import Note, export_osu, parse_osu
from manialab.planner import plan
from manialab.storage import Projects

ROOT=Path(__file__).parents[1]


def test_full_remap_export(tmp_path):
    app=create_app(ROOT,tmp_path); client=app.test_client(); headers={'X-ManiaLab-Token':app.config['TOKEN']}
    original=export_osu([Note(1000,4000,0),Note(1000,5000,3),Note(2000,2000,8)],10,ROOT/'vendor/mapperatorinator/template.osu',audio='song.wav')
    data={'files':[(io.BytesIO(original.encode()),'chart.osu'),(io.BytesIO(b'fake-audio'),'song.wav')]}
    response=client.post('/api/import',data=data,headers=headers)
    assert response.status_code==200,response.json
    pid=response.json['id']
    denied=client.post(f'/api/projects/{pid}/generate',json={})
    assert denied.status_code==403
    response=client.post(f'/api/projects/{pid}/generate',json={'mode':'remap','settings':{'keys':10}},headers=headers)
    jid=response.json['id']
    for _ in range(100):
        job=client.get('/api/jobs/'+jid).json
        if job['status'] in ('done','error'):break
        time.sleep(.02)
    assert job['status']=='done',job
    exported=client.get(f'/api/projects/{pid}/export/osu')
    assert exported.status_code==200
    parsed=parse_osu(exported.data.decode())
    assert [(n.start,n.end) for n in parsed['notes']]==[(1000,4000),(1000,5000),(2000,2000)]
    package=client.get(f'/api/projects/{pid}/export/osz')
    with zipfile.ZipFile(io.BytesIO(package.data)) as z:
        assert set(z.namelist())=={'ManiaLab.osu','song.wav'}
    assert client.get('/api/status',headers={'Host':'evil.example'}).status_code==403
    assert client.get('/api/status',headers={'Origin':'https://evil.example'}).status_code==403
    app.extensions['executor'].shutdown()


def test_real_http_planner_mock():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path=='/v1/chat/completions'
            assert 'segments' in body['messages'][1]['content']
            result={'choices':[{'message':{'content':json.dumps({'preferences':{'ln_ratio':.8},'explanation':'test'})}}]}
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(result).encode())
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        result=plan(f'http://127.0.0.1:{server.server_port}/v1','test','fake-test-key','prefer sustain',[{'start_ms':0,'end_ms':1000}],{})
        assert result['preferences']['ln_ratio']==.8
    finally:server.shutdown();thread.join();server.server_close()


def test_audio_generation_uses_confirmed_grid_and_real_ln_target(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    state = client.post('/api/import', data={'files': (io.BytesIO(b'placeholder'), 'song.wav')}, headers=headers).json
    pid = state['id']
    notes = [Note(1014 + i * 1000, 1764 + i * 1000, source='pitch', id=str(i)) for i in range(20)]
    state['analysis'] = {'revision': 3, 'candidates': [n.data() for n in notes], 'tempo': {'bpm': 121, 'offset': 20, 'stable': False}}
    Projects(tmp_path / 'projects').save(state)
    payload = {'mode': 'audio', 'settings': {'ln_ratio': .5}, 'bpm': 120, 'offset': 0, 'division': 4,
               'snap': True, 'timing_confirmed': True}
    job = client.post(f'/api/projects/{pid}/generate', json=payload, headers=headers).json
    for _ in range(300):
        result = client.get('/api/jobs/' + job['id']).json
        if result['status'] in ('done', 'error'):
            break
        time.sleep(.02)
    assert result['status'] == 'done', result
    output = client.get(f'/api/projects/{pid}').json['variants'][0]
    assert output['report']['ln'] == 10
    assert output['selection']['alignment']['moved'] == 20
    assert all(n['start'] % 1000 == 0 for n in output['notes'])
    parsed = parse_osu(client.get(f'/api/projects/{pid}/export/osu').data.decode())
    assert '0.000,500.00000000,4,2,0,70,1,0' in parsed['timing']
    assert output['report']['grid']['off_grid'] == 0
    assert client.post(f'/api/projects/{pid}/timing-check', json={'bpm': 0}, headers=headers).status_code == 400
    app.extensions['executor'].shutdown()


def wait_job(client, job):
    for _ in range(500):
        result=client.get('/api/jobs/'+job['id']).json
        if result['status'] in ('done','error'):
            return result
        time.sleep(.02)
    raise AssertionError('job timed out')


def test_original_model_requires_confirmation_and_preserves_provided_clock(tmp_path,monkeypatch):
    from manialab.upstream import MapperEngine
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    projects=Projects(tmp_path/'projects')
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];base=f'/api/projects/{pid}'
    (projects.path(pid)/'preview.wav').write_bytes(b'placeholder')
    seen=[]
    def model(self,audio,output,prefs,progress,cancelled,seconds,tempo):
        seen.append(tempo)
        output.mkdir(parents=True);target=output/'generated.osu'
        target.write_text(export_osu([Note(1092,2092,0,id='a'),Note(2595,2595,1,id='b')],10,
            ROOT/'vendor/mapperatorinator/template.osu',bpm=tempo['bpm'],offset=tempo['offset']),encoding='utf-8')
        return target
    monkeypatch.setattr(MapperEngine,'generate',model)
    payload=dict(use_timing=True,bpm=120,offset=93,division=4,audio_shift_ms=22.5,settings={'keys':10})
    try:
        rejected=wait_job(client,client.post(base+'/engine-generate',json=payload,headers=headers).json)
        assert rejected['status']=='error' and '确认' in rejected['message'] and not seen,rejected
        result=wait_job(client,client.post(base+'/engine-generate',json=payload|{'timing_confirmed':True},headers=headers).json)
        assert result['status']=='done',result
        saved=client.get(base).json;v=saved['variants'][0]
        assert seen[0]['bpm']==120 and seen[0]['offset']==93
        assert v['report']['ln_patterns'] and v['provided_timing']['offset']==93
        assert v['report']['grid']['off_grid']==0
        assert v['selection']['alignment']['moved']==2 and v['division']==4
        assert v['notes'][0]['evidence_start']==1092
        exported=parse_osu(client.get(base+'/export/osu').data.decode())
        assert [(n.start,n.end) for n in exported['notes']]==[(1093,2093),(2593,2593)]
        assert '93.000,500.00000000,4,2,0,70,1,0' in exported['timing']
        saved=projects.load(pid)
        saved['variants'][0]['notes'][0]['raw']=saved['variants'][0]['notes'][0]['raw'].replace(',1093,128,',',1092,128,')
        projects.save(saved)
        response=client.get(base+'/export/osu')
        assert response.status_code==400 and '落线' in response.json['error']
    finally:app.extensions['executor'].shutdown()


def test_mapping_clock_changes_export_once_and_audition_uses_actual_chart_time(tmp_path):
    import numpy as np
    from scipy.io import wavfile
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    projects=Projects(tmp_path/'projects')
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];base=f'/api/projects/{pid}'
    raw=json.loads(json.dumps([Note(115+i*500,865+i*500,source='pitch',id=str(i)).data() for i in range(12)]))
    state['analysis']=dict(revision=5,candidates=raw,timing_attacks=raw,tempo=dict(bpm=120,offset=115,division=4,stable=True))
    projects.save(state)
    payload=dict(mode='audio',audio_shift_ms=22.5,settings={'ln_ratio':1,'density':4})
    result=wait_job(client,client.post(base+'/generate',json=payload,headers=headers).json)
    assert result['status']=='done',result
    saved=client.get(base).json;v=saved['variants'][0]
    assert saved['analysis']['candidates']==raw
    assert v['offset']==93 and v['audio_shift_ms']==22.5
    assert all(n['start']%500==93 and n['end']-n['start']==750 for n in v['notes'])
    parsed=parse_osu(client.get(base+'/export/osu').data.decode())
    assert float(parsed['timing'][0].split(',')[0])==93
    wavfile.write(projects.path(pid)/'preview.wav',22050,np.zeros(22050*8,dtype=np.int16))
    response=client.post(base+'/audition',json={'mode':'chart','duration_ms':1000},headers=headers)
    rate,wave=wavfile.read(io.BytesIO(response.data))
    assert abs(np.flatnonzero(wave[:,0])[0]/rate*1000-93)<.1
    # Changing the convention cannot silently mix clocks in a local rewrite.
    failed=wait_job(client,client.post(base+'/generate',json=payload|{'audio_shift_ms':0,'region':[1000,3000]},headers=headers).json)
    assert failed['status']=='error' and 'Timing' in failed['message']
    app.extensions['executor'].shutdown()


def test_reference_style_preserves_audio_candidates_and_target_keys(tmp_path):
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    projects=Projects(tmp_path/'projects');state['settings']['keys']=10
    state['analysis']={'sentinel':'original evidence'};projects.save(state)
    notes=[Note(t,t+500 if i%2 else t,i%10) for i,t in enumerate(range(0,10000,250))]
    text=export_osu(notes,10,ROOT/'vendor/mapperatorinator/template.osu')
    response=client.post(f"/api/projects/{state['id']}/reference-style",data={'file':(io.BytesIO(text.encode()),'reference.osu'),
        'settings':json.dumps({'keys':6,'pitch_follow':.7})},headers=headers)
    assert response.status_code==200,response.json
    saved=response.json
    assert saved['audio']=='song.wav' and saved['analysis']==state['analysis']
    assert saved['settings']['keys']==6 and saved['settings']['ln_ratio']==.5
    assert saved['settings']['pitch_follow']==.7
    assert saved['reference_style']['keys']==10 and saved['reference_style']['objects']==40
    assert saved['variants']==[] and saved['original'] is None
    app.extensions['executor'].shutdown()


def test_audition_http_preserves_project_and_returns_aligned_wave(tmp_path):
    import numpy as np
    from scipy.io import wavfile
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];url=f'/api/projects/{pid}/audition'
    assert client.post(url,json={},headers=headers).status_code==400
    preview=tmp_path/'projects'/pid/'preview.wav'
    wavfile.write(preview,22050,np.zeros(22050*10,dtype=np.int16))
    before=(preview.parent/'state.json').read_bytes()
    result=client.post(url,json={'bpm':120,'offset':91,'start_ms':0,'duration_ms':5000},headers=headers)
    assert result.status_code==200 and result.mimetype=='audio/wav'
    sr,content=wavfile.read(io.BytesIO(result.data))
    assert sr==22050 and content.shape==(110250,2)
    first=np.flatnonzero(content[:,0])[0]
    assert abs(first/22.05-91)<.1
    assert (preview.parent/'state.json').read_bytes()==before
    assert client.post(url,json={'duration_ms':float('nan')},headers=headers).status_code==400
    assert client.post(url,json={'mode':'chart'},headers=headers).status_code==400
    app.extensions['executor'].shutdown()


@pytest.mark.parametrize('keys', range(6,11))
def test_reference_210_91_export_and_timing_guards(tmp_path,keys,monkeypatch):
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    projects=Projects(tmp_path/'projects')
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];base=f'/api/projects/{pid}'
    tick=60000/210/4
    notes=[Note(round(91+i*4*tick+24),round(91+(i*4+3)*tick+31),source='pitch',id=str(i)) for i in range(100)]
    state['analysis']={'revision':3,'candidates':[n.data() for n in notes],
        'timing_attacks':[Note(round(91+i*4*tick),round(91+i*4*tick)).data() for i in range(100)],
        'tempo':{'bpm':210,'offset':118,'stable':False,'division':4}}
    projects.save(state)
    payload=dict(mode='audio',settings={'keys':keys,'ln_ratio':1,'min_ln':100},bpm=210,offset=91,division=4)
    result=wait_job(client,client.post(base+'/generate',json=payload,headers=headers).json)
    assert result['status']=='error' and 'Timing' in result['message']
    fitted=client.post(base+'/timing-refit',json=payload,headers=headers).json
    assert fitted['offset']==91
    payload.update(timing_confirmed=True)
    result=wait_job(client,client.post(base+'/generate',json=payload,headers=headers).json)
    assert result['status']=='done',result
    response=client.get(base+'/export/osu');assert response.status_code==200,response.json
    parsed=parse_osu(response.data.decode())
    assert parsed['keys']==keys and any(n.hold for n in parsed['notes'])
    assert any(line.startswith('91.000,285.71428571,') for line in parsed['timing'])
    for n in parsed['notes']:
        assert all(abs(t-(91+round((t-91)/tick)*tick))<=.501 for t in (n.start,n.end))
    result=wait_job(client,client.post(base+'/generate',json=payload|{'offset':118,'region':[0,5000]},headers=headers).json)
    assert result['status']=='error' and 'Timing 已改变' in result['message']
    stored=projects.load(pid);stored['variants'][0]['timing_revision']=2;projects.save(stored)
    assert client.get(base+'/export/osu').status_code==400
    assert client.get(base+'/export/json').status_code==200
    monkeypatch.setattr('manialab.server.analyze',lambda *args:state['analysis'])
    result=wait_job(client,client.post(base+'/analyze',json={},headers=headers).json)
    assert result['status']=='done'
    refreshed=projects.load(pid)
    assert refreshed['active']==-1 and len(refreshed['variants'])==1
    diagnosis=client.get(base+'/export/json')
    assert diagnosis.status_code==200 and diagnosis.json['variant'] is None
    assert diagnosis.json['analysis']['tempo']['offset']==118
    app.extensions['executor'].shutdown()
