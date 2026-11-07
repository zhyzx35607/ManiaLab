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
