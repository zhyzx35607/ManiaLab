import io
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import zipfile

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
    state['analysis'] = {'revision': 2, 'candidates': [n.data() for n in notes], 'tempo': {'bpm': 121, 'offset': 20, 'stable': False}}
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
    assert client.post(f'/api/projects/{pid}/timing-check', json={'bpm': 0}, headers=headers).status_code == 400
    app.extensions['executor'].shutdown()
