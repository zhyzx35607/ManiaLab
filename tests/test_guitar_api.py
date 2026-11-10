"""Optional guitar selection through the real API and browser control; no NN."""
import io
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from manialab import server
from manialab.storage import Projects


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(server.GuitarDetector, 'status', lambda self:
        dict(available=True, default=False, predicted_releases_default=False))
    monkeypatch.setattr(server.InstrumentDetector, 'status', lambda self: dict(available=True, default=False))
    app = server.create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    response = client.post('/api/import', data={'files': (io.BytesIO(b'placeholder'), 'song.wav')}, headers=headers)
    assert response.status_code == 200
    state = response.json
    try:
        yield SimpleNamespace(app=app, client=client, headers=headers, state=state,
            store=Projects(tmp_path / 'projects'), url=f'/api/projects/{state["id"]}/analyze')
    finally:
        app.extensions['executor'].shutdown()


def _post(project, payload):
    return project.client.post(project.url, json=payload, headers=project.headers)


def _job(project, response):
    assert response.status_code == 200, response.json
    deadline = time.monotonic() + 10
    while True:
        job = project.client.get('/api/jobs/' + response.json['id']).json
        if job['status'] in ('done', 'error', 'cancelled'):
            return job
        assert time.monotonic() < deadline
        time.sleep(.01)


def _saved(project):
    return project.client.get(f'/api/projects/{project.state["id"]}').json


def _result(**kwargs):
    return dict(candidates=[], duration=1, tempo=dict(bpm=120, offset=0),
        guitar_engine='fl' if kwargs['guitar_detector'] else 'general',
        allow_guitar_predicted_releases=kwargs['allow_guitar_predicted_releases'])


def test_invalid_guitar_options_fail_before_starting_a_job(project, monkeypatch):
    monkeypatch.setattr(server, 'analyze_deep', lambda *a, **k: pytest.fail('Invalid request started analysis'))
    cases = [
        {'guitar_engine': 'other'}, {'guitar_engine': None}, {'guitar_engine': []},
        {'engine': 'basic', 'guitar_engine': 'fl'},
        {'engine': 'deep', 'guitar_engine': 'fl'},
        {'engine': 'deep6', 'allow_guitar_predicted_releases': True},
        {'engine': 'provided', 'guitar_engine': 'fl'},
        *[dict(engine='deep6', guitar_engine='fl', allow_guitar_predicted_releases=value)
          for value in ('true', 1, None, [])],
    ]
    for payload in cases:
        response = _post(project, payload)
        assert response.status_code == 400, (payload, response.json)
    assert project.app.extensions['jobs'] == {}


def test_guitar_modes_are_independent_and_persist_in_job_and_reopened_project(project, monkeypatch):
    calls = []
    def analyze(*args, **kwargs):
        calls.append(kwargs)
        return _result(**kwargs)
    monkeypatch.setattr(server, 'analyze_deep', analyze)
    status = project.client.get('/api/status').json['guitar']
    assert status['available'] and status['default'] is False and status['predicted_releases_default'] is False
    for extra in ({}, {'allow_guitar_predicted_releases': True},
                  {'instrument_engine': 'tsumugi', 'allow_predicted_releases': True}):
        payload = dict(engine='deep6', guitar_engine='fl', **extra)
        job = _job(project, _post(project, payload))
        assert job['status'] == 'done', job
        options = _saved(project)['analysis_options']
        assert options == job['identity']
        assert options['guitar_engine'] == 'fl'
        assert options['allow_guitar_predicted_releases'] is extra.get('allow_guitar_predicted_releases', False)
        assert isinstance(calls[-1]['guitar_detector'], server.GuitarDetector)
        assert calls[-1]['allow_guitar_predicted_releases'] is options['allow_guitar_predicted_releases']
        assert calls[-1]['allow_predicted_releases'] is extra.get('allow_predicted_releases', False)
        assert (calls[-1]['instrument_detector'] is not None) is ('instrument_engine' in extra)
    job = _job(project, _post(project, {'engine': 'deep'}))
    assert job['status'] == 'done'
    assert calls[-1]['guitar_detector'] is None and calls[-1]['allow_guitar_predicted_releases'] is False
    options = _saved(project)['analysis_options']
    assert options['guitar_engine'] == 'general' and options['allow_guitar_predicted_releases'] is False


def test_provided_stems_require_explicit_guitar_and_missing_model_is_rejected(project, monkeypatch):
    calls = []
    def analyze(*args, **kwargs):
        calls.append(kwargs)
        return _result(**kwargs)
    monkeypatch.setattr(server, 'analyze_deep', analyze)
    project.state['provided_stems'] = {'source': 'provided', 'voices': ['bass', 'piano']}
    project.store.save(project.state)
    response = _post(project, dict(engine='provided', guitar_engine='fl'))
    assert response.status_code == 400 and '没有吉他' in response.json['error']
    assert not calls
    project.state['provided_stems']['voices'].append('guitar')
    project.store.save(project.state)
    job = _job(project, _post(project, dict(engine='provided', guitar_engine='fl')))
    assert job['status'] == 'done' and len(calls) == 1
    monkeypatch.setattr(server.GuitarDetector, 'status', lambda self: dict(available=False))
    response = _post(project, dict(engine='provided', guitar_engine='fl'))
    assert response.status_code == 400 and '未包含吉他' in response.json['error']
    assert len(calls) == 1


@pytest.mark.parametrize('failure', [ValueError('synthetic failure'), InterruptedError('synthetic cancellation')])
def test_failed_guitar_analysis_preserves_prior_results_and_options(project, monkeypatch, failure):
    project.state['analysis'] = {'sentinel': 'prior analysis'}
    project.state['analysis_options'] = {'guitar_engine': 'general', 'allow_guitar_predicted_releases': False}
    project.store.save(project.state)
    def analyze(*args, **kwargs):
        raise failure
    monkeypatch.setattr(server, 'analyze_deep', analyze)
    job = _job(project, _post(project, dict(engine='deep6', guitar_engine='fl', allow_guitar_predicted_releases=True)))
    assert job['status'] == ('cancelled' if isinstance(failure, InterruptedError) else 'error')
    assert _saved(project) == project.state


def test_active_guitar_job_can_be_cancelled_and_blocks_a_second_job(project, monkeypatch):
    started, finish = threading.Event(), threading.Event()
    def analyze(*args, **kwargs):
        started.set()
        assert finish.wait(10)
        return _result(**kwargs)
    monkeypatch.setattr(server, 'analyze_deep', analyze)
    response = _post(project, dict(engine='deep6', guitar_engine='fl'))
    try:
        assert response.status_code == 200 and started.wait(5)
        blocked = _post(project, dict(engine='deep6', guitar_engine='fl'))
        assert blocked.status_code == 400 and '任务运行' in blocked.json['error']
        cancelled = project.client.post('/api/jobs/' + response.json['id'] + '/cancel',
                                        json={}, headers=project.headers)
        assert cancelled.status_code == 200
    finally:
        finish.set()
    assert _job(project, response)['status'] == 'cancelled'
    assert _saved(project) == project.state


def test_guitar_browser_control_defaults_eligibility_reopen_busy_and_unavailable():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the small DOM control')
    script = ROOT / 'static/guitar-detector.js'
    test = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements={};
class Element {
  constructor(){this.listeners={};this.checked=false;this.disabled=false;}
  set id(value){this._id=value;elements[value]=this;}get id(){return this._id;}
  append(){}after(){}closest(){return this;}
  addEventListener(name,fn){this.listeners[name]=fn;}
  change(){this.listeners.change?.();}
}
elements.analysisEngine=new Element();elements.analysisEngine.value='deep6';
elements.instrumentSpecialist=new Element();elements.instrumentSpecialist.checked=true;
elements.instrumentPredictedReleases=new Element();elements.instrumentPredictedReleases.checked=true;
const context={window:{},state:null,busy:false,$:id=>elements[id],document:{createElement:()=>new Element(),createTextNode:s=>s}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const api=context.window,main=elements.guitarSpecialist,tail=elements.guitarPredictedReleases;
const value=()=>JSON.parse(JSON.stringify(api.guitarSelection()));
const general={guitar_engine:'general',allow_guitar_predicted_releases:false};
assert.deepStrictEqual(value(),general);assert(main.disabled&&tail.disabled);
api.updateGuitarStatus({available:true});assert(!main.disabled&&tail.disabled);
main.checked=true;main.change();assert(!tail.disabled);tail.checked=true;tail.change();
assert.deepStrictEqual(value(),{guitar_engine:'fl',allow_guitar_predicted_releases:true});
main.checked=false;main.change();assert(!tail.checked&&tail.disabled);assert.deepStrictEqual(value(),general);
main.checked=true;
for(const engine of ['basic','deep']){elements.analysisEngine.value=engine;elements.analysisEngine.change();assert(main.disabled&&tail.disabled);assert.deepStrictEqual(value(),general);}
elements.analysisEngine.value='provided';context.state={provided_stems:{voices:['bass','piano']}};
api.lockGuitarDetector();assert(main.disabled&&tail.disabled);
context.state.provided_stems.voices.push('guitar');api.lockGuitarDetector();assert(!main.disabled);
context.state.analysis_options={guitar_engine:'fl',allow_guitar_predicted_releases:true};
api.renderGuitarDetector();assert(main.checked&&tail.checked&&value().allow_guitar_predicted_releases);
context.busy=true;api.lockGuitarDetector();assert(main.disabled&&tail.disabled);
context.busy=false;api.lockGuitarDetector();assert(!main.disabled&&!tail.disabled);
api.updateGuitarStatus({available:false});assert(main.disabled&&tail.disabled);assert.deepStrictEqual(value(),general);
api.updateGuitarStatus({available:true});assert(main.checked&&tail.checked&&!main.disabled);
context.state.analysis={guitar_engine:'fl',allow_guitar_predicted_releases:true};
context.state.analysis_options={guitar_engine:'general',allow_guitar_predicted_releases:false};
api.renderGuitarDetector();assert(!main.checked&&!tail.checked);assert.deepStrictEqual(value(),general);
delete context.state.analysis_options;api.renderGuitarDetector();assert(main.checked&&tail.checked);
context.state={provided_stems:{voices:['guitar']}};api.renderGuitarDetector();assert(!main.checked&&!tail.checked);
assert(elements.instrumentSpecialist.checked&&elements.instrumentPredictedReleases.checked);
console.log('guitar UI controls passed');
'''
    completed = subprocess.run([node, '-e', test, str(script)], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr


def test_guitar_control_is_connected_to_status_busy_render_and_request():
    markup = (ROOT / 'static/index.html').read_text(encoding='utf-8')
    assert '<script src="/static/guitar-detector.js" defer></script>' in markup
    app = (ROOT / 'static/app.js').read_text(encoding='utf-8')
    for callback in ('renderGuitarDetector?.()', 'updateGuitarStatus?.(status.guitar)',
                     'lockGuitarDetector?.()', 'guitarSelection?.()'):
        assert callback in app
