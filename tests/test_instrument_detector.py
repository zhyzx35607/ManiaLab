"""Non-neural controls for worker identity, cache integrity and cancellation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from manialab import instrument_detector as module
from manialab.domain import Note


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def create_result(command, mutate=None):
    wave = Path(command[command.index('--audio') + 1])
    output = Path(command[command.index('--output') + 1])
    expected = json.loads((wave.parent / 'request.json').read_text(encoding='utf-8'))
    output.mkdir()
    identity = dict(source_commit=module.SOURCE_COMMIT, source_files=expected['source_files'],
        model_revision=module.MODEL_REVISION, model=expected['model'], model_sha256=expected['model_sha256'],
        runtime_code_sha256=expected['runtime']['worker_sources'],
        runner_sha256=expected['runtime']['worker_sources']['scripts/transcribe_instruments.py'],
        shared_adapter_sha256=expected['runtime']['worker_sources']['scripts/research_tsumugi_transcription.py'],
        audio=str(wave.resolve()), audio_sha256=module.sha(wave), pcm_sha256=expected['pcm_sha256'],
        original_sample_rate=expected['sample_rate'], original_samples=expected['samples'],
        original_channels=expected['channels'], resampled_sample_rate=module.MODEL_RATE,
        resampled_samples=(expected['samples'] * module.MODEL_RATE + expected['sample_rate'] - 1)//expected['sample_rate'],
        resampled_pcm_sha256='4'*64, normalization=False, inference_scope='full_original_audio', stem_filter='all', evidence_revision=2)
    release = expected['allow_predicted_releases']
    reason = 'model_predicted_offset' if release else 'unresolved_or_postprocessed'
    candidate = Note(100, 700, pitch=40, strength=1., source='pitch', id='test-note',
                     voice=expected['voice'], release_reason=reason, instrument_id='test:class:8:electric_bass').data()
    event = dict(onset=.1, offset=.7, pitch=40, is_drum=False)
    trace = dict(id=candidate['id'], decision='candidate', source_voice=expected['voice'],
        candidate_start_ms=100, candidate_end_ms=700, release_reason=reason, raw_event=event,
        instrument_id=candidate['instrument_id'], authored_ln_verified=False,
        weight_semantics='unit_weight_not_confidence', allow_predicted_releases=release,
        model_predicted_release_hypothesis=release, final_endpoint_model_supported=True, final_onset_model_supported=True,
        input_start_active_interval=False, onset_basis='model_predicted_onset')
    values = {'identity.json':identity, 'notes.json':[event], 'raw-notes.json':[],
        'window-notes.json':[], 'interval-constructions.json':[], 'candidates.json':[candidate],
        'candidate-trace.json':[trace]}
    if mutate:
        mutate(values)
    for name, value in values.items():
        write(output/name, value)
    report = dict(identity=identity, context_start_seconds=0.,
        context_end_seconds=expected['samples']/expected['sample_rate'], notes=1,
        **{key:module.sha(output/name) for name,key in (
            ('notes.json','notes_sha256'),('raw-notes.json','raw_notes_sha256'),
            ('window-notes.json','window_notes_sha256'),('interval-constructions.json','interval_constructions_sha256'))})
    write(output/'report.json',report)
    result = dict(identity=identity, source_voice=expected['voice'], model=expected['model'],
        allow_predicted_releases=release, source_semantics='unit_weight_not_confidence', musical_acceptance=False,
        candidates=1, predicted_release_hypotheses=int(release),
        unobserved_onsets=0, input_start_active_intervals=0,
        evidence_sha256={name:module.sha(output/name) for name in module.EVIDENCE_FILES})
    write(output/'result.json',result)
    return output


class Finished:
    returncode = 0
    def poll(self):
        return self.returncode


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv('MANIALAB_INSTRUMENT_PYTHON', raising=False)
    monkeypatch.delenv('MANIALAB_INSTRUMENT_DEPS', raising=False)
    detector = module.InstrumentDetector(tmp_path)
    runtime = dict(kind='test', executable_sha256='1'*64, worker_sources={
        'scripts/transcribe_instruments.py':'2'*64, 'scripts/research_tsumugi_transcription.py':'3'*64})
    monkeypatch.setattr(detector, '_runtime', lambda:(['test-worker'],copy.deepcopy(runtime)))
    monkeypatch.setattr(module, 'verify_source', lambda path:{'test-source.py':'5'*64})
    monkeypatch.setattr(module, 'verify_model', lambda *args:None)
    launches = []
    def launch(command, **kwargs):
        launches.append((command,kwargs))
        create_result(command)
        return Finished()
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    samples = np.full(22050, .1, dtype=np.float32)
    return detector, samples, tmp_path/'specialist-cache', launches, runtime


def test_cache_preserves_unclipped_pcm_and_unknown_raw_tail(setup):
    detector, samples, cache, launches, _ = setup
    samples[100:103] = [1.7,-1.4,.000002]
    general=cache.parent/'bass-activations.npz';general.write_bytes(b'original-cache')
    notes, report=detector.analyze(samples,22050,cache,'stem','bass')
    assert not report['cached'] and notes[0].end==700 and notes[0].hold
    assert notes[0].release_reason=='unresolved_or_postprocessed'
    assert report['source_semantics']=='unit_weight_not_confidence'
    command,kwargs=launches[0]
    rate,wave=wavfile.read(command[command.index('--audio')+1])
    assert rate==22050 and wave.dtype==np.float32
    np.testing.assert_array_equal(wave,samples)
    assert '--allow-predicted-releases' not in command
    assert kwargs['stdout'] is not subprocess.PIPE and kwargs['stderr']==subprocess.STDOUT
    assert kwargs['creationflags']==(subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    _,second=detector.analyze(samples,22050,cache,'stem','bass')
    assert second['cached'] and len(launches)==1 and general.read_bytes()==b'original-cache'
    from manialab.arrange import parameters
    from manialab.expression import propose_expression,expression_evidence
    settings=parameters({'ln_ratio':1})
    analysis=dict(duration=1,candidates=[n.data() for n in notes],structure=dict(segments=[dict(start_ms=0,end_ms=1000)]))
    selected,_,_=expression_evidence(analysis,propose_expression(analysis,settings),settings)
    assert selected and all(not n.hold for n in selected)


@pytest.mark.parametrize('tail,reason,accepted', [
    (14894, 'unresolved_or_postprocessed', True),
    (14895, 'unresolved_or_postprocessed', False),
    (14894, 'model_predicted_offset', False),
])
def test_ceil_resampling_can_preserve_only_unknown_tail_at_recorded_model_boundary(setup, monkeypatch, tail, reason, accepted):
    detector, _, cache, _, _ = setup
    samples = np.full(656803, .1, dtype=np.float32)
    def launch(command, **kwargs):
        def mutate(values):
            values['candidates.json'][0].update(end=tail, release_reason=reason)
            trace = values['candidate-trace.json'][0]
            trace.update(candidate_end_ms=tail, release_reason=reason,
                model_predicted_release_hypothesis=reason == 'model_predicted_offset',
                final_endpoint_model_supported=reason == 'model_predicted_offset')
            values['notes.json'][0]['offset'] = 328402 / 22050
        output = create_result(command, mutate)
        result = json.loads((output/'result.json').read_text())
        result['predicted_release_hypotheses'] = int(reason == 'model_predicted_offset')
        write(output/'result.json', result)
        return Finished()
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    if accepted:
        notes, report = detector.analyze(samples, 44100, cache, 'stem', 'bass', allow_predicted_releases=True)
        assert notes[0].end == 14894 and notes[0].release_reason in module.UNOBSERVED_RELEASES
        assert report['predicted_release_hypotheses'] == 0
    else:
        with pytest.raises(ValueError):
            detector.analyze(samples, 44100, cache, 'stem', 'bass', allow_predicted_releases=True)
        assert not list(cache.glob('*/current.json'))


@pytest.mark.parametrize('change',['pcm','clock','channels','stem','voice','release','runtime'])
def test_every_input_or_mode_identity_change_gets_independent_cache(setup,change):
    detector,samples,cache,launches,runtime=setup
    detector.analyze(samples,22050,cache,'stem','bass')
    rate,stem,voice,release=22050,'stem','bass',False
    if change=='pcm':samples[0]=.2
    elif change=='clock':rate=16000
    elif change=='channels':samples=np.column_stack([samples,samples])
    elif change=='stem':stem='changed'
    elif change=='voice':voice='other'
    elif change=='release':release=True
    else:runtime['worker_sources']['scripts/transcribe_instruments.py']='6'*64
    notes,report=detector.analyze(samples,rate,cache,stem,voice,release)
    assert len(launches)==2 and not report['cached'] and len(list(cache.glob('*/current.json')))==2
    if release:
        assert notes[0].release_reason=='model_predicted_offset'
        assert '--allow-predicted-releases' in launches[-1][0]


@pytest.mark.parametrize('damage',['evidence','result','identity','input','missing_hash'])
def test_corrupt_cache_is_never_accepted_by_existence_only(setup,damage):
    detector,samples,cache,launches,_=setup
    _,report=detector.analyze(samples,22050,cache,'stem','bass')
    output=Path(report['evidence_directory']);marker=next(cache.glob('*/current.json'))
    if damage=='evidence':(output/'window-notes.json').write_text('[{}]')
    elif damage=='result':(output/'result.json').write_text('{}')
    elif damage=='identity':(output/'identity.json').write_text('{}')
    elif damage=='input':(output.parent/'input.wav').write_bytes(b'bad-wave')
    else:
        value=json.loads(marker.read_text());value.pop('result_sha256');write(marker,value)
    _,new=detector.analyze(samples,22050,cache,'stem','bass')
    assert not new['cached'] and new['cache_rejection'] and len(launches)==2
    assert Path(new['evidence_directory'])!=output and output.exists()


@pytest.mark.parametrize('damage',['clock','source','release','trace','pitch','semantics','onset','missing_result','nonzero'])
def test_invalid_fresh_worker_results_never_commit(setup,monkeypatch,damage):
    detector,samples,cache,launches,_=setup
    def launch(command,**kwargs):
        def mutate(values):
            if damage=='clock':values['identity.json']['original_samples']+=1
            elif damage=='source':values['identity.json']['runtime_code_sha256']={}
            elif damage=='release':values['candidates.json'][0]['release_reason']='model_predicted_offset'
            elif damage=='trace':values['candidate-trace.json']=[]
            elif damage=='pitch':values['candidates.json'][0]['pitch']=41
            elif damage=='semantics':values['candidate-trace.json'][0]['weight_semantics']='confidence'
            elif damage=='onset':values['candidate-trace.json'][0]['final_onset_model_supported']=False
        output=create_result(command,mutate)
        if damage=='missing_result':(output/'result.json').unlink()
        worker=Finished()
        if damage=='nonzero':worker.returncode=2
        return worker
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    with pytest.raises(ValueError):detector.analyze(samples,22050,cache,'stem','bass')
    assert not list(cache.glob('*/current.json'))
    assert list(cache.glob('*/attempt-*/worker.log'))


def test_cancel_terminates_silent_worker_and_preserves_old_analysis_cache(setup,monkeypatch):
    detector,samples,cache,_,_=setup
    detector.analyze(samples,22050,cache,'stem','bass')
    marker=next(cache.glob('*/current.json'));before=marker.read_bytes()
    class Silent:
        returncode=None
        terminated=False
        killed=False
        def poll(self):return self.returncode
        def terminate(self):self.terminated=True
        def wait(self,timeout):
            if not self.killed:raise subprocess.TimeoutExpired('test',timeout)
            self.returncode=-9
        def kill(self):self.killed=True
    worker=Silent();launched=[]
    def launch(*args,**kwargs):launched.append(True);return worker
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    samples[0]=.2
    with pytest.raises(InterruptedError):
        detector.analyze(samples,22050,cache,'stem','bass',cancelled=lambda:bool(launched))
    assert worker.terminated and worker.killed and marker.read_bytes()==before
    assert len(list(cache.glob('*/current.json')))==1


def test_cancelled_cached_result_never_starts_worker_again(setup):
    detector,samples,cache,launches,_=setup
    detector.analyze(samples,22050,cache,'stem','bass')
    calls=[]
    def cancelled():
        calls.append(True)
        return len(calls)==2
    with pytest.raises(InterruptedError):
        detector.analyze(samples,22050,cache,'stem','bass',cancelled=cancelled)
    assert len(launches)==1


@pytest.mark.parametrize('voice,release,rate,shape',[('vocals',False,22050,(22050,)),
    ('bass','true',22050,(22050,)),('bass',False,True,(22050,)),('bass',False,22050,(22050,3)),
    ('bass',False,22050,(0,))])
def test_invalid_requests_do_not_launch_worker(setup,voice,release,rate,shape):
    detector,_,cache,launches,_=setup
    with pytest.raises(ValueError):detector.analyze(np.zeros(shape),rate,cache,'stem',voice,release)
    assert not launches


def test_missing_optional_components_do_not_report_available(tmp_path,monkeypatch):
    monkeypatch.delenv('MANIALAB_INSTRUMENT_PYTHON',raising=False)
    monkeypatch.delenv('MANIALAB_INSTRUMENT_DEPS',raising=False)
    detector=module.InstrumentDetector(tmp_path)
    assert not detector.status()['available'] and not detector.status()['default']
    with pytest.raises(ValueError,match='未包含'):
        detector.analyze(np.zeros(22050),22050,tmp_path/'cache','stem','bass')
    monkeypatch.setenv('MANIALAB_INSTRUMENT_PYTHON',str(tmp_path/'missing-python.exe'))
    assert not module.InstrumentDetector(tmp_path).status()['available']


def make_runtime_sources(root):
    for name in module.RUNTIME_FILES:
        path=root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# test source '+name,encoding='utf-8')


def test_portable_runtime_requires_complete_manifest_and_checks_executable_and_code(tmp_path,monkeypatch):
    monkeypatch.delenv('MANIALAB_INSTRUMENT_PYTHON',raising=False)
    monkeypatch.delenv('MANIALAB_INSTRUMENT_DEPS',raising=False)
    detector=module.InstrumentDetector(tmp_path)
    detector.executable.parent.mkdir(parents=True)
    detector.executable.write_bytes(b'test-executable')
    (detector.assets/'source').mkdir()
    (detector.assets/'checkpoints').mkdir()
    for model in module.VOICE_MODELS.values():
        (detector.assets/'checkpoints'/('best_model_'+model+'.pth')).touch()
    write(detector.assets/'manifest.json',dict(prepared_only=True))
    assert not detector.status()['available']
    source=detector.executable.parent/'_internal/source-evidence'
    make_runtime_sources(source)
    manifest=dict(source_commit=module.SOURCE_COMMIT,model_revision=module.MODEL_REVISION,
        worker_sha256=module.sha(detector.executable),worker_sources=module.runtime_identity(source))
    write(detector.assets/'manifest.json',manifest)
    assert detector.status()['available']
    command,identity=detector._runtime()
    assert command==[str(detector.executable)] and identity['kind']=='portable'
    assert identity['worker_sources']==manifest['worker_sources']
    changed=source/module.RUNTIME_FILES[0];changed.write_text('changed')
    with pytest.raises(ValueError,match='运行源码哈希'):detector._runtime()
    make_runtime_sources(source)
    detector.executable.write_bytes(b'changed-binary')
    with pytest.raises(ValueError,match='运行程序哈希'):detector._runtime()


def test_development_runtime_tracks_code_python_and_dependency_metadata(tmp_path,monkeypatch):
    python=tmp_path/'venv/Scripts/python.exe';python.parent.mkdir(parents=True);python.write_bytes(b'fake-python')
    deps=tmp_path/'dependencies';metadata=deps/'test-1.dist-info/METADATA';metadata.parent.mkdir(parents=True)
    metadata.write_text('Version: 1')
    (metadata.parent/'RECORD').write_text('file.py,digest,1')
    make_runtime_sources(tmp_path)
    monkeypatch.setenv('MANIALAB_INSTRUMENT_PYTHON',str(python))
    monkeypatch.setenv('MANIALAB_INSTRUMENT_DEPS',str(deps))
    detector=module.InstrumentDetector(tmp_path)
    command,before=detector._runtime()
    assert command[:2]==[str(python),str(tmp_path/'scripts/transcribe_instruments.py')]
    assert command[-2:]==['--dependencies',str(deps)]
    assert before['dependency_metadata'][str(metadata)]==module.sha(metadata)
    metadata.write_text('Version: 2')
    _,after=detector._runtime()
    assert before!=after and before['worker_sources']==after['worker_sources']
    python.write_bytes(b'changed-python')
    assert detector._runtime()[1]['executable_sha256']!=after['executable_sha256']


def test_ui_defaults_eligibility_parent_toggle_and_saved_modes():
    node=shutil.which('node')
    if not node:pytest.skip('Node is needed for the small DOM control')
    script=Path(__file__).resolve().parents[1]/'static/instrument-detector.js'
    test=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements={};
class Element {
  constructor(){this.listeners={};this.checked=false;this.disabled=false;}
  set id(value){this._id=value;elements[value]=this;}get id(){return this._id;}
  append(){}after(){}closest(){return this;}
  addEventListener(name,fn){this.listeners[name]=fn;}
  change(){this.listeners.change?.();}
}
elements.analysisEngine=new Element();elements.analysisEngine.value='deep';
const context={window:{},state:null,busy:false,$:id=>elements[id],document:{createElement:()=>new Element(),createTextNode:s=>s}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const api=context.window,main=elements.instrumentSpecialist,tail=elements.instrumentPredictedReleases;
const value=()=>JSON.parse(JSON.stringify(api.instrumentSelection()));
assert.deepStrictEqual(value(),{instrument_engine:'general',allow_predicted_releases:false});
assert(main.disabled&&tail.disabled);api.updateInstrumentStatus({available:true});assert(!main.disabled&&tail.disabled);
main.checked=true;main.change();assert(!tail.disabled);tail.checked=true;tail.change();assert(value().allow_predicted_releases);
main.checked=false;main.change();assert(!tail.checked&&tail.disabled&&!value().allow_predicted_releases);
main.checked=true;elements.analysisEngine.value='basic';elements.analysisEngine.change();assert(main.disabled&&tail.disabled);assert.strictEqual(value().instrument_engine,'general');
elements.analysisEngine.value='provided';context.state={provided_stems:{voices:['piano']}};api.lockInstrumentDetector();assert(main.disabled);
context.state.provided_stems.voices=['other'];api.lockInstrumentDetector();assert(!main.disabled);
context.state.analysis={instrument_engine:'tsumugi',allow_predicted_releases:true};api.renderInstrumentDetector();assert(main.checked&&tail.checked&&value().allow_predicted_releases);
context.busy=true;api.lockInstrumentDetector();assert(main.disabled&&tail.disabled);
context.busy=false;context.state={provided_stems:{voices:['bass']}};api.renderInstrumentDetector();assert(!main.checked&&!tail.checked);
console.log('instrument UI controls passed');
'''
    completed=subprocess.run([node,'-e',test,str(script)],capture_output=True,text=True,timeout=15)
    assert completed.returncode==0,completed.stderr
