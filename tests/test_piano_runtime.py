import numpy as np
import pytest

from manialab.piano import infer,decode,OUTPUTS


def test_piano_stitching_keeps_the_same_clock_across_overlapping_windows():
    class Clock:
        def __init__(self):self.start=0
        def run(self,names,inputs):
            assert inputs['audio'].shape==(1,160000)
            values=np.broadcast_to((np.arange(1001)+self.start)[None,:,None],(1,1001,88)).copy()
            self.start+=500
            return [values for _ in names]
    result=infer(np.zeros(22050*23,dtype=np.float32),22050,Clock())
    for values in result.values():
        assert values.shape==(2300,88)
        np.testing.assert_array_equal(values[:,0],np.arange(2300))
    with pytest.raises(InterruptedError):
        infer(np.zeros(16000,dtype=np.float32),16000,Clock(),cancelled=lambda:True)


def test_piano_decoder_preserves_reattacks_and_distinguishes_forced_release():
    arrays={key:np.zeros((1000,88),dtype=np.float32) for key in OUTPUTS}
    arrays['velocity_output'][:]=.5
    def peak(key,time,pitch):
        arrays[key][time-1:time+2,pitch]=[.2,.8,.2]
    # Two attacks under sustained activation must stay separate.
    arrays['frame_output'][100:500,0]=.9
    peak('reg_onset_output',100,0);peak('reg_onset_output',300,0)
    peak('reg_offset_output',498,0)
    # No detected release: distinguish the duration limit from an actual tail.
    arrays['frame_output'][100:,1]=.9
    peak('reg_onset_output',100,1)
    notes=decode(arrays)
    assert [(n['pitch'],n['onset'],n['release_reason']) for n in notes]==[
        (21,1.,'reattack'),(21,3.,'release_regression'),(22,1.,'duration_limit')]
    assert notes[1]['release_observed'] and not notes[0]['release_observed'] and not notes[2]['release_observed']
    assert notes[0]['offset']<3. and abs(notes[1]['offset']-4.98)<1e-5


@pytest.mark.parametrize('reason',['duration_limit','input_boundary','forced_segment_boundary','unresolved_or_postprocessed'])
def test_unobserved_piano_end_keeps_attack_but_is_not_eligible_for_auto_ln(reason):
    from manialab.domain import Note
    from manialab.arrange import parameters,choose_holds
    from manialab.expression import propose_expression,expression_evidence
    p=parameters({'ln_ratio':1.})
    note=Note(1000,7000,pitch=60,strength=.9,source='pitch',voice='piano',id='p',release_reason=reason)
    selected,_=choose_holds([note],[p],p)
    assert selected[0].start==1000 and selected[0].end==1000
    analysis=dict(duration=10,candidates=[note.data()],structure=dict(segments=[dict(start_ms=0,end_ms=10000)]))
    plan=propose_expression(analysis,p)
    selected,_,trace=expression_evidence(analysis,plan,p)
    assert selected[0].start==1000 and selected[0].end==1000
    assert trace['p']['reason']=='unobserved_release'
    assert analysis['candidates'][0]['end']==7000


def test_specialist_cache_tracks_input_and_does_not_overwrite_general_activations(tmp_path,monkeypatch):
    import hashlib
    import onnxruntime as ort
    from manialab import piano
    detector=piano.PianoDetector(tmp_path)
    detector.model.parent.mkdir(parents=True)
    detector.model.write_bytes(b'fake-model')
    monkeypatch.setattr(piano,'MODEL_SHA256',hashlib.sha256(b'fake-model').hexdigest())
    monkeypatch.setattr(ort,'InferenceSession',lambda *a,**k:object())
    calls=[]
    def prediction(*args):
        calls.append(1)
        return {key:np.zeros((100,88),dtype=np.float32) for key in OUTPUTS}
    monkeypatch.setattr(piano,'infer',prediction)
    cache=tmp_path/'piano-specialist-activations.npz'
    general=tmp_path/'piano-activations.npz';general.write_bytes(b'existing-general-cache')
    samples=np.zeros(16000,dtype=np.float32)
    for count,cached in [(1,False),(1,True)]:
        notes,report=detector.analyze(samples,16000,cache,'same-stem')
        assert not notes and report['cached']==cached and len(calls)==count
    samples[100]=.5
    detector.analyze(samples,16000,cache,'same-stem')
    assert len(calls)==2
    detector.analyze(samples,16000,cache,'changed-stem-identity')
    assert len(calls)==3 and general.read_bytes()==b'existing-general-cache'


def test_deep_analysis_routes_only_piano_to_specialist(tmp_path,monkeypatch):
    from pathlib import Path
    from manialab import audio
    from manialab.arrange import parameters
    from manialab.domain import Note
    samples=np.full(audio.SR*2,.1,dtype=np.float32)
    np.save(tmp_path/'piano.npy',samples)
    report=dict(source='provided',voices=['piano'],audio_sha256='audio',set_sha256='stems',model_sha256='stems',cached=True)
    class Sources:
        def separate(self,*args):return tmp_path,report
    class Specialist:
        def analyze(self,stem,rate,cache,identity,*args):
            np.testing.assert_array_equal(stem,samples)
            assert rate==audio.SR and cache.name=='piano-specialist-activations.npz'
            return [Note(400,1200,pitch=60,source='pitch',strength=.9,id='sp',release_reason='release_regression')],{'cached':True}
    monkeypatch.setattr(audio,'decode',lambda *args:samples)
    def unused(*args):raise AssertionError('The general piano model must not also run')
    monkeypatch.setattr(audio,'infer',unused)
    result=audio.analyze_deep(Path('mix.wav'),tmp_path,Path('unused.onnx'),parameters(),Sources(),piano_detector=Specialist())
    assert result['piano_engine']=='specialist' and result['cached']
    assert result['candidates'][0]['start']==400 and result['candidates'][0]['release_reason']=='release_regression'
    assert result['voices'][0]['detector']=='piano-specialist'
    report['voices']=['bass']
    with pytest.raises(ValueError,match='钢琴声轨'):
        audio.analyze_deep(Path('mix.wav'),tmp_path,Path('unused.onnx'),parameters(),Sources(),piano_detector=Specialist())


def test_piano_api_requires_explicit_selection_and_an_isolated_piano_input(tmp_path,monkeypatch):
    import io
    from pathlib import Path
    from manialab import server
    from tests.test_server import wait_job
    calls=[]
    def analyze(*args,**kwargs):
        calls.append(kwargs.get('piano_detector'))
        return dict(candidates=[],duration=2,tempo=dict(bpm=120,offset=0),piano_engine='specialist' if calls[-1] else 'general')
    monkeypatch.setattr(server,'analyze_deep',analyze)
    monkeypatch.setattr(server.PianoDetector,'status',lambda self:dict(available=True))
    app=server.create_app(Path(__file__).parents[1],tmp_path)
    client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    try:
        pid=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json['id']
        url=f'/api/projects/{pid}/analyze'
        for engine in ('basic','deep'):
            response=client.post(url,json=dict(engine=engine,piano_engine='specialist'),headers=headers)
            assert response.status_code==400
        response=client.post(url,json=dict(engine='deep6',piano_engine='invalid'),headers=headers)
        assert response.status_code==400
        response=client.post(url,json=dict(engine='deep6',piano_engine='specialist'),headers=headers)
        assert wait_job(client,response.json)['status']=='done'
        assert isinstance(calls[-1],server.PianoDetector)
        response=client.post(url,json=dict(engine='deep6'),headers=headers)
        assert wait_job(client,response.json)['status']=='done'
        assert calls[-1] is None
    finally:
        app.extensions['executor'].shutdown()
