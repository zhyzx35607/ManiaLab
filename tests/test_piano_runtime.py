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


@pytest.mark.parametrize('reason',['duration_limit','input_boundary','forced_segment_boundary','unresolved_or_postprocessed','frame_inactive','reattack'])
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


def test_inactive_frame_is_not_a_detected_decay_or_release():
    arrays={key:np.zeros((100,88),dtype=np.float32) for key in OUTPUTS}
    # Three equal onset candidates. The middle pitch really is active for one
    # frame; the others have no sustained activation, even with an offset peak.
    arrays['reg_onset_output'][49:52,:3]=np.array([.2,.8,.2])[:,None]
    arrays['frame_output'][50,1]=.9
    arrays['reg_offset_output'][50:53,2]=[.2,.8,.2]
    notes=decode(arrays)
    assert len(notes)==3
    assert [n['release_reason'] for n in notes]==['frame_inactive','frame_decay','frame_inactive']
    assert [n['release_observed'] for n in notes]==[False,True,False]
    # No minimum-length filter: every attack and original raw interval remains.
    assert all(n['onset']==.5 and abs(n['offset']-.51)<1e-7 for n in notes)


def test_specialist_cache_tracks_input_and_does_not_overwrite_general_activations(tmp_path,monkeypatch):
    import hashlib
    import onnxruntime as ort
    from manialab import piano
    monkeypatch.setattr(piano,'MODEL_SHA256',hashlib.sha256(b'fake-model').hexdigest())
    monkeypatch.setattr(piano,'MODEL_BYTES',len(b'fake-model'))
    detector=piano.PianoDetector(tmp_path)
    detector.model.parent.mkdir(parents=True)
    detector.model.write_bytes(b'fake-model')
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


def test_specialist_cache_recovers_from_npz_crc_corruption(tmp_path,monkeypatch):
    import hashlib
    import json
    import struct
    import zipfile
    import onnxruntime as ort
    from manialab import piano
    monkeypatch.setattr(piano,'MODEL_SHA256',hashlib.sha256(b'fake-model').hexdigest())
    monkeypatch.setattr(piano,'MODEL_BYTES',len(b'fake-model'))
    detector=piano.PianoDetector(tmp_path)
    detector.model.parent.mkdir(parents=True)
    detector.model.write_bytes(b'fake-model')
    monkeypatch.setattr(ort,'InferenceSession',lambda *a,**k:object())
    calls=[]
    def prediction(*args):
        calls.append(1)
        return {key:np.zeros((100,88),dtype=np.float32) for key in OUTPUTS}
    monkeypatch.setattr(piano,'infer',prediction)

    samples=np.zeros(16000,dtype=np.float32)
    identity=dict(revision=2,model_sha256=piano.MODEL_SHA256,stem_identity='same-stem',
                  pcm_sha256=hashlib.sha256(np.asarray(samples,dtype='<f4').tobytes()).hexdigest(),
                  sample_rate=16000,samples=len(samples))
    cache=tmp_path/'piano-specialist-activations.npz'
    np.savez(cache,identity=json.dumps(identity,sort_keys=True),continuations='[]',
             **{key:np.zeros((100,88),dtype=np.float32) for key in OUTPUTS})

    damaged=bytearray(cache.read_bytes())
    with zipfile.ZipFile(cache) as archive:
        member=archive.getinfo('reg_onset_output.npy')
        assert member.compress_type==zipfile.ZIP_STORED
        name_length,extra_length=struct.unpack_from('<HH',damaged,member.header_offset+26)
        data_offset=member.header_offset+30+name_length+extra_length
        damaged[data_offset+member.file_size-1]^=1
    cache.write_bytes(damaged)

    notes,report=detector.analyze(samples,16000,cache,'same-stem')
    assert not notes and report['cached'] is False and len(calls)==1
    notes,report=detector.analyze(samples,16000,cache,'same-stem')
    assert not notes and report['cached'] is True and len(calls)==1


def test_specialist_cache_recovers_from_deflate_stream_error(tmp_path,monkeypatch):
    import hashlib
    import json
    import struct
    import zipfile
    import onnxruntime as ort
    from manialab import piano
    monkeypatch.setattr(piano,'MODEL_SHA256',hashlib.sha256(b'fake-model').hexdigest())
    monkeypatch.setattr(piano,'MODEL_BYTES',len(b'fake-model'))
    detector=piano.PianoDetector(tmp_path)
    detector.model.parent.mkdir(parents=True)
    detector.model.write_bytes(b'fake-model')
    monkeypatch.setattr(ort,'InferenceSession',lambda *a,**k:object())
    calls=[]
    def prediction(*args):
        calls.append(1)
        return {key:np.zeros((100,88),dtype=np.float32) for key in OUTPUTS}
    monkeypatch.setattr(piano,'infer',prediction)

    samples=np.zeros(16000,dtype=np.float32)
    identity=dict(revision=2,model_sha256=piano.MODEL_SHA256,stem_identity='same-stem',
                  pcm_sha256=hashlib.sha256(np.asarray(samples,dtype='<f4').tobytes()).hexdigest(),
                  sample_rate=16000,samples=len(samples))
    cache=tmp_path/'piano-specialist-activations.npz'
    np.savez_compressed(cache,identity=json.dumps(identity,sort_keys=True),continuations='[]',
                        **{key:np.zeros((100,88),dtype=np.float32) for key in OUTPUTS})

    damaged=bytearray(cache.read_bytes())
    with zipfile.ZipFile(cache) as archive:
        member=archive.getinfo('reg_onset_output.npy')
        assert member.compress_type==zipfile.ZIP_DEFLATED
        name_length,extra_length=struct.unpack_from('<HH',damaged,member.header_offset+26)
        data_offset=member.header_offset+30+name_length+extra_length
        damaged[data_offset]=0xff
    cache.write_bytes(damaged)

    notes,report=detector.analyze(samples,16000,cache,'same-stem')
    assert not notes and report['cached'] is False and len(calls)==1
    notes,report=detector.analyze(samples,16000,cache,'same-stem')
    assert not notes and report['cached'] is True and len(calls)==1


def test_deep_analysis_routes_only_piano_to_specialist(tmp_path,monkeypatch):
    import hashlib
    from pathlib import Path
    from manialab import audio
    from manialab.arrange import parameters
    from manialab.domain import Note
    samples=np.full(audio.SR*2,.1,dtype=np.float32)
    np.save(tmp_path/'piano.npy',samples)
    source=tmp_path/'mix.wav'
    source.write_bytes(b'synthetic mix')
    report=dict(source='provided',voices=['piano'],audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),set_sha256='stems',model_sha256='stems',cached=True)
    class Sources:
        def separate(self,*args):return tmp_path,report
    class Specialist:
        def analyze(self,stem,rate,cache,identity,*args):
            np.testing.assert_array_equal(stem,samples)
            assert rate==audio.SR and cache.name=='piano-specialist-activations.npz'
            return [Note(400,1200,pitch=60,source='pitch',strength=.9,id='sp',release_reason='release_regression')],{'cached':True}
    monkeypatch.setattr(audio,'decode',lambda *args,**kwargs:samples)
    def unused(*args):raise AssertionError('The general piano model must not also run')
    monkeypatch.setattr(audio,'infer',unused)
    result=audio.analyze_deep(source,tmp_path,Path('unused.onnx'),parameters(),Sources(),piano_detector=Specialist())
    assert result['piano_engine']=='specialist' and result['cached']
    assert result['candidates'][0]['start']==400 and result['candidates'][0]['release_reason']=='release_regression'
    assert result['voices'][0]['detector']=='piano-specialist'
    report['voices']=['bass']
    with pytest.raises(ValueError,match='钢琴声轨'):
        audio.analyze_deep(source,tmp_path,Path('unused.onnx'),parameters(),Sources(),piano_detector=Specialist())


def test_piano_api_requires_explicit_selection_and_an_isolated_piano_input(tmp_path,monkeypatch):
    import io
    from pathlib import Path
    from manialab import server
    from tests.test_server import wait_job
    calls=[]
    def analyze(*args,**kwargs):
        calls.append(kwargs.get('piano_detector'))
        return dict(candidates=[],duration=2,tempo=dict(bpm=120,offset=0),piano_engine='specialist' if calls[-1] else 'general')
    monkeypatch.setattr(server.PianoDetector,'status',lambda self:dict(available=True))
    app=server.create_app(Path(__file__).parents[1],tmp_path)
    monkeypatch.setattr(app.extensions['analysis_services'],'analyze_deep',analyze)
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
