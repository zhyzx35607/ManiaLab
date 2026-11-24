import hashlib
from pathlib import Path
import numpy as np
import pytest

from manialab import audio
from manialab.arrange import parameters,select_notes
from manialab.domain import Note
from manialab.stems import separation_identity, separation_label, separation_voices, stem_path


def test_current_separation_selects_its_own_stems_and_rejects_unknown_voice(tmp_path):
    assert stem_path(tmp_path,{},'bass')==tmp_path/'stems/bass.wav'
    six={'separation':{'profile':'six'}}
    assert stem_path(tmp_path,six,'piano')==tmp_path/'stems-six/piano.wav'
    four_piano={'separation':{'profile':'four-piano'}}
    assert stem_path(tmp_path,four_piano,'piano')==tmp_path/'stems-four-piano/piano.wav'
    assert separation_voices(four_piano['separation'])==('drums','piano','vocals','other')
    assert separation_label(four_piano['separation']).startswith('四主声部')
    assert separation_identity(four_piano['separation'] | {'audio_sha256':'a','model_sha256':'b'}) != \
        separation_identity(six['separation'] | {'audio_sha256':'a','model_sha256':'b'})
    for voice in ('bass','guitar'):
        with pytest.raises(ValueError):stem_path(tmp_path,four_piano,voice)
    for voice in ['piano','../bass','unknown']:
        with pytest.raises(ValueError):stem_path(tmp_path,{},voice)


def test_four_piano_profile_analyzes_only_its_four_saved_stems(tmp_path,monkeypatch):
    directory=tmp_path/'stems-four-piano';directory.mkdir()
    model=tmp_path/'basic-pitch.onnx';model.write_bytes(b'synthetic model')
    source=tmp_path/'same.wav';source.write_bytes(b'synthetic same mix')
    audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    voices=('drums','piano','vocals','other')
    for voice in voices:
        np.save(directory/f'{voice}.npy',np.zeros(audio.SR*2,dtype=np.float32))
    report=dict(profile='four-piano',audio_sha256=audio_sha256,model_sha256='six-model',cached=True)

    class Cached:
        def separate(self,*args):return directory,report

    calls=[]
    monkeypatch.setattr(audio,'decode',lambda *a,**kwargs:np.zeros(audio.SR*2,dtype=np.float32))
    def infer(*args):
        calls.append(1)
        return np.zeros((10,88)),np.zeros((10,88)),np.arange(10)/86
    monkeypatch.setattr(audio,'infer',infer)

    result=audio.analyze_deep(source,tmp_path,model,parameters(),Cached())

    assert len(calls)==3  # piano, vocals and other; drums use transient analysis
    assert {v['voice'] for v in result['voices']}==set(voices)
    assert {n['voice'] for n in result['candidates']} <= set(voices)
    assert not (directory/'bass.npy').exists() and not (directory/'guitar.npy').exists()


def test_six_source_pitch_cache_depends_on_separation_model(tmp_path,monkeypatch):
    directory=tmp_path/'stems-six';directory.mkdir()
    model=tmp_path/'basic-pitch.onnx';model.write_bytes(b'synthetic model A')
    source=tmp_path/'same.wav';source.write_bytes(b'synthetic same mix')
    audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    for voice in ('drums','bass','vocals','other','guitar','piano'):
        np.save(directory/f'{voice}.npy',np.zeros(audio.SR*2,dtype=np.float32))
    report=dict(profile='six',audio_sha256=audio_sha256,model_sha256='model-a',cached=True)
    for voice in ('bass','vocals','other','guitar','piano'):
        np.savez_compressed(directory/f'{voice}-activations.npz',version=2,
            audio_sha256=audio_sha256,separation_sha256='model-a',
            frames=np.ones((10,88)),onsets=np.ones((10,88)),times=np.arange(10)/86)
    class Cached:
        def separate(self,*args):return directory,report
    calls=[]
    monkeypatch.setattr(audio,'decode',lambda *a,**kwargs:np.zeros(audio.SR*2,dtype=np.float32))
    def infer(*args):
        calls.append(1)
        return np.zeros((10,88)),np.zeros((10,88)),np.arange(10)/86
    monkeypatch.setattr(audio,'infer',infer)
    for expected,cached in [(5,False),(5,True)]:
        result=audio.analyze_deep(source,tmp_path,model,parameters(),Cached())
        assert len(calls)==expected and result['cached']==cached
        assert {v['voice'] for v in result['voices']}=={'drums','bass','vocals','other','guitar','piano'}
    model.write_bytes(b'synthetic model B')
    result=audio.analyze_deep(source,tmp_path,model,parameters(),Cached())
    assert len(calls)==10 and not result['cached']
    report['model_sha256']='model-b'
    result=audio.analyze_deep(source,tmp_path,model,parameters(),Cached())
    assert len(calls)==15 and not result['cached']
    report['input_processing']={'revision':2,'codec':'pcm_f32le'}
    result=audio.analyze_deep(source,tmp_path,model,parameters(),Cached())
    assert len(calls)==20 and not result['cached']
    # Volatile progress/beat metadata is not part of the separated audio identity.
    report['elapsed_seconds']=500
    result=audio.analyze_deep(source,tmp_path,model,parameters(),Cached())
    assert len(calls)==20 and result['cached']


def test_six_instrument_weights_control_only_the_requested_voice():
    notes=[Note(1000,2000,pitch=60,source='pitch',voice='guitar',id='guitar'),
           Note(1000,2000,pitch=72,source='pitch',voice='piano',id='piano')]
    selected,_=select_notes(notes,parameters({'guitar_weight':0,'piano_weight':1,'ln_ratio':1}))
    assert [n.id for n in selected]==['piano']
