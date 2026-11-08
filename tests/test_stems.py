from pathlib import Path
import numpy as np
import pytest

from manialab import audio
from manialab.arrange import parameters,select_notes
from manialab.domain import Note
from manialab.stems import stem_path


def test_current_separation_selects_its_own_stems_and_rejects_unknown_voice(tmp_path):
    assert stem_path(tmp_path,{},'bass')==tmp_path/'stems/bass.wav'
    six={'separation':{'profile':'six'}}
    assert stem_path(tmp_path,six,'piano')==tmp_path/'stems-six/piano.wav'
    for voice in ['piano','../bass','unknown']:
        with pytest.raises(ValueError):stem_path(tmp_path,{},voice)


def test_six_source_pitch_cache_depends_on_separation_model(tmp_path,monkeypatch):
    directory=tmp_path/'stems-six';directory.mkdir()
    for voice in ('drums','bass','vocals','other','guitar','piano'):
        np.save(directory/f'{voice}.npy',np.zeros(audio.SR*2,dtype=np.float32))
    report=dict(profile='six',audio_sha256='same-audio',model_sha256='model-a',cached=True)
    class Cached:
        def separate(self,*args):return directory,report
    calls=[]
    monkeypatch.setattr(audio,'decode',lambda *a:np.zeros(audio.SR*2,dtype=np.float32))
    def infer(*args):
        calls.append(1)
        return np.zeros((10,88)),np.zeros((10,88)),np.arange(10)/86
    monkeypatch.setattr(audio,'infer',infer)
    for expected,cached in [(5,False),(5,True)]:
        result=audio.analyze_deep(Path('same.wav'),tmp_path,Path('model.onnx'),parameters(),Cached())
        assert len(calls)==expected and result['cached']==cached
        assert {v['voice'] for v in result['voices']}=={'drums','bass','vocals','other','guitar','piano'}
    report['model_sha256']='model-b'
    result=audio.analyze_deep(Path('same.wav'),tmp_path,Path('model.onnx'),parameters(),Cached())
    assert len(calls)==10 and not result['cached']


def test_six_instrument_weights_control_only_the_requested_voice():
    notes=[Note(1000,2000,pitch=60,source='pitch',voice='guitar',id='guitar'),
           Note(1000,2000,pitch=72,source='pitch',voice='piano',id='piano')]
    selected,_=select_notes(notes,parameters({'guitar_weight':0,'piano_weight':1,'ln_ratio':1}))
    assert [n.id for n in selected]==['piano']
