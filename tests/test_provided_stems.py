import io
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from manialab import audio
from manialab.arrange import parameters
from manialab.provided_stems import ProvidedStemSeparator,prepare_provided_stems
from manialab.stems import stem_path


def wave(path,seconds=2,shift=0):
    samples=np.zeros(round(22050*seconds),dtype=np.int16)
    samples[2205+shift:2305+shift]=12000
    wavfile.write(path,22050,samples)
    return path


def test_provided_source_retains_silence_sample_clock_and_checks_its_identity(tmp_path):
    mix=wave(tmp_path/'mix.wav');piano=wave(tmp_path/'piano.wav')
    report=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    assert report['source']=='provided' and report['voices']==['piano']
    folder,checked=ProvidedStemSeparator(report).separate(mix,tmp_path)
    samples=np.load(folder/'piano.npy')
    assert np.flatnonzero(samples)[0]==2205
    assert stem_path(tmp_path,{'separation':report},'piano')==folder/'piano.wav'
    assert checked['cached']
    assert prepare_provided_stems(mix,{'piano':piano},tmp_path)['cached']
    (folder/'piano.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='改变'):ProvidedStemSeparator(report).separate(mix,tmp_path)
    repaired=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    assert not repaired['cached']
    ProvidedStemSeparator(repaired).separate(mix,tmp_path)
    with pytest.raises(ValueError):stem_path(tmp_path,{'separation':report},'drums')
    with pytest.raises(ValueError):stem_path(tmp_path,{'separation':report|{'set_sha256':'../../escape'}},'piano')


def test_provided_import_rejects_length_mismatch_and_cancelled_set_without_committing(tmp_path):
    mix=wave(tmp_path/'mix.wav');piano=wave(tmp_path/'piano.wav',1)
    with pytest.raises(ValueError,match='长度'):
        prepare_provided_stems(mix,{'piano':piano},tmp_path)
    assert not list(tmp_path.glob('stems-provided/*/separation.json'))
    wave(piano)
    with pytest.raises(InterruptedError):prepare_provided_stems(mix,{'piano':piano},tmp_path,cancelled=lambda:True)
    assert not list(tmp_path.glob('stems-provided/*/separation.json'))


def test_imported_piano_only_analysis_needs_no_separator_and_invalidates_cache_by_source(tmp_path,monkeypatch):
    mix=wave(tmp_path/'mix.wav');piano=wave(tmp_path/'piano.wav')
    report=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    calls=[]
    def infer(*args):
        calls.append(1)
        return np.zeros((10,88)),np.zeros((10,88)),np.arange(10)/86
    monkeypatch.setattr(audio,'infer',infer)
    for expected in (1,1):
        result=audio.analyze_deep(mix,tmp_path,Path('unused.onnx'),parameters(),ProvidedStemSeparator(report))
        assert len(calls)==expected
        assert [s['voice'] for s in result['voices']]==['piano']
        assert result['tempo']['evidence_source']=='mix'
        assert 'drums' not in result['tempo']['comparisons']
        assert 'Beat This' not in result['engine']
    # Changing source content creates a new set, while old analysis stays valid.
    old=stem_path(tmp_path,{'separation':report},'piano')
    wave(piano,shift=100)
    changed=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    assert old!=stem_path(tmp_path,{'separation':changed},'piano')
    audio.analyze_deep(mix,tmp_path,Path('unused.onnx'),parameters(),ProvidedStemSeparator(changed))
    assert len(calls)==2
    ProvidedStemSeparator(report).separate(mix,tmp_path)


def test_provided_upload_keeps_existing_analysis_and_old_set_on_failure(tmp_path):
    from manialab.server import create_app
    from tests.test_server import wait_job
    from manialab.storage import Projects
    root=Path(__file__).resolve().parents[1]
    mix=wave(tmp_path/'mix.wav');short=wave(tmp_path/'short.wav',1)
    app=create_app(root,tmp_path/'data');client=app.test_client()
    headers={'X-ManiaLab-Token':app.config['TOKEN']}
    try:
        state=client.post('/api/import',data={'files':(io.BytesIO(mix.read_bytes()),'mix.wav')},headers=headers).json
        pid=state['id'];state['analysis']={'sentinel':'previous_analysis'}
        Projects(tmp_path/'data/projects').save(state)
        def upload(path):return client.post(f'/api/projects/{pid}/provided-stems',
            data={'piano':(io.BytesIO(path.read_bytes()),'钢琴.wav')},headers=headers)
        response=upload(mix);assert response.status_code==200,response.json
        assert wait_job(client,response.json)['status']=='done'
        saved=client.get(f'/api/projects/{pid}').json
        assert saved['analysis']==state['analysis'] and saved['provided_stems']['sources']['piano']['filename']=='钢琴.wav'
        response=upload(short);assert wait_job(client,response.json)['status']=='error'
        assert client.get(f'/api/projects/{pid}').json==saved
        assert client.post(f'/api/projects/{pid}/provided-stems',
            data={'unknown':(io.BytesIO(mix.read_bytes()),'bad.wav')},headers=headers).status_code==400
    finally:app.extensions['executor'].shutdown()
