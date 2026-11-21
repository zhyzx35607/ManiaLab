import io
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from manialab import audio
from manialab.arrange import parameters
from manialab.provided_stems import (
    ProvidedStemSeparator,prepare_provided_stems,provided_stem_identity,
)
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


@pytest.mark.parametrize('damage', ['list', 'null', 'decoded-null', 'voice-null', 'voice-list', 'voices'])
def test_invalid_provided_cache_manifest_is_rebuilt(tmp_path, damage):
    mix=wave(tmp_path/'mix.wav');piano=wave(tmp_path/'piano.wav')
    report=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    manifest=tmp_path/'stems-provided'/report['set_sha256']/'separation.json'
    damaged=json.loads(manifest.read_text(encoding='utf-8'))
    if damage == 'list':
        damaged=[]
    elif damage == 'null':
        damaged=None
    elif damage == 'decoded-null':
        damaged['decoded_files']=None
    elif damage == 'voice-null':
        damaged['decoded_files']['piano']=None
    elif damage == 'voice-list':
        damaged['decoded_files']['piano']=[]
    else:
        damaged['voices']=['drums']
    manifest.write_text(json.dumps(damaged),encoding='utf-8')

    repaired=prepare_provided_stems(mix,{'piano':piano},tmp_path)

    assert repaired['cached'] is False
    assert repaired['set_sha256']==report['set_sha256']
    assert json.loads(manifest.read_text(encoding='utf-8'))['decoded_files']['piano']['wav_sha256']
    assert repaired['voices']==['piano']
    ProvidedStemSeparator(repaired).separate(mix,tmp_path)
    assert not list((tmp_path/'stems-provided').glob('.provided-build-*'))
    assert not list((tmp_path/'stems-provided').glob('.provided-backup-*'))


def test_failed_cache_install_restores_the_existing_folder(tmp_path,monkeypatch):
    mix=wave(tmp_path/'mix.wav');piano=wave(tmp_path/'piano.wav')
    report=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    folder=tmp_path/'stems-provided'/report['set_sha256']
    original_files={name:(folder/name).read_bytes() for name in ('piano.npy','piano.wav')}
    (folder/'separation.json').write_text('[]',encoding='utf-8')
    real_replace=os.replace
    failed=False

    def fail_staged_install(source,destination):
        nonlocal failed
        source,destination=Path(source),Path(destination)
        if (not failed and source.parent==folder.parent
                and source.name.startswith('.provided-build-') and destination==folder):
            failed=True
            raise OSError('synthetic cache install failure')
        return real_replace(source,destination)

    monkeypatch.setattr('manialab.provided_stems.os.replace',fail_staged_install)
    with pytest.raises(OSError,match='synthetic cache install failure'):
        prepare_provided_stems(mix,{'piano':piano},tmp_path)

    assert failed
    assert {name:(folder/name).read_bytes() for name in original_files}==original_files
    assert (folder/'separation.json').read_text(encoding='utf-8')=='[]'
    assert not list(folder.parent.glob('.provided-build-*'))
    assert not list(folder.parent.glob('.provided-backup-*'))


def test_failed_install_and_restore_reports_preserved_backup(tmp_path,monkeypatch):
    mix=wave(tmp_path/'mix.wav');piano=wave(tmp_path/'piano.wav')
    report=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    folder=tmp_path/'stems-provided'/report['set_sha256']
    (folder/'separation.json').write_text('[]',encoding='utf-8')
    real_replace=os.replace
    callbacks=[]

    def fail_install_and_restore(source,destination):
        source,destination=Path(source),Path(destination)
        if source.name.startswith('.provided-build-') and destination==folder:
            raise OSError('synthetic cache install failure')
        if source.name.startswith('.provided-backup-') and destination==folder:
            raise OSError('synthetic cache restore failure')
        return real_replace(source,destination)

    monkeypatch.setattr('manialab.provided_stems.os.replace',fail_install_and_restore)
    with pytest.raises(OSError,match='synthetic cache install failure') as raised:
        prepare_provided_stems(mix,{'piano':piano},tmp_path,
                               on_new_folder=lambda path,**kwargs:callbacks.append((Path(path),kwargs)))

    replacements=[(path,details) for path,details in callbacks if 'backup' in details]
    assert len(replacements)==1
    canonical,details=replacements[0]
    backup=details['backup']
    assert canonical==folder
    assert backup.is_dir()
    assert str(backup) in '\n'.join(raised.value.__notes__)
    assert (backup/'separation.json').read_text(encoding='utf-8')=='[]'


def test_provided_import_rejects_length_mismatch_and_cancelled_set_without_committing(tmp_path):
    mix=wave(tmp_path/'mix.wav');piano=wave(tmp_path/'piano.wav',1)
    with pytest.raises(ValueError,match='长度'):
        prepare_provided_stems(mix,{'piano':piano},tmp_path)
    assert not list(tmp_path.glob('stems-provided/*/separation.json'))
    wave(piano)
    with pytest.raises(InterruptedError):prepare_provided_stems(mix,{'piano':piano},tmp_path,cancelled=lambda:True)
    assert not list(tmp_path.glob('stems-provided/*/separation.json'))


def test_provided_import_hashes_and_decodes_the_same_source_snapshots(tmp_path,monkeypatch):
    mix=tmp_path/'mix.wav';piano=tmp_path/'piano.wav'
    mix_a,piano_a=b'mix snapshot A',b'piano snapshot A'
    mix.write_bytes(mix_a);piano.write_bytes(piano_a)
    decoded=[]

    def decode_snapshot(path,work,cancelled=None):
        payload=Path(path).read_bytes()
        decoded.append(payload)
        if len(decoded)==1:
            mix.write_bytes(b'mix replacement B')
            piano.write_bytes(b'piano replacement B')
        return np.full(22050,.1,dtype=np.float32)

    monkeypatch.setattr('manialab.provided_stems.decode',decode_snapshot)
    report=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    assert decoded==[mix_a,piano_a]
    assert report['audio_sha256']==hashlib.sha256(mix_a).hexdigest()
    assert report['sources']['piano']['sha256']==hashlib.sha256(piano_a).hexdigest()
    assert not list(tmp_path.glob('.provided-inputs-*'))


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
    # These stems never pass through the model decoder; its revision is irrelevant.
    from manialab import separation
    monkeypatch.setattr(separation,'INPUT_PROCESSING',{'revision':999})
    result=audio.analyze_deep(mix,tmp_path,Path('unused.onnx'),parameters(),ProvidedStemSeparator(report))
    assert len(calls)==1 and result['cached']
    # Changing source content creates a new set, while old analysis stays valid.
    old=stem_path(tmp_path,{'separation':report},'piano')
    wave(piano,shift=100)
    changed=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    assert old!=stem_path(tmp_path,{'separation':changed},'piano')
    audio.analyze_deep(mix,tmp_path,Path('unused.onnx'),parameters(),ProvidedStemSeparator(changed))
    assert len(calls)==2
    ProvidedStemSeparator(report).separate(mix,tmp_path)


def test_imported_float_stem_preserves_overrange_samples_in_analysis(tmp_path):
    samples=np.zeros(22050,dtype=np.float32)
    samples[2205:2208]=[1.5,-1.25,.00001]
    mix=tmp_path/'mix.wav';piano=tmp_path/'piano.wav'
    wavfile.write(mix,22050,samples);wavfile.write(piano,22050,samples)
    report=prepare_provided_stems(mix,{'piano':piano},tmp_path)
    folder,_=ProvidedStemSeparator(report).separate(mix,tmp_path)
    np.testing.assert_array_equal(np.load(folder/'piano.npy'),samples)


def test_provided_upload_keeps_existing_analysis_and_old_set_on_failure(tmp_path):
    from manialab.chart_workspace import audio_work
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
        work=audio_work(Projects(tmp_path/'data/projects').path(pid),saved)
        upload_root=work/'provided-uploads'
        decoded_root=work/'stems-provided'
        previous_uploads=set(upload_root.iterdir())
        previous_decoded=set(decoded_root.iterdir())
        response=upload(short);assert wait_job(client,response.json)['status']=='error'
        assert client.get(f'/api/projects/{pid}').json==saved
        assert set(upload_root.iterdir())==previous_uploads
        assert set(decoded_root.iterdir())==previous_decoded
        assert client.post(f'/api/projects/{pid}/provided-stems',
            data={'unknown':(io.BytesIO(mix.read_bytes()),'bad.wav')},headers=headers).status_code==400
    finally:app.extensions['executor'].shutdown()


def test_upload_failure_cleans_the_folder_for_the_actual_snapshot_key(tmp_path,monkeypatch):
    from manialab.chart_workspace import audio_work
    from manialab.server import create_app
    from tests.test_server import wait_job
    from manialab.storage import Projects

    root=Path(__file__).resolve().parents[1]
    mix=wave(tmp_path/'mix.wav')
    app=create_app(root,tmp_path/'data');client=app.test_client()
    headers={'X-ManiaLab-Token':app.config['TOKEN']}
    services=app.extensions['analysis_services']
    prepare=services.prepare_provided_stems
    try:
        state=client.post('/api/import',data={
            'files':(io.BytesIO(mix.read_bytes()),'mix.wav')},headers=headers).json
        pid=state['id']

        def upload():
            return client.post(f'/api/projects/{pid}/provided-stems',
                data={'piano':(io.BytesIO(mix.read_bytes()),'piano.wav')},headers=headers)

        first=upload()
        assert first.status_code==200
        assert wait_job(client,first.json)['status']=='done'
        saved=client.get(f'/api/projects/{pid}').json
        assert saved['provided_stems']['set_sha256']
        project=Projects(tmp_path/'data/projects').path(pid)
        work=audio_work(project,saved)
        upload_root=work/'provided-uploads'
        decoded_root=work/'stems-provided'
        old_uploads=set(upload_root.iterdir())
        old_decoded=set(decoded_root.iterdir())
        old_key=saved['provided_stems']['set_sha256']
        changed_keys=[]

        def replace_then_fail(mix_path,sources,work,progress,cancelled,**kwargs):
            assert provided_stem_identity(mix_path,sources)[1]==old_key
            for path in sources.values():wave(path,shift=100)
            report=prepare(mix_path,sources,work,progress,cancelled,**kwargs)
            assert report['set_sha256']!=old_key
            changed_keys.append(report['set_sha256'])
            assert (Path(work)/'stems-provided'/report['set_sha256']/'separation.json').is_file()
            raise OSError('synthetic failure after snapshot import')

        monkeypatch.setattr(services,'prepare_provided_stems',replace_then_fail)
        failed=upload()
        assert failed.status_code==200
        result=wait_job(client,failed.json)
        assert result['status']=='error'
        assert changed_keys
        assert client.get(f'/api/projects/{pid}').json==saved
        assert set(upload_root.iterdir())==old_uploads
        assert set(decoded_root.iterdir())==old_decoded
        assert not (decoded_root/changed_keys[0]).exists()
    finally:
        app.extensions['executor'].shutdown()
