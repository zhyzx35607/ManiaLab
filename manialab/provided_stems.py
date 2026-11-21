"""User-supplied source audio on the original song clock, without reseparation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
from scipy.io import wavfile

from .audio_config import SR
from .audio_evidence import align_stem_samples
from .audio_io import decode
from .stems import VOICES,stem_path,separation_voices


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def provided_stem_identity(mix, sources):
    origin={voice:dict(filename=path.name,sha256=digest(path)) for voice,path in sources.items()}
    identity=dict(revision=1,audio_sha256=digest(mix),sources=origin)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return identity,key


def _copy_snapshot(source, target, cancelled):
    source=Path(source)
    digest_value=hashlib.sha256()
    with source.open('rb') as original, Path(target).open('xb') as copied:
        while True:
            if cancelled():raise InterruptedError('已取消声轨快照')
            chunk=original.read(1024*1024)
            if not chunk:break
            copied.write(chunk)
            digest_value.update(chunk)
    if cancelled():raise InterruptedError('已取消声轨快照')
    return digest_value.hexdigest()


def _identity_from_hashes(mix_sha256, sources, source_hashes):
    origin={voice:dict(filename=Path(path).name,sha256=source_hashes[voice])
            for voice,path in sources.items()}
    identity=dict(revision=1,audio_sha256=mix_sha256,sources=origin)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return identity,key


def _remove_path(path):
    path=Path(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _install_snapshot(staging, folder, on_new_folder):
    if not folder.exists():
        staging.replace(folder)
        if on_new_folder is not None:
            on_new_folder(folder)
        return

    backup=folder.with_name('.provided-backup-'+uuid.uuid4().hex)
    os.replace(folder,backup)
    try:
        os.replace(staging,folder)
    except BaseException as error:
        try:
            os.replace(backup,folder)
        except OSError as restore_error:
            if on_new_folder is not None:
                on_new_folder(folder, backup=backup)
            error.add_note(f'旧声轨缓存无法恢复，保留备份于 {backup}: {restore_error!r}')
        raise
    if on_new_folder is not None:
        on_new_folder(folder, backup=backup)
    else:
        _remove_path(backup)


def prepare_provided_stems(mix, sources, work, progress=lambda p,s:None, cancelled=lambda:False,
                           *, on_new_folder=None):
    """Hash and decode one private snapshot of the mix and every supplied stem."""
    if not sources or set(sources)-set(VOICES):raise ValueError('至少选择一个有效声部')
    work=Path(work)
    with tempfile.TemporaryDirectory(prefix='.provided-inputs-',dir=work) as temporary_inputs:
        snapshot_dir=Path(temporary_inputs)
        mix_path=Path(mix)
        mix_snapshot=snapshot_dir/('mix'+mix_path.suffix)
        mix_hash=_copy_snapshot(mix_path,mix_snapshot,cancelled)
        source_snapshots={}
        source_hashes={}
        for voice,path in sources.items():
            source_path=Path(path)
            snapshot=snapshot_dir/(voice+source_path.suffix)
            source_hashes[voice]=_copy_snapshot(source_path,snapshot,cancelled)
            source_snapshots[voice]=snapshot
        identity,key=_identity_from_hashes(mix_hash,sources,source_hashes)
        expected_voices=[v for v in VOICES if v in sources]
        storage_root=work/'stems-provided'
        folder=storage_root/key
        manifest=folder/'separation.json'
        if manifest.exists():
            try:
                previous=json.loads(manifest.read_text(encoding='utf-8'))
                decoded_files=previous.get('decoded_files') if isinstance(previous,dict) else None
                valid_files=isinstance(decoded_files,dict) and all(
                    isinstance(decoded_files.get(voice),dict)
                    and all(isinstance(decoded_files[voice].get(extension+'_sha256'),str)
                        and (folder/f'{voice}.{extension}').is_file()
                        and digest(folder/f'{voice}.{extension}')==
                            decoded_files[voice][extension+'_sha256']
                        for extension in ('npy','wav'))
                    for voice in sources)
                same_identity=(isinstance(previous,dict)
                    and previous.get('set_sha256')==key
                    and previous.get('audio_sha256')==identity['audio_sha256']
                    and previous.get('sources')==identity['sources']
                    and previous.get('voices')==expected_voices)
                if same_identity and valid_files:
                    return previous|{'cached':True}
            except (OSError,ValueError,TypeError):
                pass
        storage_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.provided-build-',dir=storage_root) as temporary_build:
            staging=Path(temporary_build)
            if on_new_folder is not None:
                on_new_folder(staging)
            progress(3,'核对原曲与导入声轨的时间轴')
            mixture=decode(mix_snapshot,staging,cancelled=cancelled)
            files={};padding={}
            for index,(voice,snapshot) in enumerate(source_snapshots.items()):
                if cancelled():raise InterruptedError('已取消声轨导入')
                progress(5+int(index*85/len(sources)),'读取 '+VOICES[voice])
                samples=decode(snapshot,staging,cancelled=cancelled)
                if abs(len(samples)-len(mixture))>2:
                    raise ValueError(f'{VOICES[voice]}长度 {len(samples)/SR:.3f} 秒，与原曲 {len(mixture)/SR:.3f} 秒不同；请导出同起点、同长度的声轨，保留开头静音')
                samples,padding[voice]=align_stem_samples(samples,len(mixture))
                np.save(staging/f'{voice}.npy',samples,allow_pickle=False)
                peak=max(1.,float(np.abs(samples).max()))
                wavfile.write(staging/f'{voice}.wav',SR,(np.clip(samples/peak,-1,1)*32767).astype(np.int16))
                files[voice]=dict(npy_sha256=digest(staging/f'{voice}.npy'),wav_sha256=digest(staging/f'{voice}.wav'))
            if cancelled():raise InterruptedError('已取消声轨导入')
            report=dict(**identity,source='provided',set_sha256=key,model_sha256=key,
                voices=expected_voices,duration_seconds=len(mixture)/SR,
                sample_rate=SR,decoded_files=files,end_padding_samples=padding,cached=False,
                note='用户提供的声轨；未运行分离模型。按原文件时间解码，仅允许末尾两采样以内的舍入差异。长度一致不能证明音轨内部同步，请逐声部试听。')
            temporary=staging/'separation.tmp'
            temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(staging/'separation.json')
            _install_snapshot(staging,folder,on_new_folder)
            progress(100,'声轨已保存，可按声部分析与试听')
            return report


class ProvidedStemSeparator:
    def __init__(self, report):self.report=report

    def separate(self,path,work,progress=lambda p,s:None,cancelled=lambda:False):
        report=self.report
        if not report or report.get('source')!='provided':raise ValueError('请先导入独立声轨')
        if digest(path)!=report['audio_sha256']:raise ValueError('原曲已改变，请为当前歌曲重新导入声轨')
        for voice in separation_voices(report):
            if cancelled():raise InterruptedError('已取消声部分析')
            for extension in ('npy','wav'):
                file=stem_path(Path(work),{'separation':report},voice,extension)
                if digest(file)!=report['decoded_files'][voice][extension+'_sha256']:
                    raise ValueError('导入声轨文件已改变，请重新导入')
        progress(30,'使用已导入声轨，保留原始时间轴')
        return file.parent,report|{'cached':True}
