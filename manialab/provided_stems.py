"""User-supplied source audio on the original song clock, without reseparation."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .audio import SR,align_stem_samples,decode
from .stems import VOICES,stem_path,separation_voices


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare_provided_stems(mix, sources, work, progress=lambda p,s:None, cancelled=lambda:False):
    """Decode an immutable set; a failed or cancelled import cannot select it."""
    if not sources or set(sources)-set(VOICES):raise ValueError('至少选择一个有效声部')
    work=Path(work)
    origin={voice:dict(filename=path.name,sha256=digest(path)) for voice,path in sources.items()}
    identity=dict(revision=1,audio_sha256=digest(mix),sources=origin)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=work/'stems-provided'/key
    manifest=folder/'separation.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        if all((folder/f'{v}.{ext}').is_file() and digest(folder/f'{v}.{ext}')==
               previous.get('decoded_files',{}).get(v,{}).get(ext+'_sha256')
               for v in sources for ext in ('npy','wav')):
            return previous|{'cached':True}
    folder.mkdir(parents=True,exist_ok=True)
    progress(3,'核对原曲与导入声轨的时间轴')
    mixture=decode(mix,folder)
    files={};padding={}
    for index,(voice,path) in enumerate(sources.items()):
        if cancelled():raise InterruptedError('已取消声轨导入')
        progress(5+int(index*85/len(sources)),'读取 '+VOICES[voice])
        samples=decode(path,folder)
        if abs(len(samples)-len(mixture))>2:
            raise ValueError(f'{VOICES[voice]}长度 {len(samples)/SR:.3f} 秒，与原曲 {len(mixture)/SR:.3f} 秒不同；请导出同起点、同长度的声轨，保留开头静音')
        samples,padding[voice]=align_stem_samples(samples,len(mixture))
        np.save(folder/f'{voice}.npy',samples,allow_pickle=False)
        peak=max(1.,float(np.abs(samples).max()))
        wavfile.write(folder/f'{voice}.wav',SR,(np.clip(samples/peak,-1,1)*32767).astype(np.int16))
        files[voice]=dict(npy_sha256=digest(folder/f'{voice}.npy'),wav_sha256=digest(folder/f'{voice}.wav'))
    if cancelled():raise InterruptedError('已取消声轨导入')
    report=dict(**identity,source='provided',set_sha256=key,model_sha256=key,
        voices=[v for v in VOICES if v in sources],duration_seconds=len(mixture)/SR,
        sample_rate=SR,decoded_files=files,end_padding_samples=padding,cached=False,
        note='用户提供的声轨；未运行分离模型。按原文件时间解码，仅允许末尾两采样以内的舍入差异。长度一致不能证明音轨内部同步，请逐声部试听。')
    temporary=manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(manifest)
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
