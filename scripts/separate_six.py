"""Pinned public six-source HTDemucs inference, isolated from the desktop app."""
from fractions import Fraction
import gc
import hashlib
import json
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch

MODEL_SHA256='34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd'
MODEL_URL='https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th'


def separate_six(source,output,model_path,seconds,threads,beat_model,progress,beat_evidence):
    from demucs.htdemucs import HTDemucs
    from demucs.apply import apply_model
    started=time.perf_counter()
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    with model_path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=MODEL_SHA256:
            raise ValueError('Six-source model checksum mismatch')
    with torch.serialization.safe_globals([HTDemucs,Fraction]):
        package=torch.load(model_path,map_location='cpu',weights_only=True)
    model=HTDemucs(*package['args'],**package['kwargs'])
    model.load_state_dict(package['state']);model.eval();del package
    rate,data=wavfile.read(source)
    if rate!=44100 or data.ndim!=2 or data.shape[1]!=2:
        raise ValueError('Expected stereo 44100 Hz WAV')
    samples=data.astype(np.float32)
    if np.issubdtype(data.dtype,np.integer):
        samples/=max(abs(np.iinfo(data.dtype).min),np.iinfo(data.dtype).max)
    if seconds:samples=samples[:round(seconds*rate)]
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Invalid audio input')
    mixture=torch.from_numpy(samples.T.copy())
    reference=mixture.mean(0);mean=reference.mean();scale=reference.std().clamp_min(1e-7)
    normalized=(mixture-mean)/scale
    progress(5,'Separating six instruments with HTDemucs')
    # Use upstream chunk padding, overlap-add and transformer segment limits.
    # No random shifts: repeats and cached results must be reproducible.
    with torch.inference_mode():
        estimate=apply_model(model,normalized[None],device='cpu',shifts=0,
            split=True,overlap=.25,num_workers=0,progress=False)[0]
        result=(estimate*scale+mean).mean(dim=1).numpy()
    if result.shape!=(6,len(samples)) or not np.isfinite(result).all():
        raise ValueError('Invalid six-source output length or samples')
    output.mkdir(parents=True,exist_ok=True)
    for name,stem in zip(model.sources,result):
        np.save(output/f'{name}.npy',resample_poly(stem,1,2).astype(np.float32))
    report=dict(revision=1,profile='six',sample_rate=22050,frames=(len(samples)+1)//2,
        sources={name:f'{name}.npy' for name in model.sources},model_sha256=MODEL_SHA256,
        model_url=MODEL_URL,elapsed_seconds=round(time.perf_counter()-started,2),
        reconstruction_rmse=float(np.sqrt(np.mean((result.sum(axis=0)-samples.mean(axis=1))**2))))
    if beat_model is not None:
        del model,mixture,normalized,estimate,result;gc.collect()
        report['beat_evidence']=beat_evidence(source,beat_model,seconds)
        report['elapsed_seconds']=round(time.perf_counter()-started,2)
    (output/'separation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    progress(100,'Six-source separation complete',**report)
    return report
