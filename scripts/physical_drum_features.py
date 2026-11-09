"""Exact-clock audio features and peak extraction for physical onset research."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample_poly, find_peaks
from math import gcd

RATE, HOP, FFT, BANDS = 22050, 220, 2048, 80
CLASSES = ('kick','snare','hi-hat')


def load_audio(path):
    rate, data = wavfile.read(path)
    if data.dtype.kind == 'i':
        data = data.astype(np.float32) / (2 ** (data.dtype.itemsize*8-1))
    elif data.dtype.kind == 'f':
        data = data.astype(np.float32)
    else:
        raise ValueError('Unreviewed PCM type')
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError('Invalid drum waveform')
    if rate != RATE:
        common = gcd(rate,RATE)
        data = resample_poly(data,RATE//common,rate//common).astype(np.float32)
    return data


def features(samples):
    frequency, times, spectrum = stft(samples,fs=RATE,nperseg=FFT,noverlap=FFT-HOP)
    edges = 700*np.expm1(np.linspace(np.log1p(30/700),np.log1p(10000/700),BANDS+2))
    filters = np.asarray([np.maximum(0,np.minimum((frequency-a)/(b-a),(c-frequency)/(c-b)))
        for a,b,c in zip(edges,edges[1:],edges[2:])])
    power = filters @ (abs(spectrum)**2)
    valid = times < len(samples)/RATE
    return np.log1p(power[:,valid].T*1e5).astype(np.float32), times[valid]*1000


def target_frames(times, events):
    target = np.zeros((len(times),3),np.float32)
    if not len(times):raise ValueError('Missing feature frames')
    for event in events:
        if event['kind'] not in CLASSES:continue
        center = int(np.argmin(abs(times-event['time_ms'])))
        category = CLASSES.index(event['kind'])
        for offset,value in [(-1,.25),(0,1.),(1,.25)]:
            if 0<=center+offset<len(times):target[center+offset,category]=max(target[center+offset,category],value)
    return target


def peaks(probabilities,times,thresholds):
    if probabilities.shape != (len(times),3) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid onset probabilities')
    result=[]
    for category,kind in enumerate(CLASSES):
        values=np.r_[0,probabilities[:,category],0]
        indices,_=find_peaks(values,height=thresholds[category],distance=max(1,int(np.ceil(.03*RATE/HOP))))
        for index in indices-1:
            if 0<=index<len(times):result.append(dict(time_ms=float(times[index]),kind=kind,score=float(probabilities[index,category])))
    return sorted(result,key=lambda row:(row['time_ms'],row['kind']))
