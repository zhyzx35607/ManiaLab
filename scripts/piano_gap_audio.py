"""Measure unchanged PCM around fixed gaps, without an audibility classifier."""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.signal import stft, get_window

from scripts.piano_band_evidence import _sample_coordinate, _bins, _frequency


def sample_range(begin_ms, end_ms, rate, count):
    return (max(0, int(np.ceil(_sample_coordinate(begin_ms/1000, rate)))),
            min(count, int(np.floor(_sample_coordinate(end_ms/1000, rate)))))


def ratio_db(power, reference):
    return float(10*(np.log10(power)-np.log10(reference))) if (
        power is not None and reference is not None and power > 0 and reference > 0) else None


def measure_gap(samples, rate, begin_ms, end_ms, pitches, *, sizes=(512, 2048, 8192), hop=128):
    samples = np.asarray(samples)
    if (samples.ndim != 1 or not len(samples) or not np.issubdtype(samples.dtype, np.floating)
            or not np.isfinite(samples).all() or isinstance(rate, bool)
            or not isinstance(rate, Integral) or rate <= 0):
        raise ValueError('Expected finite floating mono PCM and a positive integer sample rate')
    if any(isinstance(x, bool) or not isinstance(x, Real) or not np.isfinite(x) for x in (begin_ms, end_ms)):
        raise ValueError('Gap coordinates must be finite milliseconds')
    if not 0 <= begin_ms < end_ms <= len(samples)*1000/rate:
        raise ValueError('Gap must be inside the original waveform')
    pitches = list(pitches);sizes = tuple(sizes)
    if any(isinstance(x, bool) or not isinstance(x, Integral) or not 0 <= x <= 127 for x in pitches):
        raise ValueError('Pitch must be an integer MIDI note')
    pitches = sorted(set(map(int, pitches)))
    if (not sizes or len(set(sizes)) != len(sizes) or isinstance(hop, bool) or not isinstance(hop, Integral)
            or hop <= 0 or any(isinstance(x, bool) or not isinstance(x, Integral) or x < hop for x in sizes)):
        raise ValueError('Invalid FFT windows or hop')
    regions = {'pre': (max(0., begin_ms-500), begin_ms), 'gap': (begin_ms, end_ms),
               'post': (end_ms, min(len(samples)*1000/rate, end_ms+500))}
    limits = {name: sample_range(a, b, rate, len(samples)) for name, (a, b) in regions.items()}
    waveform = {}
    for name, (a, b) in limits.items():
        chunk = np.asarray(samples[a:b], np.float64)
        waveform[name] = dict(start_sample=a, end_sample=b, samples=max(0, b-a),
            power=float(np.mean(chunk**2)) if len(chunk) else None,
            peak=float(np.max(np.abs(chunk))) if len(chunk) else None)
    metadata = dict(rate=int(rate), samples=len(samples), gap_ms=[begin_ms, end_ms],
        regions_ms=regions, waveform=waveform,
        waveform_gap_pre_db=ratio_db(waveform['gap']['power'], waveform['pre']['power']),
        pitches=pitches, windows={}, hop_samples=int(hop), threshold_fitted=False,
        audibility_established=False, release_established=False)
    arrays = {}
    # Globally anchor the frame grid; padding context never adds audio samples.
    first = max(0, (limits['pre'][0]-max(sizes))//hop*hop)
    last = min(len(samples), limits['post'][1]+max(sizes))
    for size in sizes:
        freq = np.fft.rfftfreq(size, 1/rate)
        if last-first >= size:
            _, _, z = stft(np.asarray(samples[first:last], np.float64), fs=rate,
                window=get_window('hann', size, fftbins=True), nperseg=size, noverlap=size-hop,
                nfft=size, detrend=False, boundary=None, padded=False, scaling='spectrum')
            starts = first+np.arange(z.shape[1], dtype=np.int64)*hop
            visible = ((starts+size/2 >= limits['pre'][0]) & (starts+size/2 <= limits['post'][1]))
            starts, magnitude = starts[visible], np.abs(z[:, visible])
        else:
            starts = np.empty(0, np.int64);magnitude = np.empty((len(freq), 0))
        masks = {name: (starts >= a) & (starts+size <= b) for name, (a, b) in limits.items()}
        arrays[f'starts_{size}'] = starts
        arrays[f'times_{size}'] = (starts+size/2)/rate
        displayed = (freq > 0) & (freq <= 5000)
        arrays[f'frequencies_{size}'] = freq[displayed]
        arrays[f'magnitude_{size}'] = magnitude[displayed]
        window = dict(size=int(size), seconds=size/rate, bin_hz=rate/size,
            region_frames={name:int(mask.sum()) for name, mask in masks.items()}, bands=[])
        for pitch in pitches:
            for partial in range(1, 7):
                center = _frequency(pitch)*partial
                if center >= rate/2:continue
                width, bins = _bins(freq, center, rate/size)
                power = np.square(magnitude[bins]).sum(axis=0)
                array = f'power_{size}_{pitch}_{partial}';arrays[array] = power
                stats = {name: dict(frames=int(mask.sum()), power=float(power[mask].mean())
                    if mask.any() and len(bins) else None) for name, mask in masks.items()}
                neighbors = []
                for delta in (-1, 1):
                    neighbor = center*2**(delta/12)
                    _, other = _bins(freq, neighbor, rate/size)
                    shared = np.intersect1d(bins, other) if neighbor < rate/2 else np.empty(0, int)
                    neighbors.append(dict(semitones=delta, shared_bins=shared.tolist()))
                window['bands'].append(dict(pitch=pitch, partial=partial, center_hz=center,
                    half_width_hz=width, bins=bins.tolist(), power_array=array, regions=stats,
                    gap_pre_db=ratio_db(stats['gap']['power'], stats['pre']['power']),
                    adjacent_semitone_overlap=neighbors))
        metadata['windows'][str(size)] = window
    return metadata, arrays
