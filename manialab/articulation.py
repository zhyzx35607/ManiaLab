"""Conservative removal of false low-register reattacks on steady sound.

This never joins across a pitch-activation gap, changes pitch, or extends a
tail past existing frame support. The narrow-band waveform must be effectively
flat around the suspected retrigger. Natural timbres still need validation.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample_poly


def steady_bass_onsets(frames, onsets, times, samples, rate=22050,
                       onset_threshold=.45, frame_threshold=.3):
    removed = []
    for index in range(min(28, frames.shape[1])):  # MIDI 21..48 only
        pitch = index+21
        frequency = 440*2**((pitch-69)/12)
        peaks, _ = find_peaks(onsets[:, index], height=onset_threshold, distance=3)
        for i in peaks:
            if i < 7 or i+7 >= len(times):
                continue
            support = frames[i-7:i+8, index]
            if float(np.min(support)) < frame_threshold:
                continue
            at = float(times[i])
            lo = max(0, round((at-.12)*rate)); hi = min(len(samples), round((at+.12)*rate))
            if hi-lo < rate*.23:
                continue
            window = samples[lo:hi]
            demodulated = resample_poly(window*np.exp(-2j*np.pi*frequency*np.arange(len(window))/rate),1,10)
            envelope_rate = rate/10
            sigma = float(np.clip(2/frequency,.004,.012))*envelope_rate
            envelope = np.abs(gaussian_filter1d(demodulated.real,sigma)+1j*gaussian_filter1d(demodulated.imag,sigma))
            def part(a,b):
                left = max(0, round((at+a-lo/rate)*envelope_rate))
                right = min(len(envelope), round((at+b-lo/rate)*envelope_rate))
                return envelope[left:right]
            before, after, middle = part(-.075,-.025),part(.025,.075),part(-.025,.025)
            if not min(len(before),len(after),len(middle)):
                continue
            left,right = float(np.median(before)),float(np.median(after))
            if min(left,right)<.0001:
                continue
            ratio = right/left
            flatness = float(np.min(middle))/max(left,right)
            # Tighter than the research continuity guard: no joining audible
            # dips, even when its smoothed envelope falls only slightly.
            if .92 <= ratio <= 1.08 and flatness >= .94:
                removed.append(dict(time_ms=round(at*1000),pitch=pitch,reason='steady_low_pitch_retrigger',
                                    renewal=round(ratio,4),flatness=round(flatness,4)))
    return removed
