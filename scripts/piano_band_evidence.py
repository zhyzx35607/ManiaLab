"""Descriptive piano-band evidence on the original PCM clock.

No labels, note filters or independent-attack decisions are produced. A short
hop does not increase the time resolution of a long FFT window.
"""
from numbers import Integral, Real

import numpy as np
from scipy.signal import get_window, stft


SEGMENTS = {'pre': (-.18, -.03), 'attack': (0., .15), 'late': (.15, .35)}
CENTS = 35


def _integer(value, name, minimum, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum or maximum is not None and value > maximum):
        raise ValueError(name + ' must be an integer in the allowed range')
    return int(value)


def _frequency(pitch):
    return 440. * 2. ** ((pitch - 69) / 12.)


def _sample_coordinate(seconds, rate):
    """Remove arithmetic roundoff at sample boundaries, not real sample shifts."""
    value = seconds * rate
    nearest = round(value)
    tolerance = min(.25, 8. * np.spacing(max(1., abs(value))))
    return nearest if abs(value - nearest) <= tolerance else value


def _bins(frequencies, center, bin_hz):
    half_width = max(center * (2. ** (CENTS / 1200.) - 1.), bin_hz / 2.)
    indices = np.flatnonzero((frequencies > 0) & (np.abs(frequencies - center) <= half_width))
    return half_width, indices


def _segment_stats(envelope, indices, support_masks):
    result = {}
    for name, mask in support_masks.items():
        count = int(mask.sum())
        result[name] = dict(count=count, mean_power=(float(np.mean(np.square(envelope[mask])))
                                                    if count and len(indices) else None))
    pre, attack = result['pre']['mean_power'], result['attack']['mean_power']
    ratio = float(10. * (np.log10(attack) - np.log10(pre))) if (
        pre is not None and attack is not None and pre > 0 and attack > 0) else None
    return result, ratio


def _correlation(left, right):
    """Pearson correlation only when both finite vectors vary; no p-value."""
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    a, b = left - left.mean(), right - right.mean()
    scale_a, scale_b = np.max(np.abs(a)), np.max(np.abs(b))
    if scale_a == 0 or scale_b == 0:
        return None
    a, b = a / scale_a, b / scale_b
    value = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(value, -1., 1.)) if np.isfinite(value) else None


def analyze_window(samples, sample_rate, onset, pitch, lower_pitches, *,
                   sizes=(512, 2048, 8192), hop=44):
    """Return JSON metadata and spectra/envelopes, without changing any event.

    Input is finite floating mono PCM starting at physical time zero. MIDI
    pitches are integers 0..127; lower pitches must be below the target and are
    deduplicated. All partial centers must be strictly below Nyquist. Spectra
    retain display-range centers only, with complete unpadded input support.

    For segment statistics a frame's entire [start, start+N) sample interval
    must fit within the segment. An empty time or frequency support yields
    mean_power=None. Correlations use only attack-segment full-support frames.
    """
    signal = np.asarray(samples)
    if (signal.ndim != 1 or not len(signal) or not np.issubdtype(signal.dtype, np.floating)
            or not np.isfinite(signal).all()):
        raise ValueError('Samples must be a nonempty finite floating mono vector')
    rate = _integer(sample_rate, 'Sample rate', 1)
    if (isinstance(onset, (bool, np.bool_)) or not isinstance(onset, Real)
            or not np.isfinite(onset) or not 0 <= onset < len(signal) / rate):
        raise ValueError('Onset must be finite and inside the original PCM')
    onset = float(onset)
    target = _integer(pitch, 'Pitch', 0, 127)
    try:
        lower_values, window_values = tuple(lower_pitches), tuple(sizes)
    except TypeError as error:
        raise ValueError('Lower pitches and sizes must be iterable') from error
    lowers = sorted({_integer(value, 'Lower pitch', 0, 127) for value in lower_values})
    if any(value >= target for value in lowers):
        raise ValueError('Lower pitches must be strictly below the target')
    windows = tuple(_integer(value, 'Window size', 2) for value in window_values)
    if not windows or len(set(windows)) != len(windows):
        raise ValueError('Window sizes must be nonempty and unique')
    hop = _integer(hop, 'Hop', 1)
    if any(hop > size for size in windows):
        raise ValueError('Hop must not exceed any window size')
    nyquist = rate / 2.
    if _frequency(target) >= nyquist:
        raise ValueError('Target fundamental must be below Nyquist')

    duration = len(signal) / rate
    display = (onset - .25, onset + .45)
    left = max(0, round((display[0] - .5) * rate))
    right = min(len(signal), round((display[1] + .5) * rate))
    excerpt = signal[left:right].astype(np.float64, copy=True)
    metadata = dict(sample_rate=rate, sample_count=len(signal), duration_seconds=duration,
        onset_seconds=onset, target_pitch=target, lower_pitches=lowers,
        display_seconds=list(display), available_display_seconds=[max(0., display[0]), min(duration, display[1])],
        extraction_samples=[left, right], extraction_seconds=[left/rate, right/rate],
        segments_seconds={name: [onset+a, onset+b] for name, (a, b) in SEGMENTS.items()},
        window='periodic Hann', boundary=None, padded=False, stft_scaling='spectrum',
        band_cents=CENTS, hop_samples=hop, hop_seconds=hop/rate,
        descriptive_only=True, independent_attack_established=False,
        shared_band_warning='Shared bins are not independent evidence; unshared bins do not establish a real note.',
        time_resolution_warning='Hop spacing is not attack-time resolution; report the complete window duration.',
        windows={})
    arrays = {}
    pitches = sorted({target, *lowers})
    for size in windows:
        frequencies = np.fft.rfftfreq(size, d=1./rate)
        if len(excerpt) >= size:
            _, _, spectrum = stft(excerpt, fs=rate, window=get_window('hann', size, fftbins=True),
                nperseg=size, noverlap=size-hop, nfft=size, detrend=False,
                return_onesided=True, boundary=None, padded=False, scaling='spectrum')
            starts = left + np.arange(spectrum.shape[1], dtype=np.int64) * hop
            times = (starts + size/2.) / rate
            # Doubled samples also preserve the half-sample center of odd N.
            centers_twice = 2 * starts + size
            visible = ((centers_twice >= np.ceil(_sample_coordinate(display[0], 2*rate)))
                       & (centers_twice <= np.floor(_sample_coordinate(display[1], 2*rate))))
            starts, times = starts[visible], times[visible]
            magnitude = np.abs(spectrum[:, visible])
        else:
            starts = np.empty(0, dtype=np.int64)
            times = np.empty(0, dtype=np.float64)
            magnitude = np.empty((len(frequencies), 0), dtype=np.float64)
        if not np.isfinite(magnitude).all():
            raise ValueError('Nonfinite spectral result')
        ends = starts + size
        support_masks = {name: (starts >= np.ceil(_sample_coordinate(onset+a, rate)))
                         & (ends <= np.floor(_sample_coordinate(onset+b, rate)))
                         for name, (a, b) in SEGMENTS.items()}
        arrays[f'times_{size}'] = times
        arrays[f'frequencies_{size}'] = frequencies
        arrays[f'magnitude_{size}'] = magnitude
        bin_hz = rate / size
        window = dict(size=size, window_seconds=size/rate, bin_hz=bin_hz,
            frame_count=len(times), full_support_samples=[left, right],
            first_frame_support_samples=[int(starts[0]), int(ends[0])] if len(starts) else None,
            last_frame_support_samples=[int(starts[-1]), int(ends[-1])] if len(starts) else None,
            segment_frame_counts={name: int(mask.sum()) for name, mask in support_masks.items()},
            bands=[], correlations=[])
        # Keep every lower integer partial below Nyquist, not just its first 8.
        lower_partials = {}
        for lower in lowers:
            fundamental = _frequency(lower)
            members = []
            for partial in range(1, int(np.ceil(nyquist / fundamental))):
                center = fundamental * partial
                if center < nyquist:
                    _, indices = _bins(frequencies, center, bin_hz)
                    members.append((partial, center, set(map(int, indices))))
            lower_partials[lower] = members
        band_lookup = {}
        for current_pitch in pitches:
            for partial in range(1, 9):
                center = _frequency(current_pitch) * partial
                if center >= nyquist:
                    continue
                half_width, indices = _bins(frequencies, center, bin_hz)
                with np.errstate(over='ignore', invalid='ignore'):
                    envelope = np.sqrt(np.square(magnitude[indices]).sum(axis=0))
                if not np.isfinite(envelope).all():
                    raise ValueError('Nonfinite band energy')
                key = f'band_{size}_{current_pitch}_{partial}'
                arrays[key] = envelope
                band_lookup[current_pitch, partial] = (envelope, indices)
                statistics, ratio = _segment_stats(envelope, indices, support_masks)
                if any(row['mean_power'] is not None and not np.isfinite(row['mean_power'])
                       for row in statistics.values()):
                    raise ValueError('Nonfinite band power')
                band = dict(pitch=current_pitch, partial=partial, center_hz=center,
                    half_width_hz=half_width, bins=list(map(int, indices)),
                    frequency_supported=bool(len(indices)), array_key=key,
                    segments=statistics, attack_pre_db=ratio, neighbor_semitone_overlap=[])
                target_bins = set(map(int, indices))
                for shift in (-1, 1):
                    neighbor_center = center * 2. ** (shift / 12.)
                    _, neighbor_bins = _bins(frequencies, neighbor_center, bin_hz)
                    if neighbor_center >= nyquist:
                        neighbor_bins = np.empty(0, dtype=np.int64)
                    shared = sorted(target_bins.intersection(map(int, neighbor_bins)))
                    band['neighbor_semitone_overlap'].append(dict(semitones=shift,
                        center_hz=neighbor_center, center_below_nyquist=bool(neighbor_center < nyquist),
                        bins=list(map(int, neighbor_bins)), shared_bins=shared, shares_bin=bool(shared)))
                if current_pitch == target:
                    band['lower_overlap'] = []
                    for lower, members in lower_partials.items():
                        shared_all, overlapping = set(), []
                        for harmonic, lower_center, lower_bins in members:
                            shared = sorted(target_bins & lower_bins)
                            if shared:
                                shared_all.update(shared)
                                overlapping.append(dict(partial=harmonic, center_hz=lower_center, shared_bins=shared))
                        band['lower_overlap'].append(dict(lower_pitch=lower, checked_partials=len(members),
                            shared_bins=sorted(shared_all), shared_bin_count=len(shared_all),
                            shared_bin_fraction=len(shared_all)/len(target_bins) if target_bins else None,
                            shares_bin=bool(shared_all), overlapping_partials=overlapping))
                window['bands'].append(band)
        target_envelope, target_bins = band_lookup[target, 1]
        attack_mask = support_masks['attack']
        for lower in lowers:
            lower_envelope, lower_bins = band_lookup[lower, 1]
            value = _correlation(target_envelope[attack_mask], lower_envelope[attack_mask]) if (
                len(target_bins) and len(lower_bins)) else None
            window['correlations'].append(dict(target_pitch=target, lower_pitch=lower,
                segment='attack', count=int(attack_mask.sum()), correlation=value))
        metadata['windows'][str(size)] = window
    return metadata, arrays
