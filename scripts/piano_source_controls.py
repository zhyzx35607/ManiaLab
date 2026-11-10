"""Source-known PCM controls and descriptive acoustic-anchor onset matching.

Envelope anchors are not MIDI key times, acoustic release labels or LN targets.
This module does not load models, infer notes, normalize audio or score tails.
"""
import hashlib
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment


ANCHOR_SEMANTICS = 'Acoustic envelope anchor only; not MIDI key time, release truth or LN target.'


def _integer(value, name, minimum, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum or maximum is not None and value > maximum):
        raise ValueError(name+' must be an integer in the allowed range')
    return int(value)


def _real(value, name, minimum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or minimum is not None and value < minimum):
        raise ValueError(name+' must be finite and in the allowed range')
    return float(value)


def _pcm_hash(samples):
    """Hash contiguous little-endian PCM at its existing float precision."""
    canonical = np.ascontiguousarray(samples, dtype=np.dtype(f'<f{samples.dtype.itemsize}'))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _record(value, name):
    if not isinstance(value, dict):
        raise ValueError(name+' must be a dictionary')
    return value


def _required(row, key, name):
    if key not in row:
        raise ValueError(name+' requires '+key)
    return row[key]


def compose_case(sources, components, sample_rate=16000, duration_seconds=10., *, source_anchor_samples):
    """Place complete source arrays at sample-aligned envelope anchors.

    Source PCM must be finite float32/float64 mono. Source anchors are integer
    indices inside those arrays. Requested seconds and duration are rounded to
    nearest sample using Python's ties-to-even rule. No head or tail is cropped.
    Gains are finite nonnegative values; zero gain is permitted. Components are
    accumulated in supplied order in float64 and converted once to float32.
    A peak >=1 before OR after conversion is rejected, never clipped/rescaled.

    Metadata starts with one mixture record (including for silence), followed
    by one record per component. Hashes describe little-endian contiguous PCM;
    source hashes retain the source's original float precision.
    """
    rate = _integer(sample_rate, 'Sample rate', 1)
    duration = _real(duration_seconds, 'Duration', 0.)
    exact_count = duration*rate
    if not np.isfinite(exact_count):
        raise ValueError('Duration sample count must be finite')
    count = round(exact_count)
    if count < 1:
        raise ValueError('Duration must contain at least one sample')
    if not isinstance(sources, dict) or not isinstance(source_anchor_samples, dict):
        raise ValueError('Sources and source anchors must be dictionaries')
    if not isinstance(components, (list, tuple)):
        raise ValueError('Components must be a list or tuple')
    if set(sources) != set(source_anchor_samples):
        raise ValueError('Every source must have exactly one source anchor')
    for pitch in source_anchor_samples:
        _integer(pitch, 'Source anchor pitch', 0, 127)
    prepared = {}
    for pitch, samples in sources.items():
        pitch = _integer(pitch, 'Source pitch', 0, 127)
        signal = np.asarray(samples)
        if (signal.ndim != 1 or not len(signal) or signal.dtype.kind != 'f'
                or signal.dtype.itemsize not in (4, 8) or not np.isfinite(signal).all()):
            raise ValueError('Source PCM must be nonempty finite float32/float64 mono')
        anchor = _integer(source_anchor_samples[pitch], 'Source anchor', 0, len(signal)-1)
        prepared[pitch] = (signal, anchor, _pcm_hash(signal))
    mixed = np.zeros(count, dtype=np.float64)
    records = []
    for index, component in enumerate(components):
        component = _record(component, 'Component')
        pitch = _integer(_required(component, 'pitch', 'Component'), 'Component pitch', 0, 127)
        if pitch not in prepared:
            raise ValueError('Component pitch has no source')
        requested = _real(_required(component, 'anchor_seconds', 'Component'), 'Anchor time', 0.)
        gain = _real(_required(component, 'gain', 'Component'), 'Gain', 0.)
        desired_sample = requested*rate
        if not np.isfinite(desired_sample):
            raise ValueError('Anchor sample must be finite')
        target_anchor = round(desired_sample)
        source, source_anchor, source_hash = prepared[pitch]
        start, stop = target_anchor-source_anchor, target_anchor-source_anchor+len(source)
        if start < 0 or stop > count:
            raise ValueError('Complete source head and tail must fit inside the case')
        with np.errstate(over='ignore', invalid='ignore'):
            mixed[start:stop] += source.astype(np.float64)*gain
        if not np.isfinite(mixed[start:stop]).all():
            raise ValueError('Nonfinite mixed PCM')
        records.append(dict(kind='component', index=index, pitch=pitch, gain=gain,
            source_dtype=np.dtype(f'<f{source.dtype.itemsize}').str,
            source_samples=len(source), source_pcm_sha256=source_hash,
            source_anchor_sample=source_anchor, placement_start_sample=start,
            placement_stop_sample=stop, anchor_sample=target_anchor,
            anchor_seconds_requested=requested, anchor_seconds_realized=target_anchor/rate))
    peak64 = float(np.max(np.abs(mixed)))
    if peak64 >= 1.:
        raise ValueError('Mixed PCM peak must be below 1; clipping/normalization is disabled')
    result = mixed.astype(np.float32)
    peak32 = float(np.max(np.abs(result)))
    if peak32 >= 1.:
        raise ValueError('Float32 mixed PCM peak must be below 1')
    mixture = dict(kind='mixture', sample_rate=rate, sample_count=count,
        duration_seconds_requested=duration, duration_seconds_realized=count/rate,
        mix_dtype='<f4', mix_pcm_sha256=_pcm_hash(result), peak_float64=peak64,
        peak_float32=peak32, component_count=len(records), accumulation_dtype='float64',
        clipping_applied=False, normalization_applied=False,
        rounding='nearest sample, ties to even', source_anchor_semantics=ANCHOR_SEMANTICS)
    return result, [mixture, *records]


def score_attacks(expected, predicted, tolerance_seconds=.1):
    """Maximum-cardinality, then minimum total timing-error onset assignment.

    Only equal MIDI pitches may match. All supplied predictions remain in the
    denominator, including negative/outside-case onsets and duplicate events.
    Offset is optional; if supplied it must be finite and >=onset, but is not
    scored. Boundary comparison permits only floating-point arithmetic error
    (at most 1ns), not a wider audio-sample tolerance. At tolerance 0 only exact
    onset equality matches. Empty precision/recall/F1 denominators yield 0.
    """
    tolerance = _real(tolerance_seconds, 'Tolerance', 0.)
    if not isinstance(expected, (list, tuple)) or not isinstance(predicted, (list, tuple)):
        raise ValueError('Expected and predicted events must be lists or tuples')
    references, detections = [], []
    for row in expected:
        row = _record(row, 'Expected event')
        references.append((_integer(_required(row, 'pitch', 'Expected event'), 'Expected pitch', 0, 127),
                           _real(_required(row, 'anchor_seconds', 'Expected event'), 'Expected anchor', 0.)))
    for row in predicted:
        row = _record(row, 'Predicted event')
        pitch = _integer(_required(row, 'pitch', 'Predicted event'), 'Predicted pitch', 0, 127)
        onset = _real(_required(row, 'onset', 'Predicted event'), 'Predicted onset')
        if 'offset' in row and _real(row['offset'], 'Predicted offset') < onset:
            raise ValueError('Predicted offset must not precede onset')
        detections.append((pitch, onset))
    matches = []
    for pitch in sorted({p for p, _ in references}):
        ri = [i for i, row in enumerate(references) if row[0] == pitch]
        pi = [i for i, row in enumerate(detections) if row[0] == pitch]
        if not pi:
            continue
        anchor = np.asarray([references[i][1] for i in ri])[:, None]
        onset = np.asarray([detections[i][1] for i in pi])[None, :]
        with np.errstate(over='ignore'):
            delta = np.abs(onset-anchor)
        if tolerance:
            # Decimal boundary examples such as 1.1-1.0 should not miss .1.
            scale = np.maximum(1., np.maximum(np.abs(anchor), np.abs(onset)))
            slack = np.minimum(8*np.finfo(float).eps*scale, min(1e-9, tolerance*1e-10))
            allowed = delta <= tolerance+slack
        else:
            allowed = delta == 0
        normalized = np.zeros(delta.shape, dtype=float)
        largest = float(delta[allowed].max()) if allowed.any() else 0.
        if largest:
            # One common scale preserves the secondary total-error objective,
            # including accepted roundoff-sized excess at tolerance boundaries.
            np.divide(delta, largest, out=normalized, where=allowed)
        # Every possible sum of real-edge costs is <1, so one extra real
        # match always beats any change to the total time-error objective.
        cost = np.concatenate((np.where(allowed, normalized/(len(ri)+1), 2.),
                               np.ones((len(ri), len(ri)))), axis=1)
        assignment_rows, assignment_cols = linear_sum_assignment(cost)
        for i, j in zip(assignment_rows, assignment_cols):
            if j >= len(pi) or not allowed[i,j]:
                continue
            ref_index, pred_index = ri[i], pi[j]
            error = detections[pred_index][1]-references[ref_index][1]
            matches.append(dict(ref_index=ref_index, pred_index=pred_index, pitch=pitch,
                signed_error_seconds=error, absolute_error_seconds=abs(error)))
    matches.sort(key=lambda row: (row['ref_index'], row['pred_index']))
    found_ref = {row['ref_index'] for row in matches}
    found_pred = {row['pred_index'] for row in matches}
    missing = [i for i in range(len(references)) if i not in found_ref]
    extra = [i for i in range(len(detections)) if i not in found_pred]
    matched, nr, npred = len(matches), len(references), len(detections)
    return dict(expected=nr, predicted=npred, matched=matched, missing=len(missing), extra=len(extra),
        precision=matched/npred if npred else 0., recall=matched/nr if nr else 0.,
        f1=2*matched/(nr+npred) if nr+npred else 0., matches=matches,
        missing_ref_indices=missing, extra_pred_indices=extra, tolerance_seconds=tolerance,
        all_predictions_in_denominator=True, offsets_scored=False,
        matching_objective='Maximum cardinality, then minimum summed absolute onset error',
        anchor_semantics=ANCHOR_SEMANTICS)
