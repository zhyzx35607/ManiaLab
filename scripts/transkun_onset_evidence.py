"""Keep decoded onset proposals independently of complete MIDI intervals.

Research evidence only. No model inference, labels, timing grid or LN decision.
The original official events are preserved; an added event is an unverified
proposal and its interval is only the decoder's display estimate.
"""
from copy import deepcopy
from math import isfinite

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_onsets(proposals, candidates, tolerance):
    """Maximum one-to-one matches, then minimum total onset difference."""
    if tolerance <= 0 or not isfinite(tolerance):
        raise ValueError('Positive finite matching tolerance required')
    result = []
    for pitch in sorted({row['pitch'] for row in proposals}):
        source = [i for i, row in enumerate(proposals) if row['pitch'] == pitch]
        target = [i for i, row in enumerate(candidates) if row['pitch'] == pitch]
        if not target:
            continue
        differences = abs(np.asarray([proposals[i]['onset'] for i in source])[:, None]
                          - np.asarray([candidates[i]['onset'] for i in target])[None, :])
        allowed = differences <= tolerance
        # One additional match beats the maximum possible total timing cost.
        cost = np.concatenate((np.where(allowed, differences / (tolerance*(len(source)+1)), 2.),
                               np.ones((len(source), len(source)))), axis=1)
        rows, columns = linear_sum_assignment(cost)
        result.extend((source[i], target[j]) for i, j in zip(rows, columns)
                      if j < len(target) and allowed[i, j])
    return result


def validate_note(note):
    if (not all(isfinite(note[k]) for k in ('onset', 'offset', 'velocity'))
            or type(note['pitch']) is not int or note['pitch'] not in (-64, -67)
            and not 21 <= note['pitch'] <= 108
            or note['offset'] < note['onset']):
        raise ValueError('Invalid decoded event')


def collect_onset_evidence(trace, *, hop_samples=1024):
    rate, duration = trace['sample_rate'], trace['duration_seconds']
    if rate != 44100 or hop_samples != 1024 or not isfinite(duration) or duration <= 0:
        raise ValueError('This experiment is fixed to the verified model clock')
    tolerance = hop_samples/rate
    candidates = []
    for i, note in enumerate(trace['final_events']):
        validate_note(note)
        if note['pitch'] < 0:
            continue
        candidates.append(dict(note=deepcopy(note), origin='official', official_index=i,
                               onset_sources=[], ln_release_allowed=False))
    for expected, window in enumerate(trace['windows']):
        if window['index'] != expected or len(window['events']) != 1:
            raise ValueError('Expected ordered single-recording windows')
        origin = window['origin_seconds']
        if not isfinite(origin):
            raise ValueError('Invalid window time')
        proposals = []
        for i, raw in enumerate(window['events'][0]):
            validate_note(raw)
            if raw['pitch'] < 0 or not raw['has_onset']:
                continue
            note = dict(raw, onset=raw['onset']+origin, offset=raw['offset']+origin)
            proposals.append(dict(note, source_id=f'w{expected}:e{i}', window=expected,
                local_onset=raw['onset'], local_offset=raw['offset'],
                onset_outside_input=not 0 <= note['onset'] < duration))
        matches = assign_onsets(proposals, [c['note'] for c in candidates], tolerance)
        matched = {i: j for i, j in matches}
        for i, proposal in enumerate(proposals):
            if i not in matched:
                note = {key: proposal[key] for key in
                        ('onset', 'offset', 'pitch', 'velocity', 'has_onset', 'has_offset')}
                candidates.append(dict(note=note, origin='window_proposal', official_index=None,
                                       onset_sources=[proposal], ln_release_allowed=False))
            else:
                candidates[matched[i]]['onset_sources'].append(proposal)
    for candidate in candidates:
        note = candidate['note']
        candidate['onset_outside_input'] = not 0 <= note['onset'] < duration
        candidate['display_interval_only'] = True
        candidate['physical_release_confirmed'] = False
    candidates.sort(key=lambda row: (row['note']['onset'], row['note']['offset'], row['note']['pitch']))
    return dict(candidates=candidates, pedal_events=deepcopy(trace['pedal_events']),
                dedup_seconds=tolerance, duration_seconds=duration,
                application_enabled=False, musical_acceptance=False)
