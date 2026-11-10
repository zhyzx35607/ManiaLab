"""Audit ADT_STR custom drum tokens without silently repairing event tuples."""
from numbers import Integral

NAMES = ('Acoustic Bass Drum', 'Bass Drum 1', 'Side Stick', 'Acoustic Snare',
         'Hand Clap', 'Electric Snare', 'Floor Tom', 'Closed Hi Hat', 'Pedal Hi-Hat',
         'Open Hi-Hat', 'Mid Tom', 'Crash Cymbal', 'High Tom', 'Ride Cymbal',
         'Chinese Cymbal', 'Tambourine', 'Splash Cymbal', 'Cowbell', 'Vibraslap',
         'Congas & Timbales', 'Shaker', 'Whistle', 'Guiro', 'Claves', 'Cuica', 'Triangle')
GROUPS = {'kick': {35, 36}, 'snare': {37, 38, 39, 40}, 'hi-hat': {42, 43, 44},
          'tom': {41, 45, 47}, 'cymbal': {46, 48, 49, 51},
          'other': {50, *range(52, 61)}}


def group(pitch):
    return next((name for name, values in GROUPS.items() if pitch in values), 'unknown')


def audit_tokens(tokens, *, valid_samples=61440, max_length=512):
    """Keep contiguous valid triples, original indices and all structural issues.

    Input-padding events remain in returned diagnostics and scoring; ``in_audio``
    distinguishes them from proposals that could become audible chart events.
    An incomplete stream never becomes an implicitly complete transcription.
    """
    if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in tokens)
            or isinstance(valid_samples, bool) or not isinstance(valid_samples, Integral)
            or not 0 < valid_samples <= 61440 or not isinstance(max_length, Integral)
            or isinstance(max_length, bool) or max_length < 2):
        raise ValueError('Expected integer tokens and a valid 24kHz chunk clock')
    tokens = [int(t) for t in tokens]
    issues, events = [], []
    def issue(index, kind, **details):
        issues.append(dict(index=index, kind=kind, **details))
    eos = next((i for i, t in enumerate(tokens) if t == 3), None)
    end = len(tokens) if eos is None else eos
    if not tokens or tokens[0] != 2:
        issue(0, 'missing_initial_bos')
    if eos is None:
        issue(len(tokens), 'missing_eos')
    elif any(t != 3 for t in tokens[eos+1:]):
        issue(eos+1, 'non_eos_after_end')
    if len(tokens) > max_length:
        issue(max_length, 'sequence_exceeds_max_length')
    body_start = 1 if tokens and tokens[0] == 2 else 0
    body = tokens[body_start:end]
    if body == [0] or not body:
        return dict(events=[], issues=issues, eos_index=eos, terminated=eos is not None,
                    hit_max_length=len(tokens) >= max_length, empty_stream=True)
    i, previous, seen = body_start, -1, set()
    while i < end:
        token = tokens[i]
        if not 4 <= token < 300:
            issue(i, 'unexpected_token', token=token)
            i += 1
            continue
        if i+2 >= end or not 335 <= tokens[i+1] <= 360 or not 400 <= tokens[i+2] <= 527:
            issue(i, 'incomplete_or_invalid_triplet', tokens=tokens[i:min(i+3, end)])
            i += 1
            continue
        tick, pitch, velocity = token-4, tokens[i+1]-300, tokens[i+2]-400
        if tick >= 256:
            issue(i, 'onset_outside_chunk', tick=tick)
            i += 3
            continue
        in_audio = tick*240 < valid_samples
        if not in_audio:
            issue(i, 'onset_in_padding', tick=tick)
        if tick < previous:
            issue(i, 'nonmonotonic_onset', tick=tick, previous_tick=previous)
        if (tick, pitch) in seen:
            issue(i, 'duplicate_onset_class', tick=tick, custom_pitch=pitch)
        if velocity == 0:
            issue(i, 'zero_velocity')
        events.append(dict(token_index=i, tick=tick, custom_pitch=pitch, velocity=velocity,
                           kind=group(pitch), class_name=NAMES[pitch-35], in_audio=in_audio))
        seen.add((tick, pitch))
        previous = tick
        i += 3
    return dict(events=events, issues=issues, eos_index=eos, terminated=eos is not None,
                hit_max_length=len(tokens) >= max_length, empty_stream=False)
