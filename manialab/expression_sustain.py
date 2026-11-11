"""Bounded expression coverage for existing supported sustain candidates.

An arrangement preference, not acoustic salience or note confidence. Never
changes an endpoint, grants tail eligibility, or multiplies a chord's coverage.
"""
from collections import defaultdict
import math

from .domain import UNOBSERVED_RELEASES


def validate_weight(weight, beat_ms):
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError('持续覆盖权重必须在0至1之间')
    if weight and (isinstance(beat_ms, bool) or not isinstance(beat_ms, (int, float))
                   or not math.isfinite(beat_ms) or beat_ms <= 0):
        raise ValueError('持续覆盖需要有效拍长')


def coverage_intervals(units, metadata, trace, beat_ms, weight, score):
    validate_weight(weight, beat_ms)
    if weight == 0:
        return []
    sources = defaultdict(list)
    for unit, notes in enumerate(units):
        for note in notes:
            if (trace[note.id].get('mode') != 'sustain' or note.source != 'pitch'
                    or not note.hold or note.phrase or note.support or not note.release_reason
                    or note.release_reason in UNOBSERVED_RELEASES):
                continue
            meta = metadata[note.id]
            value = score(note)*(2 if meta['role'] == 'primary' else 1)
            if not math.isfinite(value) or value < 0:
                raise ValueError('持续覆盖候选评分无效')
            if value:
                sources[(meta['section'], note.voice)].append(
                    (note.start, min(note.end, note.start+2*beat_ms), unit, note.id, value))
    result = []
    for (section, voice), notes in sorted(sources.items()):
        events = defaultdict(list)
        for index, (begin, end, *_) in enumerate(notes):
            events[begin].append((index, True))
            events[end].append((index, False))
        active = set()
        times = sorted(events)
        for begin, end in zip(times, times[1:]):
            for index, entering in events[begin]:
                if entering:
                    active.add(index)
                else:
                    active.remove(index)
            options = defaultdict(list)
            for index in sorted(active):
                _, _, unit, identity, value = notes[index]
                options[unit].append((identity, value))
            if options:
                result.append(dict(section=section, voice=voice, start=begin, end=end,
                    factor=weight*(end-begin)/beat_ms,
                    options=[dict(unit=unit, value=max(v for _, v in candidates),
                                  notes=[identity for identity, _ in candidates])
                             for unit, candidates in sorted(options.items())]))
    return result


def add_rewards(intervals, utility, constraint):
    """Integrate the maximum selected score, once per voice and interval."""
    for row in intervals:
        options = row['options']
        previous = 0.
        for level in sorted({option['value'] for option in options}):
            auxiliary = len(utility)
            utility.append(row['factor']*(level-previous))
            constraint({auxiliary: 1, **{o['unit']: -1 for o in options if o['value'] >= level}}, 0)
            previous = level


def coverage_report(intervals, chosen, weight):
    rows = []
    for row in intervals:
        selected = [option for option in row['options'] if chosen[option['unit']]]
        value = max((o['value'] for o in selected), default=0.)
        rows.append(dict(section=row['section'], voice=row['voice'], start=row['start'], end=row['end'],
            selected_notes=sorted({identity for o in selected for identity in o['notes']}),
            selected_value=value, reward=row['factor']*value))
    return dict(weight=weight, research_only=True, horizon_beats=2,
        semantics='bounded_sustain_expression_not_acoustic_salience',
        reward=sum(r['reward'] for r in rows),
        covered_voice_ms=sum(r['end']-r['start'] for r in rows if r['selected_notes']), intervals=rows)
