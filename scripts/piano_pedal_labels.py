"""Explicit MIDI sustain targets, separate from key releases and chart LN."""
from collections import defaultdict


def sustain_labels(notes, controls, duration):
    intervals = defaultdict(list)
    down = {}
    for event in sorted(controls, key=lambda e: e['time']):
        if event['controller'] != 64:
            continue
        channel, time = event['channel'], event['time']
        if event['value'] >= 64:
            down.setdefault(channel, time)
        elif channel in down:
            intervals[channel].append((down.pop(channel), time))
    unclosed = len(down)
    for channel, start in down.items():
        intervals[channel].append((start, duration))
    next_onset = {}
    result = []
    extended = 0
    for note in sorted(notes, key=lambda n: (n['onset'], n['pitch']), reverse=True):
        channel = note.get('channel', 0)
        key = (channel, note['pitch'])
        end = note['offset']
        for start, stop in intervals[channel]:
            if start < end < stop:
                end = max(end, min(stop, next_onset.get(key, duration)))
                break
        result.append(dict(note, offset=end, key_offset=note['offset']))
        extended += end > note['offset']
        next_onset[key] = note['onset']
    result.sort(key=lambda n: (n['onset'], n['pitch'], n['offset']))
    return result, dict(extended=extended, pedal_intervals=sum(map(len, intervals.values())),
                        unclosed_pedals=unclosed, sustain_controller_threshold=64)
