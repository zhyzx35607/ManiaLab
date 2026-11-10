"""Verify SMD's MIDI/CSV key endpoints before preparing an acoustic benchmark."""
import argparse
from collections import defaultdict, deque
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha
from scripts.piano_pedal_labels import sustain_labels


def midi_events(path):
    import mido
    active = defaultdict(deque)
    notes, controls = [], []
    clock = 0.
    for event in mido.MidiFile(path):
        clock += event.time
        if event.type == 'note_on' and event.velocity:
            active[(event.channel, event.note)].append((clock, event.velocity))
        elif event.type in ('note_on', 'note_off'):
            key = (event.channel, event.note)
            if not active[key]:
                raise ValueError('Unpaired MIDI release')
            start, velocity = active[key].popleft()
            if clock <= start:
                raise ValueError('Nonpositive MIDI note')
            notes.append(dict(onset=start, offset=clock, pitch=event.note,
                              velocity=velocity, channel=event.channel, program=0, is_drum=False))
        elif event.type == 'control_change' and event.control in (64, 66, 67):
            controls.append(dict(time=clock, channel=event.channel,
                                 controller=event.control, value=event.value))
    if any(active.values()):
        raise ValueError('Unclosed MIDI keys')
    return sorted(notes, key=lambda n: (n['onset'], n['pitch'], n['offset'])), controls


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dependencies', type=Path, default=Path('test-data/piano-release-probe/dependencies'))
    args = cli.parse_args()
    sys.path.append(str(args.dependencies.resolve()))
    root = Path('test-data/smd/selected-v1')
    source = json.loads((root/'source.json').read_text(encoding='utf-8'))
    output = root/'benchmark'
    if output.exists():
        raise ValueError('Preserve prepared benchmark')
    for row in source['records']:
        if sha(root/row['member']) != row['sha256']:
            raise ValueError('Source file changed')
    output.mkdir()
    cases = []
    for composer, name in source['selection']['composers'].items():
        midi = root/'midi'/(name+'.mid')
        csv_path = root/'csv'/(name+'.csv')
        audio = root/'wav_22050_mono'/(name+'.wav')
        notes, pedals = midi_events(midi)
        with csv_path.open(newline='', encoding='utf-8-sig') as stream:
            csv_notes = [dict(onset=float(row['Start']),
                              offset=float(row['Start'])+float(row['Duration']), pitch=int(row['Pitch']))
                         for row in csv.DictReader(stream, delimiter=';')]
        csv_notes.sort(key=lambda n: (n['onset'], n['pitch'], n['offset']))
        a = np.asarray([[n[k] for k in ('onset', 'offset', 'pitch')] for n in notes])
        b = np.asarray([[n[k] for k in ('onset', 'offset', 'pitch')] for n in csv_notes])
        if a.shape != b.shape or not np.array_equal(a[:, 2], b[:, 2]):
            raise ValueError('MIDI/CSV event inventory mismatch: '+name)
        error = np.max(abs(a[:, :2]-b[:, :2]), axis=0)
        if max(error) > .0011:
            raise ValueError('CSV does not match MIDI key endpoints: '+name)
        rate, waveform = wavfile.read(audio)
        if rate != 22050 or waveform.ndim != 1 or waveform.dtype != np.int16:
            raise ValueError('Unexpected source PCM format')
        duration = len(waveform)/rate
        score_seconds, input_seconds = min(60., duration), min(70., duration)
        labels = output/(composer.lower()+'-labels.json')
        labels.write_text(json.dumps(notes, indent=2), encoding='utf-8')
        sustained, pedal_audit = sustain_labels(notes, pedals, duration)
        sustain_path = output/(composer.lower()+'-sustain-labels.json')
        sustain_path.write_text(json.dumps(sustained, indent=2), encoding='utf-8')
        (output/(composer.lower()+'-pedals.json')).write_text(json.dumps(pedals, indent=2), encoding='utf-8')
        case = dict(name=composer.lower(), original=name, composer=composer,
                    audio=str(audio), audio_sha256=sha(audio), labels=str(labels), labels_sha256=sha(labels),
                    midi_sha256=sha(midi), csv_sha256=sha(csv_path),
                    sustain_labels=str(sustain_path), sustain_labels_sha256=sha(sustain_path), pedal_audit=pedal_audit,
                    midi_csv_max_error_ms=(error*1000).tolist(), key_events=len(notes), pedal_events=len(pedals),
                    start_seconds=0., score_seconds=score_seconds, duration_seconds=input_seconds,
                    original_duration=duration, unobservable_reference_offsets=sum(
                        n['onset'] < score_seconds and n['offset'] > input_seconds for n in notes),
                    unobservable_sustain_offsets=sum(n['onset'] < score_seconds and n['offset'] > input_seconds for n in sustained))
        cases.append(case)
        print(json.dumps(case), flush=True)
    result = dict(cases=cases, source_sha256=sha(root/'source.json'),
                  protocol_sha256=sha('docs/piano-acoustic-protocol.md'),
                  pedal_protocol_sha256=sha('docs/piano-pedal-protocol.md'),
                  pedal_label_code_sha256=sha('scripts/piano_pedal_labels.py'),
                  runner_sha256=sha(__file__), endpoint_semantics='Separate MIDI key release and pedal sustain; neither is chart LN')
    (output/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
