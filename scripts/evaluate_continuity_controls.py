"""Additional fixed synthetic counterexamples for the waveform continuity guard.

Each repeated chord has an explicit 10/30/80ms silence in its own signal.
The overlapping distractor does not share its fundamental pitches. Parameters
are fixed before evaluation; these examples are not used to tune the guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.audio import analyze
from manialab.arrange import parameters
from manialab.domain import Note
from scripts.research_continuity_guard import variants, PROTOCOL
from scripts.evaluate_synthetic_mix import pitched_match


def synthesize(folder, root_pitch, gap_ms, distracting):
    rate = 22050
    signal = np.zeros(rate * 5, dtype=np.float64)
    references = []
    for index in range(8):
        start = .115 + index * .5
        length = .5 - gap_ms / 1000
        times = np.arange(round(length * rate)) / rate
        envelope = np.minimum(1, times / .005) * np.minimum(1, (length - times) / .005)
        for pitch in (root_pitch, root_pitch + 4, root_pitch + 7):
            frequency = 440 * 2 ** ((pitch - 69) / 12)
            tone = sum(np.sin(2 * np.pi * frequency * h * times) / h ** 1.5 for h in (1, 2, 3))
            a = round(start * rate)
            signal[a:a + len(times)] += .09 * tone * envelope
            references.append(Note(round(start * 1000), round((start + length) * 1000),
                                   pitch=pitch, source='pitch', id=f'{pitch}-{index}'))
    if distracting:
        times = np.arange(rate * 4) / rate
        frequency = 440 * 2 ** ((root_pitch + 14 - 69) / 12)
        tone = .1 * (np.sin(2 * np.pi * frequency * times) + .2 * np.sin(4 * np.pi * frequency * times))
        tone *= np.minimum(1, times / .015) * np.minimum(1, (4 - times) / .015)
        a = round(.115 * rate)
        signal[a:a + len(times)] += tone
    folder.mkdir(parents=True, exist_ok=True)
    wavfile.write(folder / 'repeated.wav', rate, (np.clip(signal, -.98, .98) * 32767).astype(np.int16))
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(protocol=PROTOCOL, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  description=__doc__, rows=[])
    for root_pitch in (48, 60, 72):
        for gap in (10, 30, 80):
            for distractor in (False, True):
                folder = args.work / f'{root_pitch}-{gap}-{int(distractor)}'
                reference = synthesize(folder, root_pitch, gap, distractor)
                analysis = analyze(folder / 'repeated.wav', folder, ROOT / 'assets/basic-pitch.onnx', parameters())
                modes, diagnostic = variants(folder, analysis, basic=True)
                pitches = {n.pitch for n in reference}
                results = {}
                for name, notes in modes.items():
                    notes = [n for n in notes if n.pitch in pitches and n.source == 'pitch']
                    results[name] = dict(pitched_intervals=pitched_match(notes, reference, True),
                        false_holds_over_800ms=sum(n.end - n.start >= 800 for n in notes))
                row = dict(root_pitch=root_pitch, gap_ms=gap, distracting_sustain=distractor,
                           proposed=diagnostic['proposed'], accepted=diagnostic['accepted'], results=results)
                report['rows'].append(row)
                args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
