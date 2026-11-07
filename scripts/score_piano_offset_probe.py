"""Score the isolated piano-offset probe without treating charts as full audio labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.audio import analyze
from manialab.arrange import parameters
from manialab.domain import Note, note_from, parse_osu
from manialab.timing import chart_clock_notes
from scripts.evaluate_synthetic_mix import pitched_match
from scripts.evaluate_audio import match_attacks, LOADED_SOURCE
from scripts.evaluate_ln import compare
from scripts.audit_snapshot import assert_snapshot


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--synthetic', type=Path, required=True)
    parser.add_argument('--negative', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    predictions = {name: [note_from(n) for n in read(args.inference / (name + '-notes.json'))]
                   for name in ('controlled-mix', 'repeated-chords', 'river-flows-piano')}
    reference = [note_from(n) for n in read(args.synthetic / 'ground-truth.json')]
    positive = {}
    modes = {mode: [note_from(n) for n in read(args.synthetic / mode / 'analysis.json')['candidates']]
             for mode in ('basic', 'deep')}
    modes['piano_model'] = predictions['controlled-mix']
    for name, notes in modes.items():
        positive[name] = dict(pitched_onsets=pitched_match(notes, reference),
            pitched_intervals=pitched_match(notes, reference, True),
            matched_over_2s=pitched_match(notes, [n for n in reference if n.end-n.start >= 2000], True)['matched'])
    negative_reference = [Note(round((.115 + group + i * .5) * 1000), round((.585 + group + i * .5) * 1000),
                               pitch=pitch, source='pitch')
                          for group in (0, 5) for i in range(8) for pitch in (60, 64, 67)]
    baseline = analyze(args.negative / 'repeated.wav', args.negative, ROOT / 'assets/basic-pitch.onnx', parameters())
    negative = {}
    for name, notes in {'basic': [note_from(n) for n in baseline['candidates']],
                        'piano_model': predictions['repeated-chords']}.items():
        negative[name] = dict(pitched_onsets=pitched_match(notes, negative_reference),
            pitched_intervals=pitched_match(notes, negative_reference, True),
            false_long_on_true_pitches=sum(n.pitch in (60, 64, 67) and n.end-n.start >= 800 for n in notes))
    audit = read(args.reference / 'audit.json')
    assert_snapshot(audit['source_fingerprint']['files'], LOADED_SOURCE['files'], ROOT)
    human = parse_osu((args.reference / 'reference.osu').read_text(encoding='utf-8-sig'))
    reference_notes = [n for n in human['notes'] if 0 <= n.start <= n.end < 40000]
    real = {}
    for name in ('basic', 'deep', 'piano_model'):
        notes = (predictions['river-flows-piano'] if name == 'piano_model' else
                 [note_from(n) for n in read(args.reference / name / 'analysis.json')['candidates']])
        notes = [n for n in chart_clock_notes(notes, 22.5) if 0 <= n.start <= n.end < 40000]
        real[name] = dict(onset_rows=match_attacks([n.start for n in notes], [n.start for n in reference_notes], 40),
                         ln=compare(notes, reference_notes), candidates=len(notes))
    report = dict(enabled_in_application=False,
        notice='Default official piano-specific model, evaluated without threshold changes. Synthetic harmonic tones are not natural pianos. One natural piano excerpt is an exploratory chart-style comparison, not exhaustive transcription ground truth or an independent music benchmark.',
        inference=read(args.inference / 'inference-report.json'),
        synthetic_mixture=positive, repeated_synthetic_chords=negative,
        natural_piano_excerpt=dict(beatmap_id=audit['beatmap_id'], start_ms=0, end_ms=40000,
                                  reference_constant=audit['reference']['constant_grid'],
                                  audio_shift_ms=22.5, no_grid_quantization=True, modes=real))
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
