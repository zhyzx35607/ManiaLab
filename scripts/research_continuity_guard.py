"""Research-only waveform guard for sustained-pitch reinterpretations.

The release detector is unchanged. This asks whether short waveform dips or
renewed attacks can disprove a hold suggested by continuous pitch activation.
It is not an instrument separator or a learned musical-release model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.domain import note_from, parse_osu
from manialab.arrange import parameters, select_notes
from manialab.timing import align_candidates, chart_clock_notes
from manialab.voices import apply_phrase_evidence
from scripts.research_pitch_continuity import alternatives
from scripts.evaluate_synthetic_mix import pitched_match
from scripts.evaluate_ln import compare, long_reference_recall
from scripts.evaluate_audio import LOADED_SOURCE
from scripts.audit_snapshot import assert_snapshot


PROTOCOL = {
    "purpose": "Can waveform evidence reject false connections without losing the sustained chord improvement?",
    "published_application_modified": False,
    "pitch_activation_threshold": .3,
    "dip_ratio": .45,
    "renewal_ratio": 1.8,
    "envelope_sigma_seconds": "clip(2 / pitch_frequency, .004, .012)",
    "anchor_times_changed": False,
    "human_references_used_for_guard": False,
    "limitation": "A pitch band can contain multiple instruments or harmonics; absence of a dip does not establish one continuous musical event.",
}


def pcm(path):
    rate, samples = wavfile.read(path)
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float64) / np.iinfo(samples.dtype).max
    else:
        samples = samples.astype(np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    assert rate == 22050
    return samples, rate


def guard(note, raw, samples, rate):
    """Conservative negative evidence only, with parameters fixed in PROTOCOL."""
    support = [n for n in raw if n.id in note.support and n.start > note.start + 80]
    if not support:
        return False, [{"reason": "no_secondary_support"}]
    lo = max(0, round((note.start - 150) * rate / 1000))
    hi = min(len(samples), round((note.end + 150) * rate / 1000))
    segment = samples[lo:hi]
    frequency = 440 * 2 ** ((note.pitch - 69) / 12)
    demodulated = segment * np.exp(-2j * np.pi * frequency * np.arange(len(segment)) / rate)
    # Decimation is only a computational step. All comparisons use the same
    # envelope and retain the PCM sample origin exactly.
    demodulated = resample_poly(demodulated, 1, 10)
    envelope_rate = rate / 10
    sigma = np.clip(2 / frequency, .004, .012) * envelope_rate
    envelope = np.abs(gaussian_filter1d(demodulated.real, sigma) +
                      1j * gaussian_filter1d(demodulated.imag, sigma))

    def window(t, begin, finish):
        a = max(0, round(((t + begin) / 1000 - lo / rate) * envelope_rate))
        b = min(len(envelope), round(((t + finish) / 1000 - lo / rate) * envelope_rate))
        return envelope[a:b]

    checks = []
    rejected = False
    for candidate in support:
        before = window(candidate.start, -80, -30)
        after = window(candidate.start, 20, 65)
        around = window(candidate.start, -35, 20)
        if min(len(before), len(after), len(around)) < 3:
            continue
        left, right = float(np.median(before)), float(np.median(after))
        dip = float(np.min(around)) / max(min(left, right), 1e-9)
        renewal = right / max(left, 1e-9)
        reject = dip < PROTOCOL['dip_ratio'] or renewal > PROTOCOL['renewal_ratio']
        rejected |= reject
        checks.append(dict(at=candidate.start, dip_ratio=round(dip, 4),
                           renewal_ratio=round(renewal, 4), rejected=bool(reject)))
    return not rejected and bool(checks), checks


def variants(folder, analysis, basic=False):
    raw = [note_from(n) for n in analysis['candidates']]
    proposed, accepted, diagnostics = {}, {}, []
    for voice in (('mix',) if basic else ('bass', 'other')):
        cache = folder / ('activations.npz' if basic else f'deep/stems/{voice}-activations.npz')
        wave = folder / ('preview.wav' if basic else f'deep/stems/{voice}.wav')
        samples, rate = pcm(wave)
        members = [n for n in raw if basic or n.voice == voice]
        with np.load(cache) as values:
            changes = alternatives(values['frames'], values['times'], members)
        proposed.update(changes)
        for identifier, note in changes.items():
            keep, checks = guard(note, members, samples, rate)
            if keep:
                accepted[identifier] = note
            diagnostics.append(dict(id=identifier, voice=voice, start=note.start, end=note.end,
                                    pitch=note.pitch, accepted=bool(keep), checks=checks))
    return {
        'baseline': raw,
        'unguarded': [proposed.get(n.id, n) for n in raw],
        'guarded': [accepted.get(n.id, n) for n in raw],
    }, dict(proposed=len(proposed), accepted=len(accepted), details=diagnostics)


def real_song(folder):
    audit = json.loads((folder / 'audit.json').read_text(encoding='utf-8'))
    if not audit['reference']['constant_grid']:
        return None
    equivalent = assert_snapshot(audit['source_fingerprint']['files'], LOADED_SOURCE['files'], ROOT)
    analysis = json.loads((folder / 'deep/analysis.json').read_text(encoding='utf-8'))
    modes, diagnostic = variants(folder, analysis)
    human = parse_osu((folder / 'reference.osu').read_text(encoding='utf-8-sig'))
    prefs = parameters({'keys': human['keys']})
    offset, beat = audit['reference']['timing_points'][0]
    tempo = dict(bpm=60000 / beat, offset=offset, division=4, stable=True)
    results = {}
    for name, raw in modes.items():
        raw, _ = apply_phrase_evidence(raw, [note_from(n) for n in analysis.get('phrase_candidates', [])], prefs['phrase_ln'])
        aligned, _ = align_candidates(chart_clock_notes(raw, 22.5), tempo)
        notes, _ = select_notes(aligned, prefs)
        results[name] = compare(notes, human['notes']) | {
            'long_reference': long_reference_recall(notes, human['notes']),
            'objects': len(notes), 'ln': sum(n.hold for n in notes),
        }
    return dict(id=audit['beatmap_id'], newline_equivalent_files=equivalent,
                diagnostics=diagnostic, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--mode', choices=('synthetic', 'negative', 'real'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(protocol=PROTOCOL, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.mode == 'real':
        result.update(rows=[], skipped=[])
        for path in sorted(args.folder.glob('*/audit.json')):
            row = real_song(path.parent)
            if row is None:
                result['skipped'].append(path.parent.name)
                continue
            result['rows'].append(row)
            print(json.dumps({'id': row['id'], 'proposed': row['diagnostics']['proposed'],
                              'accepted': row['diagnostics']['accepted'], 'results': row['results']}), flush=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    else:
        if args.mode == 'negative':
            from manialab.audio import analyze
            analysis = analyze(args.folder / 'repeated.wav', args.folder, ROOT / 'assets/basic-pitch.onnx', parameters())
        else:
            analysis = json.loads((args.folder / 'deep/analysis.json').read_text(encoding='utf-8'))
        modes, diagnostic = variants(args.folder, analysis, args.mode == 'negative')
        result['diagnostics'] = diagnostic
        if args.mode == 'synthetic':
            reference = [note_from(n) for n in json.loads((args.folder / 'ground-truth.json').read_text(encoding='utf-8'))]
            result['modes'] = {name: {
                'pitched_intervals': pitched_match(notes, reference, True),
                'over_2s': pitched_match(notes, [n for n in reference if n.end - n.start >= 2000], True)['matched']
            } for name, notes in modes.items()}
        else:
            result['false_long_reinterpretations'] = diagnostic['accepted']
        print(json.dumps({k: v for k, v in result.items() if k != 'diagnostics'}), flush=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
