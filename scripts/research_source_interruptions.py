"""Research trimming existing pitch candidates at a source's near-silent gap.

No new attacks or tails are invented. A >=25ms interior gap below the existing
release envelope threshold ends the original interval. The experiment is not
enabled in the application; broad-source mixtures may mask instrument gaps.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.domain import note_from, parse_osu
from manialab.arrange import parameters, select_notes
from manialab.timing import align_candidates, chart_clock_notes
from manialab.voices import apply_phrase_evidence
from scripts.research_continuity_guard import pcm
from scripts.evaluate_synthetic_mix import pitched_match
from scripts.evaluate_ln import compare, long_reference_recall
from scripts.evaluate_audio import LOADED_SOURCE
from scripts.audit_snapshot import assert_snapshot


def trim(notes, samples, rate=22050):
    step = round(rate * .005)
    padded = np.pad(samples, (0, (-len(samples)) % step))
    level = np.sqrt(np.mean(padded.reshape(-1, step) ** 2, axis=1))
    floor = max(.00001, float(np.sqrt(np.mean(samples ** 2))) * .002)
    gap_bins = int(np.ceil(.025 * rate / step))
    result = []
    changes = []
    for note in notes:
        a = max(0, int(note.start * rate / 1000 / step))
        b = min(len(level), int(np.ceil(note.end * rate / 1000 / step)))
        region = level[a:b]
        if not note.hold or len(region) < 3:
            result.append(note)
            continue
        threshold = max(floor, float(np.percentile(region, 85)) * .02)
        active = np.flatnonzero(region >= threshold)
        pauses = np.flatnonzero(np.diff(active) - 1 >= gap_bins)
        if len(pauses):
            end = max(note.start, min(note.end, round((a + int(active[pauses[0]]) + 1) * step / rate * 1000)))
            if end < note.end:
                changes.append(dict(id=note.id, start=note.start, old_end=note.end, end=end))
                note = replace(note, end=end)
        result.append(note)
    return result, changes


def variants(folder, analysis, basic=False):
    raw = [note_from(n) for n in analysis['candidates']]
    replacements = {}
    changes = []
    for voice in (('mix',) if basic else ('bass', 'other', 'vocals')):
        if basic:
            samples, rate = pcm(folder / 'preview.wav')
        else:
            samples = np.load(folder / f'deep/stems/{voice}.npy').astype(np.float64)
            rate = 22050
        members = [n for n in raw if basic or n.voice == voice]
        updated, altered = trim(members, samples, rate)
        replacements.update({n.id: n for n in updated})
        changes.extend(altered)
    return {'baseline': raw, 'probe': [replacements.get(n.id, n) for n in raw]}, changes


def evaluate(folder):
    audit = json.loads((folder / 'audit.json').read_text(encoding='utf-8'))
    if not audit['reference']['constant_grid']:
        return None
    assert_snapshot(audit['source_fingerprint']['files'], LOADED_SOURCE['files'], ROOT)
    analysis = json.loads((folder / 'deep/analysis.json').read_text(encoding='utf-8'))
    modes, changes = variants(folder, analysis)
    human = parse_osu((folder / 'reference.osu').read_text(encoding='utf-8-sig'))
    prefs = parameters({'keys': human['keys']})
    offset, beat = audit['reference']['timing_points'][0]
    tempo = dict(bpm=60000 / beat, offset=offset, division=4, stable=True)
    results = {}
    for name, notes in modes.items():
        notes, _ = apply_phrase_evidence(notes, [note_from(n) for n in analysis.get('phrase_candidates', [])], prefs['phrase_ln'])
        aligned, _ = align_candidates(chart_clock_notes(notes, 22.5), tempo)
        selected, _ = select_notes(aligned, prefs)
        results[name] = compare(selected, human['notes']) | {'objects': len(selected),
            'ln': sum(n.hold for n in selected), 'long_reference': long_reference_recall(selected, human['notes'])}
    return dict(id=audit['beatmap_id'], shortened=len(changes), results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('synthetic', 'negative', 'real'), required=True)
    parser.add_argument('--folder', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = dict(protocol=__doc__, enabled=False, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.mode == 'real':
        result.update(rows=[], skipped=[])
        for path in sorted(args.folder.glob('*/audit.json')):
            row = evaluate(path.parent)
            if row is None:
                result['skipped'].append(path.parent.name)
                continue
            result['rows'].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    else:
        if args.mode == 'negative':
            from manialab.audio import analyze
            analysis = analyze(args.folder / 'repeated.wav', args.folder, ROOT / 'assets/basic-pitch.onnx', parameters())
        else:
            analysis = json.loads((args.folder / 'deep/analysis.json').read_text(encoding='utf-8'))
        modes, changes = variants(args.folder, analysis, args.mode == 'negative')
        result['shortened'] = len(changes)
        if args.mode == 'synthetic':
            reference = [note_from(n) for n in json.loads((args.folder / 'ground-truth.json').read_text(encoding='utf-8'))]
            result['modes'] = {name: {'intervals': pitched_match(notes, reference, True),
                'long_matched': pitched_match(notes, [n for n in reference if n.end-n.start >= 2000], True)['matched']}
                for name, notes in modes.items()}
        else:
            result['modes'] = {name: {'false_long_on_true_pitches': sum(n.pitch in (60, 64, 67) and n.end-n.start >= 800 for n in notes)}
                               for name, notes in modes.items()}
        print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
