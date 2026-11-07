"""Evaluate the actual automatic timing path from pinned local analysis snapshots.

Unlike evaluate_audio.py's known-reference-grid comparison, this never supplies
human BPM or Offset to the generator. Unstable automatic timing is recorded as
a stopped generation, exactly as in the application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters, select_notes, assign_lanes
from manialab.domain import note_from, parse_osu, export_osu, validate
from manialab.timing import chart_clock_notes, align_candidates, grid_report
from manialab.voices import apply_phrase_evidence
from scripts.evaluate_audio import LOADED_SOURCE, ROOT, match_attacks, durations


def evaluate(folder, shift=22.5):
    audit = json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    if audit['source_fingerprint']['files'] != LOADED_SOURCE['files']:
        raise ValueError('Analysis source/model fingerprint differs; rerun the source audit first')
    human = parse_osu((folder/'reference.osu').read_text(encoding='utf-8-sig'))
    prefs = parameters({'keys': human['keys']})
    result = dict(beatmap_id=audit['beatmap_id'], title=audit['title'], keys=human['keys'],
                  source_fingerprint=LOADED_SOURCE['sha256'], audio_shift_ms=shift,
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), modes={})
    for mode in ('basic', 'deep'):
        analysis = json.loads((folder/mode/'analysis.json').read_text(encoding='utf-8'))
        tempo = analysis['tempo'] | {'offset': math.floor(analysis['tempo']['offset']-shift+.5)}
        if not tempo.get('stable'):
            result['modes'][mode] = dict(generated=False, reason=tempo['reliability'], bpm=tempo['bpm'], offset=tempo['offset'])
            continue
        evidence, _ = apply_phrase_evidence([note_from(n) for n in analysis['candidates']],
            [note_from(n) for n in analysis.get('phrase_candidates', [])], prefs['phrase_ln'])
        aligned, alignment = align_candidates(chart_clock_notes(evidence, shift), tempo)
        selected, selection = select_notes(aligned, prefs)
        notes = assign_lanes(selected, prefs, beat_ms=60000/tempo['bpm'])
        content = export_osu(notes, human['keys'], ROOT/'vendor/mapperatorinator/template.osu',
            bpm=tempo['bpm'], offset=tempo['offset'], grid_division=tempo.get('division', 4))
        reread = parse_osu(content)
        result['modes'][mode] = dict(generated=True, bpm=tempo['bpm'], offset=tempo['offset'],
            division=tempo.get('division', 4), validation=validate(reread['notes'], human['keys']),
            grid=grid_report(reread['notes'], tempo), holds=durations(reread['notes']),
            attack_match=match_attacks([n.start for n in reread['notes']], [n.start for n in human['notes']], 20),
            selection=selection, alignment=alignment)
        (folder/mode/'generated-auto-grid.osu').write_text(content, encoding='utf-8')
    (folder/'auto-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for path in sorted(args.folder.glob('*/audit.json')):
        try:
            result = evaluate(path.parent)
            print(json.dumps(result, ensure_ascii=True), flush=True)
        except Exception as error:
            failures.append(dict(folder=str(path.parent), error=str(error)))
            print(json.dumps(failures[-1]), flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} automatic generation evaluations failed')
