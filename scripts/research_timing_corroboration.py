"""Research-only local timing corroboration; never changes the released policy.

Keep the chosen BPM/Offset and global quality gates. A suspect 12-second
window can only be explained by the other source if it has at least ten
attacks, >=80% coverage, and <=15ms local phase deviation on that same grid.
The other source must also meet the existing global coverage/error gates.
Mixture and separated drums are correlated evidence, not independent votes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab.domain import note_from
from manialab.timing import _score, grid_diagnostics
from scripts.audit_snapshot import assert_snapshot
from scripts.evaluate_audio import LOADED_SOURCE


def evaluate(path, classification):
    audit = json.loads(path.read_text(encoding='utf-8'))
    assert_snapshot(audit['source_fingerprint']['files'], LOADED_SOURCE['files'], ROOT)
    deep = json.loads((path.parent / 'deep/analysis.json').read_text(encoding='utf-8'))
    basic = json.loads((path.parent / 'basic/analysis.json').read_text(encoding='utf-8'))
    tempo = deep['tempo']
    selected = tempo['evidence_source']
    alternate = 'mix' if selected == 'drums' else 'drums'
    sources = {'drums': [note_from(n) for n in deep['candidates'] if n['voice'] == 'drums'],
               'mix': [note_from(n) for n in basic['timing_attacks']]}
    other = sources[alternate]
    row = dict(beatmap_id=audit['beatmap_id'], group=path.parent.parent.name,
        current_accepted=tempo['stable'], selected_source=selected,
        bpm=tempo['bpm'], raw_offset=tempo['offset'], classification=classification['classification'])
    if not other:
        return row | {'probe_accepted': tempo['stable'], 'reason': 'No alternate-source attacks'}
    times = np.array([n.start for n in other], dtype=float)
    weights = np.array([max(.05, n.strength) for n in other])
    score = _score(times, weights, tempo['bpm'], tempo['offset'], tempo['division'], choose_phase=False)
    sections = {s['start_ms']: s for s in grid_diagnostics(other, tempo['bpm'], tempo['offset'], tempo['division'])}
    own_global = tempo['coverage'] >= .8 and tempo['median_error_ms'] <= 18 and 'exact_bpm' not in tempo
    other_global = score['coverage'] >= .8 and score['median_error_ms'] <= 18
    explanations = []
    unresolved = []
    for section in tempo['sections']:
        if not section['suspect']:
            continue
        candidate = sections.get(section['start_ms'])
        explained = bool(candidate and candidate['attacks'] >= 10 and candidate['coverage'] >= .8 and
                         abs(candidate['phase_error_ms']) <= 15)
        explanations.append(dict(current=section, alternate=candidate, explained=explained))
        if not explained:
            unresolved.append(section['start_ms'])
    approved = bool(tempo['stable'] or (own_global and other_global and len(tempo['sections']) >= 2 and not unresolved))
    return row | dict(probe_accepted=approved, current_global_gate=own_global,
        alternate_global_gate=other_global, alternate_score=score,
        remaining_suspect_windows=unresolved, explanations=explanations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    classifications = {(r['group'], r['beatmap_id']): r for r in
        json.loads((ROOT / 'docs/timing-acceptance.json').read_text(encoding='utf-8'))['rows'] if r['mode'] == 'deep'}
    rows = []
    for folder in args.folders:
        for path in sorted(folder.glob('*/audit.json')):
            audit = json.loads(path.read_text(encoding='utf-8'))
            row = evaluate(path, classifications[(folder.name, audit['beatmap_id'])])
            rows.append(row)
    additions = [r for r in rows if r['probe_accepted'] and not r['current_accepted']]
    report = dict(protocol=__doc__, enabled=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        notice='Existing 67-song development audit, not an unseen evaluation. Human-grid compatibility is not physical timing ground truth. No thresholds were revised after this run.',
        summary=dict(current_accepted=sum(r['current_accepted'] for r in rows),
                     probe_accepted=sum(r['probe_accepted'] for r in rows),
                     additions_by_reference_class=dict(Counter(r['classification'] for r in additions))), rows=rows)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2), flush=True)
