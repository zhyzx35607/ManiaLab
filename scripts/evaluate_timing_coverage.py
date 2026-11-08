"""Recompute source-coverage diagnostics on the frozen 67-song development audit.

This reuses already extracted attacks, not fresh model inference. It verifies
the missing-window policy only; original audio and LN scores are not recomputed.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manialab.domain import note_from
from manialab.timing import cover_missing_timing_windows


if __name__=='__main__':
    classes=json.loads((ROOT/'docs/timing-acceptance.json').read_text(encoding='utf-8'))['rows']
    rows=[];folders={}
    for group in {c['group'] for c in classes}:
        for path in (ROOT/'test-data'/group).glob('*/audit.json'):
            audit=json.loads(path.read_text(encoding='utf-8'))
            folders[(group,audit['beatmap_id'])]=path.parent
    for c in classes:
        if c['mode']!='deep':continue
        root=folders[(c['group'],c['beatmap_id'])]
        raw=(root/'deep/analysis.json').read_bytes();d=json.loads(raw)
        basic_raw=(root/'basic/analysis.json').read_bytes();b=json.loads(basic_raw)
        original=d['tempo']
        if original['evidence_source']=='drums':
            revised,_=cover_missing_timing_windows(original,[note_from(n) for n in d['timing_attacks']],
                                                   [note_from(n) for n in b['timing_attacks']])
        else:revised=original
        assert (revised['bpm'],revised['offset'])==(original['bpm'],original['offset'])
        assert not revised['stable'] or original['stable']
        rows.append(dict(group=c['group'],beatmap_id=c['beatmap_id'],classification=c['classification'],
            deep_analysis_sha256=hashlib.sha256(raw).hexdigest(),basic_analysis_sha256=hashlib.sha256(basic_raw).hexdigest(),
            before=original['stable'],after=revised['stable'],bpm=revised['bpm'],offset=revised['offset'],
            added_sections=[s for s in revised.get('sections',[]) if s.get('fallback')]))
    report=dict(protocol=__doc__,source_sha256=hashlib.sha256((ROOT/'manialab/timing.py').read_bytes()).hexdigest(),
        summary=dict(songs=len(rows),accepted_before=sum(r['before'] for r in rows),accepted_after=sum(r['after'] for r in rows),
                     revoked=[dict(beatmap_id=r['beatmap_id'],classification=r['classification']) for r in rows if r['before'] and not r['after']]),rows=rows)
    (ROOT/'docs/timing-coverage-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False))
