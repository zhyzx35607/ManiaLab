"""Pair selected reference LN intervals with existing, separated audio evidence.

No inferred source label is a human annotation. Matching a reference is not
transcription F1: authored charts deliberately omit sounds and choose different
expressions. Fixed clip edges and two explicit clock conventions are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu
from manialab.timing import BeatClock
from scripts.study_ln_references import beat_duration


def coverage(intervals, start, end):
    merged = []
    for a,b in sorted((max(start,a),min(end,b)) for a,b in intervals if b>start and a<end):
        if merged and a<=merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],b)
        else:
            merged.append([a,b])
    gaps,at = [],start
    for a,b in merged:
        gaps.append(a-at)
        at=b
    gaps.append(end-at)
    return dict(coverage=round(sum(b-a for a,b in merged)/max(1,end-start),4),
                longest_gap_ms=max(gaps,default=end-start))


def report(root, output):
    rows=[]
    for path in sorted((root/'clips').glob('*/analysis.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        meta=data['research_clip'];bid=meta['beatmap_id']
        reference=root/'audio'/str(bid)/'reference.osu'
        if hashlib.sha256(reference.read_bytes()).hexdigest()!=meta['chart_sha256']:
            raise ValueError(f'Changed chart: {bid}')
        original=json.loads((reference.parent/'reference.json').read_text(encoding='utf-8'))
        audio=reference.parent/original['audio_filename']
        if hashlib.sha256(audio.read_bytes()).hexdigest()!=meta['original_audio_sha256']:
            raise ValueError(f'Changed audio: {bid}')
        chart=parse_osu(reference.read_text(encoding='utf-8-sig'))
        clock=BeatClock(chart['timing'])
        begin,finish=meta['start_ms']+meta['excluded_edge_ms'],meta['end_ms']-meta['excluded_edge_ms']
        holds=[n for n in chart['notes'] if n.hold and begin<=n.start<n.end<=finish]
        long=[n for n in holds if beat_duration(n.start,n.end,clock)>=2-.01]
        candidates=data['candidates']
        voice_names=sorted({n['voice'] for n in candidates if n['source']=='pitch'})
        examples=[]
        for n in long:
            support=[]
            # The reference chart clock is left untouched. A generic 22.5 ms
            # convention is shown as sensitivity, never silently fitted per song.
            for shift in (0,22.5):
                a,b=n.start-meta['start_ms']+shift,n.end-meta['start_ms']+shift
                for voice in voice_names:
                    overlaps=[c for c in candidates if c['source']=='pitch' and c['voice']==voice and c['end']>a and c['start']<b]
                    times=[c['end'] for c in candidates if c['source']=='pitch' and c['voice']==voice and c['start']<b]
                    head_times=[c['start'] for c in candidates if c['source']=='pitch' and c['voice']==voice]
                    support.append(dict(voice=voice,audio_shift_ms=shift,
                        **coverage([(c['start'],c['end']) for c in overlaps],a,b),
                        detected_pitches=len({c['pitch'] for c in overlaps}),
                        inside_detected_attacks=sum(a+50<c['start']<b-50 for c in overlaps),
                        nearest_head_ms=round(min((abs(t-a) for t in head_times),default=1e9),1),
                        nearest_tail_ms=round(min((abs(t-b) for t in times),default=1e9),1)))
            support.sort(key=lambda r:(-r['coverage'],r['nearest_head_ms']))
            examples.append(dict(start_ms=n.start,end_ms=n.end,lane=n.lane+1,
                duration_beats=round(beat_duration(n.start,n.end,clock),3),source_evidence=support))
        rows.append(dict(beatmap_id=bid,title=chart['metadata'].get('TitleUnicode') or chart['metadata']['Title'],
            keys=chart['keys'],clip=meta,analysis_seconds=data['elapsed_seconds'],
            candidates=len(candidates),phrase_candidates=len(data['phrase_candidates']),
            interior_reference_holds=len(holds),interior_reference_holds_two_beats=len(long),examples=examples))
    result=dict(protocol=__doc__,rows=rows,trained=False,
        limits=['Source separation and pitch estimates can bleed or miss events.',
                'Overlapping several pitches does not establish that an LN follows vocals or one syllable.',
                'A reference tail need not equal an acoustic decay endpoint.',
                'These inspected songs are development material, not held-out evaluation.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(clips=len(rows),long_hold_intervals=sum(len(r['examples']) for r in rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('test-data/chinese-ln'))
    p.add_argument('--output',type=Path,default=Path('test-data/chinese-ln/audio-evidence.json'))
    a=p.parse_args();report(a.root,a.output)
