"""Audit basic/deep evidence against a small local set of human chart references.

Matching a human map is a measurable style comparison, not proof that all
audible events were transcribed or that a new chart plays well. Known reference
timing is used for arrangement so timing estimation is reported independently.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from manialab.audio import analyze, analyze_deep
from manialab.arrange import assign_lanes, parameters, select_notes, layout_metrics
from manialab.domain import export_osu, note_from, parse_osu, validate
from manialab.separation import Separator
from manialab.timing import align_candidates, chart_clock_notes, grid_report
from manialab.voices import apply_phrase_evidence

ROOT = Path(__file__).resolve().parents[1]


def source_fingerprint():
    files = [*sorted((ROOT/'manialab').glob('*.py')), Path(__file__), ROOT/'assets/pattern-prior.json']
    hashes = {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return dict(sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),files=hashes)


LOADED_SOURCE = source_fingerprint()


def constant_reference(points):
    """Equal BPM alone is not enough: a red line can also reset beat phase."""
    if not points:
        return False
    origin, beat = points[0]
    if max(p[1] for p in points) - min(p[1] for p in points) >= .01:
        return False
    return all(abs((offset - origin + beat / 2) % beat - beat / 2) <= 1.
               for offset, _ in points)


def match_attacks(predicted, reference, tolerance):
    a, b = sorted(set(predicted)), sorted(set(reference))
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j] - tolerance:
            i += 1
        elif b[j] < a[i] - tolerance:
            j += 1
        else:
            matched += 1;i += 1;j += 1
    precision, recall = matched / max(1, len(a)), matched / max(1, len(b))
    return dict(predicted=len(a), reference=len(b), matched=matched,
                precision=round(precision,4), recall=round(recall,4),
                f1=round(2*precision*recall/max(1e-9,precision+recall),4), tolerance_ms=tolerance)


def durations(notes):
    lengths = np.array([n.end-n.start for n in notes if n.hold])
    return dict(count=len(lengths), ratio=round(len(lengths)/max(1,len(notes)),4),
                percentiles_ms=np.round(np.percentile(lengths,[25,50,75,90,100])).astype(int).tolist() if len(lengths) else [],
                over_1s=int((lengths>=1000).sum()), over_3s=int((lengths>=3000).sum()))


def run(folder, modes, audio_shift_ms=22.5):
    metadata = json.loads((folder/'reference.json').read_text(encoding='utf-8'))
    audio = folder/metadata.get('audio_filename','audio.opus')
    parsed = parse_osu((folder/'reference.osu').read_text(encoding='utf-8-sig'))
    points=[]
    for line in parsed['timing']:
        p=line.split(',')
        if len(p)>=7 and p[6]=='1' and float(p[1])>0:
            points.append((float(p[0]),float(p[1])))
    if not points:
        raise ValueError('Reference has no red timing point')
    constant=constant_reference(points)
    beat=points[0][1]
    tempo=dict(bpm=60000/beat,offset=points[0][0],division=4,stable=True)
    prefs=parameters({'keys':parsed['keys']})
    reference=parsed['notes']
    output=dict(beatmap_id=metadata['beatmap_id'],title=metadata['title'],keys=parsed['keys'],
        source_fingerprint=LOADED_SOURCE, audio_shift_ms=audio_shift_ms,
        reference=dict(objects=len(reference),timing_points=points,constant_grid=constant,holds=durations(reference)),modes={})
    for mode in modes:
        started=time.perf_counter();work=folder/mode;work.mkdir(exist_ok=True)
        def progress(percent,message):
            if percent in (2,31,46,61,76,95,100):
                print(json.dumps(dict(id=metadata['beatmap_id'],mode=mode,percent=percent,message=message)),flush=True)
        if mode=='deep':
            analysis=analyze_deep(audio,work,ROOT/'assets/basic-pitch.onnx',prefs,Separator(ROOT),progress)
        else:
            analysis=analyze(audio,work,ROOT/'assets/basic-pitch.onnx',prefs,progress)
        (work/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False),encoding='utf-8')
        raw=[note_from(n) for n in analysis['candidates']]
        evidence,phrase_report=apply_phrase_evidence(raw,[note_from(n) for n in analysis.get('phrase_candidates',[])],prefs['phrase_ln'])
        result=dict(seconds=round(time.perf_counter()-started,2),tempo=analysis['tempo'],
            candidates=len(raw),candidate_voices=dict(Counter(n.voice for n in raw)),
            raw_attack_match={str(t):match_attacks([n.start for n in raw],[n.start for n in reference],t) for t in (20,40)},
            raw_hold_candidates=durations(raw),phrase_evidence=phrase_report)
        mapped = chart_clock_notes(raw,audio_shift_ms)
        result['mapping_attack_match'] = {str(t):match_attacks([n.start for n in mapped],[n.start for n in reference],t) for t in (20,40)}
        # Phase comparison is meaningful only on a shared constant grid.
        detected=analysis['tempo']
        ratio=detected['bpm']/tempo['bpm']
        same_unit=min(abs(ratio-v) for v in (.5,1.,2.))<.001
        if constant and same_unit:
            phase=lambda x:round((x-tempo['offset']+beat/8)%(beat/4)-beat/8,3)
            result['timing_comparison']=dict(raw_phase_error_ms=phase(detected['offset']),
                mapping_phase_error_ms=phase(detected['offset']-audio_shift_ms),
                bpm_ratio=round(ratio,6),notice='Reference chart phase is a mapping convention, not the physical waveform onset')
        if constant:
            aligned,alignment=align_candidates(chart_clock_notes(evidence,audio_shift_ms),tempo)
            selected,selection=select_notes(aligned,prefs)
            arranged=assign_lanes(selected,prefs,beat_ms=beat)
            result.update(generation=dict(grid_source='human reference timing',selection=selection,
                validation=validate(arranged,prefs['keys']),grid=grid_report(arranged,tempo),holds=durations(arranged),
                layout=layout_metrics(arranged,prefs['keys'],beat),
                attack_match=match_attacks([n.start for n in arranged],[n.start for n in reference],20)))
            (work/'generated-reference-grid.osu').write_text(export_osu(arranged,prefs['keys'],ROOT/'vendor/mapperatorinator/template.osu',
                bpm=tempo['bpm'],offset=tempo['offset'],grid_division=4),encoding='utf-8')
        else:
            result['generation']=dict(skipped='Changing tempo or beat phase; do not flatten reference red lines')
        output['modes'][mode]=result
        (folder/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=metadata['beatmap_id'],mode=mode,seconds=result['seconds'],
                             attack_match=result['raw_attack_match'],holds=result.get('generation',{}).get('holds'))),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=Path('test-data/audio-audit'))
    parser.add_argument('--modes',nargs='+',choices=['basic','deep'],default=['basic','deep'])
    parser.add_argument('--audio-shift-ms',type=float,default=22.5)
    args=parser.parse_args()
    failures = []
    for item in sorted(args.folder.glob('*/reference.json')):
        try:
            run(item.parent.resolve(),args.modes,args.audio_shift_ms)
        except Exception as error:
            import traceback
            print(json.dumps(dict(failed=str(item.parent),error=str(error))),flush=True);traceback.print_exc()
            failures.append(str(item.parent))
    if failures:
        raise SystemExit(f'{len(failures)} audio audits failed')
