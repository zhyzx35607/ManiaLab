"""Inspect automatic timing acceptance against reference mapping conventions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def run(folders):
    rows=[]
    for folder in folders:
        for path in sorted(folder.glob('*/audit.json')):
            audit=json.loads(path.read_text(encoding='utf-8'))
            for mode in ('basic','deep'):
                analysis=json.loads((path.parent/mode/'analysis.json').read_text(encoding='utf-8'))
                tempo=analysis['tempo'];constant=audit['reference']['constant_grid']
                row=dict(group=folder.name,beatmap_id=audit['beatmap_id'],mode=mode,accepted=tempo['stable'],
                    reference_constant=constant,bpm=tempo['bpm'],suspect_sections=sum(s['suspect'] for s in tempo['sections']))
                if not constant:
                    row['classification']='variable_reference_requires_review'
                else:
                    offset,beat=audit['reference']['timing_points'][0]
                    ratio=tempo['bpm']/(60000/beat)
                    unit=min((.25,.5,1.,2.,4.),key=lambda v:abs(math.log2(ratio/v)))
                    drift=analysis['duration']*1000*abs(1-ratio/unit)
                    mapped=math.floor(tempo['offset']-audit['audio_shift_ms']+.5)
                    phase=(mapped-offset+beat/8)%(beat/4)-beat/8
                    compatible=abs(phase)<=20 and drift<=20
                    row.update(mapping_phase_error_ms=round(phase,3),whole_song_rate_drift_ms=round(drift,3),
                        beat_unit_ratio=unit,reference_compatible=compatible,
                        classification=('accepted_' if tempo['stable'] else 'blocked_')+('compatible' if compatible else 'incompatible'))
                rows.append(row)
    summary={mode:dict(Counter(row['classification'] for row in rows if row['mode']==mode)) for mode in ('basic','deep')}
    return dict(notice='Reference-grid compatibility diagnostic, not physical audio ground truth or proof that all accepted notes align.',
        criterion='Constant reference grid only. Integer mapped offset phase within 20ms on the reference quarter-beat grid; rate drift <=20ms over the full audio, allowing 1/4, 1/2, 1, 2 or 4 beat units. Variable references are reported separately.',
        policy='No acceptance threshold changed in response to this report.',summary=summary,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('docs/timing-acceptance.json'))
    args=parser.parse_args();report=run(args.folders)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2))
