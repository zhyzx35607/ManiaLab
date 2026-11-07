"""Evaluate imported-reference settings separately from default generation.

This deliberately supplies the human chart's timing and numerical style. It
does not copy any notes or lanes and must not be called an automatic result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters,select_notes,assign_lanes
from manialab.domain import note_from,parse_osu,export_osu,validate
from manialab.style import describe_chart
from manialab.timing import chart_clock_notes,align_candidates,grid_report
from manialab.voices import apply_phrase_evidence
from scripts.evaluate_audio import ROOT,LOADED_SOURCE,match_attacks,durations
from scripts.evaluate_ln import compare,long_reference_recall
from scripts.audit_snapshot import assert_snapshot


def evaluate(folder):
    audit=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    assert_snapshot(audit['source_fingerprint']['files'], LOADED_SOURCE['files'], ROOT)
    result=dict(beatmap_id=audit['beatmap_id'],source_fingerprint=LOADED_SOURCE['sha256'],modes={})
    if not audit['reference']['constant_grid']:
        return result|{'skipped':'Changing reference tempo or phase'}
    human=parse_osu((folder/'reference.osu').read_text(encoding='utf-8-sig'))
    style=describe_chart(human)
    prefs=parameters({'keys':human['keys']}|style['preferences'])
    offset,beat=audit['reference']['timing_points'][0]
    tempo=dict(bpm=60000/beat,offset=offset,division=4,stable=True)
    result.update(style=style,settings=prefs,reference_tempo=tempo,audio_shift_ms=audit['audio_shift_ms'])
    for mode in ('basic','deep'):
        analysis=json.loads((folder/mode/'analysis.json').read_text(encoding='utf-8'))
        raw,_=apply_phrase_evidence([note_from(n) for n in analysis['candidates']],
            [note_from(n) for n in analysis.get('phrase_candidates',[])],prefs['phrase_ln'])
        aligned,_=align_candidates(chart_clock_notes(raw,audit['audio_shift_ms']),tempo)
        selected,selection=select_notes(aligned,prefs)
        output=assign_lanes(selected,prefs,beat_ms=beat)
        text=export_osu(output,human['keys'],ROOT/'vendor/mapperatorinator/template.osu',
            bpm=tempo['bpm'],offset=offset,grid_division=4)
        reread=parse_osu(text)['notes']
        result['modes'][mode]=dict(validation=validate(reread,human['keys']),grid=grid_report(reread,tempo),
            holds=durations(reread),selection=selection,
            attacks=match_attacks([n.start for n in reread],[n.start for n in human['notes']],20),
            ln=compare(reread,human['notes']),long_reference=long_reference_recall(reread,human['notes']))
        (folder/mode/'generated-reference-style.osu').write_text(text,encoding='utf-8')
    (folder/'style-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('docs/style-audit.json'))
    args=parser.parse_args();rows=[];errors=[]
    for group in args.folders:
        for path in sorted(group.glob('*/audit.json')):
            try:
                result=evaluate(path.parent)|{'group':group.name}
                rows.append(result)
                print(json.dumps(dict(group=group.name,id=result['beatmap_id'],skipped=result.get('skipped'),
                    matches={k:v['ln'] for k,v in result['modes'].items()})),flush=True)
            except Exception as error:
                errors.append(dict(group=group.name,folder=path.parent.name,error=str(error)))
                print(json.dumps(errors[-1]),flush=True)
    report=dict(purpose='Human reference timing and imported numerical style; not automatic generation or audio ground truth',
        source_fingerprint=LOADED_SOURCE['sha256'],songs=rows,errors=errors)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if errors:raise SystemExit(f'{len(errors)} style-conditioned evaluations failed')
