"""Compose an onset-only source sketch from frozen acoustic development models.

Author columns label the trained source tasks; no authored HitObjects are read.
Timing is explicitly supplied from the known source-reference file for this
diagnostic. Unreliable instrument offsets are not converted to LN lengths.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import Note, sections, export_osu, validate
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output',type=Path,default=Path('test-data/sakura-benchmark/detected-onset-sketch-v1'))
    args=cli.parse_args();root=args.root
    if args.output.exists():raise ValueError('Preserve completed sketches')
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    reference=root/'reference.osu';parts=sections(reference.read_text(encoding='utf-8-sig'))
    # Author events and bookmarks are neither inputs to the composer nor used
    # to choose/correct any generated onset. Only red timing lines survive.
    reds=[line for line in parts['TimingPoints'] if len(line.split(','))>=8 and line.split(',')[6].strip()=='1']
    timing=[]
    for line in reds:
        fields=line.split(',');fields[7]='0';timing.append(','.join(fields))
    points=sorted((float(line.split(',')[0]),float(line.split(',')[1])) for line in timing)
    def snap(time):
        offset,beat=max((p for p in points if p[0]<=time),default=points[0])
        return max(0,round(offset+round((time-offset)/(beat/4))*(beat/4)))
    notes=[];identities={};events=[]
    for role in ('piano','other'):
        folder=root/(role+'-yourmt3-layers')
        identities[role]=dict(events_sha256=sha(folder/'events.json'),
            model_sha256=sha(folder/'development-model.npz'),report_sha256=sha(folder/'report.json'))
        capacity=2 if role=='piano' else 4;first=4 if role=='piano' else 6
        for index,event in enumerate(read(folder/'events.json')):
            count=event['layers']
            if type(count)!=int or not 0<=count<=capacity:raise ValueError('Invalid learned layer count')
            time=snap(event['start_ms'])
            events.append(dict(role=role,candidate=index,raw_start_ms=event['start_ms'],grid_start_ms=time,layers=count))
            notes.extend(Note(time,time,first+layer,voice=role,id=f'{role}:{index}:{layer}') for layer in range(count))
    folder=root/'drum-classifier'
    identities['drums']=dict(events_sha256=sha(folder/'events.json'),
        model_sha256=sha(folder/'development-model.npz'),report_sha256=sha(folder/'report.json'))
    snare=0
    for index,event in enumerate(read(folder/'events.json')):
        kind=event['kind'];time=snap(event['start'])
        if kind not in ('kick','snare','hi-hat'):raise ValueError('Unknown drum source class')
        lane=0 if kind=='kick' else 3 if kind=='hi-hat' else 1+snare%2
        if kind=='snare':snare+=1
        notes.append(Note(time,time,lane,voice='drums',id=f'drums:{index}'))
        events.append(dict(role=kind,candidate=index,raw_start_ms=event['start'],grid_start_ms=time,layers=1))
    # Snapping can collide two detections in the same source lane. Keep one
    # and retain all IDs in the audit; never fabricate a spare-lane layer.
    unique={};duplicates=[]
    for note in notes:
        key=(note.start,note.lane)
        if key in unique:duplicates.append(dict(kept=unique[key].id,duplicate=note.id))
        else:unique[key]=note
    notes=sorted(unique.values(),key=lambda n:(n.start,n.lane))
    output=export_osu(notes,10,Path('vendor/mapperatorinator/template.osu'),
        title='Sakura Grass source-onset diagnostic',version='Detected onset sources',audio='accompaniment.wav',
        bpm=60000/points[0][1],offset=points[0][0])
    output_parts=sections(output);output_parts['TimingPoints']=timing;output_parts['Editor']=[]
    output='\n'.join(['\n'.join(output_parts.get('',['osu file format v14']))]+
        [f'[{name}]\n'+'\n'.join(lines) for name,lines in output_parts.items() if name])+'\n'
    args.output.mkdir(parents=True)
    (args.output/'sketch.osu').write_text(output,encoding='utf-8')
    (args.output/'decisions.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    report=dict(identity=identities,sketch_sha256=sha(args.output/'sketch.osu'),runner_sha256=sha(__file__),
        supplied_timing=timing,validation=validate(notes,10),source_counts=dict(Counter(n.voice for n in notes)),
        grid_collisions=duplicates,enabled_in_application=False,
        protocol=['Same-song development source models; fit 0-162340ms, calibrate 162340-188286ms.',
            'Known author red lines and four-way subdivision explicitly supplied; not automatic Timing acceptance.',
            'Every generated object is a tap; uncertain acoustic tails are not used as LN evidence.',
            'Piano 5/6 and other 7-10 source layers are learned counts. No final performance layout.',
            'Kick lane1, single snare alternating lanes2/3, hi-hat/other drum classifier lane4; no inferred snare doubles.',
            'No original Reference HitObjects or Bookmarks enter composition or correction. Finished chart is not read.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
