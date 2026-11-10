"""Free-generate author-labelled sources with identical baseline/adapter inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, note_from
from scripts.prepare_sakura_training import SOURCE_LANES
from scripts.probe_mapper_adapter import event_similarity, sha
from scripts.research_source_attack_classes import drum_event_counts, CLASSES

CONDITIONS = dict(add_mapper_token=False, add_year_token=False,
                  add_descriptors=False, add_song_position_token=False)


def source_scores(reference, predicted, voice, start, end):
    expected=[n for n in reference if start<=n.start<end]
    actual=[n for n in predicted if start<=n.start<end]
    result=dict(semantic_lane_fraction=sum(n.lane in SOURCE_LANES[voice] for n in actual)/max(1,len(actual)),
        onset_multiplicity=event_similarity([(n.start,) for n in expected],[(n.start,) for n in actual],[40]),
        note_column_match=event_similarity([(n.start,n.lane) for n in expected],[(n.start,n.lane) for n in actual],[40,.1]))
    # The second comparison dimension is a column, not milliseconds.
    result['note_column_match']['mean_absolute_error_column']=result['note_column_match']['mean_absolute_error_ms'].pop()
    if voice=='drums':
        human=drum_event_counts(expected);generated=drum_event_counts(actual)
        result['predominant_drum_classes']={kind:event_similarity(
            [(t,) for t,c in human if c==kind],[(t,) for t,c in generated if c==kind],[40]) for kind in CLASSES}
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset',type=Path,default=Path('test-data/sakura-benchmark/source-mmrs'))
    cli.add_argument('--adapter',type=Path,default=Path('test-data/sakura-benchmark/source-lora/adapter'))
    cli.add_argument('--output',type=Path,default=Path('test-data/sakura-benchmark/source-free-215853-12'))
    cli.add_argument('--start-ms',type=int,default=215853)
    cli.add_argument('--seconds',type=int,default=12)
    cli.add_argument('--voices',nargs='+',choices=list(SOURCE_LANES),default=list(SOURCE_LANES))
    cli.add_argument('--variants',nargs='+',choices=['baseline','adapter'],default=['baseline','adapter'])
    args=cli.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    condition_file=args.output/'conditioning.json'
    if condition_file.exists() and json.loads(condition_file.read_text())!=CONDITIONS:
        raise ValueError('Source probe conditioning changed')
    condition_file.write_text(json.dumps(CONDITIONS,indent=2),encoding='utf-8')
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    if args.adapter.exists() and json.loads((args.adapter/'conditioning.json').read_text())!=CONDITIONS:
        raise ValueError('Adapter and baseline conditioning differ')
    results={}
    for voice in args.voices:
        row=next(r for r in manifest['records'] if r['split']=='validation' and r['source_role']==voice)
        a,b=args.start_ms,args.start_ms+args.seconds*1000
        if not any(lo<=a<b<=hi for lo,hi in row['allowed_audio_ranges_ms']):
            raise ValueError('Probe interval is outside held-back source sections')
        output=args.output/voice
        command=[sys.executable,str(Path(__file__).with_name('probe_mapper_adapter.py')),
            '--dataset',str(args.dataset),'--adapter',str(args.adapter),'--output',str(output),
            '--beatmap-id',str(row['beatmap_id']),'--start-ms',str(a),'--seconds',str(args.seconds),
            '--conditioning',str(condition_file),'--variants',*args.variants]
        subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        path=args.dataset/'validation/data'/str(row['beatmap_id'])/'reference.osu'
        human=parse_osu(path.read_text(encoding='utf-8'))['notes']
        results[voice]={}
        for variant in args.variants:
            result_path=output/variant/'result.json'
            result=json.loads(result_path.read_text(encoding='utf-8'))
            raw=output/variant/result['raw_file']
            if sha(raw)!=result['raw_sha256']:raise ValueError('Raw prediction changed')
            notes=parse_osu(raw.read_text(encoding='utf-8-sig'))['notes']
            results[voice][variant]=dict(timing=result['comparison'],source=source_scores(human,notes,voice,a,b),
                validation=result['validation'],result_sha256=sha(result_path))
        report=dict(validation_scope=manifest['validation_scope'],results=results,
            notice='Semantic columns are a source-transcription task, not performance layout. No source labels enter audio inference.')
        (args.output/'source-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(voice=voice,results=results[voice])),flush=True)


if __name__=='__main__':main()
