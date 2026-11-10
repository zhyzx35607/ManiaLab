"""Test arranging from the author's onset sketch, explicitly NOT blind audio.

The Reference is supplied as GD context; the finished chart remains evaluation
only. This isolates expression/layout from transcription errors.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters
from manialab.domain import parse_osu, validate, sections
from manialab.upstream import MapperEngine
from scripts.probe_mapper_adapter import sha, timing_comparison, write_review


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--start-ms',type=int,default=6664)
    cli.add_argument('--seconds',type=int,default=26)
    cli.add_argument('--adapter',type=Path)
    cli.add_argument('--sketch',type=Path,help='Explicit input sketch; its hash is recorded separately from the finished target')
    cli.add_argument('--sketch-kind',choices=('author','detected'),default='author',help='Provenance label for a supplied experimental sketch')
    cli.add_argument('--conditioning',type=Path,help='Identical prefix flags for a baseline/adapter comparison')
    cli.add_argument('--output',type=Path)
    cli.add_argument('--state-constraints',action='store_true',help='Experimental event grammar; no default change')
    cli.add_argument('--existing-raw',type=Path,help='Recover a completed raw output whose scoring failed; never regenerate it')
    args=cli.parse_args();root=args.root
    if args.sketch_kind=='detected' and not args.sketch:raise ValueError('Detected sketch requires an explicit path')
    output=args.output or root/f'expression-{args.start_ms}-{args.seconds}'
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for role in ('reference','finished'):
        if sha(root/(role+'.osu'))!=manifest['charts'][role]['chart_sha256']:
            raise ValueError('Author chart identity changed')
    audio=root/'mapper-6664-26/accompaniment.wav'
    if not audio.exists():raise ValueError('First prepare the fixed non-vocal accompaniment probe')
    human=parse_osu((root/'finished.osu').read_text(encoding='utf-8-sig'))['notes']
    sketch_path=args.sketch or root/'reference.osu'
    sketch=parse_osu(sketch_path.read_text(encoding='utf-8-sig'))['notes']
    ratio=sum(n.hold for n in human)/len(human)
    start,end=args.start_ms,args.start_ms+args.seconds*1000
    engine=MapperEngine(Path('test-data').resolve(),Path.cwd())
    condition_file=args.conditioning or (args.adapter/'conditioning.json' if args.adapter else None)
    conditions=json.loads(condition_file.read_text()) if condition_file else None
    identity=dict(task='arrangement_from_author_sketch_not_blind_audio',audio=sha(audio),
        sketch_codec='map-delimited-sketch-v1',codec_sha256=sha(Path('assets/mapper_sketch_context.py')),
        sketch=sha(sketch_path),start_ms=start,seconds=args.seconds,keys=10,
        ln_ratio=ratio,difficulty=4.56727,seed=42,conditioning=conditions,
        adapter=sha(args.adapter/'adapter_model.safetensors') if args.adapter else None,
        source_commit=engine.status()['source_commit'],model_revision=engine.status()['model_revision'])
    if args.sketch_kind=='detected':
        identity['task']='arrangement_from_detected_sketch_known_timing_same_song_development'
    result_path=output/'result.json'
    if args.state_constraints:
        identity['state_constraints']=dict(codec='mania-event-state-v1',source_sha256=sha(Path('assets/mania_event_constraints.py')))
    if result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if result['identity']!=identity or sha(output/result['raw_file'])!=result['raw_sha256']:
            raise ValueError('Expression probe cache changed')
    else:
        last=[0.]
        def progress(percent,message):
            if time.monotonic()-last[0]>30:print(message,flush=True);last[0]=time.monotonic()
        begin=time.monotonic()
        if args.existing_raw:
            path=args.existing_raw
            if path.resolve().parent!=output.resolve() or path.name.endswith('context.osu'):
                raise ValueError('Recovered raw output must be a generated chart in this run folder')
            log=(output/'inference.log').read_text(encoding='utf-8')
            if 'ManiaLab sketch context active:' not in log or str(path.resolve()) not in log:
                raise ValueError('Run log does not confirm the generated raw chart and active sketch')
            if args.state_constraints and 'ManiaLab research event constraints active: mania-event-state-v1' not in log:
                raise ValueError('Recovered run does not confirm active event constraints')
            engine.last_warnings=[s.strip() for s in log.splitlines() if 'Warning:' in s][:30]
        else:
            path=engine.generate(audio,output,parameters(dict(keys=10,ln_ratio=ratio,seed=42)),progress,lambda:False,
                start_ms=start,seconds=args.seconds,arrangement_reference=sketch_path,
                sketch_codec='map-delimited-sketch-v1',
                state_constraints=args.state_constraints,
                lora_path=args.adapter,difficulty=4.56727,conditioning=conditions)
        content=path.read_text(encoding='utf-8-sig')
        objects=[s for s in sections(content).get('HitObjects',[]) if s.strip() and not s.startswith('//')]
        notes=parse_osu(content)['notes'] if objects else []
        checked=validate(notes,10)
        if not notes:checked.update(valid=False,issues=['Empty generated chart'],issue_count=1)
        result=dict(identity=identity,raw_file=path.relative_to(output).as_posix(),raw_sha256=sha(path),
            notes=[n.data() for n in notes if n.start<end and n.end>=start],
            comparison=timing_comparison(human,notes,start,end),validation=checked,
            warnings=engine.last_warnings,elapsed_seconds=None if args.existing_raw else time.monotonic()-begin,
            generation_outcome='events_generated' if notes else 'empty_generation_failure')
        result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    selected=lambda notes:dict(notes=[n.data() for n in notes if n.start<end and n.end>=start])
    write_review(output,audio,start,end,dict(reference=selected(sketch),finished=selected(human),arrangement=result),
        labels=dict(reference='输入 · 检测声部起音草稿' if args.sketch_kind=='detected' else '输入 · 作者声部采音草稿',
            finished='评价 · 作者成品 10K',arrangement='模型 · 已知草稿后的编排'),
        sources={v:dict(label='Nova '+v,path=root/f'bundled/{v}/preview.wav') for v in ('drums','piano','other')},
        notice=('输入为音频检测后组成的起音草稿；使用已知作者 Timing、全曲难度和 LN 条件。声部分类器在同曲前段学习，尚非独立歌曲端到端测试。成品音符未进入推理。'
            if args.sketch_kind=='detected' else
            '这一实验把作者 Reference 的音符作为输入，专门检查采音已正确时的编排能力。它不是盲采音，也不是端到端效果。成品音符未进入推理。'))
    print(json.dumps(result['comparison']),flush=True)


if __name__=='__main__':main()
