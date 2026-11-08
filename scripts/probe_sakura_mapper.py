"""Blind Mapperatorinator probes against the user's two Sakura Grass charts.

Only authored TimingPoints and declared chart difficulty/LN conditions enter
generation. Source labels, bookmarks and HitObjects are evaluation-only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters
from manialab.domain import parse_osu, validate
from manialab.upstream import MapperEngine
from scripts.probe_mapper_adapter import sha, timing_comparison, write_review


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--start-ms',type=int,default=6664)
    cli.add_argument('--seconds',type=int,default=26)
    cli.add_argument('--voices',nargs='+',choices=['mix','accompaniment','piano','drums'],default=['mix','accompaniment'])
    args=cli.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    sources={r['id']:root/r['preview'] for r in manifest['audio']}
    for row in manifest['audio']:
        if sha(root/row['path'])!=row['sha256']:raise ValueError('Source audio identity changed')
    reference=parse_osu((root/'reference.osu').read_text(encoding='utf-8-sig'))['notes']
    finished=parse_osu((root/'finished.osu').read_text(encoding='utf-8-sig'))['notes']
    output=root/f'mapper-{args.start_ms}-{args.seconds}';output.mkdir(exist_ok=True)
    if 'accompaniment' in args.voices:
        tracks=[wavfile.read(sources[v]) for v in ('drums','piano','bass','guitar','other')]
        if any(rate!=22050 or data.shape!=tracks[0][1].shape for rate,data in tracks):raise ValueError('Source clocks differ')
        mix=sum(data.astype(np.float32)/32768 for _,data in tracks)
        gain=min(1.,.98/max(.001,float(abs(mix).max())))
        path=output/'accompaniment.wav';wavfile.write(path,22050,np.round(mix*gain*32767).astype(np.int16))
        sources['accompaniment']=path
        (output/'accompaniment.json').write_text(json.dumps(dict(gain=gain,sources=['drums','piano','bass','guitar','other'],vocals=False)),encoding='utf-8')
    engine=MapperEngine(Path('test-data').resolve(),Path.cwd())
    start,end=args.start_ms,args.start_ms+args.seconds*1000
    selected=lambda notes:[n.data() for n in notes if n.start<end and n.end>=start]
    report=dict(reference=dict(notes=selected(reference)),finished=dict(notes=selected(finished)))
    for voice in args.voices:
        folder=output/voice;folder.mkdir(exist_ok=True)
        result_path=folder/'result.json'
        ln_ratio=0. if voice in ('piano','drums') else sum(n.hold for n in finished)/len(finished)
        identity=dict(audio_sha256=sha(sources[voice]),timing_sha256=sha(root/'finished.osu'),keys=10,
            start_ms=start,seconds=args.seconds,seed=42,ln_ratio=ln_ratio,difficulty=4.56727,
            source_commit=engine.status()['source_commit'],model_revision=engine.status()['model_revision'])
        if result_path.exists():
            row=json.loads(result_path.read_text(encoding='utf-8'))
            if row['identity']!=identity or sha(folder/row['raw_file'])!=row['raw_sha256']:raise ValueError('Probe cache differs')
        else:
            last=[0.]
            def progress(percent,message):
                if time.monotonic()-last[0]>30:
                    print(json.dumps(dict(voice=voice,message=message)),flush=True);last[0]=time.monotonic()
            began=time.monotonic()
            path=engine.generate(sources[voice],folder,parameters(dict(keys=10,ln_ratio=ln_ratio,seed=42)),progress,lambda:False,
                start_ms=start,seconds=args.seconds,timing_reference=root/'finished.osu',difficulty=4.56727)
            notes=parse_osu(path.read_text(encoding='utf-8-sig'))['notes']
            target=[n for n in reference if (4<=n.lane<6 if voice=='piano' else n.lane<4 if voice=='drums' else True)]
            row=dict(identity=identity,raw_file=path.relative_to(folder).as_posix(),raw_sha256=sha(path),
                notes=selected(notes),validation=validate(notes,10),warnings=engine.last_warnings,
                elapsed_seconds=time.monotonic()-began,
                reference_comparison=timing_comparison(target,notes,start,end),
                finished_comparison=timing_comparison(finished,notes,start,end))
            result_path.write_text(json.dumps(row,indent=2),encoding='utf-8')
        report[voice]=row
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(voice=voice,reference=row['reference_comparison'],finished=row['finished_comparison'])),flush=True)
    labels=dict(reference='作者 Reference · 乐器采音',finished='作者成品 10K',mix='原模型 · 混音输入',
        accompaniment='原模型 · Nova 伴奏输入',piano='原模型 · Nova 钢琴输入',drums='原模型 · Nova 鼓声输入')
    write_review(output,sources['mix'],start,end,report,labels=labels,
        sources={v:dict(label='Nova '+v,path=sources[v]) for v in ('drums','piano','other','bass','vocals')},
        notice='Reference 的列表示作者确认的声部，成品列表示演奏布局。模型没有收到原谱音符或书签。伴奏输入由作者使用的 Nova 声轨合成并排除人声；这仍不是程序自行分离的端到端验收。')


if __name__=='__main__':main()
