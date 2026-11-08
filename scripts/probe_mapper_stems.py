"""Run the real mania checkpoint on a mix and aligned separated stems.

This is an ablation, not instrument transcription truth. Raw event timing is
retained; no quantization, new LN rules or lane reassignment are applied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manialab.arrange import parameters
from manialab.domain import parse_osu, validate
from manialab.upstream import MapperEngine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--stems',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--engine-root',type=Path,default=Path('test-data'))
    p.add_argument('--voices',nargs='*',choices=['vocals','piano','drums','bass','guitar','other'],default=['vocals','piano'])
    p.add_argument('--adapter',type=Path)
    p.add_argument('--conditioning',type=Path)
    p.add_argument('--start-ms',type=int,default=24000)
    p.add_argument('--seconds',type=int,default=12)
    p.add_argument('--bpm',type=float,required=True)
    p.add_argument('--offset',type=float,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    conditioning_path=args.conditioning or (args.adapter/'conditioning.json' if args.adapter else None)
    if args.conditioning and not args.conditioning.is_file():raise ValueError('Explicit conditioning file does not exist')
    conditioning=json.loads(conditioning_path.read_text(encoding='utf-8')) if conditioning_path and conditioning_path.exists() else None
    separation=json.loads((args.stems/'separation.json').read_text(encoding='utf-8'))
    if separation.get('audio_sha256')!=hashlib.sha256(args.audio.read_bytes()).hexdigest():
        raise ValueError('Separated stems do not match the input audio bytes')
    engine=MapperEngine(args.engine_root.resolve(),root)
    args.output.mkdir(parents=True,exist_ok=True)
    report=[]
    for voice in ['mix',*args.voices]:
        source=args.audio if voice=='mix' else args.stems/(voice+'.wav')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        folder=args.output/voice;manifest=folder/'result.json'
        context=dict(probe_revision=2,audio_sha256=digest,start_ms=args.start_ms,seconds=args.seconds,
            bpm=args.bpm,offset=args.offset,keys=10,seed=42,ln_ratio=.55,
            source_commit=engine.status()['source_commit'],model_revision=engine.status()['model_revision'])
        if args.adapter is not None:
            context['adapter_sha256']=hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
            context['adapter_config_sha256']=hashlib.sha256((args.adapter/'adapter_config.json').read_bytes()).hexdigest()
        if conditioning is not None:
            context['conditioning']=conditioning
        if manifest.exists():
            saved=json.loads(manifest.read_text(encoding='utf-8'))
            if saved['context']!=context:raise ValueError('Probe cache differs; use a new output directory')
            if hashlib.sha256((folder/saved['raw_file']).read_bytes()).hexdigest()!=saved['raw_sha256']:
                raise ValueError('Raw model output changed after evaluation')
            report.append(saved);continue
        last=[0.]
        def progress(percent,message):
            if time.monotonic()-last[0]>30:
                print(json.dumps(dict(voice=voice,progress=percent,message=message),ensure_ascii=True),flush=True)
                last[0]=time.monotonic()
        started=time.monotonic()
        result=engine.generate(source,folder,parameters(),progress,lambda:False,seconds=args.seconds,
            tempo=dict(bpm=args.bpm,offset=args.offset),start_ms=args.start_ms,lora_path=args.adapter,conditioning=conditioning)
        notes=parse_osu(result.read_text(encoding='utf-8-sig'))['notes']
        local=[n for n in notes if n.start<args.start_ms+args.seconds*1000 and n.end>=args.start_ms]
        saved=dict(voice=voice,context=context,elapsed_seconds=round(time.monotonic()-started,2),
            raw_file=result.relative_to(folder).as_posix(),raw_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
            validation=validate(local,10),warnings=engine.last_warnings,
            notes=[n.data() for n in local],
            notice='Stem-conditioned chart predictions; not exhaustive instrument note labels. Not enabled in the app.')
        manifest.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
        report.append(saved)
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(voice=voice,seconds=saved['elapsed_seconds'],validation=saved['validation'])),flush=True)


if __name__=='__main__':main()
