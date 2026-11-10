"""Compare real cached training prefixes with the wrapper's inference context.

This does not run a model or provide finished events to inference. It checks
the declared oracle difficulty/LN conditions and the supplied input sketch.
Token-level differences remain in the private experiment directory.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',type=Path,default=Path('test-data/sakura-benchmark/expression-lora-prefix-v1'))
    cli.add_argument('--context',type=Path,default=Path('test-data/sakura-benchmark/expression-cold-215853-12/adapter/arrangement-context.osu'))
    args=cli.parse_args();root=Path(__file__).resolve().parents[1]
    source=root/'test-data/mapperatorinator/source'
    sys.path[:0]=[str(root),str(root/'assets'),str(source/'osuT5'),str(source)]
    import torch
    from omegaconf import OmegaConf
    from scipy.io import wavfile
    from slider import Beatmap
    import osuT5.config
    import osuT5.tokenizer
    sys.modules['osuT5.osuT5']=sys.modules['osuT5']
    sys.modules['osuT5.osuT5.config']=osuT5.config
    sys.modules['osuT5.osuT5.tokenizer']=osuT5.tokenizer
    from osuT5.event import ContextType
    from osuT5.inference.processor import Processor,generation_config_from_beatmap
    from config import InferenceConfig
    from mapper_sketch_context import tokenizer_with_sketch
    cfg=OmegaConf.load(args.run/'resolved-config.yaml')
    cfg.data.context_types=[{'in':[ContextType.TIMING,ContextType.GD],'out':[ContextType.MAP]}]
    tokenizer=osuT5.tokenizer.Tokenizer()
    tokenizer.load_state_dict(tokenizer_with_sketch(json.loads(
        (root/'test-data/mapperatorinator/model/gamemode=3/tokenizer.json').read_text())))
    inventory=json.loads((args.run/'samples.json').read_text())
    identity=inventory['identity']
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    dataset=Path(cfg.data.train_dataset_path).parent
    manifest=json.loads((dataset/'manifest.json').read_text())
    inference=InferenceConfig();inference.train=cfg
    processor=Processor(inference,None,tokenizer)
    rows=[];configs=[]
    for chart in inventory['charts']:
        record=next(r for r in manifest['records'] if r['beatmap_id']==chart['id'])
        folder=dataset/chart['split']/'data'/str(chart['id'])
        rate,samples=wavfile.read(folder/'audio.wav',mmap=True)
        song_length=len(samples)/rate*1000
        entries=torch.load(args.run.parent/'encoded'/f'{key}-{chart["id"]}.pt',weights_only=True)['samples']
        # Test all retained cold windows; no target history enters inference.
        for context_path in (folder/'sketch.osu',args.context):
            inputs=processor.get_in_context(in_context=[ContextType.TIMING,ContextType.GD],
                beatmap_path=str(context_path),song_length=song_length)
            conditions=generation_config_from_beatmap(Beatmap.from_path(context_path),context_path,tokenizer)
            conditions.difficulty=record['difficulty'];conditions.hold_note_ratio=record['ln_ratio']
            outputs=processor.get_out_context(out_context=[ContextType.MAP],generation_config=conditions,
                given_context=[],beatmap_path=None,song_length=song_length,verbose=False)
            configs.append(dict(split=chart['split'],context=str(context_path),target_conditions=asdict(conditions)))
            required=['last_sv'] if cfg.data.add_sv_special_token and cfg.data.add_mania_sv else []
            for entry in entries:
                if entry['window']['kind']!='cold_start':continue
                start=entry['window']['start_ms']
                prompt=processor.get_prompt(processor.prepare_context_sequences(inputs,start,False,required),
                    processor.prepare_context_sequences(outputs,start,True,required))[0].tolist()
                tokens=entry['batch']['decoder_input_ids'][0].tolist()
                prefix_end=tokens.index(tokenizer.sos_id)+2
                expected=tokens[:prefix_end]
                differences=[dict(index=i,training=a,inference=b) for i,(a,b) in enumerate(zip(expected,prompt)) if a!=b]
                rows.append(dict(split=chart['split'],context=str(context_path),window=entry['window'],
                    training_length=len(expected),inference_length=len(prompt),differences=differences,
                    equal=expected==prompt))
    report=dict(identity=identity,rows=rows,conditions=configs,all_equal=all(r['equal'] for r in rows),
        notice='Real input-prefix comparison only; no model or musical quality acceptance.')
    (args.run/'real-prefix-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(all_equal=report['all_equal'],comparisons=len(rows),
        mismatches=[{k:r[k] for k in ('split','context','window','training_length','inference_length','differences')}
                    for r in rows if not r['equal']]),indent=2),flush=True)


if __name__=='__main__':main()
