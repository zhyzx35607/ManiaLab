"""Check teacher-forced vs cold-prefix logits on saved real training windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run',type=Path,default=Path('test-data/sakura-benchmark/expression-lora-cold-v2'))
    args=cli.parse_args();root=Path(__file__).resolve().parents[1]
    source=root/'test-data/mapperatorinator/source'
    sys.path[:0]=[str(source/'osuT5'),str(source)]
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from peft import PeftModel
    from omegaconf import OmegaConf
    from transformers.modeling_outputs import BaseModelOutput
    from osuT5.utils.model_utils import load_model
    import osuT5.config
    torch.set_num_threads(2)
    cfg=OmegaConf.to_object(OmegaConf.load(args.run/'resolved-config.yaml'))
    model,tokenizer=load_model(str(root/'test-data/mapperatorinator/model/gamemode=3'),cfg,
        device='cpu',precision='fp32',attn_implementation='sdpa',gamemode=3)
    inventory=json.loads((args.run/'samples.json').read_text())
    identity=inventory['identity']
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    examples=[]
    for chart in inventory['charts']:
        cache=args.run.parent/'encoded'/f'{key}-{chart["id"]}.pt'
        entries=torch.load(cache,weights_only=True)['samples']
        for entry in [r for r in entries if r['window']['kind']=='cold_start'][:2]:
            examples.append((chart['split'],entry))
    from osuT5.event import EventType
    first_time=tokenizer.event_start[EventType.TIME_SHIFT];last_time=tokenizer.event_end[EventType.TIME_SHIFT]
    rows=[]
    for variant in ('baseline','adapter','merged_adapter'):
        if variant=='adapter':model=PeftModel.from_pretrained(model,str(args.run/'adapter'))
        if variant=='merged_adapter':model=model.merge_and_unload()
        model.eval()
        for split,entry in examples:
            batch={k:v for k,v in entry['batch'].items() if k not in ('labels','sample_weights')}
            tokens=batch['decoder_input_ids'][0]
            target_sos=int((tokens==tokenizer.sos_id).nonzero()[0])
            prefix_end=target_sos+2 # SOS then MAP delimiter, before the first event.
            expected=int(entry['batch']['labels'][0,prefix_end-1])
            if not first_time<=expected<last_time:raise ValueError('Cold target does not start with time')
            encoder=BaseModelOutput(last_hidden_state=entry['hidden'])
            with torch.inference_mode():
                full=model(**batch,encoder_outputs=encoder,use_cache=False).logits[0,prefix_end-1].clone()
                prefix=dict(batch,decoder_input_ids=batch['decoder_input_ids'][:,:prefix_end],
                    decoder_attention_mask=batch['decoder_attention_mask'][:,:prefix_end])
                partial=model(**prefix,encoder_outputs=encoder,use_cache=False).logits[0,-1]
            probs=partial.softmax(-1);top=probs.topk(5)
            time_top=probs[first_time:last_time].topk(5)
            row=dict(variant=variant,split=split,window=entry['window'],expected=expected,
                max_logit_difference=float(abs(full-partial).max()),next_token=int(probs.argmax()),
                time_probability=float(probs[first_time:last_time].sum()),
                time_top=[dict(time_ms=int(t)*10,probability=float(p)) for p,t in zip(time_top.values,time_top.indices)],
                top=[dict(token=int(t),probability=float(p)) for p,t in zip(top.values,top.indices)])
            rows.append(row);print(json.dumps(row),flush=True)
    report=dict(identity=identity,rows=rows,causal=all(r['max_logit_difference']<.001 for r in rows),
        notice='Prefix consistency and first-token behavior only; not music-quality acceptance.')
    (args.run/'causal-prefix-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
