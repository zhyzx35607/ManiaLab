"""CPU-capable LoRA experiment using Mapperatorinator's real audio/event path.

Reuses the pinned upstream model, tokenizer, OsuParser and MMRS sequence
builder. Trains decoder attention adapters on mixed audio and authored events.
Teacher-forced scores are diagnostics, never a music-quality acceptance gate.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

from mapper_training_context import context_dataset_class, typed_time_positions, has_musical_target


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=Path('test-data/mapperatorinator/source'))
    cli.add_argument('--checkpoint',type=Path,default=Path('test-data/mapperatorinator/model/gamemode=3'))
    cli.add_argument('--dataset',type=Path,default=Path('test-data/mapper-training/mmrs'))
    cli.add_argument('--output',type=Path,default=Path('test-data/mapper-training/lora'))
    cli.add_argument('--steps',type=int,default=128)
    cli.add_argument('--threads',type=int,default=2)
    cli.add_argument('--windows-per-chart',type=int,default=2)
    cli.add_argument('--prepare-only',action='store_true')
    opts=cli.parse_args()
    if not 1<=opts.steps<=100000 or not 1<=opts.windows_per_chart<=100 or opts.threads<1:raise ValueError('Invalid training size or thread count')
    opts.source=opts.source.resolve();opts.checkpoint=opts.checkpoint.resolve();opts.dataset=opts.dataset.resolve();opts.output=opts.output.resolve()
    if (opts.output/'report.json').exists():
        raise ValueError('A run already exists in this output directory; choose a new --output to preserve evidence. Encoded caches can be reused.')
    opts.output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(__file__,opts.output/'runner-snapshot.py')
    shutil.copyfile(Path(__file__).with_name('mapper_training_context.py'),opts.output/'context-builder-snapshot.py')
    os.environ.update(HF_HUB_OFFLINE='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(opts.source/'osuT5'),str(opts.source)]
    import numpy as np
    import torch
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from peft import LoraConfig,get_peft_model
    from transformers.modeling_outputs import BaseModelOutput
    from osuT5.event import EventType,ContextType
    from osuT5.dataset.osu_parser import OsuParser
    from osuT5.utils.model_utils import load_model,save_lora_checkpoint_metadata
    import osuT5.config
    torch.set_num_threads(opts.threads);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    MmrsDataset=context_dataset_class()
    manifest=json.loads((opts.dataset/'manifest.json').read_text(encoding='utf-8'))
    with initialize_config_dir(str(opts.source/'configs/train'),version_base='1.1'):
        cfg=compose(config_name='lora_v32')
    overrides=dict(device='cpu',compile=False,mixed_precision='no',precision='fp32',attn_implementation='sdpa',
        pretrained_path=str(opts.checkpoint),pretrained_gamemode=3,
        lora=dict(r=8,lora_alpha=16,lora_dropout=0.,bias='none',init_lora_weights=True,
            target_modules=r'transformer\.model\.decoder\.layers\.\d+\.(self_attn|encoder_attn)\.(Wq|Wkv|Wqkv|Wo)'),
        lora_metadata=dict(ckpt_subfolders=['gamemode=3']),
        data=dict(dataset_type='mmrs',gamemodes=[3],ranked_statuses=[1,2,4],min_year=None,max_year=None,
            train_dataset_path=str(opts.dataset/'train'),test_dataset_path=str(opts.dataset/'validation'),
            train_dataset_start=0,train_dataset_end=manifest['counts']['train']['charts'],
            test_dataset_start=0,test_dataset_end=manifest['counts']['validation']['charts'],
            src_seq_len=2048,tgt_seq_len=2560,per_track=True,cycle_length=1,add_empty_sequences=False,
            add_descriptors=False,add_year_token=False,add_mapper_token=False,add_song_position_token=False,
            add_pre_tokens=False,frame_offset_augment_prob=0.,dt_augment_prob=0.,flip_horizontal_prob=0.,flip_vertical_prob=0.,
            timing_random_offset=0,timing_random_offset_2=0,snapping_random_prob=0.,lookback_prob=0.,lookback=0.,
            label_smoothing=0.,rhythm_weight=1.,context_types=[{'in':[ContextType.TIMING],'out':[ContextType.MAP]}],context_weights=[1]),
        logging=dict(mode='disabled',grad_l2=False,weights_l2=False),dataloader=dict(num_workers=0,pin_memory=False,balancer_buffer_size=0))
    cfg=OmegaConf.merge(cfg,overrides);args=OmegaConf.to_object(cfg)
    OmegaConf.save(cfg,opts.output/'resolved-config.yaml')
    model,tokenizer=load_model(str(opts.checkpoint),args,device='cpu',precision='fp32',attn_implementation='sdpa',eval_mode=True,gamemode=3)
    model.config.use_cache=False;model.transformer.config.use_cache=False
    # Freeze the audio encoder; cached representations are valid only because
    # the adapter regex exclusively targets decoder self/cross attention.
    for parameter in model.parameters():parameter.requires_grad_(False)
    parser=OsuParser(args,tokenizer);encoder=model.get_encoder().eval()
    source_hash=hashlib.sha256(''.join(digest(p) for p in sorted((opts.source/'osuT5').rglob('*.py'))).encode()).hexdigest()
    identity=dict(manifest=digest(opts.dataset/'manifest.json'),weights=digest(opts.checkpoint/'model.safetensors'),
        tokenizer=digest(opts.checkpoint/'tokenizer.json'),upstream_source_sha256=source_hash,
        context_builder_sha256=digest(Path(__file__).with_name('mapper_training_context.py')),
        config=hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest(),windows=opts.windows_per_chart)
    cache=opts.output.parent/'encoded';cache.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    samples=[];inventory=[]
    for index,row in enumerate(manifest['records']):
        folder=opts.dataset/row['split']/'data'/str(row['beatmap_id'])
        if digest(folder/'reference.osu')!=row['prepared_chart_sha256'] or digest(folder/'audio.wav')!=row['decoded_audio_sha256']:
            raise ValueError('Prepared training pair changed')
        path=cache/(fingerprint[:16]+'-'+str(row['beatmap_id'])+'.pt')
        if path.exists():
            cached=torch.load(path,weights_only=True);selected=cached['samples'];audit=cached['audit']
        else:
            dataset=MmrsDataset(args.data,parser,tokenizer,subset_ids=[row['beatmap_id']],test=row['split']=='validation')
            # Use deterministic validation preprocessing for both pools, then
            # reservoir-sample windows over the WHOLE song, not only its intro.
            dataset.test=True
            dataset.path=opts.dataset/row['split']
            rng=random.Random(row['beatmap_id']);selected=[];eligible=0
            for ordinal,item in enumerate(dataset):
                labels=item['labels']
                musical_ids={tokenizer.event_start[k] for k in
                    (EventType.CIRCLE,EventType.HOLD_NOTE,EventType.HOLD_NOTE_END,EventType.HOLD_NOTE_SUSTAIN)}
                if not has_musical_target(labels.tolist(),musical_ids):continue
                entry=dict(ordinal=ordinal,window=item['context_window'],
                           batch={k:v.unsqueeze(0) for k,v in item.items() if torch.is_tensor(v)})
                eligible+=1
                if len(selected)<opts.windows_per_chart:selected.append(entry)
                else:
                    slot=rng.randrange(eligible)
                    if slot<len(selected):selected[slot]=entry
            if not selected:raise ValueError(f"No active windows: {row['beatmap_id']}")
            for entry in selected:
                batch=entry['batch']
                with torch.no_grad():hidden=encoder(frames=batch.pop('frames'),return_dict=True).last_hidden_state.detach().clone()
                entry['hidden']=hidden
                batch.pop('beatmap_idx',None)
                # SDPA handles padding; removing only right padding reduces
                # CPU work without truncating any target or timing context.
                end=int(torch.nonzero(batch['decoder_attention_mask'][0],as_tuple=False)[-1])+1
                for key in ('decoder_input_ids','decoder_attention_mask','labels'):batch[key]=batch[key][:,:end]
            audit=dataset.audit
            torch.save(dict(samples=selected,audit=audit),path)
        for entry in selected:samples.append(dict(entry,row=row))
        inventory.append(dict(id=row['beatmap_id'],keys=row['keys'],band=row['band'],split=row['split'],
            audit=audit,windows=[dict(ordinal=e['ordinal'],window=e['window'],tokens=int((e['batch']['labels']!=-100).sum())) for e in selected]))
        print(json.dumps(dict(stage='prepare',chart=index+1,total=len(manifest['records']),id=row['beatmap_id'],windows=len(selected))),flush=True)
    (opts.output/'samples.json').write_text(json.dumps(dict(identity=identity,charts=inventory),indent=2),encoding='utf-8')
    if opts.prepare_only:return
    model=get_peft_model(model,LoraConfig(**args.lora))
    trained=[p for p in model.parameters() if p.requires_grad]
    if not trained:raise ValueError('No adapter parameters')
    if any('encoder.layers' in n for n,p in model.named_parameters() if p.requires_grad):raise ValueError('Encoder cache is invalid for trainable encoder')
    train=[s for s in samples if s['row']['split']=='train'];val=[s for s in samples if s['row']['split']=='validation']
    types={k:[tokenizer.event_start[t],tokenizer.event_end[t]] for k,t in
        [('time',EventType.TIME_SHIFT),('column',EventType.MANIA_COLUMN),('ln_head_type',EventType.HOLD_NOTE),
         ('ln_tail_type',EventType.HOLD_NOTE_END),('tap_type',EventType.CIRCLE)]}
    starts={t.value:n for t,n in tokenizer.event_start.items()};ends={t.value:n for t,n in tokenizer.event_end.items()}
    def forward(sample):return model(**sample['batch'],encoder_outputs=BaseModelOutput(last_hidden_state=sample['hidden']),use_cache=False)
    def evaluate():
        model.eval();results=[]
        with torch.no_grad():
            for sample in val:
                out=forward(sample);y=sample['batch']['labels'];pred=out.logits.argmax(-1)
                loss=torch.nn.functional.cross_entropy(out.logits.transpose(1,2),y,reduction='none',ignore_index=-100)
                stats={}
                masks={name:(y>=lo)&(y<hi) for name,(lo,hi) in {'all':[0,tokenizer.vocab_size_out],**types}.items()}
                for name,indices in typed_time_positions(y[0].tolist(),starts,ends).items():
                    mask=torch.zeros_like(y,dtype=torch.bool);mask[0,indices]=True;masks[name]=mask
                for name,mask in masks.items():
                    count=int(mask.sum())
                    stats[name]=dict(tokens=count,nll=float(loss[mask].mean()) if count else None,
                        accuracy=float((pred[mask]==y[mask]).float().mean()) if count else None)
                results.append(dict(id=sample['row']['beatmap_id'],keys=sample['row']['keys'],ordinal=sample['ordinal'],stats=stats))
        macro={}
        for name in ['all',*types,'ln_head_time','ln_tail_time','tap_time']:
            songs=defaultdict(list)
            for s,r in zip(val,results):
                if r['stats'][name]['tokens']:songs[s['row']['group']].append(r['stats'][name])
            macro[name]=dict(groups=len(songs),nll=float(np.mean([np.mean([v['nll'] for v in values]) for values in songs.values()])) if songs else None,
                accuracy=float(np.mean([np.mean([v['accuracy'] for v in values]) for values in songs.values()])) if songs else None)
        return dict(macro=macro,windows=results)
    report=dict(identity=identity,trainable_parameters=sum(p.numel() for p in trained),steps_requested=opts.steps,
        runner_sha256=digest(opts.output/'runner-snapshot.py'),
        enabled_in_application=False,measure='teacher-forced authored event prediction; not free-generation music quality',
        baseline=evaluate(),training=[])
    report_path=opts.output/'report.json'
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    buckets=defaultdict(list)
    for sample in train:buckets[(sample['row']['keys'],sample['row']['band'])].append(sample)
    rng=random.Random(42);schedule=[10,6,10,7,10,8,10,9];optimizer=torch.optim.AdamW(trained,lr=5e-5,weight_decay=.01)
    started=time.monotonic();seen=set();queues={}
    for step in range(opts.steps):
        key=schedule[step%len(schedule)];bands=sorted(b for k,b in buckets if k==key)
        band=bands[(step//len(schedule))%len(bands)];bucket=(key,band)
        if not queues.get(bucket):
            queues[bucket]=list(buckets[bucket]);rng.shuffle(queues[bucket])
        sample=queues[bucket].pop()
        model.train();optimizer.zero_grad(set_to_none=True)
        out=forward(sample);loss=out.loss
        if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(trained,1.,error_if_nonfinite=True);optimizer.step()
        seen.add(sample['row']['beatmap_id'])
        row=dict(step=step+1,id=sample['row']['beatmap_id'],keys=key,band=band,loss=float(loss.detach()),grad_norm=float(norm),
                 elapsed_seconds=round(time.monotonic()-started,2))
        report['training'].append(row);print(json.dumps(row),flush=True)
        if (step+1)%16==0:report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['after']=evaluate();report['charts_seen_in_optimizer']=sorted(seen)
    adapter=opts.output/'adapter';model.save_pretrained(adapter)
    save_lora_checkpoint_metadata(adapter,args)
    (adapter/'conditioning.json').write_text(json.dumps({key:getattr(args.data,key) for key in
        ('add_mapper_token','add_year_token','add_descriptors','add_song_position_token')},indent=2),encoding='utf-8')
    report['adapter_sha256']=digest(adapter/'adapter_model.safetensors')
    report['training_seconds']=round(time.monotonic()-started,2)
    report['completed_steps']=len(report['training'])
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(stage='complete',baseline=report['baseline']['macro'],after=report['after']['macro'])),flush=True)


if __name__=='__main__':main()
