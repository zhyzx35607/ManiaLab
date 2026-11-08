"""Integration audit: real tokenizer/parser consume sketches as unlabelled input."""
from datetime import datetime
import json
from pathlib import Path
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'test-data/mapperatorinator/source'
    sys.path[:0]=[str(root),str(root/'assets'),str(source/'osuT5'),str(source)]
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf
    from scipy.io import wavfile
    from osuT5.tokenizer import Tokenizer
    from osuT5.dataset.osu_parser import OsuParser
    from osuT5.event import ContextType,EventType
    # Root config.py spells the training package as osuT5.osuT5.config.
    # Keep one TrainConfig class in this isolated cross-path audit process.
    import osuT5.config
    sys.modules['osuT5.osuT5']=sys.modules['osuT5']
    sys.modules['osuT5.osuT5.config']=osuT5.config
    sys.modules['osuT5.osuT5.tokenizer']=sys.modules['osuT5.tokenizer']
    from osuT5.inference.processor import Processor,generation_config_from_beatmap
    from config import InferenceConfig
    from slider import Beatmap
    from manialab.domain import Note,export_osu
    from mapper_sketch_context import tokenizer_with_sketch
    from scripts.mapper_training_context import context_dataset_class
    cfg=OmegaConf.load(root/'test-data/sakura-benchmark/expression-lora-v2/resolved-config.yaml')
    folder=root/'test-data/mapper-sketch-fixture';track=folder/'data/1';track.mkdir(parents=True,exist_ok=True)
    template=root/'vendor/mapperatorinator/template.osu'
    target=[Note(1000,19000,0),Note(9000,9000,1),Note(17000,17000,2)]
    (track/'reference.osu').write_text(export_osu(target,10,template,audio='audio.wav'),encoding='utf-8')
    wavfile.write(track/'audio.wav',16000,np.zeros(24*16000,dtype=np.int16))
    metadata=[dict(BeatmapSetId=1,Id=bid,BeatmapSetFolder='1',BeatmapFile=name,AudioFile='audio.wav',
        ModeInt=3,Ranked=0,DifficultyRating=1.,StarRating=[1.]*7,SubmittedDate=datetime(2020,1,1),
        RankedDate=datetime(2020,1,1),UserId=-1) for bid,name in [(10,'sketch.osu'),(20,'reference.osu')]]
    pd.DataFrame(metadata).to_parquet(folder/'metadata.parquet',index=False)
    cfg.data.test_dataset_path=str(folder);cfg.data.test_dataset_start=0;cfg.data.test_dataset_end=1
    cfg.data.context_types=[{'in':[ContextType.TIMING,ContextType.GD],'out':[ContextType.MAP]}]
    tokenizer=Tokenizer();tokenizer.load_state_dict(tokenizer_with_sketch(json.loads(
        (root/'test-data/mapperatorinator/model/gamemode=3/tokenizer.json').read_text())))
    column=tokenizer.event_start[EventType.MANIA_COLUMN]
    results=[];cold_results=[]
    for lane in (8,9):
        # Same final event time makes the parser's inferred Timing extent equal.
        sketch=[Note(500,500,lane),Note(8500,8500,lane),Note(17000,17000,lane)]
        (track/'sketch.osu').write_text(export_osu(sketch,10,template,audio='audio.wav'),encoding='utf-8')
        dataset=context_dataset_class()(cfg.data,OsuParser(cfg,tokenizer),tokenizer,subset_ids=[1],test=True,target_ids=[20])
        rows=list(dataset)
        assert len(rows)==3,'Input-only sketch was trained as a reverse target'
        for row in rows:
            tokens=row['decoder_input_ids'].tolist();labels=row['labels'].tolist()
            indices=[i for i,t in enumerate(tokens) if t==column+lane]
            assert indices,'Sketch source columns are missing from decoder input'
            assert all(labels[max(0,i-1)]==-100 for i in indices),'Sketch tokens became targets'
            assert labels.count(column+lane)==0
        results.append(rows)
        standalone=context_dataset_class()(cfg.data,OsuParser(cfg,tokenizer),tokenizer,
            subset_ids=[1],test=True,target_ids=[20],cold_start_windows=True)
        expanded=list(standalone)
        history=[r for r in expanded if r['context_window']['kind']=='history']
        cold=[r for r in expanded if r['context_window']['kind']=='cold_start']
        assert len(history)==3 and len(cold)==2
        for original,retained in zip(rows,history):
            assert original['context_window']==retained['context_window']
            for field in ('frames','decoder_input_ids','labels'):
                assert original[field].equal(retained[field]),'Cold augmentation changed a history example'
        for row in cold:
            labels=row['labels'].tolist();active=set();current_column=None
            first_time=None
            for token in labels:
                if tokenizer.event_start[EventType.TIME_SHIFT]<=token<tokenizer.event_end[EventType.TIME_SHIFT]:
                    if first_time is None:first_time=token-tokenizer.event_start[EventType.TIME_SHIFT]
                elif column<=token<tokenizer.event_end[EventType.MANIA_COLUMN]:
                    current_column=token-column
                elif token==tokenizer.event_start[EventType.HOLD_NOTE]:active.add(current_column)
                elif token in (tokenizer.event_start[EventType.HOLD_NOTE_END],tokenizer.event_start[EventType.HOLD_NOTE_SUSTAIN]):
                    assert current_column in active,('Standalone supervision contains an incoming LN without its head',
                        row['context_window'],current_column,token,labels[:120])
                    if token==tokenizer.event_start[EventType.HOLD_NOTE_END]:active.remove(current_column)
            start=row['context_window']['start_ms']
            expected_first=1000 if start==0 else 9000
            assert first_time==int((expected_first-start)/10),'Standalone first attack is not supervised'
            assert row['context_window']['target_ms']==start
            sketch_positions=[i for i,t in enumerate(row['decoder_input_ids'].tolist()) if t==column+lane]
            assert sketch_positions and all(labels[max(0,i-1)]==-100 for i in sketch_positions)
        cold_results.append(cold)
        # Build a real empty-output inference prompt for the same 16 s window.
        # Only the requested target conditions are supplied, never target events.
        inference=InferenceConfig();inference.train=cfg
        processor=Processor(inference,None,tokenizer)
        song_length=24008.
        inputs=processor.get_in_context(in_context=[ContextType.TIMING,ContextType.GD],
            beatmap_path=str(track/'sketch.osu'),song_length=song_length)
        for context in inputs:
            if context['context_type']==ContextType.GD:
                sketch_config=generation_config_from_beatmap(Beatmap.from_path(track/'sketch.osu'),track/'sketch.osu',tokenizer)
                sketch_config.difficulty=1.
                context['class']=processor.get_class_vector(sketch_config,song_length)
        target_config=generation_config_from_beatmap(Beatmap.from_path(track/'reference.osu'),track/'reference.osu',tokenizer)
        target_config.difficulty=1.
        outputs=processor.get_out_context(out_context=[ContextType.MAP],generation_config=target_config,
            given_context=[],beatmap_path=None,song_length=song_length,verbose=False)
        required=['last_sv'] if cfg.data.add_sv_special_token and cfg.data.add_mania_sv else []
        for row in cold:
            start=row['context_window']['start_ms']
            prompt=processor.get_prompt(processor.prepare_context_sequences(inputs,start,False,required),
                processor.prepare_context_sequences(outputs,start,True,required))[0].tolist()
            tokens=row['decoder_input_ids'].tolist()
            differences=[(i,a,b) for i,(a,b) in enumerate(zip(tokens,prompt)) if a!=b]
            assert tokens[:len(prompt)]==prompt,('Training and inference prompt differ',start,differences[:12],len(prompt))
            assert row['labels'][len(prompt)-1]>=tokenizer.event_start[EventType.TIME_SHIFT]
    expected_types={EventType.CIRCLE:2,EventType.HOLD_NOTE:1,EventType.HOLD_NOTE_END:1}
    for kind,count in expected_types.items():
        assert sum(r['labels'].tolist().count(tokenizer.event_start[kind]) for r in results[0])==count
    for left,right in zip(*results):
        assert left['context_window']==right['context_window']
        assert left['labels'].tolist()==right['labels'].tolist(),'Changing input sketch altered target supervision'
        assert left['decoder_input_ids'].tolist()!=right['decoder_input_ids'].tolist(),'Input sketch was ignored'
    for left,right in zip(*cold_results):
        assert left['labels'].equal(right['labels'])
        assert not left['decoder_input_ids'].equal(right['decoder_input_ids'])
    report=dict(passed=True,windows=3,input_only_target_excluded=True,sketch_changes_decoder_input=True,
        sketch_labels_ignored=True,finished_labels_unchanged=True,codec='map-delimited-sketch-v1',
        cold_start_windows=2,cold_start_first_attack_supervised=True,cold_start_orphan_lns_removed=True,
        original_history_unchanged=True,cold_start_inference_prompt_matches_training=True)
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
