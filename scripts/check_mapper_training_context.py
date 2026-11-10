"""Integration check with real upstream parser/tokenizer and synthetic long LNs."""
from datetime import datetime
import json
from pathlib import Path
import sys

from mapper_training_context import context_dataset_class


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'test-data/mapperatorinator/source'
    sys.path[:0]=[str(root),str(source/'osuT5'),str(source)]
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf
    from scipy.io import wavfile
    from osuT5.tokenizer import Tokenizer
    from osuT5.dataset.osu_parser import OsuParser
    from osuT5.event import ContextType
    from manialab.domain import Note,export_osu
    cfg=OmegaConf.load(root/'test-data/mapper-training/lora-context/resolved-config.yaml')
    folder=root/'test-data/mapper-context-fixture';track=folder/'data/1';track.mkdir(parents=True,exist_ok=True)
    notes=[Note(1000,26000,0),Note(8183,12000,1),Note(8184,20000,2),Note(16368,16368,3),Note(24551,24551,4),Note(24552,30000,5)]
    (track/'reference.osu').write_text(export_osu(notes,10,root/'vendor/mapperatorinator/template.osu',audio='audio.wav'),encoding='utf-8')
    wavfile.write(track/'audio.wav',16000,np.zeros(32*16000,dtype=np.int16))
    pd.DataFrame([dict(BeatmapSetId=1,Id=1,BeatmapSetFolder='1',BeatmapFile='reference.osu',AudioFile='audio.wav',
        ModeInt=3,Ranked=1,DifficultyRating=1.,StarRating=[1.]*7,SubmittedDate=datetime(2020,1,1),RankedDate=datetime(2020,1,1),UserId=-1)]).to_parquet(folder/'metadata.parquet',index=False)
    cfg.data.test_dataset_path=str(folder);cfg.data.test_dataset_start=0;cfg.data.test_dataset_end=1
    cfg.data.context_types=[{'in':[ContextType.TIMING],'out':[ContextType.MAP]}]
    tokenizer=Tokenizer();tokenizer.load_state_dict(json.loads((root/'test-data/mapperatorinator/model/gamemode=3/tokenizer.json').read_text()))
    starts={t.value:n for t,n in tokenizer.event_start.items()};ends={t.value:n for t,n in tokenizer.event_end.items()}
    dataset=context_dataset_class()(cfg.data,OsuParser(cfg,tokenizer),tokenizer,subset_ids=[1],test=True)
    expected={'hold_note':4,'hold_note_end':4,'circle':2};actual=dict.fromkeys(expected,0);tails=0;windows=[]
    for sample in dataset:
        tokens=sample['decoder_input_ids'].tolist();labels=sample['labels'].tolist();active=set();column=None
        for name in expected:actual[name]+=labels.count(starts[name])
        for i,token in enumerate(tokens):
            if starts['column']<=token<ends['column']:column=token-starts['column']
            elif token in (starts['hold_note'],starts['hold_note_sustain']):active.add(column)
            elif token==starts['hold_note_end']:
                if i>0 and labels[i-1]==token:
                    assert column in active, f'LN tail lacks history in {sample["context_window"]}'
                    tails+=1
                active.discard(column)
        windows.append(sample['context_window'])
    assert actual==expected,(actual,expected)
    assert tails==4 and dataset.audit['over_capacity']==0
    report=dict(passed=True,expected=expected,actual=actual,conditioned_tails=tails,windows=windows,audit=dataset.audit)
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
