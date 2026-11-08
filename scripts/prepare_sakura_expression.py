"""Build one-way Reference -> finished GD pairs for expression development."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, sections
from scripts.prepare_sakura_training import TRAIN_RANGE, VALIDATION_RANGE
from scripts.probe_mapper_adapter import sha


def prepare_chart(text, clear_structure=False):
    parts=sections(text);parts['Editor']=[]
    parts['General']=[re.sub(r'^AudioFilename:.*','AudioFilename: audio.wav',line) for line in parts['General']]
    if clear_structure:
        timing=[]
        for line in parts['TimingPoints']:
            fields=line.split(',')
            if len(fields)>=8:fields[7]=str(int(fields[7])&~1)
            timing.append(','.join(fields))
        parts['TimingPoints']=timing
    return '\n'.join(['\n'.join(parts.get('', ['osu file format v14']))]+
        [f'[{name}]\n'+'\n'.join(body) for name,body in parts.items() if name])+'\n'


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--output',type=Path,default=Path('test-data/sakura-benchmark/expression-mmrs'))
    cli.add_argument('--detected-sketch',type=Path,help='Frozen audio-derived source sketch, replacing the author input only')
    args=cli.parse_args()
    import pandas as pd
    import rosu_pp_py
    manifest=json.loads((args.root/'manifest.json').read_text(encoding='utf-8'))
    texts={}
    for role in ('reference','finished'):
        path=args.root/(role+'.osu')
        if sha(path)!=manifest['charts'][role]['chart_sha256']:raise ValueError('Author chart changed')
        texts[role]=prepare_chart(path.read_text(encoding='utf-8-sig'),clear_structure=role=='reference')
    if args.detected_sketch:
        detected=parse_osu(args.detected_sketch.read_text(encoding='utf-8-sig'))
        if detected['keys']!=10:raise ValueError('Detected source sketch must use the confirmed 10K source representation')
        texts['reference']=prepare_chart(args.detected_sketch.read_text(encoding='utf-8-sig'),clear_structure=True)
    source=args.root/'mapper-6664-26/accompaniment.wav'
    rate,audio=wavfile.read(source)
    if rate!=22050 or audio.ndim!=1:raise ValueError('Accompaniment clock changed')
    audio=resample_poly(audio.astype(np.float32)/32768,320,441)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.json').exists():raise ValueError('Choose a fresh experiment dataset')
    records=[]
    for index,(split,bounds) in enumerate([('train',TRAIN_RANGE),('validation',VALIDATION_RANGE)]):
        sid=900000100+index;folder=args.output/split/'data'/str(sid);folder.mkdir(parents=True,exist_ok=True)
        wavfile.write(folder/'audio.wav',16000,np.round(np.clip(audio,-1,1)*32767).astype(np.int16))
        metadata=[];stars={}
        for role,filename,bid in [('reference','sketch.osu',5969640),('finished','reference.osu',5969639)]:
            path=folder/filename;path.write_text(texts[role],encoding='utf-8')
            stars[role]=rosu_pp_py.Difficulty().calculate(rosu_pp_py.Beatmap(path=str(path))).stars
            metadata.append(dict(BeatmapSetId=sid,Id=bid,BeatmapSetFolder=str(sid),BeatmapFile=filename,
                AudioFile='audio.wav',ModeInt=3,Ranked=0,DifficultyRating=stars[role],StarRating=[stars[role]]*7,
                SubmittedDate=datetime(1970,1,1),RankedDate=datetime(1970,1,1),UserId=-1))
        pd.DataFrame(metadata).to_parquet(args.output/split/'metadata.parquet',index=False)
        notes=parse_osu(texts['finished'])['notes']
        records.append(dict(beatmap_id=sid,target_beatmap_id=5969639,beatmapset_id=2644803,
            group='sakura-grass-development',keys=10,band='expression',source_role='accompaniment',split=split,
            allowed_audio_ranges_ms=[list(bounds)],prepared_chart_sha256=sha(folder/'reference.osu'),
            decoded_audio_sha256=sha(folder/'audio.wav'),input_charts=[dict(file='sketch.osu',sha256=sha(folder/'sketch.osu'))],
            difficulty=stars['finished'],ln_ratio=sum(n.hold for n in notes)/len(notes),source_audio_sha256=sha(source)))
    result=dict(revision=1,records=records,input_contexts=['TIMING','GD'],ranked_statuses=[0],
        context_codec='map-delimited-sketch-v1',
        counts={s:dict(charts=1) for s in ('train','validation')},
        validation_scope='Same-song later-section expression given author onset sketch; not blind audio or independent-song testing.',
        protocol=['Only finished chart is a target. Reference sketch is input-only; never train reverse task.',
            'Input sketch contains source columns/notes and red lines; bookmarks and Kiai flags removed.',
            'Entire audio/history/input-sketch/target window lies within its declared split.',
            'Audio is the author Nova non-vocal accompaniment. Oracle whole-chart difficulty/LN ratio.'])
    if args.detected_sketch:
        result['detected_sketch_sha256']=sha(args.detected_sketch)
        result['validation_scope']='Same-song later-section expression from frozen audio-detected onset sketch; known author Timing, not independent-song testing.'
        result['protocol'][0]='Only finished chart is a target. Detected source sketch is input-only; never train reverse task.'
        result['protocol'][1]='Input sketch contains detected source onsets/layers and supplied author red lines; bookmarks and Kiai flags removed.'
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(records=len(records),validation_scope=result['validation_scope'])),flush=True)


if __name__=='__main__':main()
