"""Build hash-verified 6-10K audio/chart MMRS data for upstream training.

Only original mixed audio is paired with a full human chart. Separated stems
are NOT assigned the complete chart as invented instrument-specific labels.
"""
from collections import Counter, defaultdict
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def grouped_split(rows, seed='mapper-audio-20261108'):
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(rows):
        identities=['audio:'+r['audio_sha256'],'set:'+str(r['beatmapset_id']),
                    *['song:'+s for s in r['song_keys']]]
        for key in identities:
            if key in seen:parent[find(i)]=find(seen[key])
            seen[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    ranked=sorted(groups,key=lambda g:hashlib.sha256((seed+min(r['audio_sha256'] for r in groups[g])).encode()).hexdigest())
    selected=set()
    # Explicitly reserve a separate 10K LN song. A validation set without any
    # sustained 10K writing cannot answer this project's primary question.
    targets=[(10,'ln'),(6,None),(7,None),(8,None),(9,None),(10,'tap'),(10,'hybrid')]
    def covers(g,key,band):
        return any(r['keys']==key and (band is None or r['band']==band) for r in groups[g])
    def leaves_training(g):
        remaining=[r for h in groups if h not in selected|{g} for r in groups[h]]
        return set(r['keys'] for r in remaining)=={6,7,8,9,10} and any(r['keys']==10 and r['band']=='ln' for r in remaining)
    for key,band in targets:
        if any(covers(g,key,band) for g in selected):continue
        candidate=next((g for g in ranked if g not in selected and covers(g,key,band) and leaves_training(g)),None)
        if candidate is None:raise ValueError(f'Insufficient independent songs for validation: {key}K {band}')
        selected.add(candidate)
    for g in ranked:
        if len(selected)>=max(len(targets),round(len(groups)*.2)):break
        if g not in selected and leaves_training(g):selected.add(g)
    result=[]
    for g,items in groups.items():
        identity=hashlib.sha256('|'.join(sorted(r['audio_sha256'] for r in items)).encode()).hexdigest()
        result.extend(dict(r,split='validation' if g in selected else 'train',group=identity) for r in items)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,default=Path('test-data/audio-choice-training-final/report.json'))
    p.add_argument('--extra',type=Path,default=Path('test-data/mapper-training-extra'))
    p.add_argument('--snapshot',type=Path,default=Path('test-data/corpus-expanded/snapshot-verified'))
    p.add_argument('--output',type=Path,default=Path('test-data/mapper-training/mmrs'))
    p.add_argument('--ffmpeg',type=Path,required=True)
    args=p.parse_args()
    import pandas as pd
    import rosu_pp_py
    snapshot=[json.loads(l) for l in (args.snapshot/'manifest-snapshot.jsonl').read_text(encoding='utf-8').splitlines()]
    accepted={r['sha256']:r for r in snapshot if r.get('eligible') and r.get('reference_role')=='expression_and_layout'
              and r['keys'] in range(6,11) and (r.get('official_status') in ('ranked','approved','loved') or r.get('approved') in (1,2,4))}
    inputs=[]
    for row in json.loads(args.pairs.read_text(encoding='utf-8'))['manifest']:
        inputs.append((Path(row['source']).parent.parent,row['audio_sha256'],row['snapshot_sha256']))
    for path in sorted(args.extra.glob('*/reference.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        inputs.append((path.parent,r['audio_sha256'],r['chart_sha256']))
    rows=[];seen=set()
    for folder,audio_hash,chart_hash in inputs:
        if chart_hash in seen:continue
        record=accepted.get(chart_hash)
        if not record:raise ValueError('Chart is not in the pinned eligible snapshot')
        chart=args.snapshot/'maps'/f"{record['beatmap_id']}.osu"
        if sha(chart)!=chart_hash:raise ValueError('Pinned chart changed')
        raw=(folder/'reference.osu').read_bytes()
        if raw.replace(b'\r\n',b'\n')!=chart.read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('Paired chart changed')
        audio=next((f for f in folder.iterdir() if f.suffix.lower() in ('.mp3','.wav','.opus','.flac','.ogg') and sha(f)==audio_hash),None)
        if audio is None:raise ValueError('Paired audio changed')
        parsed=parse_osu(raw.decode('utf-8-sig'));ratio=sum(n.hold for n in parsed['notes'])/len(parsed['notes'])
        rows.append(dict(beatmap_id=int(parsed['metadata']['BeatmapID']),beatmapset_id=int(record['beatmapset_id']),
            chart=str(chart.resolve()),chart_sha256=chart_hash,audio=str(audio.resolve()),audio_sha256=audio_hash,
            keys=parsed['keys'],ln_ratio=ratio,band='ln' if ratio>=.5 else 'hybrid' if ratio>=.15 else 'tap',
            song_keys=sorted(set([record['song_key'],*record.get('related_song_keys',[])])),
            status=record.get('official_status') or {1:'ranked',2:'approved',4:'loved'}[record['approved']],
            status_scope='official' if record.get('official_status') else record.get('status_scope','mirror')))
        seen.add(chart_hash)
    rows=grouped_split(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=defaultdict(list)
    for i,r in enumerate(rows):
        # Each chart has its own track folder even when a set contains distinct
        # encodings; grouping for validation remains the original song/set.
        sid=r['beatmap_id'];folder=args.output/r['split']/'data'/str(sid);folder.mkdir(parents=True,exist_ok=True)
        wav=folder/'audio.wav';provenance=folder/'source.json'
        cache=json.loads(provenance.read_text()) if provenance.exists() else {}
        if not wav.exists() or cache.get('source_audio_sha256')!=r['audio_sha256'] or cache.get('decoded_audio_sha256')!=sha(wav):
            subprocess.run([str(args.ffmpeg),'-v','error','-nostdin','-y','-i',r['audio'],
                '-vn','-ar','16000','-ac','1','-c:a','pcm_s16le',str(wav)],check=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        text=Path(r['chart']).read_text(encoding='utf-8-sig')
        text=re.sub(r'(?m)^AudioFilename:.*$', 'AudioFilename: audio.wav',text)
        target=folder/'reference.osu';target.write_text(text,encoding='utf-8')
        b=rosu_pp_py.Beatmap(path=r['chart']);stars=rosu_pp_py.Difficulty().calculate(b).stars
        r.update(prepared_chart_sha256=sha(target),decoded_audio_sha256=sha(wav),difficulty=stars)
        provenance.write_text(json.dumps(dict(source_audio_sha256=r['audio_sha256'],decoded_audio_sha256=r['decoded_audio_sha256'])),encoding='utf-8')
        metadata[r['split']].append(dict(BeatmapSetId=sid,Id=sid,BeatmapSetFolder=str(sid),BeatmapFile='reference.osu',
            AudioFile='audio.wav',ModeInt=3,Ranked={'ranked':1,'approved':2,'loved':4}[r['status']],
            DifficultyRating=stars,StarRating=[stars]*7,SubmittedDate=datetime(1970,1,1),RankedDate=datetime(1970,1,1),UserId=-1))
        print(json.dumps(dict(prepared=i+1,total=len(rows),id=sid,split=r['split'],keys=r['keys'])),flush=True)
    for split,items in metadata.items():pd.DataFrame(items).to_parquet(args.output/split/'metadata.parquet',index=False)
    report=dict(revision=1,seed='mapper-audio-20261108',records=rows,
        counts={s:dict(charts=len(items),groups=len({r['group'] for r in rows if r['split']==s}),
            keys=dict(Counter(r['keys'] for r in rows if r['split']==s)),
            bands=dict(Counter(r['band'] for r in rows if r['split']==s))) for s,items in metadata.items()},
        protocol=['Original mixed audio and full human chart; no invented per-instrument labels.',
            'Unions of related songs, original mapsets, exact source audio hashes stay in one split.',
            'Validation is for this fine-tuning experiment, not an untouched base-model test.',
            'No LN-ratio exclusion; independent 10K LN song required in both splits.',
            'Year and mapper unknown: disable conditioning. StarRating only valid at speed 1: disable speed augmentation.',
            'Original audio clock retained; no fixed 22.5ms correction applied to human chart labels.'])
    (args.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['counts']),flush=True)


if __name__=='__main__':main()
