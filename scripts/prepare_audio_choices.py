"""Add balanced verified audio/map pairs, including LN, hybrid and tap maps.

One request and one CPU analysis at a time. Sources and clips stay private.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.fetch_paired_maps import fetch
from manialab.audio import analyze_deep,run_hidden
from manialab.arrange import parameters
from manialab.separation import Separator
from manialab.domain import parse_osu
import imageio_ffmpeg


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--seconds',type=int,default=60)
    args=parser.parse_args();root=Path.cwd();snapshot=root/'test-data/corpus-expanded/snapshot-verified'
    records=[json.loads(l) for l in (snapshot/'manifest-snapshot.jsonl').read_text(encoding='utf-8').splitlines()]
    existing={p.parent.name for group in ('audio-choice-pairs','audio-additional-10k','audio-extension2-10k','chinese-ln/audio') for p in (root/'test-data'/group).glob('*/reference.json')}
    eligible=[r for r in records if r.get('eligible') and r.get('reference_role')=='expression_and_layout' and str(r['beatmap_id']) not in existing]
    eligible.sort(key=lambda r:hashlib.sha256(('paired-audio-20261108:'+r['song_key']).encode()).hexdigest())
    chosen=[];songs=set()
    # First cover sustained mapping in each keycount, then hybrid and taps.
    for keys,band in [(k,b) for b in ('ln_majority','hybrid','tap_majority') for k in (10,6,7,8,9)]:
        match=next((r for r in eligible if r['keys']==keys and r.get('ln_class')==band and r['song_key'] not in songs),None)
        if match:
            chosen.append(match);songs.add(match['song_key'])
        if len(chosen)>=args.count:break
    output=root/'test-data/audio-choice-pairs';output.mkdir(exist_ok=True)
    # Resume precisely the first selection; completed caches must not cause a
    # rerun to silently select a different cohort.
    selection=output/'selection.json'
    if selection.exists():chosen=json.loads(selection.read_text(encoding='utf-8'))
    else:selection.write_text(json.dumps(chosen,ensure_ascii=False,indent=2),encoding='utf-8')
    reports=[]
    for index,r in enumerate(chosen):
        folder=output/str(r['beatmap_id']);work=folder/'deep';result=work/'analysis.json'
        try:
            meta=fetch(r,snapshot,output);work.mkdir(exist_ok=True)
            if not result.exists():
                parsed=parse_osu((folder/'reference.osu').read_text(encoding='utf-8-sig'))
                start=max(0,min(n.start for n in parsed['notes'])+12000)
                clip=work/'clip.wav'
                run_hidden([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(folder/meta['audio_filename']),
                    '-ss',str(start/1000),'-t',str(args.seconds),'-ac','2','-ar','44100','-c:a','pcm_s16le',str(clip)],check=True,timeout=120)
                def progress(p,message):
                    if p in (31,46,61,76,94,100):print(json.dumps(dict(index=index+1,total=len(chosen),id=r['beatmap_id'],percent=p)),flush=True)
                a=analyze_deep(clip,work,root/'assets/basic-pitch.onnx',parameters({'keys':r['keys']}),Separator(root),progress)
                a['research_clip']=dict(start_ms=start,end_ms=start+round(a['duration']*1000),excluded_edge_ms=3000,
                    original_audio_sha256=meta['audio_sha256'],chart_sha256=meta['chart_sha256'])
                result.write_text(json.dumps(a,ensure_ascii=False),encoding='utf-8')
            row=dict(id=r['beatmap_id'],keys=r['keys'],ln_class=r['ln_class'],ok=True)
        except Exception as e:row=dict(id=r['beatmap_id'],ok=False,error=str(e))
        reports.append(row);(output/'prepare-report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True);time.sleep(1)


if __name__=='__main__':main()
