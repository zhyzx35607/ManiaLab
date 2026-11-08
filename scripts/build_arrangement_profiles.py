"""Summarize audited native Ranked/Loved references by keycount and song group.

All difficulties contribute to each song group's median, then each song has
equal influence on published density quantiles. Exact layout duplicates and
asymmetric/scratch layouts are excluded. This produces descriptive calibration,
not a star rating, acoustic labels, or a guarantee of musical quality.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu,validate
from manialab.style import describe_chart
from manialab.timing import BeatClock
from scripts.study_ln_references import beat_duration


def build(roots,output,report_path):
    selected={}
    for root in roots:
        for manifest in sorted(root.glob('manifest-*.jsonl')):
            for line in manifest.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                r=json.loads(line)
                selected.pop(r['beatmap_id'],None)
                if r.get('approved') not in (1,2,4) or not r.get('eligible'):continue
                selected[r['beatmap_id']]=(r,root/'maps'/f"{r['beatmap_id']}.osu")
    by_key=defaultdict(list);seen=set();excluded=Counter();manifest=[]
    for index,(bid,(r,path)) in enumerate(sorted(selected.items())):
        if index%250==0:print(json.dumps(dict(stage='profiles',visited=index,total=len(selected),accepted=len(manifest))),flush=True)
        if r.get('reference_role')=='layout_only':
            excluded['layout_only_reference']+=1;continue
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:
            excluded['hash_mismatch']+=1;continue
        chart=parse_osu(raw.decode('utf-8-sig'));keys=chart['keys']
        style=' '.join(str(r.get(k,'')) for k in ('title','version','tags')).casefold()
        if keys==10 and re.search(r'\b(?:4k?\s*\+\s*6k?|6k?\s*\+\s*4k?)\b',style):
            excluded['asymmetric_10k']+=1;continue
        if keys==8 and re.search(r'\b(?:7k?\s*\+\s*1k?|1k?\s*\+\s*7k?|iidx)\b',style):
            excluded['scratch_8k']+=1;continue
        if not validate(chart['notes'],keys)['valid']:
            excluded['structure']+=1;continue
        fingerprint=hashlib.sha256(json.dumps([keys,sorted((n.start,n.end,n.lane) for n in chart['notes'])]).encode()).hexdigest()
        if fingerprint in seen:
            excluded['duplicate_layout']+=1;continue
        seen.add(fingerprint)
        profile=describe_chart(chart);clock=BeatClock(chart['timing'])
        durations=[beat_duration(n.start,n.end,clock) for n in chart['notes'] if n.hold]
        span=max(n.end for n in chart['notes'])-min(n.start for n in chart['notes'])
        heads={n.start for n in chart['notes']};tails={n.end for n in chart['notes'] if n.hold}
        group=r['song_key']
        if 'various' in r.get('artist','').lower():
            # Metadata-only packs often have different songs as difficulties.
            # Keep them together conservatively instead of inventing identities.
            group='pack:'+str(r['beatmapset_id'])
        metric=dict(active_nps95=profile['active_nps95'],ln_ratio=len(durations)/len(chart['notes']),
            occupancy=sum(n.end-n.start for n in chart['notes'])/max(1,span*keys),
            ln_beats_median=float(np.median(durations)) if durations else 0.,
            independent_release_fraction=len(tails-heads)/max(1,len(tails)))
        style_class='ln_majority' if metric['ln_ratio']>=.5 else 'hybrid' if metric['ln_ratio']>=.15 else 'tap_majority'
        by_key[keys].append(dict(song=group,style_class=style_class,**metric))
        manifest.append(dict(id=bid,keys=keys,sha256=r['sha256'],layout_sha256=fingerprint,song=group,
            status=r['approved'],status_scope=r.get('status_scope','source_metadata'),style_class=style_class,metrics=metric))
    def aggregate(rows):
        songs=defaultdict(list)
        for row in rows:songs[row['song']].append(row)
        metric_names=('active_nps95','ln_ratio','occupancy','ln_beats_median','independent_release_fraction')
        aggregates={metric:[float(np.median([r[metric] for r in group])) for group in songs.values()] for metric in metric_names}
        return dict(charts=len(rows),songs=len(songs),**{metric:{str(q):round(float(np.percentile(values,q)),4) for q in (10,25,50,75,90)} for metric,values in aggregates.items()}) if rows else dict(charts=0,songs=0)
    keys={}
    for key,rows in sorted(by_key.items()):
        keys[str(key)]=aggregate(rows)
        keys[str(key)]['styles']={name:aggregate([r for r in rows if r['style_class']==name]) for name in ('ln_majority','hybrid','tap_majority')}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    payload=dict(revision=1,manifest_sha256=digest,keys=keys,
        weighting='Median within song/pack group, equal weight between groups',
        purpose='Descriptive density and LN-expression calibration only; not audio truth or star ratings')
    output.parent.mkdir(parents=True,exist_ok=True);report_path.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    report_path.write_text(json.dumps(dict(protocol=__doc__,excluded=dict(excluded),manifest=manifest,**payload),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(charts=len(manifest),keys=keys,excluded=dict(excluded))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',type=Path,nargs='+')
    p.add_argument('--output',type=Path,default=Path('assets/arrangement-profiles.json'))
    p.add_argument('--report',type=Path,default=Path('test-data/corpus-expanded/arrangement-report.json'))
    a=p.parse_args();build(a.roots,a.output,a.report)
