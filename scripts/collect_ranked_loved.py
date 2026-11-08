"""Resume a public, all-language native 6-10K Ranked/Approved/Loved inventory.

The public Sayo mirror provides discovery metadata and chart ZIP ranges. Cached
official metadata can verify individual status and MD5. Raw bytes, hashes,
conversion markers and failures remain auditable. No audio is downloaded.
The mirror's global `results` count is NOT the count of this query. Completion
means exhausting this mirror query, never proving possession of every osu map.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu, validate
from scripts.collect_corpus import song_identity
from scripts.sayo_charts import fetch_charts

QUERY=dict(cmd='beatmaplist',type='search',keyword='',mode=8,**{'class':5},cs=[6,10],limit=100)
_lock=threading.Lock()
_last=0.
_last_page=0.
_cooldown=0.


def request(url, body=None, headers=None, include_headers=False):
    global _last,_last_page,_cooldown
    official_page=url.startswith('https://osu.ppy.sh/beatmapsets/')
    for attempt in range(5):
        try:
            with _lock:
                now=time.monotonic()
                time.sleep(max(0,_last+.4-now,_cooldown-now,(_last_page+1.5-now) if official_page else 0))
                _last=time.monotonic()
                if official_page:_last_page=_last
            req=urllib.request.Request(url,data=json.dumps(body).encode() if body else None,
                headers={'User-Agent':'ManiaLab-public-chart-research','Content-Type':'application/json'}|(headers or {}))
            with urllib.request.urlopen(req,timeout=45) as response:
                raw=response.read(8*1024*1024+1)
                response_headers=dict(response.headers);response_url=response.url
            if len(raw)>8*1024*1024:
                raise ValueError('Unexpectedly large metadata/chart response')
            return (raw,response_headers,response_url) if include_headers else raw
        except (OSError,ValueError) as error:
            if isinstance(error,urllib.error.HTTPError) and error.code==429:
                try:delay=max(30,float(error.headers.get('Retry-After',60)))
                except ValueError:delay=60
                with _lock:_cooldown=max(_cooldown,time.monotonic()+delay)
                print(json.dumps(dict(stage='rate_limit',retry_after=delay)),flush=True)
            if attempt==4:raise
            time.sleep(2*(attempt+1))


def save_json(path, data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def collect(root, workers=2, max_pages=None, source='sayobot', fallback_official=False, include_sets=()):
    for name in ('pages','sets','maps','records','official','mirror'):(root/name).mkdir(parents=True,exist_ok=True)
    config=root/'query.json'
    if config.exists() and json.loads(config.read_text(encoding='utf-8'))!=QUERY:
        raise ValueError('This directory belongs to a different query')
    save_json(config,QUERY)
    cursor,seen,sets=0,set(),{}
    inventory=dict(query=QUERY,complete=False,started=time.time(),pages=0,sets=0)
    try:
        while cursor not in seen:
            seen.add(cursor)
            path=root/'pages'/f'{cursor}.json'
            if not path.exists():
                raw=request('https://api.sayobot.cn/?post',QUERY|dict(offset=cursor))
                data=json.loads(raw)
                if data.get('status')!=0 or not isinstance(data.get('data'),list):
                    raise ValueError('Invalid discovery page')
                save_json(path,data)
            data=json.loads(path.read_text(encoding='utf-8'))
            for row in data['data']:
                if row.get('approved') not in (1,2,4):
                    raise ValueError('Mirror ignored the Ranked/Loved query; refusing unlabelled expansion')
                sets[row['sid']]=row
            inventory.update(pages=len(seen),sets=len(sets),last_cursor=cursor)
            save_json(root/'inventory-status.json',inventory)
            print(json.dumps(dict(stage='inventory',pages=len(seen),sets=len(sets),cursor=cursor)),flush=True)
            if not data['data']:
                inventory['complete']=True
                break
            next_cursor=data.get('endid')
            if next_cursor==0 and len(data['data'])<QUERY['limit']:
                inventory['complete']=True
                inventory['terminal']='short final page with zero endid'
                break
            if not isinstance(next_cursor,int) or next_cursor<=cursor:
                raise ValueError('Discovery cursor stopped advancing; inventory is incomplete')
            cursor=next_cursor
            if max_pages is not None and len(seen)>=max_pages:
                break
    except (OSError,ValueError) as error:
        inventory['error']=str(error)
    inventory['finished']=time.time()
    save_json(root/'inventory-status.json',inventory)
    supplements=[]
    supplement_path=root/'supplement-selection.json'
    remembered=json.loads(supplement_path.read_text(encoding='utf-8')) if supplement_path.exists() else []
    for sid in sorted(set(include_sets)|set(remembered)):
        if sid in sets:continue
        path=root/'official'/f'{sid}.json'
        if not path.exists():raise ValueError('Explicit supplements need cached official status')
        official=json.loads(path.read_text(encoding='utf-8'))
        if official.get('ranked') not in (1,2,4):raise ValueError('Supplement is not currently Ranked/Loved')
        sets[sid]=dict(sid=sid,approved=official['ranked']);supplements.append(sid)
    save_json(supplement_path,supplements)
    save_json(root/'set-selection.json',list(sets.values()))

    previous_by_set={}
    for record_path in (root/'records').glob('*.json'):
        row=json.loads(record_path.read_text(encoding='utf-8'))
        previous_by_set.setdefault(row['beatmapset_id'],[]).append(row)

    def exclude_previous(sid, records, allowed, reason):
        # Retractions must supersede earlier snapshots, not merely disappear
        # from this manifest and silently re-enter through an older source.
        for saved in previous_by_set.get(sid,[]):
            if saved['beatmap_id'] not in allowed:
                record=dict(saved,eligible=False,suspected_conversion=False,
                    approved=-1,exclusion_reason=reason,audit_revision=4)
                save_json(root/'records'/f"{record['beatmap_id']}.json",record)
                records.append(record)

    def fetch_set(entry):
        sid=entry['sid'];path=root/'sets'/f'{sid}.json'
        records,failures=[],[]
        try:
            if not path.exists():
                value=json.loads(request(f'https://api.sayobot.cn/v2/beatmapinfo?K={sid}'))
                if value.get('status')!=0:raise ValueError('Set metadata unavailable')
                save_json(path,value)
            metadata=json.loads(path.read_text(encoding='utf-8'))['data']
            official_path=root/'official'/f'{sid}.json'
            if metadata.get('approved') not in (1,2,4) and not (official_path.exists() and
                    json.loads(official_path.read_text(encoding='utf-8')).get('ranked') in (1,2,4)):
                exclude_previous(sid,records,set(),'set_status_changed')
                return records,[dict(set_id=sid,error='Set status changed; excluded')]
            if not official_path.exists() and source=='official':
                page=request(f'https://osu.ppy.sh/beatmapsets/{sid}').decode('utf-8')
                match=re.search(r'<script[^>]*id="json-beatmapset"[^>]*>(.*?)</script>',page,re.S)
                if not match:raise ValueError('Official beatmap metadata absent')
                official=json.loads(match[1])
                if official.get('id')!=sid:raise ValueError('Official set identity differs')
                save_json(official_path,official)
            verified=official_path.exists()
            official=json.loads(official_path.read_text(encoding='utf-8')) if verified else None
            if metadata['approved']==4 and not verified:
                exclude_previous(sid,records,set(),'loved_difficulty_status_unverified')
                return records,[dict(set_id=sid,error='Loved difficulty status not verified; excluded')]
            difficulties=official['beatmaps'] if official else [dict(id=d['bid'],mode='mania' if d['mode']==3 else 'other',
                cs=d['CS'],status={1:'ranked',2:'approved',4:'loved'}[metadata['approved']],version=d['version']) for d in metadata['bid_data']]
            allowed={d['id'] for d in difficulties if d.get('mode')=='mania' and d.get('cs') in range(6,11)
                and not d.get('convert') and d.get('status') in ('ranked','approved','loved')}
            exclude_previous(sid,records,allowed,'difficulty_no_longer_ranked_loved_or_supported')
            mirror_charts=None;mirror_failure=None
            for diff in difficulties:
                if diff.get('mode')!='mania' or diff.get('cs') not in range(6,11) or diff.get('convert'):continue
                if diff.get('status') not in ('ranked','approved','loved'):continue
                bid=diff['id'];record_path=root/'records'/f'{bid}.json';map_path=root/'maps'/f'{bid}.osu'
                try:
                    saved={}
                    if record_path.exists() and map_path.exists():
                        saved=json.loads(record_path.read_text(encoding='utf-8'))
                        raw=map_path.read_bytes()
                        if saved.get('audit_revision',0)>=4 and saved.get('approved')=={'ranked':1,'approved':2,'loved':4}[diff['status']] and hashlib.sha256(raw).hexdigest()==saved['sha256'] and (not verified or (saved.get('status_scope')=='beatmap' and hashlib.md5(raw).hexdigest()==diff['checksum'])):
                            records.append(saved);continue
                    expected=diff.get('checksum')
                    if verified and not expected:raise ValueError('Official checksum missing')
                    raw=map_path.read_bytes() if map_path.exists() else b''
                    chart_source=saved.get('source') if raw and hashlib.sha256(raw).hexdigest()==saved.get('sha256') else None
                    if chart_source is None:chart_source='local cache; original transfer source unavailable'
                    if not raw or (expected and hashlib.md5(raw).hexdigest()!=expected):
                        if source=='sayobot':
                            try:
                                if mirror_failure is not None:raise mirror_failure
                                if mirror_charts is None:
                                    try:files,receipt=fetch_charts(sid,request)
                                    except (OSError,ValueError,zipfile.BadZipFile,zlib.error) as error:
                                        mirror_failure=error;raise
                                    save_json(root/'mirror'/f'{sid}.json',receipt)
                                    mirror_charts=[]
                                    for filename,content in files:
                                        try:parsed_member=parse_osu(content.decode('utf-8-sig'))
                                        except (ValueError,UnicodeError):continue
                                        mirror_charts.append((filename,content,parsed_member))
                                def normalized(version):return re.sub(r'^\[\d+K\]\s*','',version,flags=re.I).strip()
                                matches=[(filename,content) for filename,content,parsed_member in mirror_charts
                                    if parsed_member['keys']==diff['cs'] and
                                    (hashlib.md5(content).hexdigest()==expected if expected else
                                     normalized(parsed_member['metadata'].get('Version',''))==normalized(diff['version']))]
                                if len(matches)!=1:raise ValueError('Mirror chart identity/checksum is ambiguous or differs')
                                filename,raw=matches[0]
                                chart_source=f'https://txy1.sayobot.cn/beatmaps/download/mini/{sid}#'+filename
                            except (OSError,ValueError,zipfile.BadZipFile,zlib.error):
                                if not fallback_official or not verified:raise
                                chart_source=f'https://osu.ppy.sh/osu/{bid}'
                                raw=request(chart_source)
                        else:
                            chart_source=f'https://osu.ppy.sh/osu/{bid}'
                            raw=request(chart_source)
                        if expected and hashlib.md5(raw).hexdigest()!=expected:
                            raise ValueError('Chart checksum differs from official metadata')
                        map_path.write_bytes(raw)
                    parsed=parse_osu(raw.decode('utf-8-sig'))
                    if parsed['keys']!=diff['cs']:
                        raise ValueError('Official chart keycount differs from official metadata')
                    check=validate(parsed['notes'],parsed['keys'])
                    label=' '.join(parsed['metadata'].get(k,'') for k in ('Title','Version','Tags')).casefold()
                    # These are exclusions for learning authentic expression,
                    # not a judgement of a converted chart's play quality.
                    suspected=any(word in label for word in ('convert','inverse edit','full ln','fln trainer','autogen','rate edit'))
                    fingerprint=hashlib.sha256(json.dumps(sorted((n.start,n.end,n.lane) for n in parsed['notes'])).encode()).hexdigest()
                    song=parsed['metadata']
                    record=dict(beatmap_id=bid,beatmapset_id=sid,keys=parsed['keys'],
                        approved={'ranked':1,'approved':2,'loved':4}[diff['status']],
                        artist=song.get('Artist',''),title=song.get('Title',''),creator=song.get('Creator',''),
                        title_unicode=song.get('TitleUnicode',''),artist_unicode=song.get('ArtistUnicode',''),
                        version=song.get('Version',''),language_id=metadata.get('language'),genre_id=metadata.get('genre'),
                        tags=metadata.get('tags',''),source=chart_source,
                        discovery='https://api.sayobot.cn/?post',last_updated=metadata.get('last_update'),
                        status_scope='beatmap' if verified else 'mirror_beatmapset',audit_revision=4,
                        official_verification_source=f'https://osu.ppy.sh/beatmapsets/{sid}' if verified else None,
                        official_status=diff['status'] if verified else None,
                        official_md5=expected,embedded_beatmap_id=song.get('BeatmapID'),
                        identity_check='official checksum' if verified else 'mirror set, unique difficulty name and keycount',
                        sha256=hashlib.sha256(raw).hexdigest(),layout_sha256=fingerprint,
                        audio_filename=parsed['general'].get('AudioFilename',''),report=check,
                        suspected_conversion=suspected,eligible=check['valid'] and len(parsed['notes'])>=100 and not suspected)
                    record['song_key']=song_identity(record)
                    save_json(record_path,record);records.append(record)
                except (OSError,ValueError,KeyError,zipfile.BadZipFile,zlib.error) as error:
                    failures.append(dict(set_id=sid,beatmap_id=bid,error=str(error)))
        except (OSError,ValueError,KeyError,zipfile.BadZipFile,zlib.error) as error:
            failures.append(dict(set_id=sid,error=str(error)))
        return records,failures
    records,failures=[],[]
    manifest=root/'manifest-ranked-loved.jsonl'
    with manifest.open('w',encoding='utf-8',newline='\n') as stream,ThreadPoolExecutor(max_workers=workers) as pool:
        for index,(batch,failed) in enumerate(pool.map(fetch_set,sets.values())):
            records.extend(batch);failures.extend(failed)
            for row in batch:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            stream.flush()
            if index%10==0 or failed:
                save_json(root/'fetch-status.json',dict(complete=False,sets_done=index+1,sets=len(sets),charts=len(records),failures=failures))
                print(json.dumps(dict(stage='charts',sets_done=index+1,sets=len(sets),charts=len(records),failures=len(failures))),flush=True)
    summary=dict(complete=inventory['complete'] and not failures,inventory_complete=inventory['complete'],
        sets_done=len(sets),supplement_sets=supplements,charts=len(records),eligible=sum(r['eligible'] for r in records),
        keys={str(k):sum(r['keys']==k for r in records) for k in range(6,11)},failures=failures,
        notice='Coverage of this public mirror query only; no claim to include deleted, unavailable or unindexed charts.')
    save_json(root/'fetch-status.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('test-data/corpus-expanded'))
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    p.add_argument('--max-pages',type=int)
    p.add_argument('--source',choices=('sayobot','official'),default='sayobot')
    p.add_argument('--fallback-official',action='store_true',help='Fetch missing or stale mirror charts only when official status/checksum is cached')
    p.add_argument('--include-set',type=int,action='append',default=[],help='Known official Ranked/Loved set absent from the mirror discovery query')
    a=p.parse_args();collect(a.output,a.workers,a.max_pages,a.source,a.fallback_official,a.include_set)
