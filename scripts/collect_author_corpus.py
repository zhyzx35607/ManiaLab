"""Collect public ranked/loved mania charts by author, retaining provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.parse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_corpus import request_bytes,save_sample


def collect(root,authors,pages=3):
    cache=root/'authors';cache.mkdir(parents=True,exist_ok=True)
    existing=[json.loads(line) for p in root.glob('manifest-*.jsonl') for line in p.read_text(encoding='utf-8').splitlines() if line]
    known={r['beatmap_id'] for r in existing};sets={}
    for author in authors:
        for status in ('ranked','loved'):
            for page in range(pages):
                output=cache/f'{author}-{status}-{page}.json'
                try:
                    if not output.exists():output.write_bytes(request_bytes(f'https://osu.ppy.sh/users/{author}/beatmapsets/{status}?limit=100&offset={page*100}'))
                    batch=json.loads(output.read_bytes())
                    if not isinstance(batch,list):raise ValueError('Expected a public beatmapset list')
                    for item in batch:sets[item['id']]=item
                    print(json.dumps(dict(author=author,status=status,page=page,sets=len(batch))),flush=True)
                    if len(batch)<100:break
                except (ValueError,OSError) as error:
                    print(json.dumps(dict(author=author,status=status,error=str(error))),flush=True);break
                time.sleep(.2)
    tasks=[(s,b) for s in sets.values() for b in s.get('beatmaps',[])
        if b.get('mode')=='mania' and 6<=b.get('cs',0)<=10 and b['id'] not in known]
    def fetch(pair):
        s,b=pair;path=cache/f"{b['id']}.osu"
        try:
            if not path.exists():path.write_bytes(request_bytes(f"https://osu.ppy.sh/osu/{b['id']}"))
            raw=path.read_text(encoding='utf-8-sig')
            metadata=dict(content=raw,beatmap_id=b['id'],beatmapset_id=s['id'],mode=3,diff_size=b['cs'],artist=s['artist'],title=s['title'],
                creator=s['creator'],creator_id=s['user_id'],version=b['version'],approved=b.get('ranked'),tags=s.get('tags',''),bpm=b['bpm'])
            return save_sample(dict(beatmaps=[metadata]),str(s['id']),root,f"https://osu.ppy.sh/osu/{b['id']}")
        except (OSError,ValueError) as error:
            print(json.dumps(dict(id=b['id'],error=str(error))),flush=True);return []
    records=[];stamp=hashlib.sha256(','.join(map(str,sorted(authors))).encode()).hexdigest()[:12]
    manifest=root/f'manifest-authors-{stamp}.jsonl'
    if manifest.exists():
        records=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index,result in enumerate(pool.map(fetch,tasks)):
            records.extend(result)
            if index%25==0:
                manifest.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records),encoding='utf-8')
                print(json.dumps(dict(downloaded=index,total=len(tasks),kept=len(records))),flush=True)
    manifest.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records),encoding='utf-8')
    print(json.dumps(dict(completed=True,records=len(records),manifest=str(manifest))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=Path('test-data/corpus'))
    parser.add_argument('--authors',type=int,nargs='*')
    parser.add_argument('--pages',type=int,default=3)
    args=parser.parse_args()
    if args.authors is None:
        records=[json.loads(line) for p in args.folder.glob('manifest-*.jsonl') for line in p.read_text(encoding='utf-8').splitlines() if line]
        authors=sorted({int(r.get('creator_id') or r.get('user_id')) for r in records if r['keys']>=8 and (r.get('creator_id') or r.get('user_id'))})
    else:authors=args.authors
    collect(args.folder,authors,args.pages)
