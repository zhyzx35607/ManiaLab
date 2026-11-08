"""Precompute versioned fixed-rhythm features while chart discovery continues.

This does not train or select a model. The final frozen snapshot still decides
eligibility and grouped splits; cache entries are keyed by content and row seed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from manialab.domain import parse_osu,validate
from scripts.train_layout_network import samples


def prepare_one(item):
    row,path,cache=item;digest=row['sha256'];bid=row['beatmap_id']
    target=cache/(digest+'-'+str(bid)+'-network-ln-v2.npz')
    if target.exists():
        try:
            with np.load(target,allow_pickle=False) as data:
                if data['positive'].shape==data['negative'].shape:return 'cached'
        except (ValueError,OSError,EOFError,zipfile.BadZipFile):pass
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:return 'hash_mismatch'
    parsed=parse_osu(raw.decode('utf-8-sig'))
    if not validate(parsed['notes'],parsed['keys'])['valid']:return 'invalid'
    a,b=samples(parsed,int(bid))
    temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,positive=a,negative=b);temp.replace(target)
    return 'prepared'


def run(args):
    args.cache.mkdir(parents=True,exist_ok=True);records={}
    for root in args.roots:
        if (root/'records').is_dir():
            values=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'records').glob('*.json'))]
        else:values=[json.loads(line) for p in root.glob('manifest-*.jsonl') for line in p.read_text(encoding='utf-8').splitlines() if line]
        for row in values:
            records.pop(row['beatmap_id'],None)
            if row.get('approved') in (1,2,4) and (row.get('eligible') or row.get('suspected_conversion')):
                records[row['beatmap_id']]=(row,root/'maps'/f"{row['beatmap_id']}.osu")
    counts={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index,status in enumerate(pool.map(prepare_one,((row,path,args.cache) for row,path in records.values()))):
            counts[status]=counts.get(status,0)+1
            if index%100==0:print(json.dumps(dict(visited=index+1,eligible=len(records),counts=counts)),flush=True)
    print(json.dumps(dict(eligible=len(records),counts=counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+')
    p.add_argument('--cache',type=Path,default=Path('test-data/corpus-expanded/network-cache'))
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    run(p.parse_args())
