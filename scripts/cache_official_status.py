"""Cache per-difficulty official status for Loved sets and ambiguous mirror IDs.

Chart bytes still come from the mirror. This small metadata audit distinguishes
Loved difficulties from non-Loved guests in the same set, without downloading
official chart files. Requests are serial, paced and honor 429 cooldowns.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_ranked_loved import request,save_json


def run(root,extra_roots=()):
    selection=json.loads((root/'set-selection.json').read_text(encoding='utf-8'))
    failures=json.loads((root/'fetch-status.json').read_text(encoding='utf-8')).get('failures',[])
    ids={r['sid'] for r in selection if r['approved']==4}|{r['set_id'] for r in failures}
    for extra in extra_roots:
        for manifest in extra.glob('manifest-*.jsonl'):
            for line in manifest.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                if row.get('approved')==4:ids.add(row['beatmapset_id'])
    ids=sorted(ids)
    errors=[];new=0
    for index,sid in enumerate(ids):
        path=root/'official'/f'{sid}.json'
        if not path.exists():
            try:
                page=request(f'https://osu.ppy.sh/beatmapsets/{sid}').decode('utf-8')
                match=re.search(r'<script[^>]*id="json-beatmapset"[^>]*>(.*?)</script>',page,re.S)
                if not match:raise ValueError('Official status absent')
                data=json.loads(match[1])
                if data.get('id')!=sid:raise ValueError('Official set ID differs')
                save_json(path,data);new+=1
            except (ValueError,OSError) as error:errors.append(dict(set_id=sid,error=str(error)))
            time.sleep(2)
        if index%10==0:print(json.dumps(dict(done=index+1,total=len(ids),new=new,errors=len(errors))),flush=True)
    report=dict(sets=len(ids),new=new,failures=errors)
    save_json(root/'official-status-audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--extra-root',type=Path,action='append',default=[])
    a=p.parse_args();run(a.root,a.extra_root)
