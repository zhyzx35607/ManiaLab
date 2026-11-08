"""Fetch public .osu references from a locally saved 10K community table.

Save original charts only under test-data. Selection is frozen before fetching;
every failure and keycount mismatch stays in the manifest. No song audio is
downloaded and no community chart is redistributed with the application.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manialab.domain import parse_osu,validate


def run(table_path,folder):
    table=json.loads(table_path.read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'docs/training-report.json').read_text(encoding='utf-8'))['split_manifest']
    membership={(r['id'],r['sha256']):r['split'] for r in frozen}
    candidates={}
    for row in table['rows']:
        if len(row)<7 or '+' not in row[4]['text'] or not row[5]['text'].isdigit():continue
        for cell in row:
            for link in cell['links']:
                url=urllib.parse.parse_qs(urllib.parse.urlparse(link).query).get('q',[link])[0]
                if urllib.parse.urlparse(url).hostname!='osu.ppy.sh':continue
                match=re.search(r'(?:/beatmaps/|#mania/)(\d+)',url)
                if match:
                    mid=int(match[1]);level=int(row[5]['text'])
                    candidates[mid]=dict(beatmap_id=mid,keys=10,level=level,orientation=row[4]['text'],
                                         label=row[1]['text'],source_url=url)
    selected=[]
    # Fixed skill bands, with stronger noodle orientation first, then SHA of id.
    for low,high in [(3,6),(7,10),(11,14),(15,18),(19,24)]:
        band=[r for r in candidates.values() if low<=r['level']<=high]
        band.sort(key=lambda r:(-len(r['orientation']),hashlib.sha256(str(r['beatmap_id']).encode()).hexdigest()))
        selected.extend(band[:3])
    folder.mkdir(parents=True,exist_ok=True);(folder/'maps').mkdir(exist_ok=True)
    manifest=dict(protocol=__doc__,table_url=table['url'],table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
                  selection=selected,rows=[])
    path=folder/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for entry in selected:
        result=dict(entry)
        try:
            url=f"https://osu.ppy.sh/osu/{entry['beatmap_id']}"
            request=urllib.request.Request(url,headers={'User-Agent':'ManiaLab reference research'})
            with urllib.request.urlopen(request,timeout=30) as response:raw=response.read(4*1024*1024)
            chart=parse_osu(raw.decode('utf-8-sig'))
            if chart['keys']!=10:raise ValueError(f"Expected 10K, received {chart['keys']}K")
            report=validate(chart['notes'],10)
            (folder/'maps'/f"{entry['beatmap_id']}.osu").write_bytes(raw)
            digest=hashlib.sha256(raw).hexdigest()
            result.update(passed=True,sha256=digest,report=report,
                          layout_membership=membership.get((entry['beatmap_id'],digest),'absent_map_may_share_training_song'),
                          duration_ms=max(n.end for n in chart['notes'])-min(n.start for n in chart['notes']))
        except Exception as error:result.update(passed=False,error=str(error))
        manifest['rows'].append(result)
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('label','source_url')}),flush=True)
    valid=[{k:r[k] for k in ('beatmap_id','sha256','keys','layout_membership')} for r in manifest['rows'] if r.get('passed') and r['report']['valid']]
    (folder/'selection.json').write_text(json.dumps(valid,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table',type=Path,default=ROOT/'test-data/ln-design/references/ln-table-0.json')
    p.add_argument('--output',type=Path,default=ROOT/'test-data/ln-reference')
    a=p.parse_args();run(a.table,a.output)
