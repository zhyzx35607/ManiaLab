"""Run full real charts through a local actual EXE, without publishing originals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.domain import parse_osu,validate


def signature(notes):
    return Counter(tuple(n.raw.split(',')[1:]) for n in notes)


def run(url,selection,corpus,output):
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    html=http.open(url).read().decode()
    token=re.search(r'name="manialab-token" content="([^"]+)"',html).group(1)
    rows=[]
    def request(path,payload=None,upload=None):
        body=None;headers={}
        if upload is not None:
            boundary='ManiaLabFullChartVerification'
            body=(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="reference.osu"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+upload+f'\r\n--{boundary}--\r\n'.encode())
            headers['Content-Type']='multipart/form-data; boundary='+boundary
        elif payload is not None:
            body=json.dumps(payload).encode();headers['Content-Type']='application/json'
        if body is not None:headers['X-ManiaLab-Token']=token
        response=http.open(urllib.request.Request(url+path,data=body,headers=headers),timeout=30)
        raw=response.read()
        return json.loads(raw) if 'application/json' in response.headers.get('Content-Type','') else raw
    version=request('/api/status')['version']
    for entry in json.loads(selection.read_text(encoding='utf-8')):
        started=time.perf_counter();result={k:entry[k] for k in ('beatmap_id','keys','sha256','duration','layout_membership','song_group_sha256')}
        try:
            raw=(corpus/'maps'/f"{entry['beatmap_id']}.osu").read_bytes()
            assert hashlib.sha256(raw).hexdigest()==entry['sha256'],'Source SHA mismatch'
            original=parse_osu(raw.decode('utf-8-sig'))
            project=request('/api/import',upload=raw);base='/api/projects/'+project['id']
            job=request(base+'/generate',dict(mode='remap',settings={'keys':entry['keys']}))
            deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                state=request('/api/jobs/'+job['id'])
                if state['status'] in ('done','error','cancelled'):break
                time.sleep(.5)
            else:raise TimeoutError('Full chart layout exceeded ten minutes')
            assert state['status']=='done',state
            exported=request(base+'/export/osu').decode('utf-8');parsed=parse_osu(exported)
            assert signature(parsed['notes'])==signature(original['notes']),'Non-lane object fields changed'
            assert parsed['timing']==original['timing'],'TimingPoints changed'
            report=validate(parsed['notes'],entry['keys']);assert report['valid'],report
            reds=[line for line in original['timing'] if len(line.split(','))>=7 and line.split(',')[6]=='1']
            result.update(passed=True,notes=report['notes'],ln=report['ln'],max_held=report['max_held'],
                red_lines=len(reds),preserved_nonlane_fields=True,preserved_timing=True,seconds=round(time.perf_counter()-started,2))
        except Exception as error:
            result.update(passed=False,error=str(error),seconds=round(time.perf_counter()-started,2))
        rows.append(result)
        output.write_text(json.dumps(dict(version=version,notice='Full-chart structural and preservation verification through actual local EXE. Not player quality evaluation. 8K maps appeared in the frozen training snapshot, but their learning prior is disabled.',
            selection='Deterministic SHA order per key, prefer maps absent from frozen snapshot; 600–6000 objects, >=50 LN, >=90 seconds, distinct known song groups; frozen before running.',
            rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
    if any(not r['passed'] for r in rows):raise SystemExit('Full chart verification failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True);parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,default=Path('test-data/corpus'))
    parser.add_argument('--output',type=Path,default=Path('docs/full-remap-validation.json'))
    args=parser.parse_args();run(args.url,args.selection,args.corpus,args.output)
