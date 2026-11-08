"""Exercise the actual Windows EXE with an isolated data directory and clean PATH."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manialab import __version__
from manialab.domain import parse_osu, validate
from manialab.timing import grid_report


def synthetic(path, seconds=30):
    rate = 22050
    windows = [(1,3,261.63),(5,9,329.63),(12,13,392.),(20,23,440.),(26,28,523.25)]
    data = bytearray(rate*seconds*2)
    for start,end,pitch in windows:
        for i in range(round(start*rate), min(round(end*rate),rate*seconds)):
            t=i/rate-start
            fade=min(1.,t/.005,(end-i/rate)/.005)
            value=.6*math.sin(2*math.pi*pitch*t)*fade
            struct.pack_into('<h',data,i*2,round(value*32767))
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(rate);stream.writeframes(data)


def run(executable, work):
    executable, work = executable.resolve(), work.resolve()
    work.mkdir(parents=True,exist_ok=True)
    data=work/'独立项目';data.mkdir(exist_ok=True)
    source=work/'音频测试.wav';synthetic(source)
    environment={k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','MANIALAB_DEEP_PYTHON','MANIALAB_DEEP_MODEL')}
    environment['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
    environment.update(HTTP_PROXY='http://127.0.0.1:9',HTTPS_PROXY='http://127.0.0.1:9',NO_PROXY='127.0.0.1,localhost',PYTHONUTF8='1')
    process=None;log=None;url='';token='';checks=[]
    local_http=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start():
        nonlocal process,log,url,token
        marker=data/'current-url.txt';marker.unlink(missing_ok=True)
        log=(work/f'launch-{len(checks)}.log').open('w',encoding='utf-8')
        process=subprocess.Popen([str(executable),'--no-browser','--data-dir',str(data)],cwd=work,
            env=environment,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+60
        while not marker.exists():
            if process.poll() is not None:raise RuntimeError('Packaged process exited; inspect launch log')
            if time.monotonic()>deadline:raise TimeoutError('Packaged startup exceeded 60s')
            time.sleep(.2)
        url=marker.read_text(encoding='utf-8').strip()
        while True:
            if process.poll() is not None:raise RuntimeError('Packaged process exited before HTTP readiness; inspect launch log')
            try:
                html=local_http.open(url,timeout=2).read().decode()
                break
            except (urllib.error.URLError,TimeoutError):
                if time.monotonic()>deadline:raise TimeoutError('Packaged HTTP service did not become ready')
                time.sleep(.2)
        token=re.search(r'name="manialab-token" content="([^"]+)"',html).group(1)

    def stop():
        if process is not None and process.poll() is None:
            process.terminate();process.wait(timeout=15)
        if log:log.close()

    def request(path,payload=None,upload=None):
        headers={};body=None
        if upload:
            field,name,content=upload;boundary='ManiaLabPortableVerificationBoundary'
            body=(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+content+f'\r\n--{boundary}--\r\n'.encode())
            headers['Content-Type']='multipart/form-data; boundary='+boundary
        elif payload is not None:
            body=json.dumps(payload).encode();headers['Content-Type']='application/json'
        if body is not None:headers['X-ManiaLab-Token']=token
        with local_http.open(urllib.request.Request(url+path,data=body,headers=headers),timeout=30) as response:
            raw=response.read()
            return json.loads(raw) if 'application/json' in response.headers.get('Content-Type','') else raw

    def wait(job,timeout=240):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            result=request('/api/jobs/'+job['id'])
            if result['status'] in ('done','error','cancelled'):return result
            time.sleep(.25)
        raise TimeoutError('Packaged job exceeded timeout')

    try:
        release_manifest=json.loads((executable.parent/'release-manifest.json').read_text(encoding='utf-8'))
        assert release_manifest['version']==__version__
        assert release_manifest['executable_sha256']==hashlib.sha256(executable.read_bytes()).hexdigest()
        for name,digest in release_manifest['sources'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f'Packaged source is stale: {name}'
        assert release_manifest['worker_sha256']==hashlib.sha256((executable.parent/'_internal/assets/deep/ManiaLabAudio/ManiaLabAudio.exe').read_bytes()).hexdigest()
        checks.append(dict(check='release_manifest',source_files=len(release_manifest['sources']),executable_sha256=release_manifest['executable_sha256']))
        expected_prior=hashlib.sha256((ROOT/'assets/pattern-prior.json').read_bytes()).hexdigest()
        bundled_prior=hashlib.sha256((executable.parent/'_internal/assets/pattern-prior.json').read_bytes()).hexdigest()
        assert bundled_prior==expected_prior, 'Packaged layout model differs from the source release'
        checks.append(dict(check='bundled_model',sha256=bundled_prior))
        start();status=request('/api/status')
        assert status['version']==__version__ and status['separator']['available'],status
        checks.append(dict(check='startup',version=status['version'],separator=status['separator'],prior=status['pattern_prior']))
        state=request('/api/import',upload=('files',source.name,source.read_bytes()));pid=state['id'];base='/api/projects/'+pid
        job=wait(request(base+'/analyze',dict(engine='deep',settings={},audio_shift_ms=22.5)))
        assert job['status']=='done',job
        state=request(base);raw=state['analysis']['candidates']
        assert len(raw)==5,(len(raw),raw)
        expected=[(1000,3000),(5000,9000),(12000,13000),(20000,23000),(26000,28000)]
        assert all(abs(n['start']-a)<=20 and abs(n['end']-b)<=20 for n,(a,b) in zip(raw,expected))
        checks.append(dict(check='deep_analysis',seconds=state['analysis']['elapsed_seconds'],candidates=len(raw),engine=state['analysis']['engine']))
        for keys in range(6,11):
            payload=dict(mode='audio',bpm=120,offset=-22,division=4,audio_shift_ms=22.5,timing_confirmed=True,settings={'keys':keys,'ln_ratio':.6})
            job=wait(request(base+'/generate',payload));assert job['status']=='done',job
            text=request(base+'/export/osu').decode('utf-8');parsed=parse_osu(text)
            report=validate(parsed['notes'],keys)
            assert report['valid'] and report['notes']==5 and report['ln']==3,report
            grid=grid_report(parsed['notes'],dict(bpm=120,offset=-22,division=4));assert grid['off_grid']==0,grid
            checks.append(dict(check='generate_export',keys=keys,notes=5,ln=3,off_grid=0))
        with zipfile.ZipFile(io.BytesIO(request(base+'/export/osz'))) as archive:
            assert set(archive.namelist())=={'ManiaLab.osu',source.name},archive.namelist()
        audio=request(base+'/audition',dict(mode='chart',start_ms=0,duration_ms=12000));assert audio[:4]==b'RIFF'
        window=request(base+'/waveform?start_ms=900&duration_ms=500');assert window['sample_rate']==22050
        state=request(base);saved_notes=state['variants'][state['active']]['notes']
        stop();start();restored=request(base)
        assert restored['variants'][restored['active']]['notes']==saved_notes
        checks.append(dict(check='restart',preserved=True,unicode_audio=source.name))
        # Both packaged separators must run with no Python or model downloads.
        assert status['separator']['profiles']['six']['available'],status['separator']
        job=wait(request(base+'/analyze',dict(engine='deep6',settings={})),timeout=360)
        assert job['status']=='done',job
        state=request(base)
        assert state['analysis']['separation']['profile']=='six'
        assert {v['voice'] for v in state['analysis']['voices']}=={'drums','bass','vocals','other','guitar','piano'}
        for voice in ('guitar','piano'):
            assert request(base+'/stem/'+voice)[:4]==b'RIFF'
            assert request(base+f'/waveform?voice={voice}&start_ms=900&duration_ms=500')['sample_rate']==22050
        job=wait(request(base+'/generate',dict(mode='audio',bpm=120,offset=-22,division=4,
            audio_shift_ms=22.5,timing_confirmed=True,settings={'keys':10,'guitar_weight':0})))
        assert job['status']=='done',job
        state=request(base);candidate=state['variants'][state['active']]
        assert candidate['report']['grid']['off_grid']==0 and all(n['voice']!='guitar' for n in candidate['notes'])
        checks.append(dict(check='six_stem_analysis_export',seconds=state['analysis']['elapsed_seconds'],
            voices=[v['voice'] for v in state['analysis']['voices']],notes=len(candidate['notes']),off_grid=0))
        # Cancel a fresh separation, then confirm the same server remains usable.
        second=request('/api/import',upload=('files','取消测试.wav',source.read_bytes()+b'\x00\x00'))
        second_base='/api/projects/'+second['id'];job=request(second_base+'/analyze',dict(engine='deep',settings={}))
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            running=request('/api/jobs/'+job['id'])
            if running['status']=='running' and running['progress']>=2:break
            time.sleep(.1)
        request('/api/jobs/'+job['id']+'/cancel',{})
        cancelled=wait(job,30);assert cancelled['status']=='cancelled',cancelled
        assert not request(second_base).get('analysis')
        bad=request('/api/import',upload=('files','bad.wav',b'invalid audio'))
        failure=wait(request('/api/projects/'+bad['id']+'/analyze',dict(engine='basic',settings={})))
        assert failure['status']=='error'
        assert request('/api/status')['version']==__version__
        checks.append(dict(check='cancel_and_failure',cancelled=cancelled['status'],bad_audio=failure['status'],service_alive=True))
    finally:
        stop()
        (work/'report.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('--work',type=Path,default=ROOT/'test-data/portable-alpha4')
    args=parser.parse_args();run(args.executable,args.work)
