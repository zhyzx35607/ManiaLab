"""Run fixed acoustic manifests sequentially with resumable identity checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--python',type=Path,default=Path('test-data/yourmt3/venv/Scripts/python.exe'))
    args=cli.parse_args();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    manifest=read(args.manifest)
    identity=dict(manifest_sha256=sha(args.manifest),transcription_runner_sha256=sha('scripts/research_instrument_transcription.py'),
        audit_runner_sha256=sha('scripts/audit_instrument_releases.py'),phases=[0.,.5],context_seconds=2.1)
    args.output.mkdir(parents=True,exist_ok=True);plan_path=args.output/'plan.json'
    if plan_path.exists():
        if read(plan_path)['identity']!=identity:raise ValueError('Existing run identity differs')
    else:
        plan_path.write_text(json.dumps(dict(identity=identity,manifest=manifest,
            created_utc=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')
    for case in manifest['cases']:
        if sha(case['audio'])!=case['audio_sha256']:raise ValueError('Audio changed: '+case['name'])
        for phase,name in ((0.,'baseline'),(.5,'shifted')):
            output=args.output/case['name']/name;output.parent.mkdir(exist_ok=True)
            print(json.dumps(dict(stage='start',case=case['name'],phase=phase)),flush=True)
            if (output/'report.json').exists():
                recorded=read(output/'report.json')['identity']
                if (recorded['sources']!={case['audio']:case['audio_sha256']} or
                    recorded['start_ms']!=case['start_ms'] or recorded['seconds']!=case['seconds'] or
                    recorded['context_seconds']!=2.1 or recorded['segment_phase']!=phase or
                    recorded['runner_sha256']!=identity['transcription_runner_sha256']):
                    raise ValueError('Cached prediction differs from frozen plan')
            else:
                cmd=[str(args.python),'scripts/research_instrument_transcription.py','--audio',case['audio'],
                    '--start-ms',str(case['start_ms']),'--seconds',str(case['seconds']),
                    '--context-seconds','2.1','--segment-phase',str(phase),'--output',str(output)]
                with (output.parent/(name+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (output.parent/(name+'-audit.log')).open('w',encoding='utf-8') as log:
                subprocess.run([str(args.python),'scripts/audit_instrument_releases.py','--runs',str(output)],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            print(json.dumps(dict(stage='done',case=case['name'],phase=phase)),flush=True)
    (args.output/'completed.json').write_text(json.dumps(dict(identity=identity,
        cases=len(manifest['cases']),passes=2*len(manifest['cases']))),encoding='utf-8')


if __name__=='__main__':main()
