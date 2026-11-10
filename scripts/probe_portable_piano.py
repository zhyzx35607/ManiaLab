"""Run the portable piano path on fixed boundary tests and the Nova piano stem."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manialab.piano import infer,decode
from scripts.probe_mapper_adapter import sha


def main():
    exported=Path('test-data/piano-onnx/export-v1')
    verification=json.loads((exported/'runtime-verification.json').read_text())
    model=exported/'piano.onnx'
    if sha(model)!=verification['graph_sha256']:raise ValueError('Verified export changed')
    output=Path('test-data/piano-onnx/portable-v1')
    if output.exists():raise ValueError('Preserve portable runs')
    manifest=json.loads(Path('test-data/smd/selected-v1/benchmark/manifest.json').read_text())
    cases=[dict(case) for i,case in enumerate(manifest['cases']) if i in (0,5,10)]
    raw=Path('test-data/sakura-benchmark/raw-pcm-audit-v1')
    original=json.loads((raw/'verification.json').read_text())
    piano=next(row for row in original['rows'] if row['role']=='piano')
    cases.append(dict(name='sakura-nova-piano',audio=str(raw/'piano.wav'),
                      audio_sha256=piano['raw_float_sha256'],duration_seconds=0))
    output.mkdir(parents=True)
    identity=dict(model_sha256=sha(model),runner_sha256=sha(__file__),runtime_sha256=sha('manialab/piano.py'),
                  cases=cases,scope='SMD indices 0/5/10 and complete original-amplitude Sakura Nova piano; no chart inputs')
    (output/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    reports=[]
    for case in cases:
        if sha(case['audio'])!=case['audio_sha256']:raise ValueError('Audio changed')
        rate,raw=wavfile.read(case['audio'])
        if raw.ndim!=1 or raw.dtype.kind not in ('i','f'):raise ValueError('Invalid piano input')
        samples=raw.astype(np.float32)
        if raw.dtype.kind=='i':samples/=2**(raw.dtype.itemsize*8-1)
        if case['duration_seconds']:samples=samples[:round(case['duration_seconds']*rate)]
        began=time.monotonic()
        last=[-1]
        def progress(percent,message):
            if percent//20!=last[0]:
                print(json.dumps(dict(case=case['name'],percent=percent,message=message),ensure_ascii=True),flush=True)
                last[0]=percent//20
        activations=infer(samples,rate,session,progress)
        notes=decode(activations)
        np.savez_compressed(output/(case['name']+'-activations.npz'),**activations)
        path=output/(case['name']+'-notes.json')
        path.write_text(json.dumps(notes,indent=2),encoding='utf-8')
        report=dict(name=case['name'],notes=len(notes),notes_sha256=sha(path),audio_sha256=case['audio_sha256'],
                    elapsed_seconds=time.monotonic()-began,duration_seconds=len(samples)/rate)
        reports.append(report)
        (output/'progress.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)
    (output/'report.json').write_text(json.dumps(dict(identity=identity,cases=reports,enabled_in_application=False),indent=2),encoding='utf-8')


if __name__=='__main__':main()
