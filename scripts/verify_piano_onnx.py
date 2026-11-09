"""Check exported piano probabilities and decoded events in the app runtime."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_mapper_adapter import sha
from manialab.piano import decode


def main():
    root=Path('test-data/piano-onnx/export-v1')
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    for file,key in [('piano.onnx','graph_sha256'),('input.npy','input_sha256'),('torch-reference.npz','reference_sha256')]:
        if sha(root/file)!=report[key]:raise ValueError('Export evidence changed')
    if (root/'runtime-verification.json').exists():raise ValueError('Preserve prior verification')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(root/'piano.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    started=time.monotonic()
    outputs=session.run(report['output_names'],{'audio':np.load(root/'input.npy',allow_pickle=False)[None]})
    elapsed=time.monotonic()-started
    actual=dict(zip(report['output_names'],outputs));rows=[]
    with np.load(root/'torch-reference.npz',allow_pickle=False) as reference:
        for key,value in actual.items():
            delta=abs(value-reference[key])
            np.testing.assert_allclose(value,reference[key],rtol=1e-3,atol=1e-4)
            rows.append(dict(output=key,shape=list(value.shape),maximum_absolute_error=float(delta.max()),
                             mean_absolute_error=float(delta.mean())))
        expected_events=decode({key:reference[key][0,:1000] for key in reference.files})
    events=decode({key:value[0,:1000] for key,value in actual.items()})
    if len(events)!=len(expected_events):raise ValueError('Export changed decoded event inventory')
    keys=('onset','offset','pitch','velocity')
    a=np.asarray([[event[k] for k in keys] for event in events])
    b=np.asarray([[event[k] for k in keys] for event in expected_events])
    np.testing.assert_array_equal(a[:,2:],b[:,2:])
    np.testing.assert_allclose(a[:,:2],b[:,:2],rtol=0,atol=.001)
    if [n['release_reason'] for n in events]!=[n['release_reason'] for n in expected_events]:
        raise ValueError('Export changed release provenance')
    np.savez_compressed(root/'onnx-output.npz',**actual)
    result=dict(graph_sha256=report['graph_sha256'],ort=ort.__version__,elapsed_seconds=elapsed,outputs=rows,
                decoded_events=len(events),max_event_time_error_ms=float(abs(a[:,:2]-b[:,:2]).max()*1000),
                runner_sha256=sha(__file__),enabled_in_application=False)
    (root/'runtime-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
