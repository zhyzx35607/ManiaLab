"""Replay actual application stitching with cached windows and independent scores."""
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manialab.piano import OUTPUTS,infer,decode
from scripts.check_expression_current_replay import read,save,sha
from scripts.audit_piano_window_continuity import cardinality


class Replay:
    def __init__(self,pcm,windows):self.pcm,self.windows,self.index=pcm,windows,0
    def run(self,names,inputs):
        expected=np.zeros(160000,np.float32)
        part=self.pcm[self.index*80000:self.index*80000+160000]
        expected[:len(part)]=part
        np.testing.assert_array_equal(inputs['audio'],expected[None])
        value=self.windows[self.index];self.index+=1
        return [value[k][None] for k in names]


def read_windows(folder,rows,*,batched=False):
    windows=[]
    for row in rows:
        path=folder/row.get('path',row.get('file'))
        assert sha(path)==row['sha256']
        with np.load(path,allow_pickle=False) as z:
            windows.append({k:z[k][0] if batched else z[k] for k in OUTPUTS})
    return windows


def run(output):
    output=output.resolve();assert not output.exists()
    paths=('scripts/verify_piano_continuity_runtime.py','manialab/piano.py','manialab/piano_continuity.py',
        'scripts/audit_piano_window_continuity.py')
    sources={p:sha(ROOT/p) for p in paths}
    save(output/'plan.json',dict(sources=sources,neural_inference=False,test_read=False,
        discarded_single_window_frame_is_dummy=True))
    for name in sources:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    results=[];model_windows=0
    def verify(name,pcm,windows,expected):
        nonlocal model_windows
        replay=Replay(pcm,windows);trace=[]
        actual=infer(pcm,16000,replay,continuation_trace=trace)
        assert replay.index==len(windows)
        model_windows+=replay.index
        with np.load(expected/'candidate-activations.npz',allow_pickle=False) as z:
            for k in OUTPUTS:np.testing.assert_array_equal(actual[k],z[k])
        assert trace==read(expected/'trace.json') and decode(actual)==read(expected/'candidate.json')
        return dict(name=name,windows=len(windows),proposals=len(trace),frames=len(actual[OUTPUTS[0]]),
            expected_sha256=sha(expected/'candidate-activations.npz'))
    for directory,evaluation in (('piano-window-smd-v1','piano-window-continuity-regression-v1'),
                                  ('piano-window-sakura-v1','piano-continuity-development-v1')):
        folder=ROOT/'test-data'/directory;report=read(folder/'report.json')
        assert sha(folder/'plan.json')==report['plan_sha256']
        for name,digest in read(folder/'plan.json')['inputs'].items():assert sha(ROOT/name)==digest
        for name in sorted({r['name'] for r in report['windows']}):
            rows=sorted([r for r in report['windows'] if r['name']==name],key=lambda r:r['start_sample'])
            pcm_path=folder/name/'pcm.npy';assert sha(pcm_path)==rows[0]['pcm_sha256']
            results.append(verify(name,np.load(pcm_path,allow_pickle=False),read_windows(folder,rows),ROOT/'test-data'/evaluation/name))
            print(name,'application replay matched',flush=True)
    cache=ROOT/'test-data/piano-onset-head-cache-v1';cached=read(cache/'report.json')
    assert sha(cache/'report.json')=='29a84fe641efbda504cd815dfea777de81ae8452fb6fc3fad1ebb87e36ad0a8f'
    allowed={'449','1046','518','1176','926','1222','1219','628','1134','752'}
    assert {r['id'] for r in cached['records']}==allowed
    development=ROOT/'test-data/piano-continuity-development-v1'
    for record in cached['records']:
        name=record['id'];rows=sorted([r for r in cached['windows'] if r['id']==name],key=lambda r:r['index'])
        pcm_path=cache/(name+'-16k.npy');assert sha(pcm_path)==record['pcm_sha256']
        results.append(verify(name,np.load(pcm_path,allow_pickle=False),read_windows(cache,rows,batched=True),development/name))
        print(name,'application replay matched',flush=True)
    # Original controls are exactly one window; the discarded 1001st output
    # frame was not saved. A dummy value is never decoded or used for repair.
    controls=0
    dataset=ROOT/'test-data/piano-isolated-controls-data-v1';manifest=read(dataset/'manifest.json')
    for directory in ('piano-isolated-controls-run-v1','piano-matched-level-controls-v1'):
        folder=ROOT/'test-data'/directory;report=read(folder/'report.json')
        for file,digest in report['artifacts_sha256'].items():
            if not file.endswith('-activations.npz'):continue
            assert sha(folder/file)==digest
            name=file.removesuffix('-activations.npz')
            if directory=='piano-isolated-controls-run-v1':
                case=next(c for c in manifest['cases'] if c['name']==name)
                wav=dataset/case['file'];assert sha(wav)==case['wav_sha256']
            else:
                wav=folder/(name+'.wav');assert sha(wav)==report['artifacts_sha256'][wav.name]
            rate,pcm=wavfile.read(wav);assert rate==16000 and pcm.shape==(160000,) and pcm.dtype==np.float32
            with np.load(folder/file,allow_pickle=False) as z:values={k:z[k] for k in OUTPUTS}
            padded={k:np.concatenate([v,np.zeros((1,88),v.dtype)]) for k,v in values.items()}
            trace=[];actual=infer(pcm,rate,Replay(pcm,[padded]),continuation_trace=trace)
            assert not trace
            for k in OUTPUTS:np.testing.assert_array_equal(actual[k],values[k])
            assert decode(actual)==decode(values);controls+=1
    assert controls==91
    # Independent maximum-cardinality calculation, not the scorer's Hungarian
    # implementation, for both endpoint semantics and both tolerances.
    report=read(development/'report.json');score_checks=0
    for case in report['cases']:
        name=case['name']
        if name=='sakura':continue
        record=next(r for r in cached['records'] if r['id']==name)
        path=cache/(name+'-labels.json');assert sha(path)==record['labels_sha256'];labels=read(path)
        for variant in ('baseline','candidate'):
            predicted=[n for n in read(development/name/(variant+'.json')) if 0<=n['onset']<170]
            for semantics,field in (('key','notes'),('pedal','sustain_notes')):
                reference=[n for n in labels[field] if 0<=n['onset']<170]
                for mode,strict in (('standard',False),('strict',True)):
                    for kind,offsets in (('onsets',False),('intervals',True)):
                        observed=case['scores'][semantics][variant][mode][kind]
                        assert observed['matched']==cardinality(reference,predicted,strict,offsets)
                        assert observed['reference']==len(reference) and observed['predicted']==len(predicted)
                        score_checks+=1
    assert sources=={p:sha(ROOT/p) for p in sources}
    save(output/'report.json',dict(cases=results,replayed_long_windows=model_windows,
        single_window_controls=controls,independent_development_scores=score_checks,
        neural_inference=False,test_read=False,musical_acceptance=False))
    print('verified',len(results),'recordings,',controls,'controls,',score_checks,'independent scores',flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();run(args.output)
