"""Run the pinned STRAdi offline violin model against all frozen violin clips.

No app default changes. Retain the author's decoder and raw frame heads; its
artificial duration/retrigger/end-of-input tails are not trusted LN releases.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_instrument_benchmark import window_metrics

COMMIT = '6b9612056b2a31775ef833dc38337ed6040b1f8c'
CHECKPOINT = '2b344f91fe5d49499faf553b982ab5944ee6b9be46de07879ca727fa54f64123'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--root',type=Path,default=Path('test-data/instrument-model-extra-search-2026-11-10'))
    args=cli.parse_args()
    if args.output.exists():raise ValueError('Preserve existing outputs')
    source=args.root/'source-stradi'; checkpoint=args.root/'violin_transcription_offline.pth'
    def git(*a):return subprocess.check_output(['git','-C',str(source),*a],text=True).strip()
    if git('rev-parse','HEAD')!=COMMIT or git('diff','HEAD','--name-only') or sha(checkpoint)!=CHECKPOINT:
        raise ValueError('STRAdi source or weights differ from reviewed identity')
    sources={name:sha(source/name) for name in git('ls-files','*.py','LICENSE').splitlines()}
    for path in ('test-data/piano-release-probe/dependencies','test-data/rosvot-runtime',source):
        sys.path.insert(0,str(Path(path).resolve()))
    import numpy as np
    import torch
    import soundfile as sf
    from violin_transcription.models import offline
    from violin_transcription.inference import predict,load_audio
    from violin_transcription.decoding import output_to_notes
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    allowed=[(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,np.dtypes.Float64DType]
    with torch.serialization.safe_globals(allowed):
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
    config=loaded['config']
    assert config['model_type']=='NonCausalViolinTranscription' and config['classes_num']==58 and config['begin_note']==54
    assert config['sample_rate']==16000 and config['frames_per_second']==100
    state={k[7:] if k.startswith('module.') else k:v for k,v in loaded['model'].items()}
    offline.F0_BINS=state['f0_model.fc.weight'].shape[0]
    model=offline.NonCausalViolinTranscription(frames_per_second=100,classes_num=58).eval()
    model.load_state_dict(state,strict=True)
    model.to('cpu')
    manifest_path=ROOT/'test-data/urmp/comparison-extension/manifest.json'
    baseline_root=ROOT/'test-data/tsumugi-urmp-comparison-v1'
    baseline_path=baseline_root/'comparison.json'
    assert sha(baseline_path)=='ff98a2e3c53671c1bcf4ec16a6c3cdff3215e17e239101dfe6883f884fbe2e75'
    baseline=read(baseline_path)
    assert sha(manifest_path)==baseline['identity']['manifest_sha256']
    cases=[c for c in read(manifest_path)['cases'] if c['kind']=='isolated' and c['instrument']=='vn']
    assert len(cases)==12
    prior={row['name']:row for row in baseline['rows']}
    identity=dict(source_commit=COMMIT,source_files=sources,checkpoint_sha256=CHECKPOINT,
        runner_sha256=sha(__file__),strict_weights=True,checkpoint_config=config,
        model_mode='offline',intra_threads=2,inter_threads=1,thresholds=dict(onset=.5,frame=.5,offset=.5),
        normalization='author_whole_clip_peak',timing_quantization=False,
        versions={p:importlib.metadata.version(p) for p in ('torch','torchlibrosa','librosa','numpy','scipy','soundfile')},
        torchlibrosa_source_sha256=sha(sys.modules['torchlibrosa.stft'].__file__),
        manifest_sha256=sha(manifest_path),baseline_sha256=sha(baseline_path),cases=[c['name'] for c in cases])
    args.output.mkdir(parents=True)
    save(args.output/'plan.json',identity)
    def infer(pcm,folder,source_identity):
        started=time.monotonic()
        heads=predict(model,pcm,torch.device('cpu'))
        assert set(heads)=={'onset','offset','frame','f0'}
        assert all(a.ndim==2 and a.shape[0]==len(pcm)//160+1 and np.isfinite(a).all() for a in heads.values())
        raw=output_to_notes(heads)
        notes=[dict(onset=n['onset'],offset=n['offset'],pitch=n['midi'],program=40,is_drum=False,
                    strength=1.,strength_semantics='unit_not_confidence',
                    release_reason='unresolved_or_postprocessed') for n in raw]
        folder.mkdir()
        np.savez_compressed(folder/'heads.npz',**heads)
        save(folder/'notes.json',notes)
        save(folder/'raw-notes.json',raw)
        report=dict(identity=identity,input=source_identity,samples=len(pcm),sample_rate=16000,
            model_input_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
            heads_sha256=sha(folder/'heads.npz'),notes_sha256=sha(folder/'notes.json'),notes=len(notes),
            seconds=time.monotonic()-started,tail_evidence='decoder_outputs_not_observed_releases')
        save(folder/'report.json',report)
        return notes,report
    _,silence=infer(np.zeros(4*16000,dtype=np.float32),args.output/'silence',dict(control='zero_pcm'))
    print(json.dumps(dict(control='silence',notes=silence['notes'],seconds=silence['seconds'])),flush=True)
    rows=[]
    for index,case in enumerate(cases):
        name=case['name'];old=prior[name];clip=baseline_root/(name+'.wav')
        assert sha(clip)==old['clip_sha256'] and sha(case['labels_path'])==case['labels_sha256']
        original,rate=sf.read(clip,dtype='float32')
        assert rate==16000 and original.ndim==1 and 1<=len(original)/rate<=40
        pcm=load_audio(clip)
        peak=float(np.abs(original).max())
        assert np.array_equal(pcm,original/peak if peak else original)
        notes,report=infer(pcm,args.output/name,dict(audio=str(clip),sha256=sha(clip),
            original_samples=len(original),original_peak=peak,normalization_gain=1/peak if peak else 1))
        metrics,scope=window_metrics(read(case['labels_path']),notes,begin=0,end=30,
                                    context_start=0,context_end=len(original)/rate)
        metrics={key:{m:v for m,v in values.items() if 'instrument' not in m} for key,values in metrics.items()}
        row=dict(name=name,metrics=metrics,tsumugi=old['metrics'],application=old['application'],yourmt3=old['yourmt3'],
            scope=scope,seconds=report['seconds'],report_sha256=sha(args.output/name/'report.json'))
        rows.append(row);save(args.output/name/'metrics.json',row)
        print(json.dumps(dict(index=index+1,name=name,seconds=report['seconds'],
            onset=metrics['standard']['pitched_onsets']['f1'],interval=metrics['standard']['complete_intervals']['f1'])),flush=True)
    totals={}
    for variant in ('application','yourmt3','tsumugi','metrics'):
        totals[variant]={}
        for metric in ('pitched_onsets','complete_intervals'):
            total={k:sum(r[variant]['standard'][metric][k] for r in rows) for k in ('reference','predicted','matched')}
            total['f1']=2*total['matched']/(total['reference']+total['predicted'])
            totals[variant][metric]=total
    assert sha(__file__)==identity['runner_sha256'] and sources=={name:sha(source/name) for name in sources}
    save(args.output/'report.json',dict(identity=identity,rows=rows,totals=totals,seconds=sum(r['seconds'] for r in rows),
        silence_notes=silence['notes'],enabled_in_application=False,musical_acceptance=False))
    print(json.dumps(dict(stage='complete',totals=totals)),flush=True)


if __name__=='__main__':main()
