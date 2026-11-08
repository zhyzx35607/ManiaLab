"""Isolated All-In-One structure probe using author-supplied Nova stems.

Uses upstream madmom features/model/postprocessing. Only the neighborhood
attention kernel is replaced by a checked CPU tensor implementation. No author
bookmarks, chart notes or timing enter inference. Nova source audio differs
from the model's default HTDemucs preprocessing; results remain experimental.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def package(name,path):
    """Load only needed upstream modules, avoiding unrelated compiled imports."""
    module=types.ModuleType(name);module.__path__=[str(path.resolve())]
    sys.modules[name]=module


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path('test-data/sakura-benchmark'))
    cli.add_argument('--source',type=Path,default=Path('test-data/allin1/source'))
    cli.add_argument('--madmom',type=Path,default=Path('test-data/allin1/madmom-source'))
    cli.add_argument('--output',type=Path,default=Path('test-data/sakura-benchmark/structure-allin1'))
    cli.add_argument('--weights',type=Path,nargs='+',required=True)
    cli.add_argument('--features-from',type=Path,help='Reuse identity-checked upstream features from an earlier run')
    cli.add_argument('--threads',type=int,default=2)
    args=cli.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():raise ValueError('Preserve finished experiments; use a new output folder')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from research_neighborhood_attention import install,check
    torch.set_num_threads(args.threads)
    kernel_check=check();install()
    for name,base in [('allin1',args.source/'src/allin1'),('madmom',args.madmom/'madmom')]:
        package(name,base)
        for sub in (('models','postprocessing') if name=='allin1' else ('audio','io')):
            package(name+'.'+sub,base/sub)
    from madmom.audio.signal import Signal,FramedSignalProcessor
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.audio.spectrogram import FilteredSpectrogramProcessor,LogarithmicSpectrogramProcessor
    from madmom.processors import SequentialProcessor
    from allin1.models.allinone import AllInOne
    from allin1.postprocessing.functional import postprocess_functional_structure
    checkpoint=torch.load(args.weights[0],map_location='cpu',weights_only=True)
    config=OmegaConf.create(checkpoint['config'])
    model=AllInOne(config).eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
    sources={v:args.root/f'bundled/{v}.mp3' for v in ('bass','drums','other','guitar','piano','vocals')}
    manifest=json.loads((args.root/'manifest.json').read_text(encoding='utf-8'))
    expected={a['id']:a['sha256'] for a in manifest['audio']}
    if any(sha(p)!=expected[v] for v,p in sources.items()):raise ValueError('Source identity changed')
    commit=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    identity=dict(source_commit=commit(args.source),madmom_commit=commit(args.madmom),weights_sha256={p.name:sha(p) for p in args.weights},
        runner_sha256=sha(__file__),kernel_sha256=sha(Path(__file__).with_name('research_neighborhood_attention.py')),
        audio_sha256={v:sha(p) for v,p in sources.items()},source_profile='Nova six stems merged into bass/drums/other/vocals')
    print(json.dumps(dict(stage='loaded',identity=identity,kernel_check=kernel_check)),flush=True)
    feature_root=args.features_from or args.output
    cache=feature_root/'spectrogram.npy';cache_meta=feature_root/'spectrogram-identity.json'
    feature_identity=dict(audio_sha256=identity['audio_sha256'],madmom_commit=identity['madmom_commit'],
        sample_rate=44100,fft_size=2048,fps=100,bands=12,fmin=30,fmax=17000)
    begin=time.monotonic()
    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())==feature_identity:
        spec=np.load(cache,allow_pickle=False)
    else:
        if args.features_from:raise ValueError('Requested cached features do not match source identity')
        ffmpeg=next(Path('assets').rglob('ffmpeg*.exe'),None)
        if ffmpeg is None:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        signals={}
        for voice,path in sources.items():
            result=subprocess.run([str(ffmpeg),'-v','error','-nostdin','-i',str(path),'-ac','1','-ar','44100','-f','f32le','pipe:1'],
                capture_output=True,check=True,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            signals[voice]=np.frombuffer(result.stdout,dtype='<f4').copy()
        if len({len(y) for y in signals.values()})!=1:raise ValueError('Unsynchronized stems')
        signals['other']=signals['other']+signals.pop('guitar')+signals.pop('piano')
        processor=SequentialProcessor([FramedSignalProcessor(frame_size=2048,fps=100),ShortTimeFourierTransformProcessor(),
            FilteredSpectrogramProcessor(num_bands=12,fmin=30,fmax=17000,norm_filters=True),
            LogarithmicSpectrogramProcessor(mul=1,add=1)])
        features=[]
        for voice in ('bass','drums','other','vocals'):
            features.append(np.asarray(processor(Signal(signals[voice],sample_rate=44100,num_channels=1))))
            print(json.dumps(dict(stage='features',voice=voice,shape=list(features[-1].shape))),flush=True)
        spec=np.stack(features);np.save(cache,spec,allow_pickle=False)
        cache_meta.write_text(json.dumps(feature_identity,indent=2),encoding='utf-8')
    if spec.ndim!=3 or spec.shape[0]!=4 or spec.shape[2]!=81 or not np.isfinite(spec).all():
        raise ValueError('Unexpected model features')
    names=('logits_beat','logits_downbeat','logits_section','logits_function')
    total={}
    for index,path in enumerate(args.weights):
        if index:
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            other=OmegaConf.create(checkpoint['config'])
            if any(other[k]!=config[k] for k in ('fps','sample_rate','hop_size','min_hops_per_beat')):
                raise ValueError('Ensemble clocks differ')
            model=AllInOne(other).eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
        with torch.inference_mode():output=model(torch.from_numpy(spec).float().unsqueeze(0))
        for name in names:
            value=getattr(output,name)
            if not torch.isfinite(value).all():raise ValueError('Non-finite structure logits')
            total[name]=total.get(name,0)+value
        print(json.dumps(dict(stage='inference',fold=index+1,total=len(args.weights))),flush=True)
    from allin1.typings import AllInOneOutput
    output=AllInOneOutput(**{name:value/len(args.weights) for name,value in total.items()})
    np.savez_compressed(args.output/'logits.npz',**{name:getattr(output,name).numpy() for name in names})
    segments=postprocess_functional_structure(output,config)
    report=dict(identity=identity,kernel_check=kernel_check,elapsed_seconds=time.monotonic()-begin,
        segments=[asdict(s) for s in segments],enabled_in_application=False,
        folds=len(args.weights),ensemble='Mean logits, as in the upstream ensemble',
        notice='Nova input domain; CPU tensor fallback not compared to a compiled kernel. Not accepted as automatic musical truth.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
