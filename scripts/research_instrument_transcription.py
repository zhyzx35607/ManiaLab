"""Isolated YourMT3 CPU probe; no chart/timing/section labels enter inference.

Uses the author's frozen CPU Space source and checkpoint configuration. Source
audio and event outputs stay in test-data; no application defaults are changed.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load_sources(paths):
    """Keep original floating PCM and source gains, including peaks above one."""
    import numpy as np
    from scipy.io import wavfile
    signals=[];rate=None;length=None;inputs=[]
    if not paths:raise ValueError('At least one PCM source is required')
    for path in paths:
        sr,y=wavfile.read(path)
        if y.ndim!=1 or not len(y) or sr<=0 or y.dtype not in (np.int16,np.float32):
            raise ValueError('Expected nonempty mono int16 or float32 PCM')
        if rate is not None and (rate!=sr or length!=len(y)):
            raise ValueError('Source clocks differ')
        signal=y.astype(np.float32)/32768 if y.dtype==np.int16 else y.copy()
        if not np.isfinite(signal).all():raise ValueError('Nonfinite PCM')
        inputs.append(dict(path=str(path),file_sha256=sha(path),sample_rate=sr,samples=len(y),
            input_dtype=str(y.dtype),pcm_sha256=hashlib.sha256(signal.astype('<f4').tobytes()).hexdigest(),
            peak=float(np.max(abs(signal))),normalized=False,clipped=False))
        rate=sr;length=len(y);signals.append(signal)
    audio=np.sum(signals,axis=0,dtype=np.float32)
    if not np.isfinite(audio).all():raise ValueError('Nonfinite source sum')
    return rate,audio,inputs


def load_checkpoint(path):
    import numpy as np
    import torch
    from utils.task_manager import TaskManager
    from utils.tokenizer import NoteEventTokenizer
    from utils.event_codec import FastCodec
    from utils.note_event_dataclasses import Event,EventRange
    # Reviewed data containers in the fixed source have no custom unpickling
    # hooks. Keep weights_only enabled and reject any other pickle globals.
    allow=[TaskManager,NoteEventTokenizer,FastCodec,Event,EventRange,np.ndarray,np.dtype,
        (np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),
        (np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),
        type(np.dtype('float64')),type(np.dtype('int64'))]
    with torch.serialization.safe_globals(allow):
        return torch.load(path,map_location='cpu',weights_only=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=Path('test-data/yourmt3/source'))
    cli.add_argument('--model',type=Path,default=Path('test-data/yourmt3/model'))
    cli.add_argument('--audio',type=Path,nargs='+',required=True,help='Synchronized PCM sources to sum without normalization')
    cli.add_argument('--start-ms',type=int,default=215853)
    cli.add_argument('--seconds',type=float,default=12)
    cli.add_argument('--context-seconds',type=float,default=2.1)
    cli.add_argument('--segment-phase',type=float,choices=(0.,.5),default=0.,
        help='Research second pass: shift segmentation by half a model window, retaining physical time')
    cli.add_argument('--threads',type=int,default=2)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.start_ms<0 or args.seconds<=0 or args.context_seconds<0:raise ValueError('Invalid interval')
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.glob('*.json')) or (args.output/'tokens.npz').exists():
        raise ValueError('Preserve previous output; use a new folder')
    shutil.copyfile(__file__,args.output/'runner-snapshot.py')
    rate,audio,inputs=load_sources(args.audio)
    sys.path.insert(0,str((args.source/'amt/src').resolve()))
    os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1')
    import numpy as np
    import torch
    import torchaudio
    from model.ymt3 import YourMT3
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes
    torch.set_num_threads(args.threads);torch.manual_seed(42)
    source=json.loads((args.model/'source.json').read_text())
    checkpoint=args.model/source['path']
    commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    repository=subprocess.check_output(['git','-C',str(args.source),'rev-parse','--show-toplevel'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(args.source),'diff','HEAD','--name-only'],text=True).strip()
    if Path(repository).resolve()!=args.source.resolve() or changes:
        raise ValueError('Expected the unmodified standalone source checkout')
    if commit!=source['revision'] or sha(checkpoint)!=source['sha256']:
        raise ValueError('Source/checkpoint identity changed')
    checkpoint_data=load_checkpoint(checkpoint)
    hparams=checkpoint_data['hyper_parameters']
    hparams['write_output_dir']=None
    model=YourMT3(**hparams).cpu().eval()
    weights=checkpoint_data['state_dict']
    model.load_state_dict(weights,strict=True)
    identity=dict(source_commit=commit,checkpoint_sha256=source['sha256'],runner_sha256=sha(__file__),
        sources={row['path']:row['file_sha256'] for row in inputs},pcm_inputs=inputs,
        sum_pcm_sha256=hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest(),
        start_ms=args.start_ms,seconds=args.seconds,
        context_seconds=args.context_seconds,task=model.task_manager.task_name,
        segment_phase=args.segment_phase,
        decoder_channels=model.task_manager.num_decoding_channels,strict_weights=True)
    print(json.dumps(dict(stage='loaded',identity=identity)),flush=True)
    begin=max(0,round((args.start_ms/1000-args.context_seconds)*rate))
    finish=min(len(audio),round((args.start_ms/1000+args.seconds+args.context_seconds)*rate))
    if finish<=begin:raise ValueError('Requested interval is outside audio')
    samples=torchaudio.functional.resample(torch.from_numpy(audio[begin:finish]),rate,model.audio_cfg['sample_rate'])
    size=model.audio_cfg['input_frames'];phase_samples=int(size*args.segment_phase)
    samples=samples[phase_samples:]
    if not len(samples):raise ValueError('Interval is shorter than segment phase')
    real_start=begin/rate+phase_samples/model.audio_cfg['sample_rate']
    identity['resampled_pcm_sha256']=hashlib.sha256(samples.numpy().astype('<f4').tobytes()).hexdigest()
    identity['resampled_samples']=len(samples)
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    count=(len(samples)+size-1)//size
    segments=torch.nn.functional.pad(samples,(0,count*size-len(samples))).reshape(count,1,size)
    predictions=[];began=time.monotonic()
    with torch.inference_mode():
        for index,segment in enumerate(segments):
            predictions.append(model.inference(segment[None]).cpu().numpy())
            print(json.dumps(dict(stage='inference',segment=index+1,total=count,
                elapsed_seconds=round(time.monotonic()-began,2))),flush=True)
    all_notes=[];errors=Counter();starts=[real_start+size*i/model.audio_cfg['sample_rate'] for i in range(count)]
    for channel in range(model.task_manager.num_decoding_channels):
        tokens=[p[:,channel,:] for p in predictions]
        zipped,_,decode_errors=model.task_manager.detokenize_list_batches(tokens,starts,return_events=True)
        notes,merge_errors=merge_zipped_note_events_and_ties_to_notes(zipped)
        all_notes.append(notes);errors.update(decode_errors);errors.update(merge_errors)
    notes=mix_notes(all_notes)
    rows=[asdict(n) for n in notes]
    (args.output/'notes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    np.savez_compressed(args.output/'tokens.npz',**{str(i):p for i,p in enumerate(predictions)})
    report=dict(identity=identity,elapsed_seconds=time.monotonic()-began,decoded_notes=len(rows),
        notes_sha256=sha(args.output/'notes.json'),tokens_sha256=sha(args.output/'tokens.npz'),
        by_program=dict(Counter('drums' if n.is_drum else str(n.program) for n in notes)),
        decoding_errors=dict(errors),sample_rate=model.audio_cfg['sample_rate'],
        segment_samples=size,context_start_seconds=real_start,context_end_seconds=finish/rate,
        enabled_in_application=False,
        notice='Frozen instrument transcription without authored timing or events. Note selection and LN expression remain separate tasks.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
